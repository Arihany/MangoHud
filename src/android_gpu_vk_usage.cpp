#include "android_gpu_vk_usage.h"
#include <vulkan/vulkan.h>
#include <chrono>
#include <mutex>
#include <cmath>
#include <vector>
#include <limits>
#include <cstdint>
#include <spdlog/spdlog.h>
#include <cstdlib>
#include <atomic>
#include <condition_variable>

// ====================== 내부 상태 ======================

struct AndroidVkGpuContext {
    VkPhysicalDevice        phys_dev      = VK_NULL_HANDLE;
    VkDevice                device        = VK_NULL_HANDLE;
    AndroidVkGpuDispatch    disp{};

    float                   ts_period_ns  = 0.0f;   // ns per tick
    uint32_t                ts_valid_bits = 0;
    uint64_t                ts_mask       = ~0ULL;
    std::atomic<bool>       ts_supported{false};

    VkQueryPool             query_pool    = VK_NULL_HANDLE;
    uint32_t                queue_family_index = VK_QUEUE_FAMILY_IGNORED;

    // 링 버퍼 / 슬롯 설정
    static constexpr uint32_t MAX_FRAMES            = 16;   // 슬롯 개수
    static constexpr uint32_t MAX_QUERIES_PER_FRAME = 128;  // 슬롯당 쿼리 개수 (start/end 2개씩 → submit 최대 64개)
    static constexpr uint32_t FRAME_LAG             = 3;    // 몇 프레임 뒤에 슬롯을 읽고 해제할지
    static constexpr uint32_t WARMUP_TIMESTAMP_PAIRS= 2;    // 프레임당 미리 확보할 timestamp pair 수

struct FrameResources {
    VkCommandPool   cmd_pool        = VK_NULL_HANDLE;
    uint32_t        query_start     = 0;
    uint32_t        query_capacity  = 0;
    uint32_t        query_used      = 0;
    bool            has_queries     = false;
    uint64_t        frame_serial    = std::numeric_limits<uint64_t>::max();
    std::vector<VkCommandBuffer> timestamp_cmds;
};

    FrameResources          frames[MAX_FRAMES]{};
    std::atomic<uint64_t>   frame_index{0};
    uint64_t                read_serial = 0;

    // CPU/GPU 사용률 + smoothing용 상태
    std::mutex                                      lock;
    // [LIFETIME GUARD]
    // on_present()가 락 밖에서 GetQueryPoolResults를 호출하는 동안 destroy()가 리소스를 파괴하면 UAF 위험.
    // 그래서 "in_flight" 카운트가 0이 될 때까지 destroy()가 기다리도록 만든다.
    std::condition_variable                         cv;
    bool                                            destroying = false;
    uint32_t                                        in_flight  = 0;
    std::chrono::steady_clock::time_point           last_present{};   // 마지막 present 시각

    std::chrono::steady_clock::time_point           window_start{};   // smoothing 윈도우 시작 시각
    double                                          acc_cpu_ms_sampled = 0.0;
    uint32_t                                        acc_frames_sampled = 0;
    double                                          acc_gpu_ms     = 0.0;   // "회수 성공한" GPU ms만 누적
    double                                          acc_cpu_ms     = 0.0;   // 모든 프레임의 CPU ms 누적
    uint32_t                                        acc_frames     = 0;     // 모든 프레임 카운트
    uint32_t                                        acc_gpu_samples= 0;     // GPU 측정 회수 성공 카운트

    float                                           smooth_gpu_ms = 0.0f;
    float                                           smooth_usage  = 0.0f;

    float                                           last_gpu_ms   = 0.0f;
    float                                           last_usage    = 0.0f;
    bool                                            have_metrics  = false;

    // ===== queue_submit용 scratch 버퍼 (vkQueueSubmit) =====
    std::vector<VkSubmitInfo>           scratch_wrapped;      // 크기: submitCount
    std::vector<VkCommandBuffer>        scratch_cmds;         // 평면화된 pCommandBuffers
    std::vector<uint32_t>               scratch_cb_offsets;   // submit별 offset
    std::vector<uint32_t>               scratch_cb_counts;    // submit별 CB 개수
    std::vector<uint8_t>                scratch_instrument;   // submit별 계측 여부 (0/1)

    // ===== queue_submit2용 scratch 버퍼 (vkQueueSubmit2) =====
    std::vector<VkSubmitInfo2>          scratch_wrapped2;     // 크기: submitCount
    std::vector<VkCommandBufferSubmitInfo> scratch_cmd_infos; // 평면화된 pCommandBufferInfos
    struct CpuSample { uint64_t serial; float ms; };
    static constexpr uint32_t CPU_RING = 64; // 16보다 크게 (NotReady 오래가도 안전)
    CpuSample cpu_ring[CPU_RING]{};

    // CPU_RING은 bitmask 인덱싱을 쓰므로 반드시 2의 거듭제곱이어야 한다.
    static_assert((CPU_RING & (CPU_RING - 1u)) == 0u, "CPU_RING must be power-of-two");
};

// ====================== 환경 변수 플래그 ======================
// 정책: 기본 OFF(fdinfo + kgsl 사용). 오직 MANGOHUD_VKP=1 일 때만 Vulkan timestamp 백엔드 ON.
// 캐시: -1 = 미초기화, 0 = 비활성, 1 = 활성
namespace {
    bool android_gpu_usage_env_enabled()
    {
        static int cached = -1;
        if (cached != -1)
            return cached != 0;

        const char* env = std::getenv("MANGOHUD_VKP");

        // 오직 "1"만 활성. 그 외(unset/빈문자/"0"/"true"/"yes"/뭐든) 전부 비활성.
        const bool enabled = (env && env[0] == '1' && env[1] == '\0');

        cached = enabled ? 1 : 0;

        if (!enabled) {
            SPDLOG_INFO(
                "MANGOHUD_VKP!=1 -> Vulkan GPU usage backend disabled (fdinfo + kgsl remains default)"
            );
            return false;
        }

        SPDLOG_INFO("MANGOHUD_VKP=1 -> Vulkan GPU usage backend enabled");
        return true;
    }
}

// [샘플링 전략] 짝수 프레임만 계측하여 오버헤드를 50%로 줄임
static inline bool
android_gpu_usage_should_sample(const AndroidVkGpuContext* ctx)
{
    if (!ctx)
        return false;
    const uint64_t fi = ctx->frame_index.load(std::memory_order_relaxed);
    return (fi & 1u) == 0u;
}

// ====================== 헬퍼: 타임스탬프 리소스 초기화 ======================

static inline void
android_gpu_usage_consume_slot(AndroidVkGpuContext::FrameResources& fr);

static void
android_gpu_usage_destroy_timestamp_resources(AndroidVkGpuContext* ctx)
{
    if (!ctx || ctx->device == VK_NULL_HANDLE)
        return;

if (ctx->disp.DestroyCommandPool) {
    for (uint32_t i = 0; i < AndroidVkGpuContext::MAX_FRAMES; ++i) {
        auto& fr = ctx->frames[i];
        if (fr.cmd_pool != VK_NULL_HANDLE) {
            ctx->disp.DestroyCommandPool(ctx->device, fr.cmd_pool, nullptr);
            fr.cmd_pool = VK_NULL_HANDLE;
            fr.timestamp_cmds.clear();
        }
        fr.query_used   = 0;
        fr.has_queries  = false;
        fr.frame_serial = std::numeric_limits<uint64_t>::max();
    }
}

    if (ctx->query_pool != VK_NULL_HANDLE && ctx->disp.DestroyQueryPool) {
        ctx->disp.DestroyQueryPool(ctx->device, ctx->query_pool, nullptr);
        ctx->query_pool = VK_NULL_HANDLE;
    }
    ctx->queue_family_index = VK_QUEUE_FAMILY_IGNORED;
}

static bool
android_gpu_usage_init_timestamp_resources(AndroidVkGpuContext* ctx,
                                           uint32_t queue_family_index)
{
    if (!ctx || !ctx->ts_supported.load(std::memory_order_relaxed))
        return false;

    // 이미 초기화되어 있으면 그대로 사용
    if (ctx->query_pool != VK_NULL_HANDLE) {
        return true;
    }

    ctx->queue_family_index = queue_family_index;

    // queue family의 timestampValidBits가 0이면 timestamps 미지원 :contentReference[oaicite:4]{index=4}
    uint32_t qf_count = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(ctx->phys_dev, &qf_count, nullptr);
    if (queue_family_index >= qf_count) {
        SPDLOG_WARN("Android GPU usage: bad queue_family_index={} (qf_count={})", queue_family_index, qf_count);
        ctx->ts_supported.store(false, std::memory_order_relaxed);
        return false;
    }

    std::vector<VkQueueFamilyProperties> qf(qf_count);
    vkGetPhysicalDeviceQueueFamilyProperties(ctx->phys_dev, &qf_count, qf.data());

    const uint32_t vb = qf[queue_family_index].timestampValidBits;
    if (vb == 0) {
        SPDLOG_WARN("Android GPU usage: queue family {} has timestampValidBits=0 -> disabling", queue_family_index);
        ctx->ts_supported.store(false, std::memory_order_relaxed);
        return false;
    }

    // [ACCURACY GUARD] 게임/렌더링 계측 목적이면 그래픽 큐를 우선한다.
    // 그래픽 비트 없는 큐를 첫 init 대상으로 잡아버리면 "프레임" 계측이 삐끗하기 쉽다.
    if ((qf[queue_family_index].queueFlags & VK_QUEUE_GRAPHICS_BIT) == 0) {
        SPDLOG_DEBUG("Android GPU usage: queue family {} is not GRAPHICS -> skip init", queue_family_index);
        return false;
    }

    ctx->ts_valid_bits = vb;
    ctx->ts_mask = (vb >= 64) ? ~0ULL : ((1ULL << vb) - 1ULL);
    
    // 1) CommandPool 관련 함수 포인터부터 체크
    if (!ctx->disp.CreateCommandPool || !ctx->disp.ResetCommandPool) {
        SPDLOG_WARN(
            "Android GPU usage: Create/ResetCommandPool not available → disabling timestamp backend"
        );
        ctx->ts_supported = false;
        return false;
    }

    // 2) QueryPool 생성 (슬롯 × 쿼리 수)
    VkQueryPoolCreateInfo qp{};
    qp.sType              = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
    qp.queryType          = VK_QUERY_TYPE_TIMESTAMP;
    qp.queryCount         = AndroidVkGpuContext::MAX_FRAMES *
                            AndroidVkGpuContext::MAX_QUERIES_PER_FRAME;
    qp.flags              = 0;
    qp.pipelineStatistics = 0;

    if (!ctx->disp.CreateQueryPool ||
        ctx->disp.CreateQueryPool(ctx->device, &qp, nullptr, &ctx->query_pool) != VK_SUCCESS) {
        SPDLOG_WARN(
            "Android GPU usage: CreateQueryPool failed (qcount={}) → disabling timestamp backend",
            qp.queryCount
        );
        ctx->query_pool   = VK_NULL_HANDLE;
        ctx->ts_supported = false;
        return false;
    }

    // 3) 프레임별 커맨드 풀 만들기
    for (uint32_t i = 0; i < AndroidVkGpuContext::MAX_FRAMES; ++i) {
        VkCommandPoolCreateInfo cp{};
        cp.sType            = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        cp.queueFamilyIndex = queue_family_index;
        cp.flags            = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT |
                              VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;

        VkCommandPool pool = VK_NULL_HANDLE;
        if (ctx->disp.CreateCommandPool(ctx->device, &cp, nullptr, &pool) != VK_SUCCESS) {
            SPDLOG_WARN(
                "Android GPU usage: CreateCommandPool failed at slot {} → disabling timestamp backend",
                i
            );
            android_gpu_usage_destroy_timestamp_resources(ctx);
            ctx->ts_supported = false;
            return false;
        }

        auto& fr          = ctx->frames[i];
        fr.cmd_pool       = pool;
        fr.query_start    = i * AndroidVkGpuContext::MAX_QUERIES_PER_FRAME;
        fr.query_capacity = AndroidVkGpuContext::MAX_QUERIES_PER_FRAME;
        fr.query_used     = 0;
        fr.has_queries    = false;
        fr.frame_serial   = std::numeric_limits<uint64_t>::max();
        fr.timestamp_cmds.clear();

        // Warm-up: 프레임당 소량의 timestamp CB pair 미리 확보
        if (ctx->disp.AllocateCommandBuffers &&
            AndroidVkGpuContext::WARMUP_TIMESTAMP_PAIRS > 0) {
            const uint32_t max_pairs =
                AndroidVkGpuContext::MAX_QUERIES_PER_FRAME / 2u;
            uint32_t warm_pairs =
                AndroidVkGpuContext::WARMUP_TIMESTAMP_PAIRS > max_pairs
                    ? max_pairs
                    : AndroidVkGpuContext::WARMUP_TIMESTAMP_PAIRS;

            if (warm_pairs > 0) {
                VkCommandBufferAllocateInfo ai{};
                ai.sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
                ai.commandPool        = fr.cmd_pool;
                ai.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
                ai.commandBufferCount = warm_pairs * 2u;

                std::vector<VkCommandBuffer> tmp(ai.commandBufferCount, VK_NULL_HANDLE);
                if (ctx->disp.AllocateCommandBuffers(
                        ctx->device, &ai, tmp.data()) == VK_SUCCESS) {
                    fr.timestamp_cmds.reserve(ai.commandBufferCount);
                    for (uint32_t j = 0; j < ai.commandBufferCount; ++j)
                        fr.timestamp_cmds.push_back(tmp[j]);
                }
            }
        }
    }

    SPDLOG_INFO(
        "Android GPU usage: timestamp resources initialized (qf_index={} qpool={} slots={} qpf={})",
        queue_family_index,
        static_cast<void*>(ctx->query_pool),
        AndroidVkGpuContext::MAX_FRAMES,
        AndroidVkGpuContext::MAX_QUERIES_PER_FRAME
    );

    return true;
}

static bool
android_gpu_usage_begin_frame(AndroidVkGpuContext* ctx,
                              uint32_t frame_idx,
                              uint64_t frame_serial)
{
    auto& fr = ctx->frames[frame_idx];

    // 같은 프레임 serial이면(한 프레임 내 다중 submit) 계속 사용
    if (fr.frame_serial == frame_serial)
        return true;

    // 아직 결과를 회수 못한 슬롯이면 덮어쓰지 않는다(정확도 핵심)
    // 단, 너무 오래(=링 한 바퀴 이상) 미회수면 계측이 영구 정지될 수 있으니 강제 드랍 안전장치.
    if (fr.has_queries && fr.frame_serial != std::numeric_limits<uint64_t>::max()) {
        uint64_t age = (frame_serial > fr.frame_serial)
            ? (frame_serial - fr.frame_serial)
            : std::numeric_limits<uint64_t>::max();

        if (age < AndroidVkGpuContext::MAX_FRAMES) {
            return false; // 이번 프레임은 계측 포기(패스스루)
        }

        // 너무 오래(NotReady가 링 한 바퀴 이상)면 이 슬롯은 폐기해서 계측이 영구정지 되는 걸 막는다.
        // 다음 사용 시 각 pair에서 vkCmdResetQueryPool을 하기 때문에 재사용 안전성은 유지된다.
        SPDLOG_DEBUG("Android GPU usage: stale slot (serial={} age={}) -> drop slot to avoid stall",
                     fr.frame_serial, age);
        android_gpu_usage_consume_slot(fr);
        return false;
    }

    fr.frame_serial = frame_serial;
    fr.query_used   = 0;
    fr.has_queries  = false;

    // timestamp CB가 하나라도 있으면 커맨드풀 리셋
    if (ctx->disp.ResetCommandPool && fr.cmd_pool != VK_NULL_HANDLE &&
        !fr.timestamp_cmds.empty()) {
        ctx->disp.ResetCommandPool(ctx->device, fr.cmd_pool, 0);
    }

    return true;
}

// submit 하나에 대한 begin/end CB + 쿼리 인덱스 할당 및 녹화
static bool
android_gpu_usage_record_timestamp_pair(AndroidVkGpuContext*            ctx,
                                        AndroidVkGpuContext::FrameResources& fr,
                                        uint32_t&                       out_query_first,
                                        VkCommandBuffer&                out_cmd_begin,
                                        VkCommandBuffer&                out_cmd_end)
{
    if (!ctx->query_pool)
        return false;
    
    if (!ctx->disp.AllocateCommandBuffers ||
        !ctx->disp.BeginCommandBuffer ||
        !ctx->disp.EndCommandBuffer ||
        !ctx->disp.CmdResetQueryPool ||
        !ctx->disp.CmdWriteTimestamp)
        return false;
    // 쿼리 2개(start/end) 남았는지 확인
    if (fr.query_used + 2 > fr.query_capacity)
        return false;

    // 이 submit이 사용하는 쿼리 페어 인덱스
    const uint32_t pair_index = fr.query_used / 2;

    const uint32_t query_first = fr.query_start + fr.query_used;

    // 필요한 CB 슬롯 인덱스 (begin/end)
    const uint32_t begin_idx = pair_index * 2;
    const uint32_t end_idx   = begin_idx + 1;

    // 타임스탬프용 CB 핸들 부족하면 2개씩 추가 할당 (1회성 증가, 상한 = MAX_QUERIES_PER_FRAME)
    if (fr.timestamp_cmds.size() <= end_idx) {
        if (fr.timestamp_cmds.size() + 2 > AndroidVkGpuContext::MAX_QUERIES_PER_FRAME) {
            // 이 프레임은 그냥 계측 포기
            return false;
        }
        VkCommandBufferAllocateInfo ai{};
        ai.sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        ai.commandPool        = fr.cmd_pool;
        ai.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        ai.commandBufferCount = 2;

        VkCommandBuffer pair[2] = { VK_NULL_HANDLE, VK_NULL_HANDLE };
        if (ctx->disp.AllocateCommandBuffers(ctx->device, &ai, pair) != VK_SUCCESS)
            return false;

        fr.timestamp_cmds.push_back(pair[0]);
        fr.timestamp_cmds.push_back(pair[1]);
    }

    out_cmd_begin = fr.timestamp_cmds[begin_idx];
    out_cmd_end   = fr.timestamp_cmds[end_idx];

    VkCommandBufferBeginInfo bi{};
    bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

    // begin CB: (pair-local) reset + start timestamp
    if (ctx->disp.BeginCommandBuffer(out_cmd_begin, &bi) != VK_SUCCESS)
        return false;
    
    // 이 submit이 사용할 2개 쿼리만 reset (프레임 전체 reset_cmd 제거)
    ctx->disp.CmdResetQueryPool(out_cmd_begin, ctx->query_pool, query_first, 2);
    
    ctx->disp.CmdWriteTimestamp(out_cmd_begin,
        VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
        ctx->query_pool,
        query_first);
    
    if (ctx->disp.EndCommandBuffer(out_cmd_begin) != VK_SUCCESS)
        return false;

    // end CB: End timestamp만 기록
    if (ctx->disp.BeginCommandBuffer(out_cmd_end, &bi) != VK_SUCCESS)
        return false;

    ctx->disp.CmdWriteTimestamp(
        out_cmd_end,
        VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
        ctx->query_pool,
        query_first + 1u
    );

    if (ctx->disp.EndCommandBuffer(out_cmd_end) != VK_SUCCESS)
        return false;

    // 여기까지 왔으면 진짜로 쿼리 2개를 쓴 것 → 상태 커밋
    out_query_first   = query_first;
    fr.query_used    += 2;
    fr.has_queries    = true;

    return true;
}

enum class AndroidGpuReadStatus : uint8_t {
    Ready,
    NotReady,
    Error,
    DeviceLost
};

static AndroidGpuReadStatus
android_gpu_usage_query_range_gpu_ms(AndroidVkGpuContext* ctx,
                                     uint32_t query_start,
                                     uint32_t query_count,
                                     float* out_gpu_ms)
{
    if (!ctx || !out_gpu_ms ||
        !ctx->ts_supported.load(std::memory_order_relaxed) ||
        ctx->query_pool == VK_NULL_HANDLE ||
        !ctx->disp.GetQueryPoolResults)
        return AndroidGpuReadStatus::Error;

    // query_count는 (start,end) 페어라서 항상 2의 배수여야 한다.
    // 깨졌다는 건 "아직 준비 안 됨"이 아니라 내부 상태 오염이므로 Error로 보내서 슬롯을 폐기하게 한다.
    if (query_count < 2 || (query_count & 1u))
        return AndroidGpuReadStatus::Error;

    thread_local std::vector<uint64_t> scratch;
    const uint32_t needed_u64 = query_count * 2u;
    if (scratch.size() < needed_u64)
        scratch.resize(needed_u64);

    VkResult r = ctx->disp.GetQueryPoolResults(
        ctx->device,
        ctx->query_pool,
        query_start,
        query_count,
        needed_u64 * sizeof(uint64_t),
        scratch.data(),
        sizeof(uint64_t) * 2,
        VK_QUERY_RESULT_64_BIT |
        VK_QUERY_RESULT_WITH_AVAILABILITY_BIT
    );

    if (r == VK_ERROR_DEVICE_LOST)
        return AndroidGpuReadStatus::DeviceLost;
   if (r == VK_NOT_READY)
       return AndroidGpuReadStatus::NotReady;
    if (r < 0)
        return AndroidGpuReadStatus::Error;

    const uint32_t pair_count = query_count / 2u;

    // 핵심: 하나라도 avail=0이면 NotReady. 부분 결과로 계산하지 않는다.
    for (uint32_t i = 0; i < pair_count; ++i) {
        const uint32_t qs = 2u * i;
        const uint32_t qe = qs + 1u;
        const uint64_t as = scratch[qs * 2u + 1u];
        const uint64_t ae = scratch[qe * 2u + 1u];
        if (!as || !ae)
            return AndroidGpuReadStatus::NotReady;
    }

    uint64_t min_start_ts = std::numeric_limits<uint64_t>::max();
    uint64_t max_end_ts   = 0;
    double   sum_ms       = 0.0;

    for (uint32_t i = 0; i < pair_count; ++i) {
        const uint32_t qs = 2u * i;
        const uint32_t qe = qs + 1u;

        uint64_t start_ts = scratch[qs * 2u + 0u] & ctx->ts_mask;
        uint64_t end_ts   = scratch[qe * 2u + 0u] & ctx->ts_mask;

        if (ctx->ts_valid_bits > 0 && ctx->ts_valid_bits < 64 && end_ts < start_ts)
            end_ts += (1ULL << ctx->ts_valid_bits);

        if (end_ts <= start_ts)
            continue;

        if (start_ts < min_start_ts) min_start_ts = start_ts;
        if (end_ts > max_end_ts)     max_end_ts   = end_ts;

        const uint64_t delta = end_ts - start_ts;
        const double ns = double(delta) * double(ctx->ts_period_ns);
        const double ms = ns * 1e-6;
        if (ms > 0.0 && std::isfinite(ms))
            sum_ms += ms;
    }

    double range_ms = 0.0;
    if (max_end_ts > min_start_ts && min_start_ts != std::numeric_limits<uint64_t>::max()) {
        const uint64_t range_ticks = max_end_ts - min_start_ts;
        range_ms = double(range_ticks) * double(ctx->ts_period_ns) * 1e-6;
    }

    // GPU usage(바쁨) 용도면 sum_ms가 더 정직하다.
    double gpu_ms = 0.0;
    if (sum_ms > 0.0 && std::isfinite(sum_ms)) gpu_ms = sum_ms;
    else if (range_ms > 0.0 && std::isfinite(range_ms)) gpu_ms = range_ms;

    if (!(gpu_ms > 0.0) || !std::isfinite(gpu_ms))
        return AndroidGpuReadStatus::Error;

    *out_gpu_ms = (float)gpu_ms;
    return AndroidGpuReadStatus::Ready;
}

static inline void
android_gpu_usage_consume_slot(AndroidVkGpuContext::FrameResources& fr)
{
    fr.has_queries  = false;
    fr.query_used   = 0;
    fr.frame_serial = std::numeric_limits<uint64_t>::max();
}

static float
android_gpu_usage_collect_frame_gpu_ms(AndroidVkGpuContext* ctx,
                                       AndroidVkGpuContext::FrameResources& fr)
{
    if (!ctx ||
        !ctx->ts_supported.load(std::memory_order_relaxed) ||
        ctx->query_pool == VK_NULL_HANDLE)
        return 0.0f;

    if (!fr.has_queries || fr.query_used < 2)
        return 0.0f;

    float gpu_ms = 0.0f;
    const auto st = android_gpu_usage_query_range_gpu_ms(
        ctx, fr.query_start, fr.query_used, &gpu_ms
    );

    if (st == AndroidGpuReadStatus::DeviceLost) {
        SPDLOG_WARN("Android GPU usage: DEVICE_LOST on GetQueryPoolResults, disabling timestamp backend");
        ctx->ts_supported = false;
        android_gpu_usage_consume_slot(fr);
        android_gpu_usage_destroy_timestamp_resources(ctx);
        return 0.0f;
    }

    if (st == AndroidGpuReadStatus::Error) {
        // 오류면 슬롯을 붙잡고 있으면 영구 정지될 수 있으니 소비하고 버린다.
        android_gpu_usage_consume_slot(fr);
        return 0.0f;
    }

    if (st == AndroidGpuReadStatus::NotReady) {
        // 핵심: 미준비면 슬롯 유지
        return 0.0f;
    }

    // Ready
    android_gpu_usage_consume_slot(fr);
    return gpu_ms;
}

// ====================== 외부 API ======================

AndroidVkGpuContext*
android_gpu_usage_create(VkPhysicalDevice            phys_dev,
                         VkDevice                    device,
                         float                       timestamp_period_ns,
                         uint32_t                    timestamp_valid_bits,
                         const AndroidVkGpuDispatch& disp)
{
    auto ctx = new AndroidVkGpuContext{};
    ctx->phys_dev      = phys_dev;
    ctx->device        = device;
    ctx->disp          = disp;
    ctx->ts_period_ns  = (timestamp_period_ns > 0.0f) ? timestamp_period_ns : 0.0f;
    ctx->ts_valid_bits = timestamp_valid_bits;

    // 정책: 기본 OFF. MANGOHUD_VKP=1일 때만 Vulkan 경로 활성.
    if (!android_gpu_usage_env_enabled()) {
        ctx->ts_supported = false;
        SPDLOG_INFO(
            "Android GPU usage: backend disabled (MANGOHUD_VKP!=1), context will be no-op"
        );
        return ctx;
    }

    if (ctx->ts_valid_bits == 0 || ctx->ts_valid_bits >= 64) {
        ctx->ts_mask = ~0ULL;
    } else {
        ctx->ts_mask = (1ULL << ctx->ts_valid_bits) - 1ULL;
    }
    ctx->last_present  = std::chrono::steady_clock::now();
    ctx->last_gpu_ms   = 0.0f;
    ctx->last_usage    = 0.0f;

    bool has_submit =
        (disp.QueueSubmit  != nullptr) ||
        (disp.QueueSubmit2 != nullptr) ||
        (disp.QueueSubmit2KHR != nullptr);

    bool dispatch_ok =
        has_submit &&
        disp.CreateQueryPool &&
        disp.DestroyQueryPool &&
        disp.GetQueryPoolResults &&
        disp.CreateCommandPool &&
        disp.DestroyCommandPool &&
        disp.ResetCommandPool &&
        disp.AllocateCommandBuffers &&
        disp.BeginCommandBuffer &&
        disp.EndCommandBuffer &&
        disp.CmdWriteTimestamp &&
        disp.CmdResetQueryPool;

    ctx->ts_supported =
        (ctx->ts_period_ns > 0.0f) &&
        dispatch_ok; // timestampValidBits는 init에서 queue family 기준으로 재확인 :contentReference[oaicite:3]{index=3}

    SPDLOG_INFO(
        "Android GPU usage: create ctx={} ts_period_ns={} ts_valid_bits={} dispatch_ok={} ts_supported={}",
        static_cast<void*>(ctx),
        ctx->ts_period_ns,
        ctx->ts_valid_bits,
        dispatch_ok,
        ctx->ts_supported
    );

    if (!ctx->ts_supported.load(std::memory_order_relaxed)) {
        SPDLOG_WARN("Android GPU usage: Vulkan timestamps not supported, backend will be disabled");
    }

    // 약한 기기용: 흔한 submit 규모만큼 미리 reserve 해서 realloc 스파이크 제거
    ctx->scratch_wrapped.reserve(32);
    ctx->scratch_cmds.reserve(512);
    ctx->scratch_cb_offsets.reserve(32);
    ctx->scratch_cb_counts.reserve(32);
    ctx->scratch_instrument.reserve(32);

    ctx->scratch_wrapped2.reserve(32);
    ctx->scratch_cmd_infos.reserve(512);

    return ctx;
}

void
android_gpu_usage_destroy(AndroidVkGpuContext* ctx)
{
    if (!ctx)
        return;

   {
       // on_present()가 락 밖에서 Vulkan 호출 중일 수 있으니, 그게 끝날 때까지 기다린다.
       std::unique_lock<std::mutex> lk(ctx->lock);
       ctx->destroying = true;
       ctx->cv.wait(lk, [&] { return ctx->in_flight == 0; });

       android_gpu_usage_destroy_timestamp_resources(ctx);
       ctx->have_metrics = false;
   }
    delete ctx;
}

// ====================== QueueSubmit(v1) 래핑 ======================

VkResult
android_gpu_usage_queue_submit(AndroidVkGpuContext* ctx,
                               VkQueue              queue,
                               uint32_t             queue_family_index,
                               uint32_t             submitCount,
                               const VkSubmitInfo*  pSubmits,
                               VkFence              fence)
{
    if (!ctx || !ctx->disp.QueueSubmit || !pSubmits || submitCount == 0)
        return (ctx && ctx->disp.QueueSubmit)
            ? ctx->disp.QueueSubmit(queue, submitCount, pSubmits, fence)
            : VK_ERROR_INITIALIZATION_FAILED;

    if (!android_gpu_usage_env_enabled())
        return ctx->disp.QueueSubmit(queue, submitCount, pSubmits, fence);

    if (!ctx->ts_supported.load(std::memory_order_relaxed))
        return ctx->disp.QueueSubmit(queue, submitCount, pSubmits, fence);

    // fast-path: 샘플링 안 하는 프레임이면 락을 아예 안 잡는다
    if (!android_gpu_usage_should_sample(ctx))
        return ctx->disp.QueueSubmit(queue, submitCount, pSubmits, fence);

    // [ROLLBACK GUARD] 예외로 빠져도 슬롯이 오염(제출 안 된 query)되지 않게 복구한다.
    AndroidVkGpuContext::FrameResources* fr_ptr = nullptr;
    uint64_t serial_snapshot = 0;
    uint32_t saved_query_used = 0;
    bool     saved_has_queries = false;

    try {
        std::lock_guard<std::mutex> g(ctx->lock);

        // 락 잡은 뒤에도 다시 체크(경계 상황)
        if (!ctx->ts_supported.load(std::memory_order_relaxed))
            return ctx->disp.QueueSubmit(queue, submitCount, pSubmits, fence);

        const uint64_t frame_serial = ctx->frame_index.load(std::memory_order_relaxed);
        if ((frame_serial & 1u) != 0u) // 짝수 serial만 샘플
            return ctx->disp.QueueSubmit(queue, submitCount, pSubmits, fence);

        if (!android_gpu_usage_init_timestamp_resources(ctx, queue_family_index))
            return ctx->disp.QueueSubmit(queue, submitCount, pSubmits, fence);

        // init 된 큐 패밀리랑 다르면 계측 안 함
        if (ctx->query_pool != VK_NULL_HANDLE &&
            ctx->queue_family_index != VK_QUEUE_FAMILY_IGNORED &&
            ctx->queue_family_index != queue_family_index) {
            return ctx->disp.QueueSubmit(queue, submitCount, pSubmits, fence);
        }

        const uint32_t curr_idx = (uint32_t)(frame_serial % AndroidVkGpuContext::MAX_FRAMES);
        serial_snapshot = frame_serial;
        fr_ptr = &ctx->frames[curr_idx];
        auto& fr = *fr_ptr;

        if (!android_gpu_usage_begin_frame(ctx, curr_idx, frame_serial))
            return ctx->disp.QueueSubmit(queue, submitCount, pSubmits, fence);

        saved_query_used  = fr.query_used;
        saved_has_queries = fr.has_queries;
        
        auto& wrapped  = ctx->scratch_wrapped;
        auto& flat_cbs = ctx->scratch_cmds;
        auto& offsets  = ctx->scratch_cb_offsets;
        auto& counts   = ctx->scratch_cb_counts;
        auto& inst     = ctx->scratch_instrument;

        const uint32_t n = submitCount;

        wrapped.resize(n);
        offsets.resize(n);
        counts.resize(n);
        inst.resize(n);

        uint32_t total_cmds = 0;
        uint32_t tmp_used   = fr.query_used;
        bool any_inst = false;

        constexpr uint32_t MAX_PAIRS_PER_SAMPLED_FRAME = 16;
        const uint32_t pair_budget =
            std::min<uint32_t>(MAX_PAIRS_PER_SAMPLED_FRAME, fr.query_capacity / 2u);

        uint32_t planned_pairs = 0;

        for (uint32_t i = 0; i < n; ++i) {
            const VkSubmitInfo& src = pSubmits[i];
            const uint32_t base = src.commandBufferCount;

            const bool basic_ok = (base > 0) && src.pCommandBuffers && (fr.cmd_pool != VK_NULL_HANDLE);
            const bool room_ok  = (planned_pairs < pair_budget) &&
                                  (fr.query_used + (planned_pairs + 1u) * 2u <= fr.query_capacity);

            const bool can = basic_ok && room_ok;

            if (!can) {
                inst[i]   = 0;
                counts[i] = base;
            } else {
                inst[i]   = 1;
                any_inst  = true;
                planned_pairs += 1;
                counts[i] = base + 2; // begin + end
            }
            total_cmds += counts[i];
        }

        if (!any_inst)
            return ctx->disp.QueueSubmit(queue, submitCount, pSubmits, fence);

        if (flat_cbs.size() < total_cmds)
            flat_cbs.resize(total_cmds);

        uint32_t cursor = 0;
        for (uint32_t i = 0; i < n; ++i) {
            offsets[i] = cursor;
            cursor += counts[i];
        }

        for (uint32_t i = 0; i < n; ++i) {
            const VkSubmitInfo& src = pSubmits[i];

            if (!inst[i]) {
                wrapped[i] = src;
                continue;
            }

            VkCommandBuffer* dst_bufs = flat_cbs.data() + offsets[i];
            uint32_t idx_cb = 0;

            VkCommandBuffer cmd_begin = VK_NULL_HANDLE;
            VkCommandBuffer cmd_end   = VK_NULL_HANDLE;
            uint32_t query_first = 0;

            if (!android_gpu_usage_record_timestamp_pair(ctx, fr, query_first, cmd_begin, cmd_end)) {
                // 실패: 이 submit은 계측 포기하고 원본만
                for (uint32_t j = 0; j < src.commandBufferCount; ++j)
                    dst_bufs[idx_cb++] = src.pCommandBuffers[j];

                VkSubmitInfo dst = src;
                dst.commandBufferCount = idx_cb;
                dst.pCommandBuffers    = dst_bufs;
                wrapped[i] = dst;
                inst[i] = 0;
                continue;
            }

            dst_bufs[idx_cb++] = cmd_begin;
            for (uint32_t j = 0; j < src.commandBufferCount; ++j)
                dst_bufs[idx_cb++] = src.pCommandBuffers[j];
            dst_bufs[idx_cb++] = cmd_end;

            VkSubmitInfo dst = src;
            dst.commandBufferCount = idx_cb;
            dst.pCommandBuffers    = dst_bufs;
            wrapped[i] = dst;
        }

        VkResult vr = ctx->disp.QueueSubmit(queue, submitCount, wrapped.data(), fence);
        if (vr != VK_SUCCESS) {
            // submit 실패면 이번에 커밋한 query 사용분은 무효 처리 (측정 오염 방지)
            fr.query_used  = saved_query_used;
            fr.has_queries = saved_has_queries;

            if (vr == VK_ERROR_DEVICE_LOST) {
                ctx->ts_supported.store(false, std::memory_order_relaxed);
                android_gpu_usage_destroy_timestamp_resources(ctx);
            }
        }
        return vr;
    } catch (const std::exception& e) {
        SPDLOG_WARN("Android GPU usage: queue_submit exception: {}", e.what());
        // [ROLLBACK GUARD] 예외로 빠졌으면, 계측 슬롯 상태를 가능한 한 원복한다.
        if (ctx && fr_ptr) {
            std::lock_guard<std::mutex> g(ctx->lock);
            // 같은 serial 슬롯일 때만 되돌린다 (레이스/프레임 진행 중 오염 방지)
            if (!ctx->destroying && fr_ptr->frame_serial == serial_snapshot) {
                fr_ptr->query_used  = saved_query_used;
                fr_ptr->has_queries = saved_has_queries;
            }
        }
        return ctx->disp.QueueSubmit(queue, submitCount, pSubmits, fence);
    } catch (...) {
        SPDLOG_WARN("Android GPU usage: queue_submit unknown exception");
        if (ctx && fr_ptr) {
            std::lock_guard<std::mutex> g(ctx->lock);
            if (!ctx->destroying && fr_ptr->frame_serial == serial_snapshot) {
                fr_ptr->query_used  = saved_query_used;
                fr_ptr->has_queries = saved_has_queries;
            }
        }
        return ctx->disp.QueueSubmit(queue, submitCount, pSubmits, fence);
    }
}

// ====================== QueueSubmit2(v2) 래핑 ======================

VkResult
android_gpu_usage_queue_submit2(AndroidVkGpuContext* ctx,
                                VkQueue              queue,
                                uint32_t             queue_family_index,
                                uint32_t             submitCount,
                                const VkSubmitInfo2* pSubmits,
                                VkFence              fence)
{
    PFN_vkQueueSubmit2 fpSubmit2 = nullptr;
    if (ctx) {
        fpSubmit2 = ctx->disp.QueueSubmit2 ? ctx->disp.QueueSubmit2
                 : ctx->disp.QueueSubmit2KHR ? ctx->disp.QueueSubmit2KHR
                 : nullptr;
    }

    if (!ctx || !fpSubmit2 || !pSubmits || submitCount == 0)
        return fpSubmit2 ? fpSubmit2(queue, submitCount, pSubmits, fence)
                         : VK_ERROR_INITIALIZATION_FAILED;

    if (!android_gpu_usage_env_enabled())
        return fpSubmit2(queue, submitCount, pSubmits, fence);

    if (!ctx->ts_supported.load(std::memory_order_relaxed))
        return fpSubmit2(queue, submitCount, pSubmits, fence);

    // fast-path: 샘플링 안 하는 프레임이면 락을 아예 안 잡는다
    if (!android_gpu_usage_should_sample(ctx))
        return fpSubmit2(queue, submitCount, pSubmits, fence);

    // [ROLLBACK GUARD] 예외/실패로 “제출 안 된 query”가 슬롯에 남지 않게 복구한다.
    AndroidVkGpuContext::FrameResources* fr_ptr = nullptr;
    uint64_t serial_snapshot = 0;
    uint32_t saved_query_used = 0;
    bool     saved_has_queries = false;

    try {
        std::lock_guard<std::mutex> g(ctx->lock);

        if (!ctx->ts_supported.load(std::memory_order_relaxed))
            return fpSubmit2(queue, submitCount, pSubmits, fence);

        const uint64_t frame_serial = ctx->frame_index.load(std::memory_order_relaxed);
        if ((frame_serial & 1u) != 0u)
            return fpSubmit2(queue, submitCount, pSubmits, fence);

        if (!android_gpu_usage_init_timestamp_resources(ctx, queue_family_index))
            return fpSubmit2(queue, submitCount, pSubmits, fence);

        if (ctx->query_pool != VK_NULL_HANDLE &&
            ctx->queue_family_index != VK_QUEUE_FAMILY_IGNORED &&
            ctx->queue_family_index != queue_family_index) {
            return fpSubmit2(queue, submitCount, pSubmits, fence);
        }

        const uint32_t curr_idx = (uint32_t)(frame_serial % AndroidVkGpuContext::MAX_FRAMES);
        serial_snapshot = frame_serial;
        fr_ptr = &ctx->frames[curr_idx];
        auto& fr = *fr_ptr;

        if (!android_gpu_usage_begin_frame(ctx, curr_idx, frame_serial))
            return fpSubmit2(queue, submitCount, pSubmits, fence);

        saved_query_used  = fr.query_used;
        saved_has_queries = fr.has_queries;
        
        auto& wrapped2 = ctx->scratch_wrapped2;
        auto& infos    = ctx->scratch_cmd_infos;
        auto& offsets  = ctx->scratch_cb_offsets;
        auto& counts   = ctx->scratch_cb_counts;
        auto& inst     = ctx->scratch_instrument;

        const uint32_t n = submitCount;

        wrapped2.resize(n);
        offsets.resize(n);
        counts.resize(n);
        inst.resize(n);

        uint32_t total_infos = 0;
        uint32_t tmp_used    = fr.query_used;
        bool any_inst = false;

        // 샘플 프레임당 계측 submit 상한 (오버헤드 방지)
        constexpr uint32_t MAX_PAIRS_PER_SAMPLED_FRAME = 16; // 필요하면 32로
        const uint32_t pair_budget =
            std::min<uint32_t>(MAX_PAIRS_PER_SAMPLED_FRAME, fr.query_capacity / 2u);

        uint32_t planned_pairs = 0;

        for (uint32_t i = 0; i < n; ++i) {
            const VkSubmitInfo2& src = pSubmits[i];
            const uint32_t base = src.commandBufferInfoCount;

            const bool basic_ok = (base > 0) && src.pCommandBufferInfos && (fr.cmd_pool != VK_NULL_HANDLE);
            const bool room_ok  = (planned_pairs < pair_budget) &&
                                  (fr.query_used + (planned_pairs + 1u) * 2u <= fr.query_capacity);

            const bool can = basic_ok && room_ok;

            if (!can) {
                inst[i]   = 0;
                counts[i] = base;
            } else {
                inst[i]   = 1;
                any_inst  = true;
                planned_pairs += 1;
                counts[i] = base + 2; // begin + end
            }
            total_infos += counts[i];
        }

        if (!any_inst)
            return fpSubmit2(queue, submitCount, pSubmits, fence);

        if (infos.size() < total_infos)
            infos.resize(total_infos);

        uint32_t cursor = 0;
        for (uint32_t i = 0; i < n; ++i) {
            offsets[i] = cursor;
            cursor += counts[i];
        }

        for (uint32_t i = 0; i < n; ++i) {
            const VkSubmitInfo2& src = pSubmits[i];

            if (!inst[i]) {
                wrapped2[i] = src;
                continue;
            }

            VkCommandBufferSubmitInfo* dst_infos = infos.data() + offsets[i];
            uint32_t idx_cb = 0;

            const VkCommandBufferSubmitInfo* template_info =
                (src.commandBufferInfoCount > 0 && src.pCommandBufferInfos)
                    ? &src.pCommandBufferInfos[0]
                    : nullptr;

            auto make_cb_info = [&](VkCommandBuffer buf) {
                VkCommandBufferSubmitInfo info{};
                info.sType         = VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO;
                info.pNext         = nullptr;
                info.commandBuffer = buf;
                info.deviceMask    = template_info ? template_info->deviceMask : 0x1u;
                return info;
            };

            VkCommandBuffer cmd_begin = VK_NULL_HANDLE;
            VkCommandBuffer cmd_end   = VK_NULL_HANDLE;
            uint32_t query_first = 0;

            if (!android_gpu_usage_record_timestamp_pair(ctx, fr, query_first, cmd_begin, cmd_end)) {
                for (uint32_t j = 0; j < src.commandBufferInfoCount; ++j)
                    dst_infos[idx_cb++] = src.pCommandBufferInfos[j];

                VkSubmitInfo2 dst = src;
                dst.commandBufferInfoCount = idx_cb;
                dst.pCommandBufferInfos    = dst_infos;
                wrapped2[i] = dst;
                inst[i] = 0;
                continue;
            }

            dst_infos[idx_cb++] = make_cb_info(cmd_begin);
            for (uint32_t j = 0; j < src.commandBufferInfoCount; ++j)
                dst_infos[idx_cb++] = src.pCommandBufferInfos[j];
            dst_infos[idx_cb++] = make_cb_info(cmd_end);

            VkSubmitInfo2 dst = src;
            dst.commandBufferInfoCount = idx_cb;
            dst.pCommandBufferInfos    = dst_infos;
            wrapped2[i] = dst;
        }

        VkResult vr = fpSubmit2(queue, submitCount, wrapped2.data(), fence);
        if (vr != VK_SUCCESS) {
            // submit 실패면 이번에 커밋한 query 사용분은 무효 처리 (측정 오염 방지)
            fr.query_used  = saved_query_used;
            fr.has_queries = saved_has_queries;

            if (vr == VK_ERROR_DEVICE_LOST) {
                ctx->ts_supported.store(false, std::memory_order_relaxed);
                android_gpu_usage_destroy_timestamp_resources(ctx);
            }
        }
        return vr;
    } catch (const std::exception& e) {
        SPDLOG_WARN("Android GPU usage: queue_submit2 exception: {}", e.what());
        if (ctx && fr_ptr) {
            std::lock_guard<std::mutex> g(ctx->lock);
            if (!ctx->destroying && fr_ptr->frame_serial == serial_snapshot) {
                fr_ptr->query_used  = saved_query_used;
                fr_ptr->has_queries = saved_has_queries;
            }
        }
        return fpSubmit2(queue, submitCount, pSubmits, fence);
    } catch (...) {
        SPDLOG_WARN("Android GPU usage: queue_submit2 unknown exception");
        if (ctx && fr_ptr) {
            std::lock_guard<std::mutex> g(ctx->lock);
            if (!ctx->destroying && fr_ptr->frame_serial == serial_snapshot) {
                fr_ptr->query_used  = saved_query_used;
                fr_ptr->has_queries = saved_has_queries;
            }
        }
        return fpSubmit2(queue, submitCount, pSubmits, fence);
    }
}

// ====================== Present 시점 처리 ======================
void
android_gpu_usage_on_present(AndroidVkGpuContext*    ctx,
                             VkQueue                 queue,
                             uint32_t                queue_family_index,
                             const VkPresentInfoKHR* present_info,
                             uint32_t                swapchain_index,
                             uint32_t                image_index)
{
    (void)queue;
    (void)present_info;
    (void)swapchain_index;
    (void)image_index;

    if (!ctx)
        return;

    if (!android_gpu_usage_env_enabled())
        return;

    using clock = std::chrono::steady_clock;
    const auto now = clock::now();

    struct PendingRead {
        bool     have       = false;
        uint32_t slot_idx   = 0;
        uint64_t serial     = 0;
        uint32_t q_start    = 0;
        uint32_t q_count    = 0;
        float    cpu_ms     = 0.0f;
    } read{};

    float frame_cpu_ms = 16.0f;
    bool in_flight_armed = false; // read.have 케이스에서만 true

    try {
        // --------- (A) 짧은 락: CPU dt 갱신 + 후보 슬롯 선택 + init ---------
        {
            std::lock_guard<std::mutex> g(ctx->lock);
            if (ctx->destroying) return; // destroy 진행 중이면 더 건드리지 마

            // CPU 기준 프레임 시간
            if (ctx->last_present.time_since_epoch().count() != 0) {
                const double dt_ms =
                    std::chrono::duration<double, std::milli>(now - ctx->last_present).count();
                // 0 나눗셈만 막자. 0.001ms(=1us)면 충분.
                frame_cpu_ms = static_cast<float>(dt_ms > 0.001 ? dt_ms : 0.001);
            }
            ctx->last_present = now;

            // [PATCH] CPU dt를 "현재 serial" 슬롯에 저장해서, 나중에 GPU serial이랑 매칭한다.
            const uint64_t cur_serial = ctx->frame_index.load(std::memory_order_relaxed);
            const uint32_t cur_idx = static_cast<uint32_t>(cur_serial % AndroidVkGpuContext::MAX_FRAMES);
            auto& cs = ctx->cpu_ring[cur_serial & (AndroidVkGpuContext::CPU_RING - 1u)];
            cs.serial = cur_serial;
            cs.ms     = frame_cpu_ms;

            // GPU 후보 슬롯 선택: 가장 오래된 pending(serial)부터 처리해서 starvation 방지
            if (ctx->ts_supported.load(std::memory_order_relaxed) && ctx->query_pool != VK_NULL_HANDLE) {
                const uint64_t fi = ctx->frame_index.load(std::memory_order_relaxed);

                // read_serial이 아직 FRAME_LAG만큼 뒤처지지 않으면 읽을 게 없다
                if (ctx->read_serial + AndroidVkGpuContext::FRAME_LAG <= fi) {
                    // 한 번의 present에서 너무 오래 훑지 않게 상한
                    for (uint32_t step = 0; step < AndroidVkGpuContext::MAX_FRAMES; ++step) {
                        const uint64_t serial = ctx->read_serial;

                        if (serial + AndroidVkGpuContext::FRAME_LAG > fi)
                            break;

                        const uint32_t idx = static_cast<uint32_t>(serial % AndroidVkGpuContext::MAX_FRAMES);
                        auto& fr = ctx->frames[idx];

                        // 이 serial에 해당 슬롯이 비어있으면 다음 serial로 넘김
                        if (!(fr.frame_serial == serial && fr.has_queries && fr.query_used >= 2)) {
                            ctx->read_serial++;
                            continue;
                        }

                        // 후보 선택
                        read.have     = true;
                        read.slot_idx = idx;
                        read.serial   = serial;
                        read.q_start  = fr.query_start;
                        read.q_count  = fr.query_used;

                        const auto& cs = ctx->cpu_ring[serial & (AndroidVkGpuContext::CPU_RING - 1u)];
                        read.cpu_ms = (cs.serial == serial) ? cs.ms : 0.0f;
                        break;
                    }
                }
            }
            // [LIFETIME GUARD] 락 밖에서 GetQueryPoolResults 호출 예정이면 destroy()가 기다리게 만든다.
            if (read.have) {
                ctx->in_flight++;
                in_flight_armed = true;
            }
        }

        // --------- (B) 락 밖: QueryResults + 계산(비용 큰 부분) ---------
        float frame_gpu_ms = 0.0f;
        AndroidGpuReadStatus st = AndroidGpuReadStatus::NotReady;

        if (read.have) {
            st = android_gpu_usage_query_range_gpu_ms(ctx, read.q_start, read.q_count, &frame_gpu_ms);

            // NotReady가 너무 오래 지속되면 read_serial이 고착되니, "생존"을 위해 슬롯 폐기.
            if (st == AndroidGpuReadStatus::NotReady) {
                const uint64_t fi = ctx->frame_index.load(std::memory_order_relaxed);
                if (fi > read.serial + (AndroidVkGpuContext::MAX_FRAMES * 2u)) {
                    st = AndroidGpuReadStatus::Error; // 아래 (C)에서 폐기 루트로 보냄
                }
            }
        }

        // --------- (C) 짧은 락: 슬롯 소비(Ready일 때만) + smoothing 누적 ---------
        {
            std::lock_guard<std::mutex> g(ctx->lock);

            // [LIFETIME GUARD] 여기 들어왔다는 건 (B)가 끝났다는 뜻. 이제 destroy()를 진행 가능하게 풀어준다.
            if (in_flight_armed) {
                in_flight_armed = false;
                if (--ctx->in_flight == 0 && ctx->destroying)
                    ctx->cv.notify_all();
            }
            if (st == AndroidGpuReadStatus::DeviceLost) {
                SPDLOG_WARN("Android GPU usage: DEVICE_LOST on GetQueryPoolResults, disabling timestamp backend");
                ctx->ts_supported = false;
                ctx->have_metrics = false; // 더 이상 신뢰 불가, 값 제공 중단
                android_gpu_usage_destroy_timestamp_resources(ctx);
                frame_gpu_ms = 0.0f;

            } else if (st == AndroidGpuReadStatus::Ready && frame_gpu_ms > 0.0f) {
                auto& fr = ctx->frames[read.slot_idx];
                if (fr.frame_serial == read.serial && fr.has_queries && fr.query_used == read.q_count) {
                    android_gpu_usage_consume_slot(fr);
                    ctx->read_serial = read.serial + 1; // 다음 pending으로
                } else {
                    frame_gpu_ms = 0.0f;
                    st = AndroidGpuReadStatus::NotReady;
                }
            } else if (st == AndroidGpuReadStatus::Error) {
                // 슬롯만 폐기하고 다음으로 진행: 계측이 "멈춰버리는" 최악을 피한다.
                if (read.have) {
                    auto& fr = ctx->frames[read.slot_idx];
                    android_gpu_usage_consume_slot(fr);
                    ctx->read_serial = read.serial + 1;
                }
                frame_gpu_ms = 0.0f;
            } else {
                // NotReady: 슬롯 유지(정확도)
                frame_gpu_ms = 0.0f;
            }

            // 500ms 윈도우 누적
            if (ctx->window_start.time_since_epoch().count() == 0) {
                ctx->window_start = now;
            }

            if (st == AndroidGpuReadStatus::Ready && frame_gpu_ms > 0.0f && read.cpu_ms > 0.0f) {
                ctx->acc_cpu_ms_sampled += read.cpu_ms;
                ctx->acc_frames_sampled += 1;
            
                ctx->acc_gpu_ms         += frame_gpu_ms;
                ctx->acc_gpu_samples    += 1;
            } else {
                // NotReady/Error면 아무것도 누적 안 함 (CPU만 쌓아두면 bias 생김)
            }

            constexpr auto WINDOW = std::chrono::milliseconds(500);
            const auto elapsed = now - ctx->window_start;
            
            if (elapsed >= WINDOW) {
            
                // GPU 샘플이 한 번도 없으면 이번 윈도우는 "업데이트 스킵"
                // (0으로 덮어써서 튀는 것보다 이전 값 유지가 실전에서 덜 빡침)
                if (ctx->acc_gpu_samples > 0 && ctx->acc_frames_sampled > 0 && ctx->acc_cpu_ms_sampled > 0.0) {
            
                    const double avg_cpu_ms =
                        ctx->acc_cpu_ms_sampled / static_cast<double>(ctx->acc_frames_sampled);
            
                    const double avg_gpu_ms =
                        ctx->acc_gpu_ms / static_cast<double>(ctx->acc_gpu_samples);
            
                    float usage = 0.0f;
                    if (avg_cpu_ms > 0.0 && avg_gpu_ms > 0.0)
                        usage = static_cast<float>((avg_gpu_ms / avg_cpu_ms) * 100.0);
            
                    if (!std::isfinite(usage)) usage = 0.0f;
                    if (usage < 0.0f) usage = 0.0f;
                    if (usage > 100.0f) usage = 100.0f;
            
                    const float alpha = 0.5f;
            
                    if (!ctx->have_metrics) {
                        ctx->smooth_usage  = usage;
                        ctx->smooth_gpu_ms = static_cast<float>(avg_gpu_ms);
                    } else {
                        ctx->smooth_usage  = ctx->smooth_usage * (1.0f - alpha) + usage * alpha;
                        ctx->smooth_gpu_ms = ctx->smooth_gpu_ms * (1.0f - alpha) +
                                             static_cast<float>(avg_gpu_ms) * alpha;
                    }
            
                    ctx->last_usage   = ctx->smooth_usage;
                    ctx->last_gpu_ms  = ctx->smooth_gpu_ms;
                    ctx->have_metrics = true;
                }
            
                // 윈도우 리셋 (업데이트를 했든 말든 시간 창은 굴러야 함)
                ctx->acc_cpu_ms_sampled = 0.0;
                ctx->acc_frames_sampled = 0;
            
                ctx->acc_gpu_ms      = 0.0;
                ctx->acc_gpu_samples = 0;
            
                ctx->window_start = now;
            }

            // 다음 프레임 serial
            ctx->frame_index++;
        }
    } catch (const std::exception& e) {
        // (B)에서 예외 터지면 in_flight가 누수될 수 있음 -> 여기서 회수
        if (in_flight_armed) {
            std::lock_guard<std::mutex> g(ctx->lock);
            in_flight_armed = false;
            if (--ctx->in_flight == 0 && ctx->destroying)
                ctx->cv.notify_all();
        }
        SPDLOG_WARN("Android GPU usage: on_present exception: {}", e.what());
    } catch (...) {
        if (in_flight_armed) {
            std::lock_guard<std::mutex> g(ctx->lock);
            in_flight_armed = false;
            if (--ctx->in_flight == 0 && ctx->destroying)
                ctx->cv.notify_all();
        }
        SPDLOG_WARN("Android GPU usage: on_present unknown exception");
    }
}

bool
android_gpu_usage_get_metrics(AndroidVkGpuContext* ctx,
                              float*              out_gpu_ms,
                              float*              out_usage)
{
    if (!ctx)
        return false;

    // MANGOHUD_VKP!=1 이면 Vulkan 경로 비활성 (out_* 건들지 않음)
    if (!android_gpu_usage_env_enabled()) {
        return false;
    }

    std::lock_guard<std::mutex> g(ctx->lock);

    if (!ctx->have_metrics)
        return false;

    if (out_gpu_ms)
        *out_gpu_ms = ctx->last_gpu_ms;
    if (out_usage)
        *out_usage  = ctx->last_usage;

    return true;
}

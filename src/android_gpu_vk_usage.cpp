#include "android_gpu_vk_usage.h"
#include <vulkan/vulkan.h>
#include <chrono>
#include <mutex>
#include <cmath>
#include <vector>
#include <limits>
#include <cstdint>
#include <spdlog/spdlog.h>

// ====================== 내부 상태 ======================

struct AndroidVkGpuContext {
    VkPhysicalDevice        phys_dev      = VK_NULL_HANDLE;
    VkDevice                device        = VK_NULL_HANDLE;
    AndroidVkGpuDispatch    disp{};

    float                   ts_period_ns  = 0.0f;   // ns per tick
    uint32_t                ts_valid_bits = 0;
    bool                    ts_supported  = false;

    VkQueryPool             query_pool    = VK_NULL_HANDLE;
    uint32_t                queue_family_index = VK_QUEUE_FAMILY_IGNORED;

    // 링 버퍼 / 슬롯 설정
    static constexpr uint32_t MAX_FRAMES            = 16;   // 슬롯 개수
    static constexpr uint32_t MAX_QUERIES_PER_FRAME = 128;  // 슬롯당 쿼리 개수 (start/end 2개씩 → submit 최대 64개)
    static constexpr uint32_t FRAME_LAG             = 3;    // 몇 프레임 뒤에 슬롯을 읽고 해제할지

    struct FrameResources {
        VkCommandPool   cmd_pool        = VK_NULL_HANDLE;
        VkCommandBuffer reset_cmd       = VK_NULL_HANDLE;

        uint32_t        query_start     = 0;   // 이 슬롯의 쿼리 시작 인덱스
        uint32_t        query_capacity  = 0;   // MAX_QUERIES_PER_FRAME
        uint32_t        query_used      = 0;   // 현재까지 사용한 쿼리 수 (짝수, 2개씩)

        bool            reset_recorded  = false; // reset_cmd 안에 CmdResetQueryPool 녹화 여부
        bool            reset_submitted = false; // 이번 프레임에 reset_cmd를 실제 submit에 포함했는지
        bool            has_queries     = false; // 이 슬롯에 계측된 submit이 하나라도 있는지

        // 이 슬롯이 마지막으로 사용된 frame serial
        uint64_t        frame_serial    = std::numeric_limits<uint64_t>::max();

        // 타임스탬프용 커맨드 버퍼 재사용용
        // [2*i]   = begin CB
        // [2*i+1] = end CB
        std::vector<VkCommandBuffer> timestamp_cmds;
    };

    FrameResources          frames[MAX_FRAMES]{};

    // present 기준 frame serial (QueueSubmit는 "현재 serial"을 사용)
    uint64_t                frame_index   = 0;

    // 쿼리 결과용 스크래치 버퍼 (value + availability 쌍)
    // 최대: MAX_QUERIES_PER_FRAME * 2 * sizeof(uint64_t)
    std::vector<uint64_t>   query_scratch;

    // CPU/GPU 사용률 + smoothing용 상태
    std::mutex                                      lock;
    std::chrono::steady_clock::time_point           last_present{};   // 마지막 present 시각

    std::chrono::steady_clock::time_point           window_start{};   // smoothing 윈도우 시작 시각
    double                                          acc_gpu_ms  = 0.0;
    double                                          acc_cpu_ms  = 0.0;
    uint32_t                                        acc_frames  = 0;

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
};

// ====================== 헬퍼: 타임스탬프 리소스 초기화 ======================

static bool
android_gpu_usage_init_timestamp_resources(AndroidVkGpuContext* ctx,
                                           uint32_t queue_family_index)
{
    if (!ctx || !ctx->ts_supported)
        return false;

    // 이미 초기화되어 있으면 그대로 사용
    if (ctx->query_pool != VK_NULL_HANDLE) {
        return true;
    }

    ctx->queue_family_index = queue_family_index;

    // 1) QueryPool 생성 (슬롯 × 쿼리 수)
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

    // 2) 프레임별 커맨드 풀 만들기
    if (!ctx->disp.CreateCommandPool || !ctx->disp.ResetCommandPool) {
        SPDLOG_WARN(
            "Android GPU usage: Create/ResetCommandPool not available → disabling timestamp backend"
        );
        ctx->ts_supported = false;
        return false;
    }

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
            ctx->ts_supported = false;
            return false;
        }

        auto& fr          = ctx->frames[i];
        fr.cmd_pool       = pool;
        fr.query_start    = i * AndroidVkGpuContext::MAX_QUERIES_PER_FRAME;
        fr.query_capacity = AndroidVkGpuContext::MAX_QUERIES_PER_FRAME;
        fr.query_used     = 0;
        fr.reset_recorded = false;
        fr.reset_submitted= false;
        fr.has_queries    = false;
        fr.frame_serial   = std::numeric_limits<uint64_t>::max();
        fr.reset_cmd      = VK_NULL_HANDLE;
        fr.timestamp_cmds.clear();
    }

    SPDLOG_INFO(
        "Android GPU usage: timestamp resources initialized (qf_index={} qpool={} slots={} qpf={})",
        queue_family_index,
        static_cast<void*>(ctx->query_pool),
        AndroidVkGpuContext::MAX_FRAMES,
        AndroidVkGpuContext::MAX_QUERIES_PER_FRAME
    );

    // 쿼리 결과용 스크래치 버퍼를 한 번만 확보
    ctx->query_scratch.clear();
    ctx->query_scratch.resize(
        AndroidVkGpuContext::MAX_QUERIES_PER_FRAME * 2u
    );

    return true;
}

// 슬롯 시작: 재사용 가능성 체크 + 초기화
static bool
android_gpu_usage_begin_frame(AndroidVkGpuContext* ctx,
                              uint32_t frame_idx,
                              uint64_t frame_serial)
{
    auto& fr = ctx->frames[frame_idx];

    // 이미 이 frame_serial에 대해 초기화된 슬롯이면 그대로 사용
    if (fr.frame_serial == frame_serial)
        return true;

    // 이전에 사용된 적 있고, 너무 "가까운" 프레임이면 재사용 금지
    if (fr.frame_serial != std::numeric_limits<uint64_t>::max()) {
        uint64_t age = (frame_serial > fr.frame_serial)
                           ? (frame_serial - fr.frame_serial)
                           : std::numeric_limits<uint64_t>::max();

        if (age < AndroidVkGpuContext::FRAME_LAG) {
            // 아직 이 슬롯의 결과를 읽지 않았거나, GPU가 완전히 끝났는지 보장 불가
            return false;
        }
    }

    fr.frame_serial   = frame_serial;
    fr.query_used     = 0;
    fr.reset_recorded = false;
    fr.reset_submitted= false;
    fr.has_queries    = false;

    // 이 슬롯에서 쓰던 CB들 전부 초기화 (핸들은 유지, 내용만 리셋)
    if (ctx->disp.ResetCommandPool && fr.cmd_pool != VK_NULL_HANDLE) {
        ctx->disp.ResetCommandPool(
            ctx->device,
            fr.cmd_pool,
            VK_COMMAND_POOL_RESET_RELEASE_RESOURCES_BIT
        );
    }

    return true;
}

// 이 슬롯에서 쿼리 리셋용 커맨드 버퍼 녹화 (핸들은 1회만 할당, 이후 재녹화)
static bool
android_gpu_usage_ensure_reset_cmd(AndroidVkGpuContext*            ctx,
                                   AndroidVkGpuContext::FrameResources& fr)
{
    if (fr.reset_recorded)
        return true;

    if (!ctx->disp.AllocateCommandBuffers ||
        !ctx->disp.BeginCommandBuffer ||
        !ctx->disp.EndCommandBuffer ||
        !ctx->disp.CmdResetQueryPool)
        return false;

    VkCommandBuffer cmd = fr.reset_cmd;

    // 최초 한 번만 할당
    if (cmd == VK_NULL_HANDLE) {
        VkCommandBufferAllocateInfo ai{};
        ai.sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        ai.commandPool        = fr.cmd_pool;
        ai.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        ai.commandBufferCount = 1;

        if (ctx->disp.AllocateCommandBuffers(ctx->device, &ai, &cmd) != VK_SUCCESS)
            return false;

        fr.reset_cmd = cmd;
    }

    VkCommandBufferBeginInfo bi{};
    bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

    if (ctx->disp.BeginCommandBuffer(cmd, &bi) != VK_SUCCESS)
        return false;

    // 이 슬롯의 쿼리 범위 전체를 리셋
    ctx->disp.CmdResetQueryPool(
        cmd,
        ctx->query_pool,
        fr.query_start,
        fr.query_capacity
    );

    if (ctx->disp.EndCommandBuffer(cmd) != VK_SUCCESS)
        return false;

    fr.reset_recorded = true;
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
    if (!ctx->disp.AllocateCommandBuffers ||
        !ctx->disp.BeginCommandBuffer ||
        !ctx->disp.EndCommandBuffer ||
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

    // begin CB: Start timestamp
    if (ctx->disp.BeginCommandBuffer(out_cmd_begin, &bi) != VK_SUCCESS)
        return false;

    ctx->disp.CmdWriteTimestamp(
        out_cmd_begin,
        VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
        ctx->query_pool,
        query_first
    );

    if (ctx->disp.EndCommandBuffer(out_cmd_begin) != VK_SUCCESS)
        return false;

    // end CB: (옵션) 배리어 + End timestamp
    if (ctx->disp.BeginCommandBuffer(out_cmd_end, &bi) != VK_SUCCESS)
        return false;

    if (ctx->disp.CmdPipelineBarrier) {
        ctx->disp.CmdPipelineBarrier(
            out_cmd_end,
            VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
            VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
            0,
            0, nullptr,
            0, nullptr,
            0, nullptr
        );
    }

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

// 특정 슬롯에 대해 GPU time(ms) 합산 + 슬롯 해제
static float
android_gpu_usage_collect_frame_gpu_ms(AndroidVkGpuContext*            ctx,
                                       AndroidVkGpuContext::FrameResources& fr)
{
    if (!ctx || !ctx->ts_supported || !ctx->query_pool)
        return 0.0f;

    if (!ctx->disp.GetQueryPoolResults)
        return 0.0f;

    if (!fr.has_queries || fr.query_used < 2)
        return 0.0f;

    const uint32_t query_count = fr.query_used;
    const uint32_t pair_count  = query_count / 2;

    // 방어적 체크: 설계상 여기 걸리면 버그다
    if (query_count > AndroidVkGpuContext::MAX_QUERIES_PER_FRAME)
        return 0.0f;

    // 스크래치 버퍼 크기 보장
    const uint32_t needed_u64 = query_count * 2u;
    if (ctx->query_scratch.size() < needed_u64) {
        ctx->query_scratch.resize(needed_u64);
    }

    uint64_t* data = ctx->query_scratch.data();

    VkResult r = ctx->disp.GetQueryPoolResults(
        ctx->device,
        ctx->query_pool,
        fr.query_start,
        query_count,
        needed_u64 * sizeof(uint64_t),
        data,
        sizeof(uint64_t) * 2,
        VK_QUERY_RESULT_64_BIT |
        VK_QUERY_RESULT_WITH_AVAILABILITY_BIT |
        VK_QUERY_RESULT_WAIT_BIT
    );

    if (r < 0) {
        SPDLOG_WARN(
            "Android GPU usage: GetQueryPoolResults failed (r={} used={} start={})",
            static_cast<int>(r),
            fr.query_used,
            fr.query_start
        );
        fr.has_queries = false;
        fr.query_used  = 0;
        fr.frame_serial= std::numeric_limits<uint64_t>::max();
        return 0.0f;
    }

    uint64_t mask = 0;
    if (ctx->ts_valid_bits == 0 || ctx->ts_valid_bits >= 64) {
        mask = ~0ULL;
    } else {
        mask = (1ULL << ctx->ts_valid_bits) - 1ULL;
    }

    double   sum_ms       = 0.0;
    uint64_t min_start_ts = std::numeric_limits<uint64_t>::max();
    uint64_t max_end_ts   = 0;

    for (uint32_t i = 0; i < pair_count; ++i) {
        const uint32_t q_start = 2u * i;
        const uint32_t q_end   = q_start + 1u;

        const uint64_t start_raw = data[q_start * 2u + 0u];
        const uint64_t avail_s   = data[q_start * 2u + 1u];
        const uint64_t end_raw   = data[q_end   * 2u + 0u];
        const uint64_t avail_e   = data[q_end   * 2u + 1u];

        if (!avail_s || !avail_e)
            continue;

        uint64_t start_ts = start_raw & mask;
        uint64_t end_ts   = end_raw   & mask;

        // validBits < 64인 경우 wrap-around 보정
        if (ctx->ts_valid_bits > 0 && ctx->ts_valid_bits < 64 && end_ts < start_ts) {
            end_ts += (1ULL << ctx->ts_valid_bits);
        }

        if (end_ts <= start_ts)
            continue;

        // 1) 순수 합산 시간
        const uint64_t delta_ticks = end_ts - start_ts;
        const double   ns          = double(delta_ticks) * double(ctx->ts_period_ns);
        const double   ms          = ns * 1e-6;
        if (ms > 0.0 && std::isfinite(ms))
            sum_ms += ms;

        // 2) 활성 구간 추적
        if (start_ts < min_start_ts)
            min_start_ts = start_ts;
        if (end_ts > max_end_ts)
            max_end_ts = end_ts;
    }

    // 슬롯 상태 리셋
    fr.has_queries = false;
    fr.query_used  = 0;
    fr.frame_serial= std::numeric_limits<uint64_t>::max();

    // 활성 구간 기반 range_ms 계산
    double range_ms = 0.0;
    if (max_end_ts > min_start_ts &&
        min_start_ts != std::numeric_limits<uint64_t>::max()) {

        const uint64_t range_ticks = max_end_ts - min_start_ts;
        const double   ns          = double(range_ticks) * double(ctx->ts_period_ns);
        range_ms                   = ns * 1e-6;
    }

    double gpu_ms = 0.0;

    // 1순위: range_ms (DXVK HUD 스타일)
    if (range_ms > 0.0 && std::isfinite(range_ms)) {
        gpu_ms = range_ms;
    }
    // fallback: sum_ms
    else if (sum_ms > 0.0 && std::isfinite(sum_ms)) {
        gpu_ms = sum_ms;
    }

    // 상식적인 sanity: 진짜 말도 안 되면 백엔드 끈다.
    if (!std::isfinite(gpu_ms) || gpu_ms < 0.0) {
        SPDLOG_WARN("Android GPU usage: invalid gpu_ms={} → disabling timestamp backend", gpu_ms);
        ctx->ts_supported = false;
        return 0.0f;
    }

    // 5초 넘는 프레임은 그냥 드라이버가 맛 간 걸로 보고 끔. (1 FPS 게임은 그냥 포기)
    if (gpu_ms > 5000.0) {
        SPDLOG_WARN("Android GPU usage: insane gpu_ms={}ms (>5000) → disabling timestamp backend", gpu_ms);
        ctx->ts_supported = false;
        return 0.0f;
    }

    if (gpu_ms <= 0.0)
        return 0.0f;

    return static_cast<float>(gpu_ms);
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
        (ctx->ts_valid_bits > 0) &&
        dispatch_ok;

    SPDLOG_INFO(
        "Android GPU usage: create ctx={} ts_period_ns={} ts_valid_bits={} dispatch_ok={} ts_supported={}",
        static_cast<void*>(ctx),
        ctx->ts_period_ns,
        ctx->ts_valid_bits,
        dispatch_ok,
        ctx->ts_supported
    );

    if (!ctx->ts_supported) {
        SPDLOG_WARN("Android GPU usage: Vulkan timestamps not supported, backend will be disabled");
    }

    return ctx;
}

void
android_gpu_usage_destroy(AndroidVkGpuContext* ctx)
{
    if (!ctx)
        return;

    if (ctx->device != VK_NULL_HANDLE) {
        // 프레임별 커맨드 풀 제거
        if (ctx->disp.DestroyCommandPool) {
            for (uint32_t i = 0; i < AndroidVkGpuContext::MAX_FRAMES; ++i) {
                auto& fr = ctx->frames[i];
                if (fr.cmd_pool != VK_NULL_HANDLE) {
                    ctx->disp.DestroyCommandPool(ctx->device, fr.cmd_pool, nullptr);
                    fr.cmd_pool = VK_NULL_HANDLE;
                    fr.reset_cmd = VK_NULL_HANDLE;
                    fr.timestamp_cmds.clear();
                }
            }
        }

        // 쿼리 풀 제거
        if (ctx->query_pool != VK_NULL_HANDLE && ctx->disp.DestroyQueryPool) {
            ctx->disp.DestroyQueryPool(ctx->device, ctx->query_pool, nullptr);
            ctx->query_pool = VK_NULL_HANDLE;
        }
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
    if (!ctx || !ctx->disp.QueueSubmit || !pSubmits || submitCount == 0) {
        return (ctx && ctx->disp.QueueSubmit)
            ? ctx->disp.QueueSubmit(queue, submitCount, pSubmits, fence)
            : VK_ERROR_INITIALIZATION_FAILED;
    }

    // 타임스탬프 미지원이면 그냥 패스스루
    if (!ctx->ts_supported) {
        return ctx->disp.QueueSubmit(queue, submitCount, pSubmits, fence);
    }

    try {
        std::lock_guard<std::mutex> g(ctx->lock);

        // 리소스 lazy init
        if (!android_gpu_usage_init_timestamp_resources(ctx, queue_family_index)) {
            return ctx->disp.QueueSubmit(queue, submitCount, pSubmits, fence);
        }

        const uint64_t frame_serial = ctx->frame_index;
        const uint32_t curr_idx     = static_cast<uint32_t>(
            frame_serial % AndroidVkGpuContext::MAX_FRAMES
        );
        auto& fr = ctx->frames[curr_idx];

        // 슬롯 재사용 불가하면 이번 프레임은 계측 포기
        if (!android_gpu_usage_begin_frame(ctx, curr_idx, frame_serial)) {
            return ctx->disp.QueueSubmit(queue, submitCount, pSubmits, fence);
        }

        // ===== 1-pass: 어떤 submit을 계측할지 + CB 개수 계산 =====
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
        bool     any_inst   = false;

        for (uint32_t i = 0; i < n; ++i) {
            const VkSubmitInfo& src = pSubmits[i];

            const uint32_t base_count = src.commandBufferCount;

            bool can_instrument =
                (base_count > 0) &&
                (src.pCommandBuffers != nullptr) &&
                (ctx->query_pool != VK_NULL_HANDLE) &&
                (fr.cmd_pool   != VK_NULL_HANDLE) &&
                (tmp_used + 2 <= fr.query_capacity);

            if (!can_instrument) {
                inst[i]   = 0;
                counts[i] = base_count;
            } else {
                inst[i]   = 1;
                any_inst  = true;
                tmp_used += 2;
                counts[i] = base_count + 2; // begin + end
            }

            total_cmds += counts[i];
        }

        // 이 프레임에 계측할 submit이 하나도 없으면 패스스루
        if (!any_inst) {
            return ctx->disp.QueueSubmit(queue, submitCount, pSubmits, fence);
        }

        // reset_cmd 준비 (1회만)
        if (!fr.reset_recorded) {
            if (!android_gpu_usage_ensure_reset_cmd(ctx, fr)) {
                // reset 안 되면 깔끔하게 계측 포기
                return ctx->disp.QueueSubmit(queue, submitCount, pSubmits, fence);
            }
        }

        // reset_cmd를 첫 번째 계측 submit 앞에 한 번만 삽입
        int32_t reset_target = -1;
        if (fr.reset_cmd != VK_NULL_HANDLE && !fr.reset_submitted) {
            for (uint32_t i = 0; i < n; ++i) {
                if (inst[i]) {
                    reset_target = static_cast<int32_t>(i);
                    counts[i]   += 1;
                    total_cmds  += 1;
                    break;
                }
            }
        }

        // flat_cbs 크기 확보 (capacity 증가는 드물게만 일어남)
        if (flat_cbs.size() < total_cmds)
            flat_cbs.resize(total_cmds);

        // offset 계산
        uint32_t cursor = 0;
        for (uint32_t i = 0; i < n; ++i) {
            offsets[i] = cursor;
            cursor     += counts[i];
        }

        // ===== 2-pass: scratch에 CB 채우고 VkSubmitInfo 구성 =====
        for (uint32_t i = 0; i < n; ++i) {
            const VkSubmitInfo& src = pSubmits[i];
            VkSubmitInfo&       dst = wrapped[i];

            const bool do_inst   = (inst[i] != 0);
            const bool add_reset = (reset_target == static_cast<int32_t>(i));

            // 계측도 없고 reset도 없으면 그냥 복사
            if (!do_inst && !add_reset) {
                dst = src;
                continue;
            }

            VkCommandBuffer* dst_bufs = flat_cbs.data() + offsets[i];
            uint32_t         idx_cb   = 0;

            // reset_cmd를 이 submit 앞에 한 번만 삽입
            if (add_reset && fr.reset_cmd != VK_NULL_HANDLE) {
                dst_bufs[idx_cb++] = fr.reset_cmd;
                fr.reset_submitted = true;
            }

            VkCommandBuffer cmd_begin = VK_NULL_HANDLE;
            VkCommandBuffer cmd_end   = VK_NULL_HANDLE;

            if (do_inst) {
                uint32_t query_first = 0;
                if (!android_gpu_usage_record_timestamp_pair(
                        ctx, fr, query_first, cmd_begin, cmd_end)) {
                    // 드문 실패 케이스: 이 submit만 계측 포기하고 그냥 원본 CB만 쓴다.
                    for (uint32_t j = 0; j < src.commandBufferCount; ++j)
                        dst_bufs[idx_cb++] = src.pCommandBuffers[j];

                    dst = src;
                    dst.commandBufferCount = idx_cb;
                    dst.pCommandBuffers    = dst_bufs;
                    inst[i] = 0;
                    continue;
                }

                // begin timestamp CB
                dst_bufs[idx_cb++] = cmd_begin;
            }

            // 앱의 원래 커맨드 버퍼들
            for (uint32_t j = 0; j < src.commandBufferCount; ++j) {
                dst_bufs[idx_cb++] = src.pCommandBuffers[j];
            }

            // end timestamp CB
            if (do_inst) {
                dst_bufs[idx_cb++] = cmd_end;
            }

            dst = src;
            dst.commandBufferCount = idx_cb;
            dst.pCommandBuffers    = dst_bufs;
        }

        // 실제 QueueSubmit 호출
        return ctx->disp.QueueSubmit(
            queue,
            submitCount,
            wrapped.data(),
            fence
        );
    } catch (const std::exception& e) {
        SPDLOG_WARN("Android GPU usage: queue_submit exception: {}", e.what());
        return ctx->disp.QueueSubmit(queue, submitCount, pSubmits, fence);
    } catch (...) {
        SPDLOG_WARN("Android GPU usage: queue_submit unknown exception");
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
        if (ctx->disp.QueueSubmit2)
            fpSubmit2 = ctx->disp.QueueSubmit2;
        else if (ctx->disp.QueueSubmit2KHR)
            fpSubmit2 = ctx->disp.QueueSubmit2KHR;
    }

    if (!ctx || !fpSubmit2 || !pSubmits || submitCount == 0) {
        return fpSubmit2
            ? fpSubmit2(queue, submitCount, pSubmits, fence)
            : VK_ERROR_INITIALIZATION_FAILED;
    }

    // 타임스탬프 미지원이면 그냥 패스스루
    if (!ctx->ts_supported) {
        return fpSubmit2(queue, submitCount, pSubmits, fence);
    }

    try {
        std::lock_guard<std::mutex> g(ctx->lock);

        // 리소스 lazy init
        if (!android_gpu_usage_init_timestamp_resources(ctx, queue_family_index)) {
            return fpSubmit2(queue, submitCount, pSubmits, fence);
        }

        const uint64_t frame_serial = ctx->frame_index;
        const uint32_t curr_idx     = static_cast<uint32_t>(
            frame_serial % AndroidVkGpuContext::MAX_FRAMES
        );
        auto& fr = ctx->frames[curr_idx];

        // 슬롯 재사용 불가하면 이번 프레임은 계측 포기
        if (!android_gpu_usage_begin_frame(ctx, curr_idx, frame_serial)) {
            return fpSubmit2(queue, submitCount, pSubmits, fence);
        }

        // ===== 1-pass: 어떤 submit을 계측할지 + CommandBufferInfo 개수 계산 =====
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
        bool     any_inst    = false;

        for (uint32_t i = 0; i < n; ++i) {
            const VkSubmitInfo2& src = pSubmits[i];

            const uint32_t base_count = src.commandBufferInfoCount;

            bool can_instrument =
                (base_count > 0) &&
                (src.pCommandBufferInfos != nullptr) &&
                (ctx->query_pool != VK_NULL_HANDLE) &&
                (fr.cmd_pool   != VK_NULL_HANDLE) &&
                (tmp_used + 2 <= fr.query_capacity);

            if (!can_instrument) {
                inst[i]   = 0;
                counts[i] = base_count;
            } else {
                inst[i]   = 1;
                any_inst  = true;
                tmp_used += 2;
                counts[i] = base_count + 2; // begin + end
            }

            total_infos += counts[i];
        }

        // 이 프레임에 계측할 submit이 하나도 없으면 패스스루
        if (!any_inst) {
            return fpSubmit2(queue, submitCount, pSubmits, fence);
        }

        // reset_cmd 준비 (1회만)
        if (!fr.reset_recorded) {
            if (!android_gpu_usage_ensure_reset_cmd(ctx, fr)) {
                return fpSubmit2(queue, submitCount, pSubmits, fence);
            }
        }

        // reset_cmd를 첫 번째 계측 submit 앞에 한 번만 삽입
        int32_t reset_target = -1;
        if (fr.reset_cmd != VK_NULL_HANDLE && !fr.reset_submitted) {
            for (uint32_t i = 0; i < n; ++i) {
                if (inst[i]) {
                    reset_target = static_cast<int32_t>(i);
                    counts[i]   += 1;
                    total_infos  += 1;
                    break;
                }
            }
        }

        if (infos.size() < total_infos)
            infos.resize(total_infos);

        // offset 계산
        uint32_t cursor = 0;
        for (uint32_t i = 0; i < n; ++i) {
            offsets[i] = cursor;
            cursor     += counts[i];
        }

        // ===== 2-pass: scratch에 CommandBufferSubmitInfo 채우고 VkSubmitInfo2 구성 =====
        for (uint32_t i = 0; i < n; ++i) {
            const VkSubmitInfo2& src = pSubmits[i];
            VkSubmitInfo2&       dst = wrapped2[i];

            const bool do_inst   = (inst[i] != 0);
            const bool add_reset = (reset_target == static_cast<int32_t>(i));

            if (!do_inst && !add_reset) {
                dst = src;
                continue;
            }

            VkCommandBufferSubmitInfo* dst_infos = infos.data() + offsets[i];
            uint32_t                   idx_cb    = 0;

            const VkCommandBufferSubmitInfo* template_info =
                (src.commandBufferInfoCount > 0 && src.pCommandBufferInfos)
                    ? &src.pCommandBufferInfos[0]
                    : nullptr;

            auto make_cb_info = [&](VkCommandBuffer buf) {
                VkCommandBufferSubmitInfo info{};
                info.sType        = VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO;
                info.pNext        = nullptr;
                info.commandBuffer= buf;
                info.deviceMask   = template_info ? template_info->deviceMask : 0x1u;
                return info;
            };

            // reset_cmd를 이 submit 앞에 한 번만 삽입
            if (add_reset && fr.reset_cmd != VK_NULL_HANDLE) {
                dst_infos[idx_cb++] = make_cb_info(fr.reset_cmd);
                fr.reset_submitted  = true;
            }

            VkCommandBuffer cmd_begin = VK_NULL_HANDLE;
            VkCommandBuffer cmd_end   = VK_NULL_HANDLE;

            if (do_inst) {
                uint32_t query_first = 0;
                if (!android_gpu_usage_record_timestamp_pair(
                        ctx, fr, query_first, cmd_begin, cmd_end)) {

                    // 계측 실패: reset이 이미 들어갔다면 그대로 두고, 나머지는 원본 CB만 복사
                    for (uint32_t j = 0; j < src.commandBufferInfoCount; ++j)
                        dst_infos[idx_cb++] = src.pCommandBufferInfos[j];

                    dst = src;
                    dst.commandBufferInfoCount = idx_cb;
                    dst.pCommandBufferInfos    = dst_infos;
                    inst[i] = 0;
                    continue;
                }

                dst_infos[idx_cb++] = make_cb_info(cmd_begin);
            }

            // 앱의 원래 커맨드 버퍼들
            for (uint32_t j = 0; j < src.commandBufferInfoCount; ++j) {
                dst_infos[idx_cb++] = src.pCommandBufferInfos[j];
            }

            // end timestamp CB
            if (do_inst) {
                dst_infos[idx_cb++] = make_cb_info(cmd_end);
            }

            dst = src;
            dst.commandBufferInfoCount = idx_cb;
            dst.pCommandBufferInfos    = dst_infos;
        }

        return fpSubmit2(
            queue,
            submitCount,
            wrapped2.data(),
            fence
        );
    } catch (const std::exception& e) {
        SPDLOG_WARN("Android GPU usage: queue_submit2 exception: {}", e.what());
        return fpSubmit2(queue, submitCount, pSubmits, fence);
    } catch (...) {
        SPDLOG_WARN("Android GPU usage: queue_submit2 unknown exception");
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
    (void)queue_family_index;
    (void)present_info;
    (void)swapchain_index;
    (void)image_index;

    if (!ctx)
        return;

    using clock = std::chrono::steady_clock;
    auto now = clock::now();

    try {
        std::lock_guard<std::mutex> g(ctx->lock);

        // CPU 기준 프레임 시간
        float frame_cpu_ms = 16.0f;
        if (ctx->last_present.time_since_epoch().count() != 0) {
            auto dt_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                             now - ctx->last_present)
                             .count();
            if (dt_ms <= 0)
                dt_ms = 1;
            frame_cpu_ms = static_cast<float>(dt_ms);
        }
        ctx->last_present = now;

        // GPU 기준 프레임 시간 (FRAME_LAG 프레임 뒤 슬롯에서 읽기)
        float frame_gpu_ms = 0.0f;

        if (ctx->ts_supported && ctx->query_pool != VK_NULL_HANDLE) {
            if (ctx->frame_index >= AndroidVkGpuContext::FRAME_LAG) {
                const uint64_t read_serial = ctx->frame_index - AndroidVkGpuContext::FRAME_LAG;
                const uint32_t read_idx    = static_cast<uint32_t>(
                    read_serial % AndroidVkGpuContext::MAX_FRAMES
                );

                auto& fr = ctx->frames[read_idx];
                if (fr.frame_serial == read_serial) {
                    frame_gpu_ms = android_gpu_usage_collect_frame_gpu_ms(ctx, fr);
                }
            }
        }

        // 500ms 윈도우 누적 + smoothing
        if (ctx->window_start.time_since_epoch().count() == 0) {
            ctx->window_start = now;
        }

        ctx->acc_cpu_ms += frame_cpu_ms;
        ctx->acc_gpu_ms += frame_gpu_ms;
        ctx->acc_frames += 1;

        constexpr auto WINDOW = std::chrono::milliseconds(500);
        auto elapsed = now - ctx->window_start;

        if (elapsed >= WINDOW && ctx->acc_cpu_ms > 0.0 && ctx->acc_frames > 0) {
            const double avg_cpu_ms = ctx->acc_cpu_ms / static_cast<double>(ctx->acc_frames);
            const double avg_gpu_ms = ctx->acc_gpu_ms / static_cast<double>(ctx->acc_frames);

            float usage = 0.0f;
            if (avg_cpu_ms > 0.0) {
                usage = static_cast<float>((avg_gpu_ms / avg_cpu_ms) * 100.0);
            }

            if (!std::isfinite(usage))
                usage = 0.0f;
            if (usage < 0.0f)
                usage = 0.0f;
            if (usage > 150.0f)
                usage = 150.0f;

            const float alpha = 0.5f;

            if (!ctx->have_metrics) {
                ctx->smooth_usage  = usage;
                ctx->smooth_gpu_ms = static_cast<float>(avg_gpu_ms);
            } else {
                ctx->smooth_usage  =
                    ctx->smooth_usage * (1.0f - alpha) + usage * alpha;
                ctx->smooth_gpu_ms =
                    ctx->smooth_gpu_ms * (1.0f - alpha) +
                    static_cast<float>(avg_gpu_ms) * alpha;
            }

            ctx->last_usage   = ctx->smooth_usage;
            ctx->last_gpu_ms  = ctx->smooth_gpu_ms;
            ctx->have_metrics = true;

            ctx->acc_cpu_ms   = 0.0;
            ctx->acc_gpu_ms   = 0.0;
            ctx->acc_frames   = 0;
            ctx->window_start = now;
        }

        // 다음 프레임 serial
        ctx->frame_index++;
    } catch (const std::exception& e) {
        SPDLOG_WARN("Android GPU usage: on_present exception: {}", e.what());
    } catch (...) {
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

    std::lock_guard<std::mutex> g(ctx->lock);

    if (!ctx->have_metrics)
        return false;

    if (out_gpu_ms)
        *out_gpu_ms = ctx->last_gpu_ms;
    if (out_usage)
        *out_usage  = ctx->last_usage;

    return true;
}

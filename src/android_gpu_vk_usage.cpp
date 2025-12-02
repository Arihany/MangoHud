#include "android_gpu_vk_usage.h"
#include <vulkan/vulkan.h>
#include <chrono>
#include <mutex>
#include <cmath>
#include <vector>
#include <limits>
#include <spdlog/spdlog.h>

// ====================== 내부 상태 ======================

struct AndroidVkGpuContext {
    VkPhysicalDevice          phys_dev      = VK_NULL_HANDLE;
    VkDevice                  device        = VK_NULL_HANDLE;
    AndroidVkGpuDispatch      disp{};

    float                     ts_period_ns  = 0.0f;   // ns per tick
    uint32_t                  ts_valid_bits = 0;
    bool                      ts_supported  = false;

    // 공용 타임스탬프 풀
    VkQueryPool               query_pool    = VK_NULL_HANDLE;
    uint32_t                  queue_family_index = VK_QUEUE_FAMILY_IGNORED;

    // 프레임 링 버퍼 설정
    static constexpr uint32_t MAX_FRAMES              = 4;    // 3프레임 지연 + 1
    static constexpr uint32_t MAX_QUERIES_PER_FRAME   = 128;  // 쿼리 128개 → submit 최대 64번

    struct FrameResources {
        VkCommandPool cmd_pool        = VK_NULL_HANDLE;
        VkCommandBuffer reset_cmd     = VK_NULL_HANDLE;

        uint32_t query_start          = 0;   // 이 프레임의 쿼리 시작 인덱스
        uint32_t query_capacity       = 0;   // MAX_QUERIES_PER_FRAME
        uint32_t query_used           = 0;   // 현재까지 사용한 쿼리 개수 (짝수, 2개씩)

        bool     reset_recorded       = false; // reset_cmd 안에 CmdResetQueryPool 녹화 여부
        bool     reset_submitted      = false; // 이번 프레임에 reset_cmd를 실제 submit에 끼워 넣었는지
        bool     has_queries          = false; // 이 프레임에서 계측된 submit이 하나라도 있는지

        uint64_t frame_serial         = std::numeric_limits<uint64_t>::max();
    };

    FrameResources            frames[MAX_FRAMES]{};

    // "현재 그리고 있는 프레임" 번호
    uint64_t                  frame_index   = 0;

    // 마지막 프레임 기준 값
    std::mutex                lock;
    std::chrono::steady_clock::time_point last_present{};
    float                     last_gpu_ms   = 0.0f;
    float                     last_usage    = 0.0f;
};

// ====================== 헬퍼: 타임스탬프 리소스 초기화 ======================

static bool android_gpu_usage_init_timestamp_resources(
    AndroidVkGpuContext* ctx,
    uint32_t queue_family_index)
{
    if (!ctx || !ctx->ts_supported) {
        SPDLOG_INFO(
            "Android GPU usage: init_timestamp_resources skipped (ctx={} ts_supported={})",
            static_cast<void*>(ctx),
            ctx ? ctx->ts_supported : false
        );
        return false;
    }

    // 이미 초기화되어 있으면 그대로 사용
    if (ctx->query_pool != VK_NULL_HANDLE) {
        SPDLOG_INFO(
            "Android GPU usage: timestamp resources already init (ctx={} qf_index={})",
            static_cast<void*>(ctx),
            queue_family_index
        );
        return true;
    }

    ctx->queue_family_index = queue_family_index;

    // 1) QueryPool 생성 (프레임 × 쿼리 수)
    VkQueryPoolCreateInfo qp{};
    qp.sType       = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
    qp.queryType   = VK_QUERY_TYPE_TIMESTAMP;
    qp.queryCount  = AndroidVkGpuContext::MAX_FRAMES *
                     AndroidVkGpuContext::MAX_QUERIES_PER_FRAME;
    qp.flags       = 0;
    qp.pipelineStatistics = 0;

    if (!ctx->disp.CreateQueryPool ||
        ctx->disp.CreateQueryPool(ctx->device, &qp, nullptr, &ctx->query_pool) != VK_SUCCESS) {
        SPDLOG_WARN(
            "Android GPU usage: CreateQueryPool failed (ctx={} qcount={}) → disabling ts",
            static_cast<void*>(ctx),
            qp.queryCount
        );
        ctx->query_pool   = VK_NULL_HANDLE;
        ctx->ts_supported = false;
        return false;
    }

    // 2) 프레임별 커맨드 풀 만들기
    if (!ctx->disp.CreateCommandPool || !ctx->disp.ResetCommandPool) {
        SPDLOG_WARN(
            "Android GPU usage: Create/ResetCommandPool not available → disabling ts");
        ctx->ts_supported = false;
        return false;
    }

    for (uint32_t i = 0; i < AndroidVkGpuContext::MAX_FRAMES; ++i) {
        VkCommandPoolCreateInfo cp{};
        cp.sType            = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        cp.queueFamilyIndex = queue_family_index;
        cp.flags            = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;

        VkCommandPool pool = VK_NULL_HANDLE;
        if (ctx->disp.CreateCommandPool(ctx->device, &cp, nullptr, &pool) != VK_SUCCESS) {
            SPDLOG_WARN(
                "Android GPU usage: CreateCommandPool failed at frame {} → disabling ts",
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
    }

    SPDLOG_INFO(
        "Android GPU usage: timestamp resources initialized (ctx={} qf_index={} qpool={} MAX_FRAMES={} MAX_QPF={})",
        static_cast<void*>(ctx),
        queue_family_index,
        static_cast<void*>(ctx->query_pool),
        AndroidVkGpuContext::MAX_FRAMES,
        AndroidVkGpuContext::MAX_QUERIES_PER_FRAME
    );
    
    return true;
}

// 프레임 시작 시, 해당 슬롯 리셋 + reset_cmd 준비
static bool android_gpu_usage_begin_frame(
    AndroidVkGpuContext* ctx,
    uint32_t frame_idx)
{
    auto& fr = ctx->frames[frame_idx];

    // 새 프레임으로 전환되었는지 체크
    if (fr.frame_serial == ctx->frame_index) {
        // 이미 이 frame_serial에 대해 초기화 끝난 상태
        return true;
    }

    fr.frame_serial   = ctx->frame_index;
    fr.query_used     = 0;
    fr.reset_recorded = false;
    fr.reset_submitted= false;
    fr.has_queries    = false;

    fr.reset_cmd      = VK_NULL_HANDLE;

    // 커맨드 풀 리셋: 이전 프레임에서 쓰던 CB들 한방에 폐기
    if (ctx->disp.ResetCommandPool && fr.cmd_pool != VK_NULL_HANDLE) {
        ctx->disp.ResetCommandPool(ctx->device, fr.cmd_pool, 0);
    }

    return true;
}

// 이 프레임에서 쿼리 리셋용 커맨드 버퍼 녹화
static bool android_gpu_usage_ensure_reset_cmd(
    AndroidVkGpuContext* ctx,
    AndroidVkGpuContext::FrameResources& fr)
{
    if (fr.reset_recorded)
        return true;

    if (!ctx->disp.AllocateCommandBuffers ||
        !ctx->disp.BeginCommandBuffer ||
        !ctx->disp.EndCommandBuffer ||
        !ctx->disp.CmdResetQueryPool)
        return false;

    // resetCmd 1개 할당
    VkCommandBufferAllocateInfo ai{};
    ai.sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    ai.commandPool        = fr.cmd_pool;
    ai.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    ai.commandBufferCount = 1;

    VkCommandBuffer cmd = VK_NULL_HANDLE;
    if (ctx->disp.AllocateCommandBuffers(ctx->device, &ai, &cmd) != VK_SUCCESS) {
        return false;
    }

    VkCommandBufferBeginInfo bi{};
    bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

    if (ctx->disp.BeginCommandBuffer(cmd, &bi) != VK_SUCCESS)
        return false;

    // 이 프레임의 쿼리 범위 전체를 리셋
    ctx->disp.CmdResetQueryPool(
        cmd,
        ctx->query_pool,
        fr.query_start,
        fr.query_capacity
    );

    if (ctx->disp.EndCommandBuffer(cmd) != VK_SUCCESS)
        return false;

    fr.reset_cmd      = cmd;
    fr.reset_recorded = true;
    return true;
}

// submit 하나에 대한 begin/end CB + 쿼리 인덱스 할당 및 녹화
static bool android_gpu_usage_record_timestamp_pair(
    AndroidVkGpuContext* ctx,
    AndroidVkGpuContext::FrameResources& fr,
    uint32_t&           out_query_first,
    VkCommandBuffer&    out_cmd_begin,
    VkCommandBuffer&    out_cmd_end)
{
    if (!ctx->disp.AllocateCommandBuffers ||
        !ctx->disp.BeginCommandBuffer ||
        !ctx->disp.EndCommandBuffer ||
        !ctx->disp.CmdWriteTimestamp)
        return false;

    // 쿼리 2개(start/end) 남았는지 확인
    if (fr.query_used + 2 > fr.query_capacity)
        return false;

    out_query_first = fr.query_start + fr.query_used;
    fr.query_used  += 2;
    fr.has_queries  = true;

    // 커맨드 버퍼 2개(begin/end) 할당
    VkCommandBufferAllocateInfo ai{};
    ai.sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    ai.commandPool        = fr.cmd_pool;
    ai.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    ai.commandBufferCount = 2;

    VkCommandBuffer pair[2] = { VK_NULL_HANDLE, VK_NULL_HANDLE };
    if (ctx->disp.AllocateCommandBuffers(ctx->device, &ai, pair) != VK_SUCCESS) {
        return false;
    }

    out_cmd_begin = pair[0];
    out_cmd_end   = pair[1];

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
        out_query_first);

    if (ctx->disp.EndCommandBuffer(out_cmd_begin) != VK_SUCCESS)
        return false;

    // end CB: (옵션) 배리어 + End timestamp
    if (ctx->disp.BeginCommandBuffer(out_cmd_end, &bi) != VK_SUCCESS)
        return false;

    if (ctx->disp.CmdPipelineBarrier) {
        // 아주 보수적으로, "모든 커맨드 끝난 뒤" 타임스탬프 찍힐 확률을 조금이라도 올리기 위해
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
        out_query_first + 1u);

    if (ctx->disp.EndCommandBuffer(out_cmd_end) != VK_SUCCESS)
        return false;

    return true;
}

// 특정 프레임 슬롯에 대해 GPU time(ms) 합산
static float android_gpu_usage_collect_frame_gpu_ms(
    AndroidVkGpuContext* ctx,
    AndroidVkGpuContext::FrameResources& fr)
{
    if (!ctx || !ctx->ts_supported || !ctx->query_pool) {
        SPDLOG_INFO(
            "Android GPU usage: collect_frame_gpu_ms skipped (ctx={} ts_supported={} qpool={})",
            static_cast<void*>(ctx),
            ctx ? ctx->ts_supported : false,
            ctx ? static_cast<void*>(ctx->query_pool) : nullptr
        );
        return 0.0f;
    }

    if (!ctx->disp.GetQueryPoolResults)
        return 0.0f;

    if (!fr.has_queries || fr.query_used < 2) {
        SPDLOG_INFO(
            "Android GPU usage: collect_frame_gpu_ms no queries (has_queries={} used={})",
            fr.has_queries,
            fr.query_used
        );
        return 0.0f;
    }

    const uint32_t query_count = fr.query_used;
    const uint32_t pair_count  = query_count / 2;

    // value + availability 쌍
    std::vector<uint64_t> data;
    data.resize(query_count * 2);

    VkResult r = ctx->disp.GetQueryPoolResults(
        ctx->device,
        ctx->query_pool,
        fr.query_start,
        query_count,
        static_cast<uint32_t>(data.size() * sizeof(uint64_t)),
        data.data(),
        sizeof(uint64_t) * 2,
        VK_QUERY_RESULT_64_BIT |
        VK_QUERY_RESULT_WITH_AVAILABILITY_BIT |
        VK_QUERY_RESULT_WAIT_BIT     // 여기서 블록해도, 이미 3프레임 지난 슬롯만 읽으므로 큰 문제 없음
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
        return 0.0f;
    }

    SPDLOG_DEBUG(
        "Android GPU usage: collect_frame_gpu_ms start (query_count={} pair_count={} valid_bits={})",
        query_count,
        pair_count,
        ctx->ts_valid_bits
    );
    
    uint64_t mask = 0;
    if (ctx->ts_valid_bits == 0 || ctx->ts_valid_bits >= 64) {
        mask = ~0ULL;
    } else {
        mask = (1ULL << ctx->ts_valid_bits) - 1ULL;
    }

    double total_ms = 0.0;

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

        if (ctx->ts_valid_bits > 0 && ctx->ts_valid_bits < 64 && end_ts < start_ts) {
            end_ts += (1ULL << ctx->ts_valid_bits);
        }

        if (end_ts <= start_ts)
            continue;

        uint64_t delta_ticks = end_ts - start_ts;
        double   ns          = double(delta_ticks) * double(ctx->ts_period_ns);
        double   ms          = ns * 1e-6;

        if (ms > 0.0 && std::isfinite(ms))
            total_ms += ms;
    }

    // 이 슬롯은 한 번 읽고 나면 비워도 된다.
    fr.has_queries = false;
    fr.query_used  = 0;

    if (total_ms <= 0.0 || !std::isfinite(total_ms)) {
        SPDLOG_DEBUG(
            "Android GPU usage: collect_frame_gpu_ms result <= 0 (total_ms={})",
            total_ms
        );
        return 0.0f;
    }

    SPDLOG_DEBUG(
        "Android GPU usage: collect_frame_gpu_ms done (total_ms={})",
        total_ms
    );
    
    return static_cast<float>(total_ms);
}

// ====================== 외부 API ======================

AndroidVkGpuContext* android_gpu_usage_create(
    VkPhysicalDevice              phys_dev,
    VkDevice                      device,
    float                         timestamp_period_ns,
    uint32_t                      timestamp_valid_bits,
    const AndroidVkGpuDispatch&   disp)
{
    auto ctx = new AndroidVkGpuContext{};
    ctx->phys_dev     = phys_dev;
    ctx->device       = device;
    ctx->disp         = disp;
    ctx->ts_period_ns = (timestamp_period_ns > 0.0f) ? timestamp_period_ns : 0.0f;
    ctx->ts_valid_bits= timestamp_valid_bits;
    ctx->last_present = std::chrono::steady_clock::now();
    ctx->last_gpu_ms  = 0.0f;
    ctx->last_usage   = 0.0f;

    bool dispatch_ok =
        disp.QueueSubmit &&
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

    return ctx;
}

void android_gpu_usage_destroy(AndroidVkGpuContext* ctx)
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

// 핵심: QueueSubmit 래핑 (샌드위치 + resetCmd)
VkResult android_gpu_usage_queue_submit(
    AndroidVkGpuContext*      ctx,
    VkQueue                   queue,
    uint32_t                  queue_family_index,
    uint32_t                  submitCount,
    const VkSubmitInfo*       pSubmits,
    VkFence                   fence)
{
    if (!ctx || !ctx->disp.QueueSubmit || !pSubmits || submitCount == 0) {
        SPDLOG_INFO(
            "Android GPU usage: queue_submit passthrough (ctx={} submitCount={} hasDispatch={})",
            static_cast<void*>(ctx),
            submitCount,
            ctx ? (ctx->disp.QueueSubmit != nullptr) : false
        );
        return (ctx && ctx->disp.QueueSubmit)
            ? ctx->disp.QueueSubmit(queue, submitCount, pSubmits, fence)
            : VK_ERROR_INITIALIZATION_FAILED;
    }

    // 타임스탬프 미지원이면 그냥 패스스루
    if (!ctx->ts_supported) {
        SPDLOG_INFO(
            "Android GPU usage: queue_submit ts_disabled (ctx={} submitCount={} qf_index={})",
            static_cast<void*>(ctx),
            submitCount,
            queue_family_index
        );
        return ctx->disp.QueueSubmit(queue, submitCount, pSubmits, fence);
    }

    std::lock_guard<std::mutex> g(ctx->lock);

    // 리소스 lazy init
    if (!android_gpu_usage_init_timestamp_resources(ctx, queue_family_index)) {
        SPDLOG_WARN(
            "Android GPU usage: init_timestamp_resources failed in queue_submit (ctx={} qf_index={})",
            static_cast<void*>(ctx),
            queue_family_index
        );
        return ctx->disp.QueueSubmit(queue, submitCount, pSubmits, fence);
    }

    const uint32_t curr_idx = static_cast<uint32_t>(
        ctx->frame_index % AndroidVkGpuContext::MAX_FRAMES
    );
    auto& fr = ctx->frames[curr_idx];

    SPDLOG_INFO(
        "Android GPU usage: queue_submit enter (ctx={} frame_index={} slot={} submitCount={} qf_index={} query_used={} cap={})",
        static_cast<void*>(ctx),
        ctx->frame_index,
        curr_idx,
        submitCount,
        queue_family_index,
        fr.query_used,
        fr.query_capacity
    );

    // 새 프레임 시작이면 해당 슬롯 리셋
    android_gpu_usage_begin_frame(ctx, curr_idx);

    std::vector<VkSubmitInfo> wrapped;
    wrapped.reserve(submitCount);

    // per-submit 확장된 CB 배열
    std::vector<std::vector<VkCommandBuffer>> per_submit_cbs(submitCount);

    for (uint32_t i = 0; i < submitCount; ++i) {
        const VkSubmitInfo& src = pSubmits[i];

        // 커맨드 버퍼가 없으면 계측할 게 없음 → 그대로 통과
        if (src.commandBufferCount == 0 ||
            !src.pCommandBuffers ||
            !ctx->ts_supported) {
            wrapped.push_back(src);
            continue;
        }

        // 이 submit을 계측할 수 있는지 체크
        bool can_instrument =
            (ctx->query_pool != VK_NULL_HANDLE) &&
            (fr.cmd_pool   != VK_NULL_HANDLE) &&
            (fr.query_used + 2 <= fr.query_capacity);

        if (!can_instrument) {
            wrapped.push_back(src);
            continue;
        }

        // 이 프레임에서 아직 resetCmd를 안 만들어놨으면, 지금 만든다.
        if (!fr.reset_recorded) {
            if (!android_gpu_usage_ensure_reset_cmd(ctx, fr)) {
                // 실패하면 이번 프레임은 그냥 계측 포기
                wrapped.push_back(src);
                continue;
            }
        }

        // 이 submit에 대한 begin/end 타임스탬프 커맨드 버퍼 준비
        uint32_t       query_first = 0;
        VkCommandBuffer cmd_begin  = VK_NULL_HANDLE;
        VkCommandBuffer cmd_end    = VK_NULL_HANDLE;

        if (!android_gpu_usage_record_timestamp_pair(
                ctx, fr, query_first, cmd_begin, cmd_end)) {
            wrapped.push_back(src);
            continue;
        }

        auto& dst_cbs = per_submit_cbs[i];
        dst_cbs.reserve(src.commandBufferCount + 3);

        // 아직 resetCmd를 큐에 넣은 적이 없으면, 이 submit 앞에 끼운다.
        if (fr.reset_recorded && !fr.reset_submitted && fr.reset_cmd != VK_NULL_HANDLE) {
            dst_cbs.push_back(fr.reset_cmd);
            fr.reset_submitted = true;
        }

        // begin timestamp CB
        dst_cbs.push_back(cmd_begin);

        // 앱의 원래 커맨드 버퍼들
        for (uint32_t j = 0; j < src.commandBufferCount; ++j) {
            dst_cbs.push_back(src.pCommandBuffers[j]);
        }

        // end timestamp CB
        dst_cbs.push_back(cmd_end);

        VkSubmitInfo dst = src;
        dst.commandBufferCount = static_cast<uint32_t>(dst_cbs.size());
        dst.pCommandBuffers   = dst_cbs.data();

        wrapped.push_back(dst);
    }

    // 실제 QueueSubmit 호출
    return ctx->disp.QueueSubmit(
        queue,
        static_cast<uint32_t>(wrapped.size()),
        wrapped.data(),
        fence
    );
}

// Present 시점: 3프레임 전 슬롯 읽어서 GPU time → usage 계산
void android_gpu_usage_on_present(
    AndroidVkGpuContext*      ctx,
    VkQueue                   queue,
    uint32_t                  queue_family_index,
    const VkPresentInfoKHR*   present_info,
    uint32_t                  swapchain_index,
    uint32_t                  image_index)
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

    std::lock_guard<std::mutex> g(ctx->lock);

    // CPU 기준 프레임 시간
    float frame_cpu_ms = 16.0f;
    if (ctx->last_present.time_since_epoch().count() != 0) {
        auto dt_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            now - ctx->last_present).count();
        if (dt_ms <= 0)
            dt_ms = 16;
        frame_cpu_ms = static_cast<float>(dt_ms);
    }
    ctx->last_present = now;

    float gpu_ms = 0.0f;

    if (ctx->ts_supported && ctx->query_pool != VK_NULL_HANDLE) {
        // 3프레임 지연 후 읽기
        if (ctx->frame_index >= (AndroidVkGpuContext::MAX_FRAMES - 1)) {
            uint64_t read_serial = ctx->frame_index - (AndroidVkGpuContext::MAX_FRAMES - 1);
            uint32_t read_idx    = static_cast<uint32_t>(
                read_serial % AndroidVkGpuContext::MAX_FRAMES
            );

            auto& fr = ctx->frames[read_idx];
            if (fr.frame_serial == read_serial) {
                gpu_ms = android_gpu_usage_collect_frame_gpu_ms(ctx, fr);
            }
        }
    }

    ctx->last_gpu_ms = gpu_ms;

    float usage = 0.0f;
    if (frame_cpu_ms > 0.5f && gpu_ms > 0.0f) {
        usage = (gpu_ms / frame_cpu_ms) * 100.0f;
    }

    if (!std::isfinite(usage))
        usage = 0.0f;

    if (usage < 0.0f)   usage = 0.0f;
    if (usage > 150.0f) usage = 150.0f;

    ctx->last_usage = usage;

    // 다음 프레임으로
    ctx->frame_index++;
}

bool android_gpu_usage_get_metrics(
    AndroidVkGpuContext*  ctx,
    float*                out_gpu_ms,
    float*                out_usage)
{
    if (!ctx)
        return false;

    std::lock_guard<std::mutex> g(ctx->lock);

    if (out_gpu_ms)
        *out_gpu_ms = ctx->last_gpu_ms;
    if (out_usage)
        *out_usage  = ctx->last_usage;

    // ts 미지원이어도, 값은 찍힐 수 있으니 true
    return true;
}

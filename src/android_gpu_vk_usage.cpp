#include "android_gpu_vk_usage.h"
#include <vulkan/vulkan.h>
#include <chrono>
#include <mutex>
#include <cmath>
#include <vector>

// ====================== 내부 상태 ======================

struct AndroidVkGpuContext {
    VkPhysicalDevice          phys_dev      = VK_NULL_HANDLE;
    VkDevice                  device        = VK_NULL_HANDLE;
    float                     ts_period_ns  = 0.0f;   // ns per tick

    AndroidVkGpuDispatch      disp{};

    bool                      ts_supported  = false;

    // 타임스탬프용 리소스
    uint32_t                  queue_family_index = VK_QUEUE_FAMILY_IGNORED;
    VkCommandPool             cmd_pool      = VK_NULL_HANDLE;
    VkQueryPool               query_pool    = VK_NULL_HANDLE;

    // submit 단위 타이밍 슬롯
    static constexpr uint32_t MAX_SLOTS = 64;

    struct SubmitSlot {
        VkCommandBuffer cmd_begin = VK_NULL_HANDLE;
        VkCommandBuffer cmd_end   = VK_NULL_HANDLE;
        uint32_t        query_first = 0;   // start = query_first, end = query_first + 1

        bool            pending    = false;  // GPU 쿼리 완료 대기 중
        uint64_t        gpu_start  = 0;
        uint64_t        gpu_end    = 0;

        std::chrono::steady_clock::time_point cpu_submit{};
    };

    SubmitSlot                slots[MAX_SLOTS]{};
    uint32_t                  slot_cursor = 0;

    // 마지막 프레임 기준 값
    std::mutex                lock;
    float                     last_gpu_ms   = 16.0f;
    float                     last_usage    = 0.0f;
    std::chrono::steady_clock::time_point last_present{};
};

// ====================== 헬퍼 함수 ======================

static bool android_gpu_usage_init_pools(AndroidVkGpuContext* ctx, uint32_t queue_family_index)
{
    if (!ctx || !ctx->ts_supported)
        return false;

    if (ctx->cmd_pool != VK_NULL_HANDLE && ctx->query_pool != VK_NULL_HANDLE)
        return true;

    ctx->queue_family_index = queue_family_index;

    // Command pool
    VkCommandPoolCreateInfo cp{};
    cp.sType            = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    cp.queueFamilyIndex = queue_family_index;
    cp.flags            = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;

    if (!ctx->disp.CreateCommandPool ||
        ctx->disp.CreateCommandPool(ctx->device, &cp, nullptr, &ctx->cmd_pool) != VK_SUCCESS) {
        ctx->cmd_pool = VK_NULL_HANDLE;
        ctx->ts_supported = false;
        return false;
    }

    // Query pool (timestamp)
    VkQueryPoolCreateInfo qp{};
    qp.sType              = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
    qp.queryType          = VK_QUERY_TYPE_TIMESTAMP;
    qp.queryCount         = AndroidVkGpuContext::MAX_SLOTS * 2;
    qp.flags              = 0;
    qp.pipelineStatistics = 0;

    if (!ctx->disp.CreateQueryPool ||
        ctx->disp.CreateQueryPool(ctx->device, &qp, nullptr, &ctx->query_pool) != VK_SUCCESS) {
        if (ctx->disp.DestroyCommandPool && ctx->cmd_pool != VK_NULL_HANDLE)
            ctx->disp.DestroyCommandPool(ctx->device, ctx->cmd_pool, nullptr);
        ctx->cmd_pool   = VK_NULL_HANDLE;
        ctx->query_pool = VK_NULL_HANDLE;
        ctx->ts_supported = false;
        return false;
    }

    // Command buffer들 할당 (슬롯당 2개)
    VkCommandBufferAllocateInfo ai{};
    ai.sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    ai.commandPool        = ctx->cmd_pool;
    ai.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    ai.commandBufferCount = AndroidVkGpuContext::MAX_SLOTS * 2;

    std::vector<VkCommandBuffer> bufs(ai.commandBufferCount);

    if (!ctx->disp.AllocateCommandBuffers ||
        ctx->disp.AllocateCommandBuffers(ctx->device, &ai, bufs.data()) != VK_SUCCESS) {
        if (ctx->disp.DestroyQueryPool && ctx->query_pool != VK_NULL_HANDLE)
            ctx->disp.DestroyQueryPool(ctx->device, ctx->query_pool, nullptr);
        if (ctx->disp.DestroyCommandPool && ctx->cmd_pool != VK_NULL_HANDLE)
            ctx->disp.DestroyCommandPool(ctx->device, ctx->cmd_pool, nullptr);
        ctx->cmd_pool   = VK_NULL_HANDLE;
        ctx->query_pool = VK_NULL_HANDLE;
        ctx->ts_supported = false;
        return false;
    }

    for (uint32_t i = 0; i < AndroidVkGpuContext::MAX_SLOTS; ++i) {
        ctx->slots[i].cmd_begin   = bufs[2u * i + 0u];
        ctx->slots[i].cmd_end     = bufs[2u * i + 1u];
        ctx->slots[i].query_first = 2u * i;
        ctx->slots[i].pending     = false;
        ctx->slots[i].gpu_start   = 0;
        ctx->slots[i].gpu_end     = 0;
    }

    return true;
}

static AndroidVkGpuContext::SubmitSlot* android_gpu_usage_alloc_slot(
    AndroidVkGpuContext* ctx,
    const std::chrono::steady_clock::time_point& now)
{
    if (!ctx)
        return nullptr;

    for (uint32_t i = 0; i < AndroidVkGpuContext::MAX_SLOTS; ++i) {
        uint32_t idx = (ctx->slot_cursor + i) % AndroidVkGpuContext::MAX_SLOTS;
        auto& slot = ctx->slots[idx];
        if (!slot.pending) {
            ctx->slot_cursor = (idx + 1u) % AndroidVkGpuContext::MAX_SLOTS;
            slot.pending     = true;
            slot.gpu_start   = 0;
            slot.gpu_end     = 0;
            slot.cpu_submit  = now;
            return &slot;
        }
    }

    // 전부 pending이면 그냥 이번 submit은 계측 포기
    return nullptr;
}

static void android_gpu_usage_record_slot_cmds(AndroidVkGpuContext* ctx,
                                               AndroidVkGpuContext::SubmitSlot& slot)
{
    if (!ctx || !ctx->ts_supported || !ctx->cmd_pool || !ctx->query_pool)
        return;

    if (!ctx->disp.ResetCommandBuffer ||
        !ctx->disp.BeginCommandBuffer ||
        !ctx->disp.EndCommandBuffer ||
        !ctx->disp.CmdWriteTimestamp)
        return;

    // begin cmd: reset + start timestamp
    ctx->disp.ResetCommandBuffer(slot.cmd_begin, 0);

    VkCommandBufferBeginInfo bi{};
    bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

    ctx->disp.BeginCommandBuffer(slot.cmd_begin, &bi);

    // 해당 슬롯의 2개 쿼리 reset
    vkCmdResetQueryPool(slot.cmd_begin, ctx->query_pool, slot.query_first, 2);

    ctx->disp.CmdWriteTimestamp(
        slot.cmd_begin,
        VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
        ctx->query_pool,
        slot.query_first);

    ctx->disp.EndCommandBuffer(slot.cmd_begin);

    // end cmd: end timestamp만
    ctx->disp.ResetCommandBuffer(slot.cmd_end, 0);
    ctx->disp.BeginCommandBuffer(slot.cmd_end, &bi);

    ctx->disp.CmdWriteTimestamp(
        slot.cmd_end,
        VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
        ctx->query_pool,
        slot.query_first + 1u);

    ctx->disp.EndCommandBuffer(slot.cmd_end);
}

static float android_gpu_usage_collect_gpu_ms_locked(AndroidVkGpuContext* ctx)
{
    if (!ctx || !ctx->ts_supported || !ctx->query_pool || !ctx->disp.GetQueryPoolResults)
        return 0.0f;

    float total_ms = 0.0f;

    for (uint32_t i = 0; i < AndroidVkGpuContext::MAX_SLOTS; ++i) {
        auto& slot = ctx->slots[i];
        if (!slot.pending)
            continue;

        uint64_t data[4] = {0, 0, 0, 0};

        VkResult r = ctx->disp.GetQueryPoolResults(
            ctx->device,
            ctx->query_pool,
            slot.query_first,
            2,                        // start, end
            sizeof(data),
            data,
            sizeof(uint64_t) * 2,     // stride: [value, avail] 한 쌍
            VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WITH_AVAILABILITY_BIT);

        if (r == VK_NOT_READY)
            continue;

        if (r < 0) {
            // 드라이버 에러 같은 건 그냥 이 슬롯 버리고 치움
            slot.pending = false;
            continue;
        }

        uint64_t start_ts = data[0];
        uint64_t avail0   = data[1];
        uint64_t end_ts   = data[2];
        uint64_t avail1   = data[3];

        if (!avail0 || !avail1) {
            // 아직 완전히 안 끝난 듯
            continue;
        }

        slot.pending = false;

        if (end_ts <= start_ts)
            continue;

        double ns = double(end_ts - start_ts) * double(ctx->ts_period_ns);
        float  ms = static_cast<float>(ns * 1e-6);

        if (ms > 0.0f && std::isfinite(ms))
            total_ms += ms;
    }

    return total_ms;
}

// ====================== 외부 API ======================

AndroidVkGpuContext* android_gpu_usage_create(
    VkPhysicalDevice                phys_dev,
    VkDevice                        device,
    float                           timestamp_period_ns,
    const AndroidVkGpuDispatch&     disp)
{
    auto ctx = new AndroidVkGpuContext{};
    ctx->phys_dev     = phys_dev;
    ctx->device       = device;
    ctx->ts_period_ns = (timestamp_period_ns > 0.0f) ? timestamp_period_ns : 0.0f;
    ctx->disp         = disp;
    ctx->last_present = std::chrono::steady_clock::now();
    ctx->last_gpu_ms  = 16.0f;
    ctx->last_usage   = 0.0f;

    bool dispatch_ok =
        disp.QueueSubmit &&
        disp.CreateQueryPool &&
        disp.DestroyQueryPool &&
        disp.GetQueryPoolResults &&
        disp.CreateCommandPool &&
        disp.DestroyCommandPool &&
        disp.AllocateCommandBuffers &&
        disp.FreeCommandBuffers &&
        disp.ResetCommandBuffer &&
        disp.BeginCommandBuffer &&
        disp.EndCommandBuffer &&
        disp.CmdWriteTimestamp;

    ctx->ts_supported = (ctx->ts_period_ns > 0.0f) && dispatch_ok;
    return ctx;
}

void android_gpu_usage_destroy(AndroidVkGpuContext* ctx)
{
    if (!ctx)
        return;

    if (ctx->device != VK_NULL_HANDLE) {
        if (ctx->cmd_pool != VK_NULL_HANDLE && ctx->disp.FreeCommandBuffers) {
            VkCommandBuffer bufs[AndroidVkGpuContext::MAX_SLOTS * 2];
            uint32_t count = 0;
            for (uint32_t i = 0; i < AndroidVkGpuContext::MAX_SLOTS; ++i) {
                if (ctx->slots[i].cmd_begin)
                    bufs[count++] = ctx->slots[i].cmd_begin;
                if (ctx->slots[i].cmd_end)
                    bufs[count++] = ctx->slots[i].cmd_end;
            }
            if (count) {
                ctx->disp.FreeCommandBuffers(ctx->device, ctx->cmd_pool, count, bufs);
            }
        }

        if (ctx->query_pool != VK_NULL_HANDLE && ctx->disp.DestroyQueryPool) {
            ctx->disp.DestroyQueryPool(ctx->device, ctx->query_pool, nullptr);
        }

        if (ctx->cmd_pool != VK_NULL_HANDLE && ctx->disp.DestroyCommandPool) {
            ctx->disp.DestroyCommandPool(ctx->device, ctx->cmd_pool, nullptr);
        }
    }

    delete ctx;
}

// 핵심: QueueSubmit 래핑
VkResult android_gpu_usage_queue_submit(
    AndroidVkGpuContext*      ctx,
    VkQueue                   queue,
    uint32_t                  queue_family_index,
    uint32_t                  submitCount,
    const VkSubmitInfo*       pSubmits,
    VkFence                   fence)
{
    if (!ctx || !pSubmits || submitCount == 0 || !ctx->disp.QueueSubmit) {
        // 그냥 패스스루
        if (ctx && ctx->disp.QueueSubmit)
            return ctx->disp.QueueSubmit(queue, submitCount, pSubmits, fence);
        return VK_ERROR_INITIALIZATION_FAILED;
    }

    // 타임스탬프 미지원이면 그냥 패스스루
    if (!ctx->ts_supported) {
        return ctx->disp.QueueSubmit(queue, submitCount, pSubmits, fence);
    }

    std::lock_guard<std::mutex> g(ctx->lock);

    // 리소스 lazy init
    if (!android_gpu_usage_init_pools(ctx, queue_family_index)) {
        return ctx->disp.QueueSubmit(queue, submitCount, pSubmits, fence);
    }

    using clock = std::chrono::steady_clock;
    auto now = clock::now();

    std::vector<VkSubmitInfo> submits;
    submits.reserve(submitCount);

    // per-submit로 확장된 커맨드 버퍼 배열을 유지
    std::vector<std::vector<VkCommandBuffer>> extra_cbs(submitCount);

    for (uint32_t i = 0; i < submitCount; ++i) {
        const VkSubmitInfo& src = pSubmits[i];

        // 커맨드 버퍼가 없으면 계측할 게 없음
        if (src.commandBufferCount == 0) {
            submits.push_back(src);
            continue;
        }

        auto* slot = android_gpu_usage_alloc_slot(ctx, now);
        if (!slot || !slot->cmd_begin || !slot->cmd_end) {
            // 슬롯 부족하면 이 submit은 그냥 통과
            submits.push_back(src);
            continue;
        }

        android_gpu_usage_record_slot_cmds(ctx, *slot);

        auto& dst_cbs = extra_cbs[i];
        dst_cbs.reserve(src.commandBufferCount + 2);

        dst_cbs.push_back(slot->cmd_begin);
        for (uint32_t j = 0; j < src.commandBufferCount; ++j)
            dst_cbs.push_back(src.pCommandBuffers[j]);
        dst_cbs.push_back(slot->cmd_end);

        VkSubmitInfo dst = src;
        dst.commandBufferCount = static_cast<uint32_t>(dst_cbs.size());
        dst.pCommandBuffers   = dst_cbs.data();

        submits.push_back(dst);
    }

    // 실제 QueueSubmit 호출
    return ctx->disp.QueueSubmit(queue,
                                 static_cast<uint32_t>(submits.size()),
                                 submits.data(),
                                 fence);
}

// Present 시점에서 CPU 프레임 간격 + GPU 타임 집계 → usage 계산
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
        auto dt = std::chrono::duration_cast<std::chrono::milliseconds>(now - ctx->last_present).count();
        if (dt <= 0)
            dt = 16;
        frame_cpu_ms = static_cast<float>(dt);
    }
    ctx->last_present = now;

    // 타임스탬프 미지원이면 그냥 "프레임 시간 기반 때려맞추기"로라도 값은 찍어주자
    if (!ctx->ts_supported || !ctx->query_pool) {
        float gpu_ms = std::min(frame_cpu_ms, 33.0f);
        float usage  = (frame_cpu_ms > 0.5f)
            ? (gpu_ms / frame_cpu_ms) * 100.0f
            : 0.0f;

        ctx->last_gpu_ms = gpu_ms;
        ctx->last_usage  = usage;
        return;
    }

    // 이 프레임까지 새로 완료된 submit들의 GPU 시간 합산
    float gpu_ms_this_frame = android_gpu_usage_collect_gpu_ms_locked(ctx);

    // EMA로 부드럽게
    const float alpha = 0.2f;

    if (gpu_ms_this_frame > 0.0f && std::isfinite(gpu_ms_this_frame)) {
        if (ctx->last_gpu_ms <= 0.0f)
            ctx->last_gpu_ms = gpu_ms_this_frame;
        else
            ctx->last_gpu_ms = (1.0f - alpha) * ctx->last_gpu_ms + alpha * gpu_ms_this_frame;
    } else {
        // 새 데이터 없으면 살짝씩 감쇠
        ctx->last_gpu_ms *= 0.98f;
    }

    float raw_usage = 0.0f;
    if (frame_cpu_ms > 0.5f && ctx->last_gpu_ms > 0.0f) {
        raw_usage = (ctx->last_gpu_ms / frame_cpu_ms) * 100.0f;
    }

    if (!std::isfinite(raw_usage))
        raw_usage = 0.0f;

    if (ctx->last_usage <= 0.0f)
        ctx->last_usage = raw_usage;
    else
        ctx->last_usage = (1.0f - alpha) * ctx->last_usage + alpha * raw_usage;

    if (ctx->last_usage < 0.0f)   ctx->last_usage = 0.0f;
    if (ctx->last_usage > 150.0f) ctx->last_usage = 150.0f;
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

    // ts 미지원이어도, 어찌됐건 무언가는 넣어주니까 true 반환
    return true;
}

#include "android_gpu_vk_usage.h"
#include <chrono>
#include <mutex>
#include <atomic>
#include <cmath>

struct AndroidVkGpuContext {
    VkPhysicalDevice          phys_dev      = VK_NULL_HANDLE;
    VkDevice                  device        = VK_NULL_HANDLE;
    float                     ts_period_ns  = 1.0f;

    AndroidVkGpuDispatch      disp{};

    // 아주 대충 흉내내는 상태값
    std::mutex                lock;
    float                     last_gpu_ms   = 16.0f;
    float                     last_usage    = 0.0f;
    std::chrono::steady_clock::time_point last_present;
};

AndroidVkGpuContext* android_gpu_usage_create(
    VkPhysicalDevice                phys_dev,
    VkDevice                        device,
    float                           timestamp_period_ns,
    const AndroidVkGpuDispatch&     disp)
{
    auto ctx = new AndroidVkGpuContext{};
    ctx->phys_dev     = phys_dev;
    ctx->device       = device;
    ctx->ts_period_ns = (timestamp_period_ns > 0.0f) ? timestamp_period_ns : 1.0f;
    ctx->disp         = disp;
    ctx->last_present = std::chrono::steady_clock::now();
    ctx->last_gpu_ms  = 16.0f;
    ctx->last_usage   = 0.0f;
    return ctx;
}

void android_gpu_usage_destroy(AndroidVkGpuContext* ctx)
{
    delete ctx;
}

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

    auto dt = std::chrono::duration_cast<std::chrono::milliseconds>(now - ctx->last_present).count();
    if (dt <= 0)
        dt = 16;

    // gpu 시간은 일단 "프레임당 60%만 일했다고 치자" 라는 병맛 가정
    float gpu_ms = static_cast<float>(dt) * 0.6f;

    // fake usage: 0~100 사이에서 완만하게 왔다갔다하는 파형
    float t = static_cast<float>(std::chrono::duration_cast<std::chrono::milliseconds>(
        now.time_since_epoch()).count()) / 1000.0f;

    float usage = 50.0f + 40.0f * std::sin(t * 0.8f);

    if (usage < 0.0f)   usage = 0.0f;
    if (usage > 100.0f) usage = 100.0f;

    ctx->last_present = now;
    ctx->last_gpu_ms  = gpu_ms;
    ctx->last_usage   = usage;
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
        *out_usage = ctx->last_usage;

    return true;
}

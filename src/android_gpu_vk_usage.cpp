// android_gpu_vk_usage.cpp
#if defined(__ANDROID__)

#include "android_gpu_vk_usage.h"

#include <chrono>
#include <cmath>
#include <algorithm>

struct AndroidVkGpuContext {
    VkPhysicalDevice phys = VK_NULL_HANDLE;
    VkDevice         device = VK_NULL_HANDLE;
    float            timestamp_period_ns = 0.0f;

    // 더미용 상태
    std::chrono::steady_clock::time_point start_time;
    std::chrono::steady_clock::time_point last_update;
    uint64_t         frame_counter = 0;

    float            last_gpu_time_ms   = 0.0f;
    float            last_gpu_usage_pct = 0.0f;
};

static inline float clampf(float v, float lo, float hi)
{
    return std::max(lo, std::min(hi, v));
}

AndroidVkGpuContext* android_gpu_usage_create(
    VkPhysicalDevice phys,
    VkDevice         device,
    float            timestamp_period_ns)
{
    auto* ctx = new AndroidVkGpuContext{};
    ctx->phys               = phys;
    ctx->device             = device;
    ctx->timestamp_period_ns = timestamp_period_ns;
    ctx->start_time         = std::chrono::steady_clock::now();
    ctx->last_update        = ctx->start_time;
    ctx->frame_counter      = 0;
    ctx->last_gpu_time_ms   = 0.0f;
    ctx->last_gpu_usage_pct = 0.0f;
    return ctx;
}

void android_gpu_usage_destroy(AndroidVkGpuContext* ctx)
{
    if (!ctx)
        return;
    delete ctx;
}

// 여기서는 진짜 GPU 타임스탬프 안 쓰고,
// 프레임마다 "요동치는" 가짜 usage를 만든다.
void android_gpu_usage_on_present(
    AndroidVkGpuContext*    ctx,
    VkQueue                 present_queue,
    uint32_t                present_queue_family,
    const VkPresentInfoKHR* pPresentInfo,
    uint32_t                swapchain_index,
    uint32_t                image_index)
{
    (void)present_queue;
    (void)present_queue_family;
    (void)pPresentInfo;
    (void)swapchain_index;
    (void)image_index;

    if (!ctx)
        return;

    ctx->frame_counter++;

    auto now = std::chrono::steady_clock::now();
    std::chrono::duration<float> since_start = now - ctx->start_time;
    std::chrono::duration<float> since_last  = now - ctx->last_update;

    // 기본 파형: 0~100% 사이에서 5초 주기로 왕복하는 사인파
    float t_sec = since_start.count();
    float base = 50.0f + 50.0f * std::sin(t_sec * 2.0f * 3.14159265f / 5.0f); // 5초 주기

    // 프레임 카운터 기반으로 살짝 톱니파 느낌 섞기
    float saw = static_cast<float>((ctx->frame_counter % 60)) / 60.0f; // 0~1
    float mod = (saw * 30.0f) - 15.0f; // -15~+15

    float usage = clampf(base + mod, 0.0f, 100.0f);

    // 가짜 GPU 프레임 시간(ms): usage와 반비례하게 대충
    // (이 값이 HUD 어디에 보이는지도 확인용)
    float cpu_dt_ms = since_last.count() * 1000.0f;
    if (cpu_dt_ms <= 0.1f)
        cpu_dt_ms = 0.1f;

    float gpu_time_ms = clampf(usage / 100.0f * 25.0f, 1.0f, 25.0f);

    ctx->last_gpu_usage_pct = usage;
    ctx->last_gpu_time_ms   = gpu_time_ms;
    ctx->last_update        = now;
}

bool android_gpu_usage_get_metrics(
    AndroidVkGpuContext* ctx,
    float*               out_gpu_time_ms,
    float*               out_gpu_usage_percent)
{
    if (!ctx || !out_gpu_time_ms || !out_gpu_usage_percent)
        return false;

    if (ctx->frame_counter == 0)
        return false; // 아직 아무 프레젠트도 안 지나감

    *out_gpu_time_ms      = ctx->last_gpu_time_ms;
    *out_gpu_usage_percent = ctx->last_gpu_usage_pct;
    return true;
}

#endif // __ANDROID__

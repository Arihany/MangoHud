#pragma once

#if defined(__ANDROID__)
#include <vulkan/vulkan.h>
#include <stdint.h>

struct AndroidVkGpuContext;

AndroidVkGpuContext* android_gpu_usage_create(
    VkPhysicalDevice        phys,
    VkDevice                device,
    float                   timestamp_period_ns);

void android_gpu_usage_destroy(AndroidVkGpuContext* ctx);

void android_gpu_usage_on_present(
    AndroidVkGpuContext*    ctx,
    VkQueue                 present_queue,
    uint32_t                present_queue_family,
    const VkPresentInfoKHR* pPresentInfo,
    uint32_t                swapchain_index,   // i in 0..swapchainCount-1
    uint32_t                image_index);

bool android_gpu_usage_get_metrics(
    AndroidVkGpuContext*    ctx,
    float*                  out_gpu_time_ms,
    float*                  out_gpu_usage_percent);

#endif // __ANDROID__

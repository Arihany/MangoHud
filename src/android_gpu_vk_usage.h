// android_gpu_vk_usage.h
#pragma once

#if defined(__ANDROID__)

#include <vulkan/vulkan.h>
#include <stdint.h>

struct AndroidVkGpuContext;

// 디바이스 단위 컨텍스트 생성
AndroidVkGpuContext* android_gpu_usage_create(
    VkPhysicalDevice        phys,
    VkDevice                device,
    float                   timestamp_period_ns);

// 삭제
void android_gpu_usage_destroy(AndroidVkGpuContext* ctx);

// QueuePresent 훅에서 호출: 현재는 "프레임 카운터 + 시간" 기반 가짜 usage만 갱신
void android_gpu_usage_on_present(
    AndroidVkGpuContext*    ctx,
    VkQueue                 present_queue,
    uint32_t                present_queue_family,
    const VkPresentInfoKHR* pPresentInfo,
    uint32_t                swapchain_index,
    uint32_t                image_index);

// HUD 스냅샷 시점에 값 읽기
// 새 값이 있으면 true, 아니면 false
bool android_gpu_usage_get_metrics(
    AndroidVkGpuContext*    ctx,
    float*                  out_gpu_time_ms,
    float*                  out_gpu_usage_percent);

#endif // __ANDROID__

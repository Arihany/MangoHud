#pragma once

#include <vulkan/vulkan.h>

// Vulkan 디스패치 테이블 모음
struct AndroidVkGpuDispatch {
    PFN_vkQueueSubmit             QueueSubmit;
    PFN_vkCreateQueryPool         CreateQueryPool;
    PFN_vkDestroyQueryPool        DestroyQueryPool;
    PFN_vkGetQueryPoolResults     GetQueryPoolResults;

    PFN_vkCreateCommandPool       CreateCommandPool;
    PFN_vkDestroyCommandPool      DestroyCommandPool;
    PFN_vkAllocateCommandBuffers  AllocateCommandBuffers;
    PFN_vkFreeCommandBuffers      FreeCommandBuffers;
    PFN_vkResetCommandBuffer      ResetCommandBuffer;
    PFN_vkBeginCommandBuffer      BeginCommandBuffer;
    PFN_vkEndCommandBuffer        EndCommandBuffer;

    PFN_vkCmdWriteTimestamp       CmdWriteTimestamp;
};

struct AndroidVkGpuContext;

// 컨텍스트 생성 / 파괴
// timestamp_period_ns: VkPhysicalDeviceProperties::limits.timestampPeriod (ns per tick)
AndroidVkGpuContext* android_gpu_usage_create(
    VkPhysicalDevice              phys_dev,
    VkDevice                      device,
    float                         timestamp_period_ns,
    const AndroidVkGpuDispatch&   disp);

void android_gpu_usage_destroy(AndroidVkGpuContext* ctx);

// vkQueueSubmit 훅에서 호출 (안드로이드 전용 경로)
// 내부에서 타임스탬프 커맨드 버퍼를 앞뒤로 붙이고 disp.QueueSubmit 호출
VkResult android_gpu_usage_queue_submit(
    AndroidVkGpuContext*          ctx,
    VkQueue                       queue,
    uint32_t                      queue_family_index,
    uint32_t                      submitCount,
    const VkSubmitInfo*           pSubmits,
    VkFence                       fence);

// vkQueuePresentKHR에서 호출
void android_gpu_usage_on_present(
    AndroidVkGpuContext*          ctx,
    VkQueue                       queue,
    uint32_t                      queue_family_index,
    const VkPresentInfoKHR*       present_info,
    uint32_t                      swapchain_index,
    uint32_t                      image_index);

// 최신 GPU time(ms), usage(%) 가져오기
bool android_gpu_usage_get_metrics(
    AndroidVkGpuContext*          ctx,
    float*                        out_gpu_ms,
    float*                        out_usage);

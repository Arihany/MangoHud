#pragma once

#include <vulkan/vulkan.h>

// 전방 선언: 구현은 android_gpu_vk_usage.cpp 안
struct AndroidVkGpuContext;

struct AndroidVkGpuDispatch {
    PFN_vkQueueSubmit            QueueSubmit;
    PFN_vkCreateQueryPool        CreateQueryPool;
    PFN_vkDestroyQueryPool       DestroyQueryPool;
    PFN_vkGetQueryPoolResults    GetQueryPoolResults;

    PFN_vkCreateCommandPool      CreateCommandPool;
    PFN_vkDestroyCommandPool     DestroyCommandPool;
    PFN_vkAllocateCommandBuffers AllocateCommandBuffers;
    PFN_vkFreeCommandBuffers     FreeCommandBuffers;
    PFN_vkResetCommandBuffer     ResetCommandBuffer;

    PFN_vkBeginCommandBuffer     BeginCommandBuffer;
    PFN_vkEndCommandBuffer       EndCommandBuffer;
    PFN_vkCmdWriteTimestamp      CmdWriteTimestamp;
};

// 컨텍스트 생성/파괴
AndroidVkGpuContext* android_gpu_usage_create(
    VkPhysicalDevice           phys,
    VkDevice                   device,
    float                      timestamp_period_ns,
    const AndroidVkGpuDispatch& dispatch);

void android_gpu_usage_destroy(AndroidVkGpuContext* ctx);

// QueuePresentKHR 훅에서 호출
void android_gpu_usage_on_present(
    AndroidVkGpuContext*      ctx,
    VkQueue                   present_queue,
    uint32_t                  queue_family_index,
    const VkPresentInfoKHR*   pPresentInfo,
    uint32_t                  swapchain_index,
    uint32_t                  image_index);

// 스냅샷에서 읽기용
bool android_gpu_usage_get_metrics(
    AndroidVkGpuContext*  ctx,
    float*                out_gpu_ms,
    float*                out_gpu_usage);

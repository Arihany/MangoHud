#pragma once

#include <vulkan/vulkan.h>

// Vulkan 디스패치 테이블 모음
struct AndroidVkGpuDispatch {
    // 큐/쿼리 관련
    PFN_vkQueueSubmit             QueueSubmit;
    PFN_vkQueueSubmit2            QueueSubmit2;
    PFN_vkQueueSubmit2KHR         QueueSubmit2KHR;
    PFN_vkCreateQueryPool         CreateQueryPool;
    PFN_vkDestroyQueryPool        DestroyQueryPool;
    PFN_vkGetQueryPoolResults     GetQueryPoolResults;

    // 커맨드 풀 / 버퍼
    PFN_vkCreateCommandPool       CreateCommandPool;
    PFN_vkDestroyCommandPool      DestroyCommandPool;
    PFN_vkResetCommandPool        ResetCommandPool;
    PFN_vkAllocateCommandBuffers  AllocateCommandBuffers;
    PFN_vkFreeCommandBuffers      FreeCommandBuffers;
    PFN_vkBeginCommandBuffer      BeginCommandBuffer;
    PFN_vkEndCommandBuffer        EndCommandBuffer;

    // 타임스탬프 / 쿼리 / 배리어
    PFN_vkCmdWriteTimestamp       CmdWriteTimestamp;
    PFN_vkCmdResetQueryPool       CmdResetQueryPool;
    PFN_vkCmdPipelineBarrier      CmdPipelineBarrier;
};

struct AndroidVkGpuContext;

// 컨텍스트 생성 / 파괴
AndroidVkGpuContext* android_gpu_usage_create(
    VkPhysicalDevice              phys_dev,
    VkDevice                      device,
    float                         timestamp_period_ns,
    uint32_t                      timestamp_valid_bits,
    const AndroidVkGpuDispatch&   disp);

void android_gpu_usage_destroy(AndroidVkGpuContext* ctx);

// vkQueueSubmit 훅에서 호출
VkResult android_gpu_usage_queue_submit(
    AndroidVkGpuContext*          ctx,
    VkQueue                       queue,
    uint32_t                      queue_family_index,
    uint32_t                      submitCount,
    const VkSubmitInfo*           pSubmits,
    VkFence                       fence);

// vkQueueSubmit2 / vkQueueSubmit2KHR 훅에서 호출
VkResult android_gpu_usage_queue_submit2(
    AndroidVkGpuContext*          ctx,
    VkQueue                       queue,
    uint32_t                      queue_family_index,
    uint32_t                      submitCount,
    const VkSubmitInfo2*          pSubmits);

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

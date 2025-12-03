#pragma once
#include <vulkan/vulkan.h>

#ifndef PFN_vkQueueSubmit2

// VkFlags 기반
typedef VkFlags VkSubmitFlags;

// forward decl
typedef struct VkSemaphoreSubmitInfo VkSemaphoreSubmitInfo;

// 스펙 그대로 레이아웃 맞춤
typedef struct VkCommandBufferSubmitInfo {
    VkStructureType sType;
    const void*     pNext;
    VkCommandBuffer commandBuffer;
    uint32_t        deviceMask;
} VkCommandBufferSubmitInfo;

typedef struct VkSubmitInfo2 {
    VkStructureType                   sType;
    const void*                       pNext;
    VkSubmitFlags                     flags;
    uint32_t                          waitSemaphoreInfoCount;
    const VkSemaphoreSubmitInfo*      pWaitSemaphoreInfos;
    uint32_t                          commandBufferInfoCount;
    const VkCommandBufferSubmitInfo*  pCommandBufferInfos;
    uint32_t                          signalSemaphoreInfoCount;
    const VkSemaphoreSubmitInfo*      pSignalSemaphoreInfos;
} VkSubmitInfo2;

// 함수 포인터 프로토타입 (vkQueueSubmit2 / vkQueueSubmit2KHR)
typedef VkResult (VKAPI_PTR *PFN_vkQueueSubmit2)(
    VkQueue            queue,
    uint32_t           submitCount,
    const VkSubmitInfo2* pSubmits,
    VkFence            fence);

typedef PFN_vkQueueSubmit2 PFN_vkQueueSubmit2KHR;

// 스펙 값과 맞춰둔 sType 상수들
#ifndef VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO
#define VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO (VkStructureType)1000267000
#endif

#ifndef VK_STRUCTURE_TYPE_SUBMIT_INFO_2
#define VK_STRUCTURE_TYPE_SUBMIT_INFO_2 (VkStructureType)1000267007
#endif

#ifndef VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO
#define VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO (VkStructureType)1000267008
#endif

#endif // PFN_vkQueueSubmit2

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
    const VkSubmitInfo2*          pSubmits,
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

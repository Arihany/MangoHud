#pragma once
#include <vulkan/vulkan.h>

/*
 * Vulkan header 호환 레이어:
 * - 코어(1.3) submit2가 있으면 그대로 사용
 * - KHR만 있으면 "코어 이름"을 KHR로 alias
 * - 둘 다 없으면(아주 구형 헤더) 최소 정의를 제공
 *
 * NOTE: sType 숫자는 Vulkan spec 값(100031400x)을 반드시 사용한다.
 */

// ------------------------------
// 1) PFN alias: KHR만 있는 경우
// ------------------------------
#if !defined(PFN_vkQueueSubmit2) && defined(PFN_vkQueueSubmit2KHR)
typedef PFN_vkQueueSubmit2KHR PFN_vkQueueSubmit2;
#endif

// ------------------------------
// 2) 코어 타입이 없고, KHR 타입/매크로가 있는 경우 alias
//    (헤더가 VK_KHR_synchronization2만 제공하는 흔한 케이스)
// ------------------------------
#if !defined(VK_STRUCTURE_TYPE_SUBMIT_INFO_2) && defined(VK_STRUCTURE_TYPE_SUBMIT_INFO_2_KHR)
  #define VK_STRUCTURE_TYPE_SUBMIT_INFO_2 VK_STRUCTURE_TYPE_SUBMIT_INFO_2_KHR
#endif
#if !defined(VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO) && defined(VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO_KHR)
  #define VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO_KHR
#endif
#if !defined(VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO) && defined(VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO_KHR)
  #define VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO_KHR
#endif

// ------------------------------
// 3) submit2 자체가 헤더에 아예 없는 경우(구형): 최소 정의
//    - 우리 코드가 실제로 사용하는 건 VkSubmitInfo2 / VkCommandBufferSubmitInfo / PFN_vkQueueSubmit2 뿐
// ------------------------------
#if !defined(PFN_vkQueueSubmit2) && !defined(PFN_vkQueueSubmit2KHR)

typedef VkFlags VkSubmitFlags;

// forward decl (우린 이 타입을 "참조"만 한다)
typedef struct VkSemaphoreSubmitInfo VkSemaphoreSubmitInfo;

#ifndef VK_STRUCTURE_TYPE_SUBMIT_INFO_2
#define VK_STRUCTURE_TYPE_SUBMIT_INFO_2 ((VkStructureType)1000314004)
#endif
#ifndef VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO
#define VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO ((VkStructureType)1000314005)
#endif
#ifndef VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO
#define VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO ((VkStructureType)1000314006)
#endif

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

typedef VkResult (VKAPI_PTR *PFN_vkQueueSubmit2)(
  VkQueue             queue,
  uint32_t            submitCount,
  const VkSubmitInfo2* pSubmits,
  VkFence             fence
);

// KHR 이름도 같이 제공(훅 코드에서 둘 다 쓰는 경우 대비)
typedef PFN_vkQueueSubmit2 PFN_vkQueueSubmit2KHR;

#endif // submit2 없는 구형 헤더

struct AndroidVkGpuDispatch {
    // 큐/쿼리 관련
    PFN_vkQueueSubmit             QueueSubmit;
    PFN_vkQueueSubmit2            QueueSubmit2;
    PFN_vkQueueSubmit2KHR         QueueSubmit2KHR;
    PFN_vkCreateQueryPool         CreateQueryPool;
    PFN_vkDestroyQueryPool        DestroyQueryPool;
    PFN_vkGetQueryPoolResults     GetQueryPoolResults;
    PFN_vkDeviceWaitIdle          DeviceWaitIdle;
    // 커맨드 풀 / 버퍼
    PFN_vkCreateCommandPool       CreateCommandPool;
    PFN_vkDestroyCommandPool      DestroyCommandPool;
    PFN_vkResetCommandPool        ResetCommandPool;
    PFN_vkAllocateCommandBuffers  AllocateCommandBuffers;
    PFN_vkBeginCommandBuffer      BeginCommandBuffer;
    PFN_vkEndCommandBuffer        EndCommandBuffer;

    // 타임스탬프 / 쿼리
    PFN_vkCmdWriteTimestamp       CmdWriteTimestamp;
    PFN_vkCmdResetQueryPool       CmdResetQueryPool;
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

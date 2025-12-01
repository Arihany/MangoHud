#if defined(__ANDROID__)
#include <vulkan/vulkan.h>
#include <stdint.h>

struct AndroidVkGpuContext;

struct AndroidVkGpuDispatch {
    PFN_vkQueueSubmit          QueueSubmit;
    PFN_vkCreateQueryPool      CreateQueryPool;
    PFN_vkDestroyQueryPool     DestroyQueryPool;
    PFN_vkGetQueryPoolResults  GetQueryPoolResults;
    PFN_vkCreateCommandPool    CreateCommandPool;
    PFN_vkDestroyCommandPool   DestroyCommandPool;
    PFN_vkAllocateCommandBuffers AllocateCommandBuffers;
    PFN_vkFreeCommandBuffers   FreeCommandBuffers;
    PFN_vkResetCommandBuffer   ResetCommandBuffer;
    PFN_vkBeginCommandBuffer   BeginCommandBuffer;
    PFN_vkEndCommandBuffer     EndCommandBuffer;
    PFN_vkCmdWriteTimestamp    CmdWriteTimestamp;
};

AndroidVkGpuContext* android_gpu_usage_create(
    VkPhysicalDevice        phys,
    VkDevice                device,
    float                   timestamp_period_ns,
    const AndroidVkGpuDispatch& dispatch);

void android_gpu_usage_destroy(AndroidVkGpuContext* ctx);

void android_gpu_usage_on_present(
    AndroidVkGpuContext*    ctx,
    VkQueue                 present_queue,
    uint32_t                present_queue_family,
    const VkPresentInfoKHR* pPresentInfo,
    uint32_t                swapchain_index,
    uint32_t                image_index);

bool android_gpu_usage_get_metrics(
    AndroidVkGpuContext*    ctx,
    float*                  out_gpu_time_ms,
    float*                  out_gpu_usage_percent);

#endif

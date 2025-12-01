struct AndroidVkGpuContext {
    VkPhysicalDevice phys;
    VkDevice         device;
    float            timestamp_period_ns;

    VkQueryPool      query_pool;
    uint32_t         query_count;
    uint32_t         cursor;

    std::chrono::steady_clock::time_point last_cpu_frame;
    float     last_gpu_ms;
    float     last_gpu_usage;

    struct PerQueue {
        VkQueue     queue;
        uint32_t    family;
        VkCommandPool cmd_pool;
        VkCommandBuffer cmd_buffer;
        bool        initialized;
    };

    std::vector<PerQueue> queues;
};

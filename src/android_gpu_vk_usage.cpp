struct AndroidVkGpuContext {
    float last_gpu_ms    = 0.0f;
    float last_gpu_usage = 0.0f;
    bool  has_value      = false;
};

AndroidVkGpuContext* android_gpu_usage_create(
    VkPhysicalDevice, VkDevice, float /*ts_ns*/)
{
    return new AndroidVkGpuContext();
}

void android_gpu_usage_destroy(AndroidVkGpuContext* ctx)
{
    delete ctx;
}

void android_gpu_usage_on_present(
    AndroidVkGpuContext* ctx,
    VkQueue,
    uint32_t,
    const VkPresentInfoKHR*,
    uint32_t,
    uint32_t)
{
    if (!ctx)
        return;

    ctx->last_gpu_ms += 1.0f;
    if (ctx->last_gpu_ms > 16.0f)
        ctx->last_gpu_ms = 0.0f;

    ctx->last_gpu_usage += 5.0f;
    if (ctx->last_gpu_usage > 100.0f)
        ctx->last_gpu_usage = 0.0f;

    ctx->has_value = true;
}

bool android_gpu_usage_get_metrics(
    AndroidVkGpuContext* ctx,
    float* out_gpu_time_ms,
    float* out_gpu_usage_percent)
{
    if (!ctx || !ctx->has_value)
        return false;

    if (out_gpu_time_ms)
        *out_gpu_time_ms = ctx->last_gpu_ms;
    if (out_gpu_usage_percent)
        *out_gpu_usage_percent = ctx->last_gpu_usage;

    return true;
}

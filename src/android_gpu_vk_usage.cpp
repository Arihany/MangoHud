#include <vulkan/vulkan.h>
#include "android_gpu_vk_usage.h"

#include <array>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <limits>
#include <mutex>
#include <vector>

#include <spdlog/spdlog.h>

namespace {
// Hard safety caps (OOM/폭주 방지). 프리징(ms 커짐)은 정상이고 입력 크기만 막는다.
constexpr uint32_t kSubmitCountHardCap = 1024;
constexpr uint64_t kFlattenHardCap     = 8192;

// Sample policy
constexpr uint32_t kMaxPairsPerSampledFrame = 16;

// Suspend policy
constexpr auto kCooldownSubmitFail   = std::chrono::milliseconds(1500);
constexpr auto kCooldownNotReadyLong = std::chrono::milliseconds(1000);
constexpr auto kCooldownStaleSlot    = std::chrono::milliseconds(1000);
constexpr auto kCooldownRecordFail   = std::chrono::milliseconds(1500);
constexpr auto kSuspendedProbeEvery  = std::chrono::milliseconds(250);
constexpr uint32_t kNotReadyLimit    = 120;

static inline uint32_t
calc_pairs_left(uint32_t query_used, uint32_t query_capacity) noexcept
{
    const uint32_t used_pairs = query_used / 2u;
    const uint32_t hard_cap   = std::min<uint32_t>(kMaxPairsPerSampledFrame, query_capacity / 2u);
    return (used_pairs < hard_cap) ? (hard_cap - used_pairs) : 0u;
}

template <typename SubmitT> struct SubmitTraits;

template <> struct SubmitTraits<VkSubmitInfo> {
    using FlatT = VkCommandBuffer;
    static inline uint32_t base_count(const VkSubmitInfo& s) noexcept { return s.commandBufferCount; }
    static inline bool has_cmds(const VkSubmitInfo& s) noexcept {
        return s.commandBufferCount > 0 && s.pCommandBuffers != nullptr;
    }
    static inline const FlatT* cmd_ptr(const VkSubmitInfo& s) noexcept { return s.pCommandBuffers; }
    static inline void set_cmds(VkSubmitInfo& dst, FlatT* p, uint32_t n) noexcept {
        dst.commandBufferCount = n;
        dst.pCommandBuffers    = p;
    }
    static inline void push_begin(FlatT* out, uint32_t& idx, VkCommandBuffer cmd, const VkSubmitInfo&) noexcept {
        out[idx++] = cmd;
    }
    static inline void push_end(FlatT* out, uint32_t& idx, VkCommandBuffer cmd, const VkSubmitInfo&) noexcept {
        out[idx++] = cmd;
    }
};

template <> struct SubmitTraits<VkSubmitInfo2> {
    using FlatT = VkCommandBufferSubmitInfo;
    static inline uint32_t base_count(const VkSubmitInfo2& s) noexcept { return s.commandBufferInfoCount; }
    static inline bool has_cmds(const VkSubmitInfo2& s) noexcept {
        return s.commandBufferInfoCount > 0 && s.pCommandBufferInfos != nullptr;
    }
    static inline const FlatT* cmd_ptr(const VkSubmitInfo2& s) noexcept { return s.pCommandBufferInfos; }
    static inline void set_cmds(VkSubmitInfo2& dst, FlatT* p, uint32_t n) noexcept {
        dst.commandBufferInfoCount = n;
        dst.pCommandBufferInfos    = p;
    }
    static inline FlatT make_cb_info(const VkSubmitInfo2& src, VkCommandBuffer buf) noexcept {
        FlatT info{};
        info.sType         = VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO;
        info.pNext         = nullptr;
        info.commandBuffer = buf;
        info.deviceMask    = (src.commandBufferInfoCount && src.pCommandBufferInfos)
                           ? src.pCommandBufferInfos[0].deviceMask
                           : 0x1u;
        return info;
    }
    static inline void push_begin(FlatT* out, uint32_t& idx, VkCommandBuffer cmd, const VkSubmitInfo2& src) noexcept {
        out[idx++] = make_cb_info(src, cmd);
    }
    static inline void push_end(FlatT* out, uint32_t& idx, VkCommandBuffer cmd, const VkSubmitInfo2& src) noexcept {
        out[idx++] = make_cb_info(src, cmd);
    }
};

template <typename SubmitT>
static inline bool
plan_instrumentation(const SubmitT* submits,
                     uint32_t n,
                     uint32_t query_used,
                     uint32_t query_capacity,
                     uint32_t pairs_left,
                     std::vector<uint8_t>& inst,
                     std::vector<uint32_t>& counts,
                     uint64_t& total_flat) noexcept
{
    inst.resize(n);
    counts.resize(n);

    bool any = false;
    total_flat = 0;

    uint32_t planned = 0;
    for (uint32_t i = 0; i < n; ++i) {
        const uint32_t base = SubmitTraits<SubmitT>::base_count(submits[i]);
        const bool can =
            SubmitTraits<SubmitT>::has_cmds(submits[i]) &&
            (planned < pairs_left) &&
            (query_used + (planned + 1u) * 2u <= query_capacity);

        if (can) {
            inst[i] = 1;
            counts[i] = base + 2u; // begin + end
            ++planned;
            any = true;
        } else {
            inst[i] = 0;
            counts[i] = base;
        }

        total_flat += (uint64_t)counts[i];
    }

    return any;
}

template <typename SubmitFn, typename SubmitT>
struct SubmitAttempt {
    VkResult vr;
    bool     instrumented_executed; // 계측 삽입 submit이 실제로 성공 실행됐는가
};

template <typename SubmitFn, typename SubmitT>
static inline SubmitAttempt
submit_with_fallback(SubmitFn&& submit_fn,
                     uint32_t n_inst, const SubmitT* s_inst,
                     uint32_t n_orig, const SubmitT* s_orig,
                     VkFence fence) noexcept
{
    VkResult vr = submit_fn(n_inst, s_inst, fence);
    if (vr == VK_SUCCESS) return { vr, true };
    if (vr == VK_ERROR_DEVICE_LOST) return { vr, true };

    VkResult vr2 = submit_fn(n_orig, s_orig, fence);
    if (vr2 == VK_SUCCESS) return { vr2, false }; // 게임은 성공, 계측은 실패
    return { vr2, false }; // 둘 다 실패면 원본 에러를 리턴(어차피 게임도 실패)
}
} // namespace

// ====================== 내부 상태 ======================

enum class BackendMode : uint8_t { Active, Suspended, Disabled };

struct AndroidVkGpuContext {
    VkPhysicalDevice        phys_dev      = VK_NULL_HANDLE;
    VkDevice                device        = VK_NULL_HANDLE;
    AndroidVkGpuDispatch    disp{};

    float                   ts_period_ns  = 0.0f;   // ns per tick
    uint32_t                ts_valid_bits = 0;
    uint64_t                ts_mask       = ~0ULL;
    std::atomic<bool>       ts_supported{false};

    // Optional host-side query reset (prevents stale query values before cmd-reset executes).
    // - Vulkan 1.2+ core: vkResetQueryPool
    // - Vulkan 1.0/1.1: VK_EXT_host_query_reset: vkResetQueryPoolEXT (if enabled)
    PFN_vkResetQueryPool     fpResetQueryPool    = nullptr;
    PFN_vkResetQueryPoolEXT  fpResetQueryPoolEXT = nullptr;

    VkQueryPool             query_pool    = VK_NULL_HANDLE;
    uint32_t                queue_family_index = VK_QUEUE_FAMILY_IGNORED;

    // 링 버퍼 / 슬롯 설정
    static constexpr uint32_t MAX_FRAMES            = 16;   // 슬롯 개수
    static constexpr uint32_t MAX_QUERIES_PER_FRAME = 128;  // 슬롯당 쿼리 개수 (start/end 2개씩 → submit 최대 64개)
    static constexpr uint32_t FRAME_LAG             = 3;    // 몇 프레임 뒤에 슬롯을 읽고 해제할지
    static constexpr uint32_t WARMUP_TIMESTAMP_PAIRS= 0; // (deprecated) 이제 prealloc로 고정

    std::atomic<BackendMode> mode{BackendMode::Active};

    // “정지 후 언제 다시 시도할지”
    std::chrono::steady_clock::time_point suspend_until{};
    // NotReady 연속/오류 연속 카운트 (복구 정책용)
    uint32_t notready_streak = 0;
    uint32_t error_streak    = 0;

    // Suspended에서 too-hot loop 방지용: 마지막 probe 시각
    std::chrono::steady_clock::time_point last_probe{};

    struct FrameResources {
        VkCommandPool cmd_pool = VK_NULL_HANDLE;
        uint32_t      query_start = 0;
        uint32_t      query_capacity = 0;
        uint32_t      query_used = 0;

        bool          has_queries = false;     // "커밋된 pair 존재" 의미
        uint64_t      valid_pairs_mask = 0;    // submit 성공한 pair만 표시 (max 64)
        std::atomic<uint32_t> in_submit{0};    // reset/reuse 보호

        uint64_t      frame_serial = std::numeric_limits<uint64_t>::max();
        std::vector<VkCommandBuffer> timestamp_cmds;
    };

    FrameResources          frames[MAX_FRAMES]{};
    std::atomic<uint64_t>   frame_index{0};
    uint64_t                read_serial = 0;

    // ---- locks ----
    std::mutex                                      lock;
    std::mutex                                      record_mtx;   // CB 녹화 직렬화(드라이버 이상 방지)
    std::mutex                                      metrics_mtx;  // smoothing/last metrics 전용
    std::condition_variable                         cv;
    std::mutex                                      destroy_mtx; // destroy wait 전용(교착 방지)
    std::atomic<bool>                               destroying{false};
    std::atomic<uint32_t>                           in_flight{0};
    // ---- metrics (guarded by metrics_mtx) ----
    std::chrono::steady_clock::time_point           last_present{};
    std::chrono::steady_clock::time_point           window_start{};
    double                                          acc_cpu_ms_sampled = 0.0;
    uint32_t                                        acc_frames_sampled = 0;
    double                                          acc_gpu_ms     = 0.0;
    uint32_t                                        acc_gpu_samples= 0;
    float                                           smooth_gpu_ms = 0.0f;
    float                                           smooth_usage  = 0.0f;
    float                                           last_gpu_ms   = 0.0f;
    float                                           last_usage    = 0.0f;
    bool                                            have_metrics  = false;

    struct CpuSample { uint64_t serial; float ms; }; // metrics_mtx
    static constexpr uint32_t CPU_RING = 64; // 16보다 크게 (NotReady 오래가도 안전)
    CpuSample cpu_ring[CPU_RING]{};

    // CPU_RING은 bitmask 인덱싱을 쓰므로 반드시 2의 거듭제곱이어야 한다.
    static_assert((CPU_RING & (CPU_RING - 1u)) == 0u, "CPU_RING must be power-of-two");
};

struct AndroidGpuApiGuard {
    AndroidVkGpuContext* ctx = nullptr;
    bool armed = false;

    explicit AndroidGpuApiGuard(AndroidVkGpuContext* c) : ctx(c) {
        if (!ctx) return;
        ctx->in_flight.fetch_add(1, std::memory_order_acq_rel);
        armed = true;
    }

    ~AndroidGpuApiGuard() {
        if (!armed || !ctx) return;
        if (ctx->in_flight.fetch_sub(1, std::memory_order_acq_rel) == 1) {
            if (ctx->destroying.load(std::memory_order_acquire))
                ctx->cv.notify_all(); // lock 필요 없음
        }
    }
};

// forward decls (submit wrapper uses these before their definitions)
static inline void
android_gpu_usage_suspend_locked(AndroidVkGpuContext* ctx,
                                 const char* reason,
                                 std::chrono::milliseconds cooldown) noexcept;

static bool
android_gpu_usage_init_timestamp_resources(AndroidVkGpuContext* ctx,
                                           uint32_t queue_family_index);

static bool
android_gpu_usage_begin_frame(AndroidVkGpuContext* ctx,
                              uint32_t frame_idx,
                              uint64_t frame_serial);

static bool
android_gpu_usage_record_timestamp_pair(AndroidVkGpuContext* ctx,
                                        AndroidVkGpuContext::FrameResources& fr,
                                        uint32_t& out_query_first,
                                        VkCommandBuffer& out_cmd_begin,
                                        VkCommandBuffer& out_cmd_end);

// ====================== submit commit/rollback helpers ======================
namespace {
static inline void
disarm_in_submit_locked(AndroidVkGpuContext* ctx, bool& armed, uint32_t slot_idx) noexcept
{
    if (!armed || !ctx) return;
    ctx->frames[slot_idx].in_submit.fetch_sub(1, std::memory_order_acq_rel);
    armed = false;
}

static inline void
rollback_slot_if_safe_locked(AndroidVkGpuContext* ctx,
                             AndroidVkGpuContext::FrameResources& fr,
                             uint64_t serial_snapshot,
                             uint32_t saved_query_used,
                             bool saved_has_queries,
                             uint32_t reserved_delta_queries) noexcept
{
    if (!ctx) return;
    if (ctx->destroying.load(std::memory_order_relaxed)) return;
    if (fr.frame_serial != serial_snapshot) return;
    if (fr.query_used != saved_query_used + reserved_delta_queries) return;
    fr.query_used  = saved_query_used;
    fr.has_queries = saved_has_queries;
}

static inline void
finalize_submit_locked(AndroidVkGpuContext* ctx,
                       AndroidVkGpuContext::FrameResources& fr,
                       VkResult vr,
                       uint64_t serial_snapshot,
                       uint32_t saved_query_used,
                       bool saved_has_queries,
                       uint32_t reserved_delta_queries,
                       uint64_t new_valid_mask,
                       const char* fail_reason) noexcept
{
    if (vr == VK_SUCCESS) {
        fr.valid_pairs_mask |= new_valid_mask;
        fr.has_queries = (fr.valid_pairs_mask != 0);
        return;
    }

    rollback_slot_if_safe_locked(ctx, fr, serial_snapshot,
                                 saved_query_used, saved_has_queries,
                                 reserved_delta_queries);

    if (vr == VK_ERROR_DEVICE_LOST) {
        ctx->ts_supported.store(false, std::memory_order_relaxed);
        ctx->mode.store(BackendMode::Disabled, std::memory_order_relaxed);
        SPDLOG_WARN("Android GPU usage: DEVICE_LOST -> disable backend");
        return;
    }
    android_gpu_usage_suspend_locked(ctx, fail_reason, kCooldownSubmitFail);
}

template <typename SubmitT>
struct TlsScratch {
    std::vector<SubmitT>                           wrapped;
    std::vector<typename SubmitTraits<SubmitT>::FlatT> flat;
    std::vector<uint32_t>                          offsets;
    std::vector<uint32_t>                          counts;
    std::vector<uint8_t>                           inst;

    struct RecordJob {
        VkCommandBuffer begin = VK_NULL_HANDLE;
        VkCommandBuffer end   = VK_NULL_HANDLE;
        uint32_t        q0    = 0;
        uint32_t        pair_index = 0;
    };
    std::vector<RecordJob>                         jobs;
};

// reserve는 락 안에서 "인덱스/핸들만 확보"하고, 실제 녹화는 락 밖에서 한다.
static bool
android_gpu_usage_reserve_timestamp_pair_locked(AndroidVkGpuContext* ctx,
                                                AndroidVkGpuContext::FrameResources& fr,
                                                uint32_t& out_query_first,
                                                uint32_t& out_pair_index,
                                                VkCommandBuffer& out_cmd_begin,
                                                VkCommandBuffer& out_cmd_end) noexcept
{
    if (!ctx || !ctx->query_pool) return false;
    if (!ctx->disp.BeginCommandBuffer || !ctx->disp.EndCommandBuffer ||
        !ctx->disp.CmdResetQueryPool || !ctx->disp.CmdWriteTimestamp)
        return false;
    if (fr.query_used + 2u > fr.query_capacity) return false;

    const uint32_t pair_index = fr.query_used / 2u;
    const uint32_t begin_idx  = pair_index * 2u;
    const uint32_t end_idx    = begin_idx + 1u;

    // prealloc이므로 "없으면 포기" (핫패스 Allocate 0회)
    if (fr.timestamp_cmds.size() <= end_idx) return false;

    const uint32_t query_first = fr.query_start + fr.query_used;
    out_query_first = query_first;
    out_pair_index  = pair_index;
    out_cmd_begin   = fr.timestamp_cmds[begin_idx];
    out_cmd_end     = fr.timestamp_cmds[end_idx];

    fr.query_used += 2u; // 예약 커밋(성공/실패 반영은 submit 성공 후 valid_mask에서)
    return true;
}

static bool
android_gpu_usage_record_timestamp_pair_unlocked(AndroidVkGpuContext* ctx,
                                                 VkCommandBuffer cmd_begin,
                                                 VkCommandBuffer cmd_end,
                                                 uint32_t query_first) noexcept
{
    if (!ctx || !ctx->query_pool) return false;
    if (!ctx->disp.BeginCommandBuffer || !ctx->disp.EndCommandBuffer ||
        !ctx->disp.CmdResetQueryPool || !ctx->disp.CmdWriteTimestamp)
        return false;

    VkCommandBufferBeginInfo bi{};
    bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

    // begin CB: reset(2) + TOP timestamp
    if (ctx->disp.BeginCommandBuffer(cmd_begin, &bi) != VK_SUCCESS) return false;
    ctx->disp.CmdResetQueryPool(cmd_begin, ctx->query_pool, query_first, 2u);
    ctx->disp.CmdWriteTimestamp(cmd_begin, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                                ctx->query_pool, query_first);
    if (ctx->disp.EndCommandBuffer(cmd_begin) != VK_SUCCESS) return false;

    // end CB: BOTTOM timestamp
    if (ctx->disp.BeginCommandBuffer(cmd_end, &bi) != VK_SUCCESS) return false;
    ctx->disp.CmdWriteTimestamp(cmd_end, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
                                ctx->query_pool, query_first + 1u);
    if (ctx->disp.EndCommandBuffer(cmd_end) != VK_SUCCESS) return false;

    return true;
}

template <typename SubmitT>
static inline void
build_wrapped_submits_locked(AndroidVkGpuContext* ctx,
                             AndroidVkGpuContext::FrameResources& fr,
                             const SubmitT* pSubmits,
                             uint32_t n,
                             TlsScratch<SubmitT>& tls,
                             uint64_t& new_valid_mask) noexcept
{
    auto& wrapped = tls.wrapped;
    auto& flat    = tls.flat;
    auto& offsets = tls.offsets;
    auto& inst    = tls.inst;
    auto& jobs    = tls.jobs;
    jobs.clear();
    jobs.reserve(kMaxPairsPerSampledFrame);

    for (uint32_t i = 0; i < n; ++i) {
        const SubmitT& src = pSubmits[i];
        if (!inst[i]) {
            wrapped[i] = src;
            continue;
        }

        auto* dst = flat.data() + offsets[i];
        uint32_t idx = 0;

        VkCommandBuffer cmd_begin = VK_NULL_HANDLE;
        VkCommandBuffer cmd_end   = VK_NULL_HANDLE;
        uint32_t query_first      = 0;
        uint32_t pair_index       = 0;

        if (!android_gpu_usage_reserve_timestamp_pair_locked(ctx, fr, query_first, pair_index,
                                                             cmd_begin, cmd_end)) {
            const uint32_t base = SubmitTraits<SubmitT>::base_count(src);
            const auto*    ptr  = SubmitTraits<SubmitT>::cmd_ptr(src);
            for (uint32_t j = 0; j < base; ++j) dst[idx++] = ptr[j];

            SubmitT dst_submit = src;
            SubmitTraits<SubmitT>::set_cmds(dst_submit, dst, idx);
            wrapped[i] = dst_submit;
            inst[i] = 0;
            continue;
        }

        if (pair_index < 64) new_valid_mask |= (1ULL << pair_index);
        jobs.push_back(typename TlsScratch<SubmitT>::RecordJob{cmd_begin, cmd_end, query_first, pair_index});

        SubmitTraits<SubmitT>::push_begin(dst, idx, cmd_begin, src);
        {
            const uint32_t base = SubmitTraits<SubmitT>::base_count(src);
            const auto*    ptr  = SubmitTraits<SubmitT>::cmd_ptr(src);
            for (uint32_t j = 0; j < base; ++j) dst[idx++] = ptr[j];
        }
        SubmitTraits<SubmitT>::push_end(dst, idx, cmd_end, src);

        SubmitT dst_submit = src;
        SubmitTraits<SubmitT>::set_cmds(dst_submit, dst, idx);
        wrapped[i] = dst_submit;
    }
}

static inline void
rollback_best_effort_locked(AndroidVkGpuContext* ctx,
                            AndroidVkGpuContext::FrameResources& fr,
                            uint64_t serial_snapshot,
                            uint32_t saved_query_used,
                            bool saved_has_queries) noexcept
{
    if (!ctx) return;
    if (ctx->destroying.load(std::memory_order_relaxed)) return;
    if (fr.frame_serial != serial_snapshot) return;
    if (fr.query_used >= saved_query_used) fr.query_used = saved_query_used;
    fr.has_queries = saved_has_queries;
}

template <typename SubmitT, typename SubmitFn>
static VkResult
android_gpu_usage_queue_submit_impl(AndroidVkGpuContext* ctx,
                                    uint32_t queue_family_index,
                                    uint32_t submitCount,
                                    const SubmitT* pSubmits,
                                    VkFence fence,
                                    SubmitFn&& submit_fn,
                                    const char* fail_reason)
{
    AndroidVkGpuContext::FrameResources* fr_ptr = nullptr;
    uint64_t serial_snapshot = 0;
    uint32_t saved_query_used = 0;
    bool     saved_has_queries = false;
    uint32_t reserved_delta_queries = 0;
    uint64_t new_valid_mask = 0;
    bool     armed_slot = false;
    uint32_t armed_slot_idx = 0;

    const SubmitT* submit_ptr = pSubmits;
    uint32_t       submit_n   = submitCount;
    
    try {
        static thread_local TlsScratch<SubmitT> tls;

        {
            std::lock_guard<std::mutex> g(ctx->lock);

            if (!ctx->ts_supported.load(std::memory_order_relaxed))
                goto pass_through_locked;

            const uint64_t frame_serial = ctx->frame_index.load(std::memory_order_relaxed);
            if ((frame_serial & 1u) != 0u)
                goto pass_through_locked;

            if (ctx->query_pool != VK_NULL_HANDLE &&
                ctx->queue_family_index != VK_QUEUE_FAMILY_IGNORED &&
                ctx->queue_family_index != queue_family_index) {
                goto pass_through_locked;
            }

            if (!android_gpu_usage_init_timestamp_resources(ctx, queue_family_index))
                goto pass_through_locked;
            
            const uint32_t curr_idx = static_cast<uint32_t>(frame_serial % AndroidVkGpuContext::MAX_FRAMES);
            serial_snapshot = frame_serial;
            fr_ptr = &ctx->frames[curr_idx];
            auto& fr = *fr_ptr;

            if (!android_gpu_usage_begin_frame(ctx, curr_idx, frame_serial))
                goto pass_through_locked;

            saved_query_used  = fr.query_used;
            saved_has_queries = fr.has_queries;

            const uint32_t n = submitCount;
            tls.wrapped.resize(n);
            tls.offsets.resize(n);
            tls.counts.resize(n);
            tls.inst.resize(n);

            const uint32_t pairs_left = calc_pairs_left(fr.query_used, fr.query_capacity);
            if (!pairs_left)
                goto pass_through_locked;

            uint64_t total_flat64 = 0;
            const bool any_inst =
                plan_instrumentation(pSubmits, n,
                                     fr.query_used, fr.query_capacity,
                                     pairs_left,
                                     tls.inst, tls.counts, total_flat64);

            if (!any_inst)
                goto pass_through_locked;

            if (total_flat64 == 0 || total_flat64 > kFlattenHardCap)
                goto pass_through_locked;

            const uint32_t total_flat = static_cast<uint32_t>(total_flat64);
            if (tls.flat.size() < total_flat)
                tls.flat.resize(total_flat);

            uint32_t cursor = 0;
            for (uint32_t i = 0; i < n; ++i) {
                tls.offsets[i] = cursor;
                cursor += tls.counts[i];
            }

            build_wrapped_submits_locked(ctx, fr, pSubmits, n, tls, new_valid_mask);

            reserved_delta_queries = fr.query_used - saved_query_used;

            if (reserved_delta_queries == 0 || new_valid_mask == 0)
                goto pass_through_locked;

            armed_slot = true;
            armed_slot_idx = curr_idx;
            fr.in_submit.fetch_add(1, std::memory_order_acq_rel);

            submit_ptr = tls.wrapped.data();
            submit_n   = submitCount;

            goto wrapped_ready_locked;

        pass_through_locked:
            goto done_locked;

        wrapped_ready_locked:
            goto done_locked;

        done_locked:
            ;
        } // unlock

        if (!armed_slot) {
            // 패스스루: 원본 submits로 그대로 호출
            return submit_fn(submitCount, pSubmits, fence);
        }
        // --------- (B) 락 밖: timestamp CB 녹화 (핫패스 alloc 0회) ---------
        bool record_ok = true;
        {
            // 드라이버가 이상하면 병렬 녹화에서 사고가 나기도 해서, 안전빵으로 직렬화.
            std::lock_guard<std::mutex> rg(ctx->record_mtx);
            for (const auto& job : tls.jobs) {
                if (!android_gpu_usage_record_timestamp_pair_unlocked(ctx, job.begin, job.end, job.q0)) {
                    record_ok = false;
                    break;
                }
            }
        }

        if (!record_ok) {
            // 녹화 실패는 "계측 자체를 포기"하고 원본 submit로 안전하게 간다.
            std::lock_guard<std::mutex> g3(ctx->lock);
            auto& fr3 = ctx->frames[armed_slot_idx];
            disarm_in_submit_locked(ctx, armed_slot, armed_slot_idx);
            rollback_slot_if_safe_locked(ctx, fr3, serial_snapshot,
                                         saved_query_used, saved_has_queries,
                                         reserved_delta_queries);
            // 부분 녹화로 cmd buffer state가 꼬였을 수 있으니 slot 커맨드풀 리셋(안전빵)
            if (ctx->disp.ResetCommandPool && fr3.cmd_pool != VK_NULL_HANDLE) {
                ctx->disp.ResetCommandPool(ctx->device, fr3.cmd_pool, 0);
            }
            android_gpu_usage_suspend_locked(ctx, "record timestamp CB failed", kCooldownRecordFail);
            return submit_fn(submitCount, pSubmits, fence);
        }

        auto res = submit_with_fallback(submit_fn,
                                        submit_n, submit_ptr,
                                        submitCount, pSubmits,
                                        fence);
        VkResult vr = res.vr;

        {
            std::lock_guard<std::mutex> g2(ctx->lock);
            auto& fr2 = ctx->frames[armed_slot_idx];
            disarm_in_submit_locked(ctx, armed_slot, armed_slot_idx);
            if (!res.instrumented_executed) {
                // 원본 submit 성공이더라도 계측 커밋은 하면 안 됨.
                rollback_slot_if_safe_locked(ctx, fr2, serial_snapshot,
                                             saved_query_used, saved_has_queries,
                                             reserved_delta_queries);
               // 녹화된 CB들이 남아있을 수 있으니 안전빵으로 리셋
               if (ctx->disp.ResetCommandPool && fr2.cmd_pool != VK_NULL_HANDLE) {
                   ctx->disp.ResetCommandPool(ctx->device, fr2.cmd_pool, 0);
               }
                android_gpu_usage_suspend_locked(ctx,
                    "instrumented submit failed, fallback used",
                    kCooldownSubmitFail);
            } else {
                finalize_submit_locked(ctx, fr2, vr,
                                       serial_snapshot, saved_query_used, saved_has_queries,
                                       reserved_delta_queries, new_valid_mask,
                                       fail_reason);
            }
        }
        return vr;
    } catch (const std::exception& e) {
        SPDLOG_WARN("Android GPU usage: submit wrapper exception: {}", e.what());
    } catch (...) {
        SPDLOG_WARN("Android GPU usage: submit wrapper unknown exception");
    }

    if (ctx && fr_ptr) {
        std::lock_guard<std::mutex> g(ctx->lock);
        disarm_in_submit_locked(ctx, armed_slot, armed_slot_idx);
        rollback_best_effort_locked(ctx, *fr_ptr, serial_snapshot, saved_query_used, saved_has_queries);
    }
    return submit_fn(submitCount, pSubmits, fence);
}
} // namespace

// ====================== 환경 변수 플래그 ======================
// 정책: 기본 OFF(fdinfo + kgsl 사용). 오직 MANGOHUD_VKP=1 일 때만 Vulkan timestamp 백엔드 ON.
// 캐시: -1 = 미초기화, 0 = 비활성, 1 = 활성
namespace {
static bool android_gpu_usage_env_enabled()
{
        // -1 = unknown, 0 = disabled, 1 = enabled
        static std::atomic<int> cached{-1};

        int v = cached.load(std::memory_order_acquire);
        if (v != -1)
            return v != 0;

        const char* env = std::getenv("MANGOHUD_VKP");
        const bool enabled = (env && env[0] == '1' && env[1] == '\0');
        const int  newv    = enabled ? 1 : 0;

        int expected = -1;
        if (cached.compare_exchange_strong(expected, newv,
                                           std::memory_order_release,
                                           std::memory_order_relaxed))
        {
            // 최초 결정한 스레드만 로그 1회
            if (!enabled) {
                SPDLOG_INFO("MANGOHUD_VKP!=1 -> Vulkan GPU usage backend disabled (fdinfo + kgsl remains default)");
            } else {
                SPDLOG_INFO("MANGOHUD_VKP=1 -> Vulkan GPU usage backend enabled");
            }
            return enabled;
        }

        // 누군가 먼저 결정했으면 그 결과를 따른다
        return cached.load(std::memory_order_acquire) != 0;
}
} // namespace

static inline void
android_gpu_usage_suspend_locked(AndroidVkGpuContext* ctx,
                                 const char* reason,
                                 std::chrono::milliseconds cooldown) noexcept
{
    if (!ctx) return;
    if (ctx->mode.load(std::memory_order_relaxed) == BackendMode::Disabled) return;

    const auto now = std::chrono::steady_clock::now();
    BackendMode m = ctx->mode.load(std::memory_order_relaxed);

    if (m == BackendMode::Suspended) {
        // 이미 Suspended면 쿨다운만 연장 (probe 폭주/스팸 방지)
        const auto t = now + cooldown;
        if (t > ctx->suspend_until) ctx->suspend_until = t;
        return;
    }

    ctx->mode.store(BackendMode::Suspended, std::memory_order_relaxed);
    ctx->suspend_until = now + cooldown;
    ctx->last_probe = {};              // 재진입 probe 폭주 방지용
    ctx->notready_streak = 0;          // 재개 직후 즉시 재정지 방지
    ctx->error_streak = 0;

    ctx->read_serial = ctx->frame_index.load(std::memory_order_relaxed);

    SPDLOG_WARN("Android GPU usage: SUSPEND ({}) -> stop instrumentation, keep last metrics", reason);
}

static inline bool
android_gpu_usage_should_sample(const AndroidVkGpuContext* ctx) noexcept
{
    return ctx &&
           ctx->mode.load(std::memory_order_relaxed) == BackendMode::Active &&
           ctx->ts_supported.load(std::memory_order_relaxed) &&
           ((ctx->frame_index.load(std::memory_order_relaxed) & 1u) == 0u);
}

// ====================== 헬퍼: 타임스탬프 리소스 초기화 ======================

static inline void
android_gpu_usage_consume_slot(AndroidVkGpuContext::FrameResources& fr);

static void
android_gpu_usage_destroy_timestamp_resources(AndroidVkGpuContext* ctx)
{
    if (!ctx || ctx->device == VK_NULL_HANDLE)
        return;

    if (ctx->disp.DestroyCommandPool) {
        for (uint32_t i = 0; i < AndroidVkGpuContext::MAX_FRAMES; ++i) {
            auto& fr = ctx->frames[i];
            if (fr.cmd_pool != VK_NULL_HANDLE) {
                ctx->disp.DestroyCommandPool(ctx->device, fr.cmd_pool, nullptr);
                fr.cmd_pool = VK_NULL_HANDLE;
                fr.timestamp_cmds.clear();
            }
            fr.query_used = 0;
            fr.has_queries = false;
            fr.valid_pairs_mask = 0;
            fr.frame_serial = std::numeric_limits<uint64_t>::max();
        }
    }

    if (ctx->query_pool != VK_NULL_HANDLE && ctx->disp.DestroyQueryPool) {
        ctx->disp.DestroyQueryPool(ctx->device, ctx->query_pool, nullptr);
        ctx->query_pool = VK_NULL_HANDLE;
    }
    ctx->queue_family_index = VK_QUEUE_FAMILY_IGNORED;
}

static bool
android_gpu_usage_init_timestamp_resources(AndroidVkGpuContext* ctx,
                                           uint32_t queue_family_index)
{
    if (!ctx || !ctx->ts_supported.load(std::memory_order_relaxed))
        return false;
    
    // Suspended/Disabled 상태에서는 절대 리소스 init 하지 않는다 (크래시 방지 우선)
    if (ctx->mode.load(std::memory_order_relaxed) != BackendMode::Active)
        return false;

    // 이미 초기화되어 있으면 그대로 사용
    if (ctx->query_pool != VK_NULL_HANDLE) {
        return true;
    }

    ctx->queue_family_index = queue_family_index;

    uint32_t qf_count = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(ctx->phys_dev, &qf_count, nullptr);
    if (queue_family_index >= qf_count) {
        SPDLOG_WARN("Android GPU usage: bad queue_family_index={} (qf_count={})", queue_family_index, qf_count);
        ctx->ts_supported.store(false, std::memory_order_relaxed);
        return false;
    }

    std::vector<VkQueueFamilyProperties> qf(qf_count);
    vkGetPhysicalDeviceQueueFamilyProperties(ctx->phys_dev, &qf_count, qf.data());

    const uint32_t vb = qf[queue_family_index].timestampValidBits;
    if (vb == 0) {
        SPDLOG_WARN("Android GPU usage: queue family {} has timestampValidBits=0 -> disabling", queue_family_index);
        ctx->ts_supported.store(false, std::memory_order_relaxed);
        return false;
    }

    // [ACCURACY GUARD] 게임/렌더링 계측 목적이면 그래픽 큐를 우선한다.
    // 그래픽 비트 없는 큐를 첫 init 대상으로 잡아버리면 "프레임" 계측이 삐끗하기 쉽다.
    if ((qf[queue_family_index].queueFlags & VK_QUEUE_GRAPHICS_BIT) == 0) {
        SPDLOG_DEBUG("Android GPU usage: queue family {} is not GRAPHICS -> skip init", queue_family_index);
        return false;
    }

    ctx->ts_valid_bits = vb;
    ctx->ts_mask = (vb >= 64) ? ~0ULL : ((1ULL << vb) - 1ULL);
    
    // 1) CommandPool 관련 함수 포인터부터 체크
    if (!ctx->disp.CreateCommandPool || !ctx->disp.ResetCommandPool) {
        SPDLOG_WARN(
            "Android GPU usage: Create/ResetCommandPool not available → disabling timestamp backend"
        );
        ctx->ts_supported = false;
        return false;
    }

    // 2) QueryPool 생성 (슬롯 × 쿼리 수)
    VkQueryPoolCreateInfo qp{};
    qp.sType              = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
    qp.queryType          = VK_QUERY_TYPE_TIMESTAMP;
    qp.queryCount         = AndroidVkGpuContext::MAX_FRAMES *
                            AndroidVkGpuContext::MAX_QUERIES_PER_FRAME;
    qp.flags              = 0;
    qp.pipelineStatistics = 0;

    if (!ctx->disp.CreateQueryPool ||
        ctx->disp.CreateQueryPool(ctx->device, &qp, nullptr, &ctx->query_pool) != VK_SUCCESS) {
        SPDLOG_WARN(
            "Android GPU usage: CreateQueryPool failed (qcount={}) → disabling timestamp backend",
            qp.queryCount
        );
        ctx->query_pool   = VK_NULL_HANDLE;
        ctx->ts_supported = false;
        return false;
    }

    // 3) 프레임별 커맨드 풀 만들기
    for (uint32_t i = 0; i < AndroidVkGpuContext::MAX_FRAMES; ++i) {
        VkCommandPoolCreateInfo cp{};
        cp.sType            = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        cp.queueFamilyIndex = queue_family_index;
        cp.flags            = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT |
                              VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;

        VkCommandPool pool = VK_NULL_HANDLE;
        if (ctx->disp.CreateCommandPool(ctx->device, &cp, nullptr, &pool) != VK_SUCCESS) {
            SPDLOG_WARN(
                "Android GPU usage: CreateCommandPool failed at slot {} → disabling timestamp backend",
                i
            );
            android_gpu_usage_destroy_timestamp_resources(ctx);
            ctx->ts_supported = false;
            return false;
        }

        auto& fr          = ctx->frames[i];
        fr.cmd_pool       = pool;
        fr.query_start    = i * AndroidVkGpuContext::MAX_QUERIES_PER_FRAME;
        fr.query_capacity = AndroidVkGpuContext::MAX_QUERIES_PER_FRAME;
        fr.query_used     = 0;
        fr.has_queries    = false;
        fr.frame_serial   = std::numeric_limits<uint64_t>::max();
        fr.timestamp_cmds.clear();

        // Prealloc: 핫패스 AllocateCommandBuffers를 0회로 만들기
        // - kMaxPairsPerSampledFrame(=16) * 2(begin/end) = 32 CB 고정 확보
        // - 쿼리는 query_used로만 소비하므로 초과분은 그냥 안 쓴다.
        if (!ctx->disp.AllocateCommandBuffers) {
            SPDLOG_WARN("Android GPU usage: AllocateCommandBuffers missing -> disable backend");
            android_gpu_usage_destroy_timestamp_resources(ctx);
            ctx->ts_supported = false;
            return false;
        }

        VkCommandBufferAllocateInfo ai{};
        ai.sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        ai.commandPool        = fr.cmd_pool;
        ai.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        ai.commandBufferCount = kMaxPairsPerSampledFrame * 2u;

        std::vector<VkCommandBuffer> tmp(ai.commandBufferCount, VK_NULL_HANDLE);
        if (ctx->disp.AllocateCommandBuffers(ctx->device, &ai, tmp.data()) != VK_SUCCESS) {
            SPDLOG_WARN("Android GPU usage: prealloc AllocateCommandBuffers failed at slot {} -> disable", i);
            android_gpu_usage_destroy_timestamp_resources(ctx);
            ctx->ts_supported = false;
            return false;
        }

        fr.timestamp_cmds.assign(tmp.begin(), tmp.end());
    }

    SPDLOG_INFO(
        "Android GPU usage: timestamp resources initialized (qf_index={} qpool={} slots={} qpf={})",
        queue_family_index,
        static_cast<void*>(ctx->query_pool),
        AndroidVkGpuContext::MAX_FRAMES,
        AndroidVkGpuContext::MAX_QUERIES_PER_FRAME
    );

    return true;
}

static bool
android_gpu_usage_begin_frame(AndroidVkGpuContext* ctx,
                              uint32_t frame_idx,
                              uint64_t frame_serial)
{
    auto& fr = ctx->frames[frame_idx];

    // 같은 프레임 serial이면(한 프레임 내 다중 submit) 계속 사용
    if (fr.frame_serial == frame_serial)
        return true;

    if (fr.has_queries && fr.frame_serial != std::numeric_limits<uint64_t>::max()) {
        const uint64_t age = (frame_serial > fr.frame_serial)
            ? (frame_serial - fr.frame_serial)
            : std::numeric_limits<uint64_t>::max();

        if (age < AndroidVkGpuContext::MAX_FRAMES)
            return false; // 아직 회수 전

        android_gpu_usage_suspend_locked(ctx, "stale slot: queries not drained", kCooldownStaleSlot);
        return false;
    }

    // slot의 커맨드풀이 submit 중이면 Reset/Re-use 금지 (크래시 방지)
    if (fr.in_submit.load(std::memory_order_acquire) != 0)
        return false;

    fr.frame_serial      = frame_serial;
    fr.query_used        = 0;
    fr.has_queries       = false;
    fr.valid_pairs_mask  = 0;

    // timestamp CB가 하나라도 있으면 커맨드풀 리셋
    if (ctx->disp.ResetCommandPool && fr.cmd_pool != VK_NULL_HANDLE &&
        !fr.timestamp_cmds.empty()) {
        ctx->disp.ResetCommandPool(ctx->device, fr.cmd_pool, 0);
    }

    return true;
}

enum class AndroidGpuReadStatus : uint8_t {
    Ready,
    NotReady,
    Error,
    DeviceLost
};

static AndroidGpuReadStatus
android_gpu_usage_query_range_gpu_ms(AndroidVkGpuContext* ctx,
                                     uint32_t query_start,
                                     uint32_t query_count,
                                     uint64_t valid_pairs_mask,
                                     float* out_gpu_ms)
{
    if (!ctx || !out_gpu_ms ||
        !ctx->ts_supported.load(std::memory_order_relaxed) ||
        ctx->query_pool == VK_NULL_HANDLE ||
        !ctx->disp.GetQueryPoolResults)
        return AndroidGpuReadStatus::Error;

    // query_count는 (start,end) 페어라서 항상 2의 배수여야 한다.
    // 깨졌다는 건 "아직 준비 안 됨"이 아니라 내부 상태 오염이므로 Error로 보내서 슬롯을 폐기하게 한다.
    if (query_count < 2 || (query_count & 1u))
        return AndroidGpuReadStatus::Error;

    thread_local std::vector<uint64_t> scratch;
    const uint32_t needed_u64 = query_count * 2u;
    if (scratch.size() < needed_u64)
        scratch.resize(needed_u64);

    VkResult r = ctx->disp.GetQueryPoolResults(
        ctx->device,
        ctx->query_pool,
        query_start,
        query_count,
        needed_u64 * sizeof(uint64_t),
        scratch.data(),
        sizeof(uint64_t) * 2,
        VK_QUERY_RESULT_64_BIT |
        VK_QUERY_RESULT_WITH_AVAILABILITY_BIT
    );

    if (r == VK_ERROR_DEVICE_LOST)
        return AndroidGpuReadStatus::DeviceLost;
    if (r == VK_NOT_READY)
        return AndroidGpuReadStatus::NotReady;
    if (r < 0)
        return AndroidGpuReadStatus::Error;

    const uint32_t pair_count = query_count / 2u;

    // 핵심: 하나라도 avail=0이면 NotReady. 부분 결과로 계산하지 않는다.
    for (uint32_t i = 0; i < pair_count; ++i) {
        if (((valid_pairs_mask >> i) & 1ULL) == 0ULL)
            continue; // submit 성공한 pair만 검사
        const uint32_t qs = 2u * i;
        const uint32_t qe = qs + 1u;
        const uint64_t as = scratch[qs * 2u + 1u];
        const uint64_t ae = scratch[qe * 2u + 1u];
        if (!as || !ae)
            return AndroidGpuReadStatus::NotReady;
    }

    // ------------------ busy_ms = union of [start,end] intervals (wrap-safe) ------------------
    // timestampValidBits < 64 인 경우 값은 "원형(mod wrap)"이므로,
    // 프레임 단위로 unwrap(한 축으로 펴기) 하지 않으면 interval 정렬/union이 깨질 수 있다.
    const bool has_wrap =
        (ctx->ts_valid_bits > 0 && ctx->ts_valid_bits < 64);
    const uint64_t wrap =
        has_wrap ? (1ULL << ctx->ts_valid_bits) : 0ULL;
    const uint64_t mask = ctx->ts_mask;

    struct Seg {
        uint64_t s_m;   // masked start
        uint64_t dur;   // duration in ticks (mod wrap diff)
    };
    thread_local std::vector<Seg> segs;
    segs.clear();
    segs.reserve(pair_count);

    for (uint32_t i = 0; i < pair_count; ++i) {
        if (((valid_pairs_mask >> i) & 1ULL) == 0ULL)
            continue;
        const uint32_t qs = 2u * i;
        const uint32_t qe = qs + 1u;

        const uint64_t s_m = scratch[qs * 2u + 0u] & mask;
        const uint64_t e_m = scratch[qe * 2u + 0u] & mask;

        uint64_t dur = 0;
        if (!has_wrap) {
            if (e_m <= s_m) continue;
            dur = e_m - s_m;
        } else {
            // 모듈러 차이: 같은 프레임에서 1회 wrap 정도는 커버 가능
            dur = (e_m - s_m) & mask;
            if (dur == 0) continue;
        }
        segs.push_back({ s_m, dur });
    }

    if (segs.empty()) {
        *out_gpu_ms = 0.0f;
        return AndroidGpuReadStatus::Ready;
    }

    struct Interval { uint64_t s; uint64_t e; };
    thread_local std::vector<Interval> iv;
    iv.clear();
    iv.reserve(segs.size());

    uint64_t min_start_u = std::numeric_limits<uint64_t>::max();
    uint64_t max_end_u   = 0;

    if (!has_wrap || segs.size() == 1) {
        // wrap이 없거나 interval이 1개면 기존처럼 단순 처리
        for (const auto& seg : segs) {
            const uint64_t s_u = seg.s_m;
            const uint64_t e_u = s_u + seg.dur;
            if (e_u <= s_u) continue;
            if (s_u < min_start_u) min_start_u = s_u;
            if (e_u > max_end_u)   max_end_u   = e_u;
            iv.push_back({ s_u, e_u });
        }
    } else {
        // (핵심) start들을 원형에서 한 축으로 unwrap
        // - start를 정렬하고, 가장 큰 gap을 wrap 경계로 선택
        // - 그 경계 다음 값을 pivot으로 잡고, delta = (s - pivot) & mask 로 한 줄에 펼친다.
        thread_local std::vector<uint64_t> starts;
        starts.clear();
        starts.reserve(segs.size());
        for (const auto& seg : segs) starts.push_back(seg.s_m);
        std::sort(starts.begin(), starts.end());

        uint64_t max_gap = 0;
        size_t   max_i   = 0;
        for (size_t i = 0; i + 1 < starts.size(); ++i) {
            const uint64_t gap = starts[i + 1] - starts[i];
            if (gap > max_gap) { max_gap = gap; max_i = i; }
        }
        // 마지막->처음 wrap gap
        const uint64_t last_gap = (starts[0] + wrap) - starts.back();
        if (last_gap > max_gap) { max_gap = last_gap; max_i = starts.size() - 1; }

        uint64_t pivot = starts[(max_i + 1) % starts.size()];
        const uint64_t span = wrap - max_gap; // unwrap 후 데이터가 차지하는 길이
        // span이 wrap에 너무 가까우면(데이터가 원형 전체에 퍼짐) 경계 선택이 의미 없다.
        // 이 경우는 어차피 신뢰도 박살 환경이므로, pivot 고정을 통해 폭발만 막는다.
        if (span > (wrap * 3ULL) / 4ULL) {
            pivot = starts[0];
        }

        for (const auto& seg : segs) {
            const uint64_t delta = (seg.s_m - pivot) & mask;
            const uint64_t s_u   = pivot + delta;
            const uint64_t e_u   = s_u + seg.dur;
            if (e_u <= s_u) continue;
            if (s_u < min_start_u) min_start_u = s_u;
            if (e_u > max_end_u)   max_end_u   = e_u;
            iv.push_back({ s_u, e_u });
        }
    }

    if (iv.empty()) {
        *out_gpu_ms = 0.0f;
        return AndroidGpuReadStatus::Ready;
    }

    std::sort(iv.begin(), iv.end(),
              [](const Interval& a, const Interval& b) {
                  if (a.s != b.s) return a.s < b.s;
                  return a.e < b.e;
              });

    uint64_t busy_ticks = 0;
    uint64_t cur_s = iv[0].s;
    uint64_t cur_e = iv[0].e;
    for (size_t k = 1; k < iv.size(); ++k) {
        const uint64_t s = iv[k].s;
        const uint64_t e = iv[k].e;
        if (s <= cur_e) {
            if (e > cur_e) cur_e = e;
        } else {
            busy_ticks += (cur_e - cur_s);
            cur_s = s;
            cur_e = e;
        }
    }
    busy_ticks += (cur_e - cur_s);

    double busy_ms = 0.0;
    if (busy_ticks > 0) {
        busy_ms = double(busy_ticks) * double(ctx->ts_period_ns) * 1e-6;
    }

    // (옵션) 완전 망한 값 방어용 fallback: union 결과가 0이면 range로만 1회 구제.
    if (!(busy_ms > 0.0) || !std::isfinite(busy_ms)) {
        double range_ms = 0.0;
        if (max_end_u > min_start_u &&
            min_start_u != std::numeric_limits<uint64_t>::max()) {
            const uint64_t range_ticks = max_end_u - min_start_u;
            range_ms = double(range_ticks) * double(ctx->ts_period_ns) * 1e-6;
        }
        if (range_ms > 0.0 && std::isfinite(range_ms)) {
            busy_ms = range_ms;
        }
    }

    if (!std::isfinite(busy_ms) || busy_ms < 0.0)
        return AndroidGpuReadStatus::Error;

    *out_gpu_ms = (float)busy_ms; // 0도 허용
    return AndroidGpuReadStatus::Ready;
}

static inline void
android_gpu_usage_consume_slot(AndroidVkGpuContext::FrameResources& fr)
{
    fr.has_queries      = false;
    fr.valid_pairs_mask = 0;
    fr.query_used       = 0;
    fr.frame_serial     = std::numeric_limits<uint64_t>::max();
}

// ====================== 외부 API ======================

AndroidVkGpuContext*
android_gpu_usage_create(VkPhysicalDevice            phys_dev,
                         VkDevice                    device,
                         float                       timestamp_period_ns,
                         uint32_t                    timestamp_valid_bits,
                         const AndroidVkGpuDispatch& disp)
{
    auto ctx = new AndroidVkGpuContext{};
    ctx->phys_dev      = phys_dev;
    ctx->device        = device;
    ctx->disp          = disp;
    ctx->ts_period_ns  = (timestamp_period_ns > 0.0f) ? timestamp_period_ns : 0.0f;
    ctx->ts_valid_bits = timestamp_valid_bits;

    // [PATCH] submit2 함수 포인터 정규화: 런타임 분기 제거
    // - KHR만 있는 환경에서도 QueueSubmit2 하나로 호출 가능하게 만든다.
    if (!ctx->disp.QueueSubmit2 && ctx->disp.QueueSubmit2KHR)
        ctx->disp.QueueSubmit2 = ctx->disp.QueueSubmit2KHR;

    // Optional: host query reset (only if supported/enabled).
    // Safe for dxvk 1.10.x / low Vulkan: if unsupported, pointers stay null.
    ctx->fpResetQueryPool =
        reinterpret_cast<PFN_vkResetQueryPool>(vkGetDeviceProcAddr(device, "vkResetQueryPool"));
    ctx->fpResetQueryPoolEXT =
        reinterpret_cast<PFN_vkResetQueryPoolEXT>(vkGetDeviceProcAddr(device, "vkResetQueryPoolEXT"));
    
    // 정책: 기본 OFF. MANGOHUD_VKP=1일 때만 Vulkan 경로 활성.
    if (!android_gpu_usage_env_enabled()) {
        ctx->ts_supported = false;
        SPDLOG_INFO(
            "Android GPU usage: backend disabled (MANGOHUD_VKP!=1), context will be no-op"
        );
        return ctx;
    }

    if (ctx->ts_valid_bits == 0 || ctx->ts_valid_bits >= 64) {
        ctx->ts_mask = ~0ULL;
    } else {
        ctx->ts_mask = (1ULL << ctx->ts_valid_bits) - 1ULL;
    }
    ctx->last_present  = std::chrono::steady_clock::now();
    ctx->last_gpu_ms   = 0.0f;
    ctx->last_usage    = 0.0f;

    bool has_submit =
        (disp.QueueSubmit  != nullptr) ||
        (disp.QueueSubmit2 != nullptr) ||
        (disp.QueueSubmit2KHR != nullptr);

    bool dispatch_ok =
        has_submit &&
        disp.CreateQueryPool &&
        disp.DestroyQueryPool &&
        disp.GetQueryPoolResults &&
        disp.CreateCommandPool &&
        disp.DestroyCommandPool &&
        disp.ResetCommandPool &&
        disp.AllocateCommandBuffers &&
        disp.BeginCommandBuffer &&
        disp.EndCommandBuffer &&
        disp.CmdWriteTimestamp &&
        disp.CmdResetQueryPool;

    ctx->ts_supported.store(dispatch_ok && (ctx->ts_period_ns > 0.0f),
                            std::memory_order_relaxed);

    SPDLOG_INFO(
        "Android GPU usage: create ctx={} ts_period_ns={} ts_valid_bits={} dispatch_ok={} ts_supported={}",
        static_cast<void*>(ctx),
        ctx->ts_period_ns,
        ctx->ts_valid_bits,
        dispatch_ok,
        ctx->ts_supported.load(std::memory_order_relaxed)
    );

    if (!ctx->ts_supported.load(std::memory_order_relaxed)) {
        SPDLOG_WARN("Android GPU usage: Vulkan timestamps not supported, backend will be disabled");
    }

// NOTE: scratch는 thread_local로 이동 (락 밖 submit 지원 + 컨텍스트 메모리 고정비 최소화)
    return ctx;
}

void
android_gpu_usage_destroy(AndroidVkGpuContext* ctx)
{
    if (!ctx)
        return;

    // 먼저 destroying 플래그를 세워서 새 작업 유입을 막는다.
    ctx->destroying.store(true, std::memory_order_release);

    // 교착 방지: ctx->lock을 잡지 말고, destroy_mtx로 in_flight==0을 기다린다.
    {
        std::unique_lock<std::mutex> lk(ctx->destroy_mtx);
        ctx->cv.wait(lk, [&] { return ctx->in_flight.load(std::memory_order_acquire) == 0; });
    }

    // 이제 어떤 스레드도 Vulkan 경로에 들어오지 않는다. (in_flight==0)
    {
        std::lock_guard<std::mutex> g(ctx->lock);
        ctx->mode.store(BackendMode::Disabled, std::memory_order_relaxed);
        ctx->ts_supported.store(false, std::memory_order_relaxed);
    }
    { std::lock_guard<std::mutex> mg(ctx->metrics_mtx); ctx->have_metrics = false; }
    android_gpu_usage_destroy_timestamp_resources(ctx);
    SPDLOG_INFO("Android GPU usage: destroy -> Vulkan resources destroyed");

    delete ctx;
}

// ====================== QueueSubmit(v1) 래핑 ======================

VkResult
android_gpu_usage_queue_submit(AndroidVkGpuContext* ctx,
                               VkQueue              queue,
                               uint32_t             queue_family_index,
                               uint32_t             submitCount,
                               const VkSubmitInfo*  pSubmits,
                               VkFence              fence)
{
    if (!ctx) return VK_ERROR_INITIALIZATION_FAILED;
    AndroidGpuApiGuard guard(ctx);

    if (!ctx->disp.QueueSubmit || !pSubmits || submitCount == 0)
        return ctx->disp.QueueSubmit
             ? ctx->disp.QueueSubmit(queue, submitCount, pSubmits, fence)
             : VK_ERROR_INITIALIZATION_FAILED;

    if (ctx->destroying.load(std::memory_order_acquire))
        return ctx->disp.QueueSubmit(queue, submitCount, pSubmits, fence);

    if (!ctx->ts_supported.load(std::memory_order_relaxed))
        return ctx->disp.QueueSubmit(queue, submitCount, pSubmits, fence);

    if (!android_gpu_usage_should_sample(ctx))
        return ctx->disp.QueueSubmit(queue, submitCount, pSubmits, fence);

    if (submitCount > kSubmitCountHardCap)
        return ctx->disp.QueueSubmit(queue, submitCount, pSubmits, fence);

    auto submit_fn = [&](uint32_t n, const VkSubmitInfo* s, VkFence f) -> VkResult {
        return ctx->disp.QueueSubmit(queue, n, s, f);
    };

    return android_gpu_usage_queue_submit_impl(ctx,
                                               queue_family_index,
                                               submitCount, pSubmits,
                                               fence,
                                               submit_fn,
                                               "QueueSubmit failed");
}

// ====================== QueueSubmit2(v2) 래핑 ======================

VkResult
android_gpu_usage_queue_submit2(AndroidVkGpuContext* ctx,
                                VkQueue              queue,
                                uint32_t             queue_family_index,
                                uint32_t             submitCount,
                                const VkSubmitInfo2* pSubmits,
                                VkFence              fence)
{
    if (!ctx) return VK_ERROR_INITIALIZATION_FAILED;
    AndroidGpuApiGuard guard(ctx);

    PFN_vkQueueSubmit2 fpSubmit2 = ctx->disp.QueueSubmit2
                                ? ctx->disp.QueueSubmit2
                                : ctx->disp.QueueSubmit2KHR;
    if (!fpSubmit2 || !pSubmits || submitCount == 0)
        return fpSubmit2 ? fpSubmit2(queue, submitCount, pSubmits, fence)
                         : VK_ERROR_INITIALIZATION_FAILED;

    if (ctx->destroying.load(std::memory_order_acquire))
        return fpSubmit2(queue, submitCount, pSubmits, fence);

    if (!ctx->ts_supported.load(std::memory_order_relaxed))
        return fpSubmit2(queue, submitCount, pSubmits, fence);

    if (!android_gpu_usage_should_sample(ctx))
        return fpSubmit2(queue, submitCount, pSubmits, fence);

    if (submitCount > kSubmitCountHardCap)
        return fpSubmit2(queue, submitCount, pSubmits, fence);

    auto submit_fn = [&](uint32_t n, const VkSubmitInfo2* s, VkFence f) -> VkResult {
        return fpSubmit2(queue, n, s, f);
    };

    return android_gpu_usage_queue_submit_impl(ctx,
                                               queue_family_index,
                                               submitCount, pSubmits,
                                               fence,
                                               submit_fn,
                                               "QueueSubmit2 failed");
}

// ====================== Present 시점 처리 ======================
void
android_gpu_usage_on_present(AndroidVkGpuContext*    ctx,
                             VkQueue                 queue,
                             uint32_t                queue_family_index,
                             const VkPresentInfoKHR* present_info,
                             uint32_t                swapchain_index,
                             uint32_t                image_index)
{
    (void)queue;
    (void)queue_family_index;
    (void)present_info;
    (void)swapchain_index;
    (void)image_index;

    if (!ctx)
        return;

    AndroidGpuApiGuard guard(ctx);

    if (!android_gpu_usage_env_enabled())
        return;

    if (ctx->destroying.load(std::memory_order_acquire))
        return;    
    
    using clock = std::chrono::steady_clock;
    const auto now = clock::now();

    struct PendingRead {
        bool     have       = false;
        uint32_t slot_idx   = 0;
        uint64_t serial     = 0;
        uint32_t q_start    = 0;
        uint32_t q_count    = 0;
        uint64_t valid_mask = 0;
        float    cpu_ms     = 0.0f;
    } read{};
    // (A) metrics lock: CPU dt + ring 기록 (submit과 분리)
    float frame_cpu_ms = 16.0f;
    const uint64_t cur_serial = ctx->frame_index.load(std::memory_order_relaxed);
    {
        std::lock_guard<std::mutex> mg(ctx->metrics_mtx);
        if (ctx->last_present.time_since_epoch().count() != 0) {
            const double dt_ms =
                std::chrono::duration<double, std::milli>(now - ctx->last_present).count();
            frame_cpu_ms = static_cast<float>(dt_ms > 0.001 ? dt_ms : 0.001);
        }
        ctx->last_present = now;

        auto& cs = ctx->cpu_ring[cur_serial & (AndroidVkGpuContext::CPU_RING - 1u)];
        cs.serial = cur_serial;
        cs.ms     = frame_cpu_ms;
    }

    auto cpu_ms_for_serial = [&](uint64_t serial) -> float {
        std::lock_guard<std::mutex> mg(ctx->metrics_mtx);
        const auto& cs2 = ctx->cpu_ring[serial & (AndroidVkGpuContext::CPU_RING - 1u)];
        return (cs2.serial == serial) ? cs2.ms : 0.0f;
    };

    try {
        // --------- (A) 짧은 락: CPU dt 갱신 + 후보 슬롯 선택 + init ---------
        {
            std::lock_guard<std::mutex> g(ctx->lock);
            if (ctx->destroying.load(std::memory_order_relaxed)) return;

            // --- Suspended 상태: 주기적으로 pending을 "안전하게" drain(=읽고 consume)해서 회복한다.
            const BackendMode mode = ctx->mode.load(std::memory_order_relaxed);
            bool probe = false;
            
            if (mode == BackendMode::Disabled) {
                // 완전 종료 상태: 프레임 인덱스만 굴리고 끝
                ctx->frame_index++;
                return;
            }
            
            if (mode == BackendMode::Suspended) {
                // 너무 자주 긁지 말고, 쿨다운 지난 뒤 일정 간격으로만 probe
                if (now >= ctx->suspend_until) {
                    if (ctx->last_probe.time_since_epoch().count() == 0 ||
                        (now - ctx->last_probe) >= kSuspendedProbeEvery) {
                        ctx->last_probe = now;
                        probe = true;
                    }
                }
            }
            
            // GPU 후보 슬롯 선택
            if (ctx->ts_supported.load(std::memory_order_relaxed) && ctx->query_pool != VK_NULL_HANDLE) {
                if (mode == BackendMode::Active) {
                    // 기존 로직 그대로: read_serial + FRAME_LAG 기반
                    const uint64_t fi = ctx->frame_index.load(std::memory_order_relaxed);
                    if (ctx->read_serial + AndroidVkGpuContext::FRAME_LAG <= fi) {
                        for (uint32_t step = 0; step < AndroidVkGpuContext::MAX_FRAMES; ++step) {
                            const uint64_t serial = ctx->read_serial;
                            if (serial + AndroidVkGpuContext::FRAME_LAG > fi) break;
            
                            const uint32_t idx = static_cast<uint32_t>(serial % AndroidVkGpuContext::MAX_FRAMES);
                            auto& fr = ctx->frames[idx];
            
                            if (!(fr.frame_serial == serial && fr.has_queries &&
                                  fr.valid_pairs_mask != 0 && fr.query_used >= 2)) {
                                ctx->read_serial++;
                                continue;
                            }
            
                            read.have     = true;
                            read.slot_idx = idx;
                            read.serial   = serial;
                            read.q_start  = fr.query_start;
                            read.q_count  = fr.query_used;
                            read.valid_mask = fr.valid_pairs_mask;
            
                            read.cpu_ms = cpu_ms_for_serial(serial);
                            break;
                        }
                    }
                } else if (probe) {
                    // Suspended probe: frames[]를 훑어서 pending 하나를 "읽기만" 해서 안전하게 drain
                    uint32_t best_idx = 0;
                    uint64_t best_serial = std::numeric_limits<uint64_t>::max();
                    for (uint32_t i = 0; i < AndroidVkGpuContext::MAX_FRAMES; ++i) {
                        auto& fr = ctx->frames[i];
                        if (fr.has_queries && fr.valid_pairs_mask != 0 &&
                            fr.query_used >= 2 && fr.frame_serial != std::numeric_limits<uint64_t>::max()) {
                            if (fr.frame_serial < best_serial) {
                                best_serial = fr.frame_serial;
                                best_idx = i;
                            }
                        }
                    }
                    if (best_serial != std::numeric_limits<uint64_t>::max()) {
                        auto& fr = ctx->frames[best_idx];
                        read.have     = true;
                        read.slot_idx = best_idx;
                        read.serial   = fr.frame_serial;
                        read.q_start  = fr.query_start;
                        read.q_count  = fr.query_used;
                        read.valid_mask = fr.valid_pairs_mask;
            
                        read.cpu_ms = cpu_ms_for_serial(read.serial);
                    } else {
                        // pending이 없으면 회복
                        ctx->mode.store(BackendMode::Active, std::memory_order_relaxed);
                        ctx->notready_streak = 0;
                        ctx->error_streak = 0;
                        SPDLOG_INFO("Android GPU usage: RESUME -> no pending queries left");
                    }
                }
            }
        }

        float frame_gpu_ms = 0.0f;
        AndroidGpuReadStatus st = AndroidGpuReadStatus::NotReady;

        if (read.have) {
            st = android_gpu_usage_query_range_gpu_ms(ctx, read.q_start, read.q_count, read.valid_mask, &frame_gpu_ms);
        }

        // --------- (C) state lock: 슬롯 소비/서스펜드/리드시리얼/프레임인덱스 ---------
        {
            std::lock_guard<std::mutex> g(ctx->lock);

            if (st == AndroidGpuReadStatus::DeviceLost) {
                ctx->ts_supported.store(false, std::memory_order_relaxed);
                ctx->mode.store(BackendMode::Disabled, std::memory_order_relaxed);
                SPDLOG_WARN("Android GPU usage: DEVICE_LOST on GetQueryPoolResults -> disable backend (no destroy)");
                frame_gpu_ms = 0.0f;
            }
            else if (st == AndroidGpuReadStatus::Ready) {
                auto& fr = ctx->frames[read.slot_idx];
                if (fr.frame_serial == read.serial && fr.has_queries && fr.query_used == read.q_count) {
                    android_gpu_usage_consume_slot(fr);
            
                    if (ctx->mode.load(std::memory_order_relaxed) == BackendMode::Active) {
                        ctx->read_serial = read.serial + 1;
                    } else {
                        ctx->read_serial = std::max<uint64_t>(ctx->read_serial, read.serial + 1);
                    }
            
                    // Ready면 결과가 0ms여도 "드레인 성공"이다.
                    ctx->notready_streak = 0;
                    ctx->error_streak = 0;
            
                    // Suspended probe 드레인 중이면 pending 다 비었을 때 resume
                    if (ctx->mode.load(std::memory_order_relaxed) == BackendMode::Suspended) {
                        bool any_pending = false;
                        for (uint32_t i = 0; i < AndroidVkGpuContext::MAX_FRAMES; ++i) {
                            if (ctx->frames[i].has_queries) { any_pending = true; break; }
                        }
                        if (!any_pending && now >= ctx->suspend_until) {
                            ctx->mode.store(BackendMode::Active, std::memory_order_relaxed);
                            ctx->notready_streak = 0;
                            ctx->error_streak = 0;
                            SPDLOG_INFO("Android GPU usage: RESUME -> drained pending queries");
                        }
                    }
                } else {
                    // 슬롯 상태가 바뀐 레이스: 이번 샘플 무효
                    frame_gpu_ms = 0.0f;
                    st = AndroidGpuReadStatus::NotReady;
                }
            }
            else if (st == AndroidGpuReadStatus::Error) {
                ctx->error_streak++;
                android_gpu_usage_suspend_locked(ctx,
                    "GetQueryPoolResults ERROR -> suspend (no consume/reuse)",
                    std::chrono::milliseconds(1500));
                frame_gpu_ms = 0.0f;
            }
            else { // NotReady 또는 (Ready but frame_gpu_ms==0)
                ctx->notready_streak++;
                if (ctx->notready_streak >= kNotReadyLimit) {
                    android_gpu_usage_suspend_locked(ctx,
                        "GetQueryPoolResults NOT_READY too long",
                        kCooldownNotReadyLong);
                    ctx->notready_streak = 0;
                }
                frame_gpu_ms = 0.0f;
            }
            ctx->frame_index++;
        }

        // --------- (D) metrics lock: smoothing/last metrics (submit과 분리) ---------
        {
            std::lock_guard<std::mutex> mg(ctx->metrics_mtx);
            if (ctx->window_start.time_since_epoch().count() == 0) {
                ctx->window_start = now;
            }

            if (st == AndroidGpuReadStatus::Ready && frame_gpu_ms > 0.0f && read.cpu_ms > 0.0f) {
                ctx->acc_cpu_ms_sampled += read.cpu_ms;
                ctx->acc_frames_sampled += 1;
                ctx->acc_gpu_ms         += frame_gpu_ms;
                ctx->acc_gpu_samples    += 1;
            }

            constexpr auto WINDOW = std::chrono::milliseconds(500);
            if ((now - ctx->window_start) >= WINDOW) {
                if (ctx->acc_gpu_samples > 0 && ctx->acc_frames_sampled > 0 && ctx->acc_cpu_ms_sampled > 0.0) {
                    const double avg_cpu_ms =
                        ctx->acc_cpu_ms_sampled / static_cast<double>(ctx->acc_frames_sampled);
                    const double avg_gpu_ms =
                        ctx->acc_gpu_ms / static_cast<double>(ctx->acc_gpu_samples);

                    float usage = 0.0f;
                    if (avg_cpu_ms > 0.0 && avg_gpu_ms > 0.0)
                        usage = static_cast<float>((avg_gpu_ms / avg_cpu_ms) * 100.0);
                    if (!std::isfinite(usage)) usage = 0.0f;
                    if (usage < 0.0f) usage = 0.0f;
                    if (usage > 100.0f) usage = 100.0f;

                    const float alpha = 0.5f;
                    if (!ctx->have_metrics) {
                        ctx->smooth_usage  = usage;
                        ctx->smooth_gpu_ms = static_cast<float>(avg_gpu_ms);
                    } else {
                        ctx->smooth_usage  = ctx->smooth_usage * (1.0f - alpha) + usage * alpha;
                        ctx->smooth_gpu_ms = ctx->smooth_gpu_ms * (1.0f - alpha) + static_cast<float>(avg_gpu_ms) * alpha;
                    }
                    ctx->last_usage   = ctx->smooth_usage;
                    ctx->last_gpu_ms  = ctx->smooth_gpu_ms;
                    ctx->have_metrics = true;
                }

                ctx->acc_cpu_ms_sampled = 0.0;
                ctx->acc_frames_sampled = 0;
                ctx->acc_gpu_ms      = 0.0;
                ctx->acc_gpu_samples = 0;
                ctx->window_start = now;
            }
        }        
    } catch (const std::exception& e) {
        SPDLOG_WARN("Android GPU usage: on_present exception: {}", e.what());
    } catch (...) {
        SPDLOG_WARN("Android GPU usage: on_present unknown exception");
    }
}

bool
android_gpu_usage_get_metrics(AndroidVkGpuContext* ctx,
                              float*              out_gpu_ms,
                              float*              out_usage)
{
    if (!ctx)
        return false;

    AndroidGpuApiGuard guard(ctx);

    // MANGOHUD_VKP!=1 이면 Vulkan 경로 비활성 (out_* 건들지 않음)
    if (!android_gpu_usage_env_enabled())
        return false;

    if (ctx->destroying.load(std::memory_order_acquire))
        return false;

    std::lock_guard<std::mutex> mg(ctx->metrics_mtx);

    if (!ctx->have_metrics)
        return false;

    if (out_gpu_ms)
        *out_gpu_ms = ctx->last_gpu_ms;
    if (out_usage)
        *out_usage  = ctx->last_usage;

    return true;
}

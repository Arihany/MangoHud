#include "iostats.h"
#include <cstdio>
#include <chrono>
#include <spdlog/spdlog.h>

#ifndef TEST_ONLY
#include "hud_elements.h"
#endif

struct iostats g_io_stats;

void getIoStats(iostats& io)
{
    Clock::time_point now = Clock::now();

#if defined(__ANDROID__)
    static bool io_disabled          = false;
    static bool io_missing_logged    = false;
    static bool io_init_logged       = false;

    if (io_disabled) {
        io.diff       = { 0.0f, 0.0f };
        io.per_second = { 0.0f, 0.0f };
        io.last_update = now;
        return;
    }
#else
    static bool io_missing_logged_desktop = false;
#endif

    // 대상 pid 기준으로 /proc/<pid>/io 경로 구성
    char path[64];
    int pid = 0;
#ifndef TEST_ONLY
    pid = HUDElements.g_gamescopePid;
#endif
    if (pid < 1)
        std::snprintf(path, sizeof(path), "/proc/self/io");
    else
        std::snprintf(path, sizeof(path), "/proc/%d/io", pid);

    // 공통 파서: 플랫폼별로 필요한 필드만 읽어서 out_read/out_write에 채움
    auto parse_io_file = [&](unsigned long long& out_read, unsigned long long& out_write) -> bool {
        out_read  = 0;
        out_write = 0;

        FILE* f = std::fopen(path, "r");
        if (!f)
            return false;

        char line[128];
        while (std::fgets(line, sizeof(line), f)) {
#if defined(__ANDROID__)
            // Android: rchar / wchar 사용
            if (std::sscanf(line, "rchar: %llu", &out_read) == 1)
                continue;
            if (std::sscanf(line, "wchar: %llu", &out_write) == 1)
                continue;
#else
            // Desktop: read_bytes / write_bytes 사용
            unsigned long long tmp = 0;
            if (std::sscanf(line, "read_bytes: %llu", &tmp) == 1) {
                out_read = tmp;
                continue;
            }
            if (std::sscanf(line, "write_bytes: %llu", &tmp) == 1) {
                out_write = tmp;
                continue;
            }
#endif
        }

        std::fclose(f);
        return true;
    };

    // ===== 첫 호출 처리: 초기값을 읽어 curr == prev 로 맞추고 diff=0 으로 시작 =====
    if (io.last_update.time_since_epoch().count() == 0) {
        unsigned long long initial_read  = 0;
        unsigned long long initial_write = 0;

        if (!parse_io_file(initial_read, initial_write)) {
#if defined(__ANDROID__)
            if (!io_missing_logged) {
                SPDLOG_DEBUG("iostats: cannot open {} (initial), disabling IO stats", path);
                io_missing_logged = true;
            }
            io_disabled = true;
#else
            if (!io_missing_logged_desktop) {
                SPDLOG_DEBUG("iostats: cannot open {} (initial), zeroing IO stats", path);
                io_missing_logged_desktop = true;
            }
#endif
            io.curr.read_bytes  = 0;
            io.curr.write_bytes = 0;
            io.prev             = io.curr;
            io.diff             = { 0.0f, 0.0f };
            io.per_second       = { 0.0f, 0.0f };
            io.last_update      = now;
            return;
        }

        io.curr.read_bytes  = initial_read;
        io.curr.write_bytes = initial_write;
        io.prev             = io.curr;
        io.diff             = { 0.0f, 0.0f };
        io.per_second       = { 0.0f, 0.0f };
        io.last_update      = now;

#if defined(__ANDROID__)
        if (!io_init_logged) {
            SPDLOG_DEBUG("iostats: Android using {} rchar/wchar as logical IO counters", path);
            io_init_logged = true;
        }
#endif
        return;
    }

    // ===== 일반 갱신 경로 =====
    std::chrono::duration<float> time_diff = now - io.last_update;
    float dt = time_diff.count();

    if (dt < 0.001f) {
        io.last_update = now;
        return;
    }

    // 이전 값 백업
    io.prev.read_bytes  = io.curr.read_bytes;
    io.prev.write_bytes = io.curr.write_bytes;

    // 새 샘플 읽기
    unsigned long long r = 0;
    unsigned long long w = 0;

    if (!parse_io_file(r, w)) {
#if defined(__ANDROID__)
        if (!io_missing_logged) {
            SPDLOG_DEBUG("iostats: cannot open {}, disabling IO stats", path);
            io_missing_logged = true;
        }
        io_disabled = true;
#else
        if (!io_missing_logged_desktop) {
            SPDLOG_DEBUG("iostats: cannot open {}, zeroing IO stats", path);
            io_missing_logged_desktop = true;
        }
#endif
        io.diff       = { 0.0f, 0.0f };
        io.per_second = { 0.0f, 0.0f };
        io.last_update = now;
        return;
    }

    io.curr.read_bytes  = r;
    io.curr.write_bytes = w;

    constexpr float TO_MIB = 1024.0f * 1024.0f;

    unsigned long long r_diff =
        (io.curr.read_bytes >= io.prev.read_bytes)
            ? (io.curr.read_bytes - io.prev.read_bytes)
            : 0ULL;

    unsigned long long w_diff =
        (io.curr.write_bytes >= io.prev.write_bytes)
            ? (io.curr.write_bytes - io.prev.write_bytes)
            : 0ULL;

    io.diff.read  = static_cast<float>(r_diff) / TO_MIB;
    io.diff.write = static_cast<float>(w_diff) / TO_MIB;

    if (dt > 0.0f) {
        io.per_second.read  = io.diff.read  / dt;
        io.per_second.write = io.diff.write / dt;
    } else {
        io.per_second = { 0.0f, 0.0f };
    }

    io.last_update = now;
}

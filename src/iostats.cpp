#include "iostats.h"
#include "string_utils.h"
#include <fstream>
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
    static bool io_disabled = false;

    if (io_disabled) {
        io.diff.read = io.diff.write = 0.0f;
        io.per_second.read = io.per_second.write = 0.0f;
        io.last_update = now;
        return;
    }

    if (io.last_update.time_since_epoch().count() == 0) {
        io.last_update = now;

        std::string f = "/proc/";
        {
    #ifndef TEST_ONLY
            const auto gs_pid = HUDElements.g_gamescopePid;
            f += (gs_pid < 1) ? "self" : std::to_string(gs_pid);
    #else
            f += "self";
    #endif
            f += "/io";
        }

        static bool logged_once = false;
        std::ifstream file(f);
        if (!file.is_open()) {
            if (!logged_once) {
                SPDLOG_DEBUG("iostats: cannot open {} on Android, IO stats disabled", f);
                logged_once = true;
            }
            io.curr.read_bytes  = 0;
            io.curr.write_bytes = 0;
            io.prev.read_bytes  = 0;
            io.prev.write_bytes = 0;
            io.diff.read = io.diff.write = 0.0f;
            io.per_second.read = io.per_second.write = 0.0f;
            io_disabled = true; // 여기서 진짜로 영구 disable
            return;
        }

        unsigned long long rchar = 0;
        unsigned long long wchar = 0;

        for (std::string line; std::getline(file, line);) {
            if (starts_with(line, "rchar:")) {
                // "rchar:" 뒤로 다 넘겨도 stoull가 공백 무시하니까 괜찮음
                try_stoull(rchar, line.substr(6));
            } else if (starts_with(line, "wchar:")) {
                try_stoull(wchar, line.substr(6));
            }
        }

        io.curr.read_bytes  = rchar;
        io.curr.write_bytes = wchar;
        io.prev.read_bytes  = rchar;
        io.prev.write_bytes = wchar;

        io.diff.read = io.diff.write = 0.0f;
        io.per_second.read = io.per_second.write = 0.0f;

        if (!logged_once) {
            SPDLOG_DEBUG("iostats: Android using /proc/<pid>/io rchar/wchar as logical IO counters");
            logged_once = true;
        }

        return;
    }

    // 이후 호출: Δrchar / Δwchar 기반으로 MiB / MiB/s 계산
    std::chrono::duration<float> time_diff = now - io.last_update;
    float dt = time_diff.count();

    // 이전 값 백업
    io.prev.read_bytes  = io.curr.read_bytes;
    io.prev.write_bytes = io.curr.write_bytes;

    std::string f = "/proc/";
    {
#ifndef TEST_ONLY
        const auto gs_pid = HUDElements.g_gamescopePid;
        f += (gs_pid < 1) ? "self" : std::to_string(gs_pid);
#else
        f += "self";
#endif
        f += "/io";
    }

    static bool io_missing_logged = false;
    std::ifstream file(f);
    if (!file.is_open()) {
        if (!io_missing_logged) {
            SPDLOG_DEBUG("iostats: cannot open {} on Android, zeroing IO stats", f);
            io_missing_logged = true;
        }
        io.diff.read = io.diff.write = 0.0f;
        io.per_second.read = io.per_second.write = 0.0f;
        io.last_update = now;
        io_disabled = true; // 여기서도 실패하면 포기
        return;
    }

    unsigned long long rchar = 0;
    unsigned long long wchar = 0;

    for (std::string line; std::getline(file, line);) {
        if (starts_with(line, "rchar:")) {
            try_stoull(rchar, line.substr(6));
        } else if (starts_with(line, "wchar:")) {
            try_stoull(wchar, line.substr(6));
        }
    }

    io.curr.read_bytes  = rchar;
    io.curr.write_bytes = wchar;

    if (dt > 0.0f) {
        constexpr float TO_MIB = 1024.0f * 1024.0f;

        unsigned long long read_diff_bytes =
            (io.curr.read_bytes >= io.prev.read_bytes)
                ? (io.curr.read_bytes - io.prev.read_bytes)
                : 0ULL;

        unsigned long long write_diff_bytes =
            (io.curr.write_bytes >= io.prev.write_bytes)
                ? (io.curr.write_bytes - io.prev.write_bytes)
                : 0ULL;

        io.diff.read  = static_cast<float>(read_diff_bytes) / TO_MIB;
        io.diff.write = static_cast<float>(write_diff_bytes) / TO_MIB;

        io.per_second.read  = io.diff.read  / dt;
        io.per_second.write = io.diff.write / dt;
    } else {
        io.diff.read = io.diff.write = 0.0f;
        io.per_second.read = io.per_second.write = 0.0f;
    }

    io.last_update = now;
    return;

#else
    // Linux Desktop 로직
    if (io.last_update.time_since_epoch().count() == 0) {
        io.last_update = now;
        return;
    }

    std::chrono::duration<float> time_diff = now - io.last_update;
    float dt = time_diff.count();

    // 이전 값 보관
    io.prev.read_bytes  = io.curr.read_bytes;
    io.prev.write_bytes = io.curr.write_bytes;

    // 대상 pid 선택 (gamescopePid가 없으면 self)
    std::string f = "/proc/";
    {
        const auto gs_pid = HUDElements.g_gamescopePid;
        f += (gs_pid < 1) ? "self" : std::to_string(gs_pid);
        f += "/io";
    }

    static bool io_missing_logged = false;
    std::ifstream file(f);

    if (!file.is_open()) {
        if (!io_missing_logged) {
            SPDLOG_DEBUG("iostats: cannot open {}, disabling IO stats", f);
            io_missing_logged = true;
        }
        // 실패 시 0 처리
        io.diff.read = io.diff.write = 0.0f;
        io.per_second.read = io.per_second.write = 0.0f;
        io.last_update = now;
        return;
    }

    // 파일 파싱
    for (std::string line; std::getline(file, line);) {
        if (starts_with(line, "read_bytes:")) {
            try_stoull(io.curr.read_bytes, line.substr(12));
        } else if (starts_with(line, "write_bytes:")) {
            try_stoull(io.curr.write_bytes, line.substr(13));
        }
    }

    // 속도 계산
    if (dt > 0.0f) {
        // Bytes -> MiB 변환
        constexpr float TO_MIB = 1024.0f * 1024.0f;
        
        // Overflow 방지
        unsigned long long read_diff_bytes = (io.curr.read_bytes >= io.prev.read_bytes) 
            ? (io.curr.read_bytes - io.prev.read_bytes) : 0;
            
        unsigned long long write_diff_bytes = (io.curr.write_bytes >= io.prev.write_bytes) 
            ? (io.curr.write_bytes - io.prev.write_bytes) : 0;

        io.diff.read  = static_cast<float>(read_diff_bytes) / TO_MIB;
        io.diff.write = static_cast<float>(write_diff_bytes) / TO_MIB;

        io.per_second.read  = io.diff.read  / dt;
        io.per_second.write = io.diff.write / dt;
    } else {
        io.diff.read = io.diff.write = 0.0f;
        io.per_second.read = io.per_second.write = 0.0f;
    }

    io.last_update = now;
#endif
}

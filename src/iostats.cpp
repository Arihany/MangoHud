#include "iostats.h"
#include "string_utils.h"
#include <fstream>
#include "hud_elements.h"

struct iostats g_io_stats;

void getIoStats(iostats& io)
{
    Clock::time_point now = Clock::now();

#if defined(__ANDROID__)
    // Android / Winlator:
    // - /proc/<pid>/io는 의미도 애매하고, VFS I/O만 낭비한다.
    // - 여기서는 지표를 완전히 비활성화하고 0만 유지.
    static bool logged_once = false;

    if (!logged_once) {
        SPDLOG_DEBUG("iostats: disabled on Android (skipping /proc/<pid>/io entirely)");
        logged_once = true;
    }

    // 최소한 last_update는 갱신해서 외부에서 time_diff 계산시 이상한 값 안 나오게 유지
    io.prev = io.curr;
    io.diff.read = io.diff.write = 0.0f;
    io.per_second.read = io.per_second.write = 0.0f;
    io.last_update = now;
    return;
#else
    std::chrono::duration<float> time_diff = now - io.last_update;

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

        // 실패 시 지표는 0으로 유지
        io.diff.read = io.diff.write = 0.0f;
        io.per_second.read = io.per_second.write = 0.0f;
        io.last_update = now;
        return;
    }

    for (std::string line; std::getline(file, line);) {
        if (starts_with(line, "read_bytes:")) {
            try_stoull(io.curr.read_bytes, line.substr(12));
        } else if (starts_with(line, "write_bytes:")) {
            try_stoull(io.curr.write_bytes, line.substr(13));
        }
    }

    if (io.last_update.time_since_epoch().count() != 0) {
        const float dt = time_diff.count();
        if (dt > 0.0f) {
            io.diff.read  = (io.curr.read_bytes  - io.prev.read_bytes)  / (1024.f * 1024.f);
            io.diff.write = (io.curr.write_bytes - io.prev.write_bytes) / (1024.f * 1024.f);

            io.per_second.read  = io.diff.read  / dt;
            io.per_second.write = io.diff.write / dt;
        } else {
            // 시간 역행/0 보호
            io.diff.read = io.diff.write = 0.0f;
            io.per_second.read = io.per_second.write = 0.0f;
        }
    }

    io.last_update = now;
#endif
}

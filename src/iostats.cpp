#include "iostats.h"
#include "string_utils.h"
#include <fstream>
#include <chrono>
#include <spdlog/spdlog.h>

#ifndef TEST_ONLY
#include "hud_elements.h"
#endif

// iostats.h에 정의되어 있지 않을 경우를 대비한 안전장치
using Clock = std::chrono::steady_clock;

struct iostats g_io_stats;

void getIoStats(iostats& io)
{
    Clock::time_point now = Clock::now();

#if defined(__ANDROID__)
    // Android / Winlator 최적화:
    // /proc/<pid>/io 파싱은 모바일에서 오버헤드가 크고 효용성이 낮음.
    // 기능을 비활성화하고 0으로 초기화하여 리소스 낭비를 막는다.
    
    static bool logged_once = false;
    if (!logged_once) {
        SPDLOG_DEBUG("iostats: disabled on Android (skipping /proc/<pid>/io entirely)");
        logged_once = true;
    }

    // [수정] 구조체 직접 대입 대신 멤버별 대입으로 변경
    io.prev.read_bytes  = io.curr.read_bytes;
    io.prev.write_bytes = io.curr.write_bytes;
    
    // 변화량 0으로 초기화
    io.diff.read = io.diff.write = 0.0f;
    io.per_second.read = io.per_second.write = 0.0f;
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

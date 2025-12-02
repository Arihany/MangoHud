#include <spdlog/spdlog.h>
#include <fstream>
#include <string>
#include <unistd.h>
#include <cstdlib> // std::strtoull

#include "memory.h"

#ifndef TEST_ONLY
#include "hud_elements.h"
#endif

float memused, memmax, swapused;
uint64_t proc_mem_resident, proc_mem_shared, proc_mem_virt;

static inline float parse_val_mb(const std::string& s) {
    const char* cstr = s.c_str();
    const char* p = cstr;

    // 숫자 시작 위치 찾기
    while (*p && (*p < '0' || *p > '9'))
        ++p;

    if (!*p)
        return 0.0f;

    char* end = nullptr;
    unsigned long long kb = std::strtoull(p, &end, 10);
    if (end == p)
        return 0.0f;

    // kB -> GiB (kB / 1024 / 1024)
    return static_cast<float>(kb) / (1024.0f * 1024.0f);
}

void update_meminfo() {
    static float cached_memtotal = 0.0f;

    std::ifstream file("/proc/meminfo");
    if (!file.is_open()) {
        SPDLOG_ERROR("can't open /proc/meminfo");
        return;
    }

    float mem_total  = cached_memtotal;
    float mem_avail  = 0.0f;
    float swap_total = 0.0f;
    float swap_free  = 0.0f;

    unsigned found = 0;
    std::string line;

    // 필요한 4개 키만 읽고 끝내기
    while (std::getline(file, line) && found < 4) {
        if (line.rfind("MemTotal:", 0) == 0) {
            mem_total = parse_val_mb(line);
            cached_memtotal = mem_total;
            ++found;
        } else if (line.rfind("MemAvailable:", 0) == 0) {
            mem_avail = parse_val_mb(line);
            ++found;
        } else if (line.rfind("SwapTotal:", 0) == 0) {
            swap_total = parse_val_mb(line);
            ++found;
        } else if (line.rfind("SwapFree:", 0) == 0) {
            swap_free = parse_val_mb(line);
            ++found;
        }
    }

    if (mem_total <= 0.0f)
        return;

    memmax   = mem_total;
    memused  = mem_total - mem_avail;
    swapused = swap_total - swap_free;
}

void update_procmem() {
    long page_size = sysconf(_SC_PAGESIZE);
    if (page_size <= 0)
        page_size = 4096;
    const unsigned long long page_bytes =
        static_cast<unsigned long long>(page_size);

    std::string f = "/proc/";

#ifndef TEST_ONLY
    {
        auto gs_pid = HUDElements.g_gamescopePid;
        f += (gs_pid < 1) ? "self" : std::to_string(gs_pid);
    }
#else
    f += "self";
#endif

    f += "/statm";

    std::ifstream file(f);
    if (!file.is_open()) {
        SPDLOG_ERROR("can't open {}", f);
        return;
    }

    std::string line;
    if (!std::getline(file, line) || line.empty())
        return;

    const char* c = line.c_str();
    char* end = nullptr;

    // /proc/<pid>/statm 포맷:
    // size resident shared ...
    unsigned long long size_pages = std::strtoull(c, &end, 10);
    if (end == c)
        return;

    unsigned long long resident_pages = std::strtoull(end, &end, 10);
    unsigned long long shared_pages   = std::strtoull(end, nullptr, 10);

    proc_mem_virt     = size_pages     * page_bytes;
    proc_mem_resident = resident_pages * page_bytes;
    proc_mem_shared   = shared_pages   * page_bytes;
}

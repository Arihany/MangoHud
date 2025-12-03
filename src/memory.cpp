#include <spdlog/spdlog.h>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <unistd.h>

#include "memory.h"

#ifndef TEST_ONLY
#include "hud_elements.h"
#endif

float memused, memmax, swapused;
uint64_t proc_mem_resident, proc_mem_shared, proc_mem_virt;

// "Label:   12345 kB" 형태에서 숫자만 추출해서 GiB로 변환
static inline float parse_kb_to_gib(const char* line)
{
    const char* p = line;

    // 숫자 시작점 찾기
    while (*p && (*p < '0' || *p > '9'))
        ++p;

    if (!*p)
        return 0.0f;

    char* end = nullptr;
    unsigned long long kb = std::strtoull(p, &end, 10);
    if (end == p)
        return 0.0f;

    // kB -> GiB
    return static_cast<float>(kb) / (1024.0f * 1024.0f);
}

void update_meminfo()
{
    static float cached_memtotal = 0.0f;

    FILE* f = std::fopen("/proc/meminfo", "r");
    if (!f) {
        static bool log_once = false;
        if (!log_once) {
            SPDLOG_ERROR("memory: can't open /proc/meminfo");
            log_once = true;
        }
        return;
    }

    float mem_total  = cached_memtotal;
    float mem_avail  = 0.0f;
    float swap_total = 0.0f;
    float swap_free  = 0.0f;

    unsigned found = 0;
    char line[256];

    // 필요한 4개 키만 찾으면 조기 종료
    while (found < 4 && std::fgets(line, sizeof(line), f)) {
        if (std::strncmp(line, "MemTotal:", 9) == 0) {
            mem_total = parse_kb_to_gib(line);
            cached_memtotal = mem_total;
            ++found;
        } else if (std::strncmp(line, "MemAvailable:", 13) == 0) {
            mem_avail = parse_kb_to_gib(line);
            ++found;
        } else if (std::strncmp(line, "SwapTotal:", 10) == 0) {
            swap_total = parse_kb_to_gib(line);
            ++found;
        } else if (std::strncmp(line, "SwapFree:", 9) == 0) {
            swap_free = parse_kb_to_gib(line);
            ++found;
        }
    }

    std::fclose(f);

    if (mem_total <= 0.0f)
        return;

    // 약간의 방어적 클램프
    if (mem_avail < 0.0f)
        mem_avail = 0.0f;
    if (mem_avail > mem_total)
        mem_avail = mem_total;

    memmax   = mem_total;
    memused  = mem_total - mem_avail;
    swapused = (swap_total > swap_free) ? (swap_total - swap_free) : 0.0f;
}

void update_procmem()
{
    // 페이지 사이즈는 런타임에 바뀔 일이 없으니 한 번만 계산
    static const long page_size = []() {
        long ps = ::sysconf(_SC_PAGESIZE);
        return (ps > 0) ? ps : 4096L;
    }();
    static const unsigned long long page_bytes =
        static_cast<unsigned long long>(page_size);

    // /proc/<pid>/statm 경로 구성
    char path[64];
    int pid = 0;
#ifndef TEST_ONLY
    pid = HUDElements.g_gamescopePid;
#endif

    if (pid < 1)
        std::snprintf(path, sizeof(path), "/proc/self/statm");
    else
        std::snprintf(path, sizeof(path), "/proc/%d/statm", pid);

    FILE* f = std::fopen(path, "r");
    if (!f) {
        static bool log_once = false;
        if (!log_once) {
            SPDLOG_DEBUG("memory: can't open {}, keeping previous proc_mem stats", path);
            log_once = true;
        }
        return;
    }

    unsigned long long size_pages     = 0;
    unsigned long long resident_pages = 0;
    unsigned long long shared_pages   = 0;

    // /proc/<pid>/statm 포맷: size resident shared text lib data dt
    int n = std::fscanf(f, "%llu %llu %llu",
                        &size_pages,
                        &resident_pages,
                        &shared_pages);
    std::fclose(f);

    if (n < 3)
        return;

    proc_mem_virt     = size_pages     * page_bytes;
    proc_mem_resident = resident_pages * page_bytes;
    proc_mem_shared   = shared_pages   * page_bytes;
}

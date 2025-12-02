#include <spdlog/spdlog.h>
#include <map>
#include <fstream>
#include <string>
#include <unistd.h>
#include <array>

#include "memory.h"
#include "hud_elements.h"

float memused, memmax, swapused;
uint64_t proc_mem_resident, proc_mem_shared, proc_mem_virt;

void update_meminfo() {
    static float cached_memtotal = 0.0f;

    std::ifstream file("/proc/meminfo");
    if (!file.is_open()) {
        SPDLOG_ERROR("can't open /proc/meminfo");
        return;
    }

    float mem_total = cached_memtotal;
    float mem_avail = 0.0f;
    float swap_total = 0.0f;
    float swap_free  = 0.0f;

    unsigned found = 0;
    std::string line;

    auto parse_val_mb = [](const std::string& s) -> float {
        size_t start = s.find_first_of("0123456789");
        if (start == std::string::npos)
            return 0.0f;
        unsigned long long kb = std::strtoull(s.c_str() + start, nullptr, 10);
        return static_cast<float>(kb) / 1024.0f / 1024.0f;
    };

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

void update_procmem()
{
    auto page_size = sysconf(_SC_PAGESIZE);
    if (page_size < 0) page_size = 4096;

    std::string f = "/proc/";

    {
        auto gs_pid = HUDElements.g_gamescopePid;
        f += gs_pid < 1 ? "self" : std::to_string(gs_pid);
        f += "/statm";
    }

    std::ifstream file(f);

    if (!file.is_open()) {
        SPDLOG_ERROR("can't open {}", f);
        return;
    }

    size_t last_idx = 0;
    std::string line;
    std::getline(file, line);

    if (line.empty())
        return;

    std::array<uint64_t, 3> meminfo;

    for (auto i = 0; i < 3; i++) {
        auto idx = line.find(" ", last_idx);
        auto val = line.substr(last_idx, idx);

        meminfo[i] = std::stoull(val) * page_size;
        last_idx = idx + 1;
    }

    proc_mem_virt = meminfo[0];
    proc_mem_resident = meminfo[1];
    proc_mem_shared = meminfo[2];
}

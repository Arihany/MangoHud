#include "cpu.h"
#include <memory>
#include <string>
#include <iostream>
#include <fstream>
#include <sstream>
#include <dirent.h>
#include <string.h>
#include <algorithm>
#include <regex>
#include <inttypes.h>
#include <spdlog/spdlog.h>
#include "string_utils.h"
#include "gpu.h"
#include "file_utils.h"
#include <cctype> // std::tolower

#if defined(__ANDROID__)
#include <chrono>
#include <unistd.h>
#include <unordered_map>
#endif

#ifndef TEST_ONLY
#include "hud_elements.h"
#endif

#ifndef PROCDIR
#define PROCDIR "/proc"
#endif

#ifndef PROCSTATFILE
#define PROCSTATFILE PROCDIR "/stat"
#endif

#ifndef PROCMEMINFOFILE
#define PROCMEMINFOFILE PROCDIR "/meminfo"
#endif

#ifndef PROCCPUINFOFILE
#define PROCCPUINFOFILE PROCDIR "/cpuinfo"
#endif

#ifdef __ANDROID__
namespace {

struct CoreCtlEntry {
    int  cpu_id     = -1;
    int  busy       = 0;
    int  online     = 1;
    bool has_busy   = false;
    bool has_online = false;
};

#if defined(__ANDROID__)
static bool g_logged_fallback_switch = false;
#endif

static bool g_logged_corectl_summary = false;
static bool g_corectl_logged_summary = false;

static bool g_logged_policy_mhz   = false;
static bool g_logged_cpufreq_mhz  = false;
static bool g_logged_mhz_missing  = false;

// ANDROID: sysfs 기반 CPU 코어 enum
static bool android_enumerate_cpus(std::vector<CPUData>& out, CPUData& total)
{
    const char* base = "/sys/devices/system/cpu";
    DIR* dir = opendir(base);
    if (!dir) {
        SPDLOG_ERROR("Android CPU: failed to open {}", base);
        return false;
    }

    std::vector<int> ids;
    struct dirent* ent = nullptr;

    while ((ent = readdir(dir)) != nullptr) {
        const char* name = ent->d_name;

        // "cpu0", "cpu1" 형태만
        if (strncmp(name, "cpu", 3) != 0)
            continue;

        char* end = nullptr;
        long id = strtol(name + 3, &end, 10);
        if (!end || *end != '\0')
            continue;
        if (id < 0 || id > 1024)
            continue;

        ids.push_back(static_cast<int>(id));
    }

    closedir(dir);

    if (ids.empty()) {
        SPDLOG_ERROR("Android CPU: no cpuN entries under {}", base);
        return false;
    }

    std::sort(ids.begin(), ids.end());
    ids.erase(std::unique(ids.begin(), ids.end()), ids.end());

    out.clear();
    out.reserve(ids.size());

    for (int id : ids) {
        CPUData cpu{};          // zero-init
        cpu.cpu_id      = id;
        cpu.percent     = 0.0f;
        cpu.totalPeriod = 0;
        out.push_back(cpu);
    }

    // total 초기화
    total = CPUData{};
    total.cpu_id = -1;

    SPDLOG_INFO("Android CPU: enumerated {} cores from sysfs", out.size());
    return true;
}


struct CoreCtlSource {
    std::string path;
    std::unique_ptr<std::ifstream> stream;
};

static std::vector<CoreCtlSource> g_corectl_sources;
static bool g_corectl_probed = false;

// 1) 한 번만 전체 /sys/devices/system/cpu를 스캔해서
//    cpuN/core_ctl/global_state 들을 캐싱
static bool probe_core_ctl_sources()
{
    if (g_corectl_probed)
        return !g_corectl_sources.empty();

    g_corectl_probed = true;
    g_corectl_sources.clear();

    const char* base = "/sys/devices/system/cpu";
    DIR* dir = opendir(base);
    if (!dir) {
        SPDLOG_DEBUG("core_ctl: failed to open {}", base);
        return false;
    }

    struct dirent* ent = nullptr;
    while ((ent = readdir(dir)) != nullptr) {
        const char* name = ent->d_name;
        if (strncmp(name, "cpu", 3) != 0)
            continue;

        char* end = nullptr;
        long id = strtol(name + 3, &end, 10);
        if (!end || *end != '\0')
            continue;
        if (id < 0 || id > 1024)
            continue;

        std::string path =
            std::string(base) + "/" + name + "/core_ctl/global_state";

        if (access(path.c_str(), R_OK) == 0) {
            CoreCtlSource src;
            src.path = path;
            SPDLOG_INFO("core_ctl: discovered global_state at {}", src.path);
            g_corectl_sources.push_back(std::move(src));
        }
    }
    closedir(dir);

    if (g_corectl_sources.empty()) {
        SPDLOG_INFO("core_ctl: no global_state found under {}", base);
        return false;
    }

    SPDLOG_INFO("core_ctl: found {} global_state file(s)", g_corectl_sources.size());
    return true;
}

// 2) Busy% / Online 파싱 (여러 global_state를 전부 합쳐서 out에 넣는다)
static bool read_core_ctl_global_state(std::unordered_map<int, CoreCtlEntry>& out)
{
    if (!probe_core_ctl_sources())
        return false;

    out.clear();

    int header_count        = 0;
    int busy_field_count    = 0;
    int online_field_count  = 0;

    // 너무 시끄럽지 않게, 최초 몇 개만 상세 로그
    static int debug_fields_logged = 0;
    constexpr int MAX_DEBUG_FIELDS = 8;

    auto trim = [](std::string& s) {
        auto first = s.find_first_not_of(" \t\r\n");
        auto last  = s.find_last_not_of(" \t\r\n");
        if (first == std::string::npos || last == std::string::npos) {
            s.clear();
            return;
        }
        s = s.substr(first, last - first + 1);
    };

    for (auto& src : g_corectl_sources) {
        if (!src.stream || !src.stream->is_open()) {
            src.stream = std::make_unique<std::ifstream>(src.path);
            if (!src.stream->is_open()) {
                SPDLOG_DEBUG("core_ctl: cannot open {}", src.path);
                src.stream.reset();
                continue;
            }
        }

        std::ifstream& file = *src.stream;
        file.clear();
        file.seekg(0, std::ios::beg);

        std::string line;
        int current_cpu       = -1;  // 실제 cpuid (CPU: N 기준)
        int current_header_id = -1;  // "CPU0" 헤더 인덱스 (디버그용)

        while (std::getline(file, line)) {
            trim(line);
            if (line.empty())
                continue;

            // 헤더 라인: "CPU0", "CPU1" ...
            if (starts_with(line, "CPU") && line.find(':') == std::string::npos) {
                std::string rest = line.substr(3);
                trim(rest);
                int hid = -1;
                if (!try_stoi(hid, rest)) {
                    SPDLOG_DEBUG("core_ctl: failed to parse header '{}'", line);
                    current_header_id = -1;
                    continue;
                }
                current_header_id = hid;
                header_count++;
                continue;
            }

            auto colon = line.find(':');
            if (colon == std::string::npos)
                continue;

            std::string key = line.substr(0, colon);
            std::string val = line.substr(colon + 1);
            trim(key);
            trim(val);

            // 디버그: 실제로 뭐가 들어오는지 몇 개만 찍어본다
            if (debug_fields_logged < MAX_DEBUG_FIELDS) {
                SPDLOG_DEBUG("core_ctl: src={} header={} raw key='{}' val='{}'",
                             src.path, current_header_id, key, val);
                debug_fields_logged++;
            }

            // "CPU: N" 라인에서 실제 cpuid를 잡는다
            if (key == "CPU") {
                int id = -1;
                if (!try_stoi(id, val)) {
                    SPDLOG_DEBUG("core_ctl: failed to parse CPU id from '{}: {}'", key, val);
                    continue;
                }
                current_cpu = id;
                auto& e = out[id];
                e.cpu_id = id;
                continue;
            }

            if (current_cpu < 0)
                continue;

            auto it = out.find(current_cpu);
            if (it == out.end())
                continue;
            auto& e = it->second;

            int v = 0;

            // Online: 조금 관대하게 매칭
            if (key == "Online" || key.find("Online") != std::string::npos) {
                if (try_stoi(v, val)) {
                    e.online     = v;
                    e.has_online = true;
                    online_field_count++;
                } else {
                    SPDLOG_DEBUG("core_ctl: failed to parse Online='{}' for cpu{}", val, current_cpu);
                }
            }
            // Busy%: '%' 유무에 상관없이 Busy가 들어가면 다 받아준다
            else if (key == "Busy%" || key == "Busy" || key.find("Busy") != std::string::npos) {
                if (try_stoi(v, val)) {
                    e.busy     = v;
                    e.has_busy = true;
                    busy_field_count++;
                } else {
                    SPDLOG_DEBUG("core_ctl: failed to parse Busy='{}' for cpu{}", val, current_cpu);
                }
            }
        }
    }

    if (out.empty()) {
        SPDLOG_INFO("core_ctl: parsed zero CPU entries from {} global_state file(s)",
                    g_corectl_sources.size());
        return false;
    }

    // 요약 로그는 딱 한 번만
    if (!g_corectl_logged_summary) {
        int busy_entries   = 0;
        int online_entries = 0;

        for (const auto& kv : out) {
            const auto& e = kv.second;
            if (e.has_busy)
                busy_entries++;
            if (e.has_online && e.online > 0)
                online_entries++;
        }

        SPDLOG_INFO(
            "core_ctl: parsed {} CPU entries from {} global_state file(s)"
            " (Busy%% fields={} -> entries with Busy%%={} , Online fields={} -> entries with online>0={})",
            out.size(), g_corectl_sources.size(),
            busy_field_count, busy_entries,
            online_field_count, online_entries);

        g_corectl_logged_summary = true;
    }

    return true;
}

// ===== cpufreq policy → cpu 매핑 =====

static bool g_cpufreq_policy_probed = false;
static std::unordered_map<int, std::string> g_cpu_policy_scaling_path;

// "0-3,5,7-8" 같은 cpulist 파싱
static void parse_cpu_list(const std::string& s, std::vector<int>& out)
{
    out.clear();
    size_t pos = 0;
    const size_t n = s.size();

    while (pos < n) {
        // 구분자 스킵
        while (pos < n &&
               (s[pos] == ' ' || s[pos] == '\t' ||
                s[pos] == ',' || s[pos] == '\n' ||
                s[pos] == '\r')) {
            ++pos;
        }
        if (pos >= n)
            break;

        size_t start = pos;
        while (pos < n &&
               s[pos] != ',' && s[pos] != ' ' &&
               s[pos] != '\t' && s[pos] != '\n' &&
               s[pos] != '\r') {
            ++pos;
        }

        std::string token = s.substr(start, pos - start);
        if (token.empty())
            continue;

        size_t dash = token.find('-');
        if (dash != std::string::npos) {
            std::string a_str = token.substr(0, dash);
            std::string b_str = token.substr(dash + 1);
            int a = -1, b = -1;
            if (try_stoi(a, a_str) && try_stoi(b, b_str) &&
                a >= 0 && b >= a && b <= 1024) {
                for (int i = a; i <= b; ++i)
                    out.push_back(i);
            }
        } else {
            int id = -1;
            if (try_stoi(id, token) && id >= 0 && id <= 1024)
                out.push_back(id);
        }
    }
}

// policy*/scaling_cur_freq → 각 cpu_id 매핑
static void android_probe_cpufreq_policies()
{
    if (g_cpufreq_policy_probed)
        return;
    g_cpufreq_policy_probed = true;

    const char* base = "/sys/devices/system/cpu/cpufreq";
    DIR* dir = opendir(base);
    if (!dir) {
        SPDLOG_DEBUG("cpufreq: failed to open {}", base);
        return;
    }

    struct dirent* ent = nullptr;
    int mapped = 0;

    while ((ent = readdir(dir)) != nullptr) {
        const char* name = ent->d_name;
        if (strncmp(name, "policy", 6) != 0)
            continue;

        std::string policy_dir = std::string(base) + "/" + name;
        std::string related_path = policy_dir + "/related_cpus";
        std::ifstream rel(related_path);
        if (!rel.is_open())
            continue;

        std::string line;
        if (!std::getline(rel, line))
            continue;

        std::vector<int> cpus;
        parse_cpu_list(line, cpus);
        if (cpus.empty())
            continue;

        std::string scaling = policy_dir + "/scaling_cur_freq";
        if (access(scaling.c_str(), R_OK) != 0)
            continue;

        for (int id : cpus) {
            // 이미 매핑된 코어는 건드리지 않는다 (첫 policy 우선)
            if (g_cpu_policy_scaling_path.find(id) == g_cpu_policy_scaling_path.end()) {
                g_cpu_policy_scaling_path[id] = scaling;
                mapped++;
            }
        }
    }

    closedir(dir);

    if (mapped > 0) {
        SPDLOG_DEBUG("cpufreq: mapped {} CPUs to policy scaling_cur_freq", mapped);
    } else {
        SPDLOG_DEBUG("cpufreq: no usable policy*/scaling_cur_freq found");
    }
}

// ===== /proc/self/stat 기반 total CPU fallback =====

static long g_clk_tck = 0;
static bool g_proc_cpu_inited = false;
static double g_last_proc_cpu = 0.0;
static std::chrono::steady_clock::time_point g_last_proc_ts;

// /proc/self/stat에서 utime+stime을 초 단위로 읽어오기
static bool android_read_self_cpu_time(double &out_sec)
{
    std::ifstream file("/proc/self/stat");
    if (!file.is_open())
        return false;

    std::string line;
    if (!std::getline(file, line))
        return false;

    // comm 필드 괄호까지 건너뛰기
    auto rparen = line.rfind(')');
    if (rparen == std::string::npos || rparen + 2 >= line.size())
        return false;

    std::string rest = line.substr(rparen + 2);
    std::istringstream iss(rest);
    std::string token;
    int idx = 0;

    unsigned long long utime  = 0;
    unsigned long long stime  = 0;
    unsigned long long cutime = 0;
    unsigned long long cstime = 0;

    // state(0) ~ ... ~ utime(11) ~ stime(12) ~ cutime(13) ~ cstime(14)
    while (iss >> token) {
        if (idx == 11) {
            utime = std::strtoull(token.c_str(), nullptr, 10);
        } else if (idx == 12) {
            stime = std::strtoull(token.c_str(), nullptr, 10);
        } else if (idx == 13) {
            cutime = std::strtoull(token.c_str(), nullptr, 10);
        } else if (idx == 14) {
            cstime = std::strtoull(token.c_str(), nullptr, 10);
            break; // 여기까지만 필요
        }
        ++idx;
    }

    if (g_clk_tck <= 0) {
        g_clk_tck = sysconf(_SC_CLK_TCK);
        if (g_clk_tck <= 0)
            g_clk_tck = 100;
    }

    const unsigned long long total_ticks = utime + stime + cutime + cstime;
    out_sec = static_cast<double>(total_ticks) / static_cast<double>(g_clk_tck);
    return true;
}

// core_ctl이 없을 때: 프로세스 CPU 사용률을 "전체 CPU %" 근사치로 계산
static bool android_update_total_cpu_fallback(CPUData &total,
                                              const std::vector<CPUData> &cpus)
{
    double now_cpu = 0.0;
    if (!android_read_self_cpu_time(now_cpu))
        return false;

    auto now_ts = std::chrono::steady_clock::now();

    if (!g_proc_cpu_inited) {
        g_proc_cpu_inited = true;
        g_last_proc_cpu   = now_cpu;
        g_last_proc_ts    = now_ts;
        total.percent     = 0.0f;
        total.totalPeriod = 0;
        SPDLOG_DEBUG("Android CPU: init /proc/self/stat fallback");
        return true;
    }

    double dt   = std::chrono::duration_cast<std::chrono::duration<double>>(now_ts - g_last_proc_ts).count();
    double dcpu = now_cpu - g_last_proc_cpu;

    g_last_proc_cpu = now_cpu;
    g_last_proc_ts  = now_ts;

    if (dt <= 0.0 || dcpu < 0.0)
        return false;

    int ncores = static_cast<int>(cpus.size());
    if (ncores <= 0)
        ncores = 1;

    double capacity = dt * static_cast<double>(ncores);
    if (capacity <= 0.0)
        return false;

    double frac = dcpu / capacity;
    double pct  = frac * 100.0;
    if (pct < 0.0) pct = 0.0;
    if (pct > 100.0) pct = 100.0;

    total.percent     = static_cast<float>(pct);
    total.totalPeriod = 100;

    SPDLOG_DEBUG("Android CPU fallback: dcpu={}s dt={}s cores={} => {}%",
                 dcpu, dt, ncores, total.percent);

    return true;
}

} // namespace

#endif // __ANDROID__

static void calculateCPUData(CPUData& cpuData,
    unsigned long long int usertime,
    unsigned long long int nicetime,
    unsigned long long int systemtime,
    unsigned long long int idletime,
    unsigned long long int ioWait,
    unsigned long long int irq,
    unsigned long long int softIrq,
    unsigned long long int steal,
    unsigned long long int guest,
    unsigned long long int guestnice)
{
    // Guest time is already accounted in usertime
    usertime = usertime - guest;
    nicetime = nicetime - guestnice;
    // Fields existing on kernels >= 2.6
    // (and RHEL's patched kernel 2.4...)
    unsigned long long int idlealltime = idletime + ioWait;
    unsigned long long int systemalltime = systemtime + irq + softIrq;
    unsigned long long int virtalltime = guest + guestnice;
    unsigned long long int totaltime = usertime + nicetime + systemalltime + idlealltime + steal + virtalltime;

    // Since we do a subtraction (usertime - guest) and cputime64_to_clock_t()
    // used in /proc/stat rounds down numbers, it can lead to a case where the
    // integer overflow.
    #define WRAP_SUBTRACT(a,b) (a > b) ? a - b : 0
    cpuData.userPeriod = WRAP_SUBTRACT(usertime, cpuData.userTime);
    cpuData.nicePeriod = WRAP_SUBTRACT(nicetime, cpuData.niceTime);
    cpuData.systemPeriod = WRAP_SUBTRACT(systemtime, cpuData.systemTime);
    cpuData.systemAllPeriod = WRAP_SUBTRACT(systemalltime, cpuData.systemAllTime);
    cpuData.idleAllPeriod = WRAP_SUBTRACT(idlealltime, cpuData.idleAllTime);
    cpuData.idlePeriod = WRAP_SUBTRACT(idletime, cpuData.idleTime);
    cpuData.ioWaitPeriod = WRAP_SUBTRACT(ioWait, cpuData.ioWaitTime);
    cpuData.irqPeriod = WRAP_SUBTRACT(irq, cpuData.irqTime);
    cpuData.softIrqPeriod = WRAP_SUBTRACT(softIrq, cpuData.softIrqTime);
    cpuData.stealPeriod = WRAP_SUBTRACT(steal, cpuData.stealTime);
    cpuData.guestPeriod = WRAP_SUBTRACT(virtalltime, cpuData.guestTime);
    cpuData.totalPeriod = WRAP_SUBTRACT(totaltime, cpuData.totalTime);
    #undef WRAP_SUBTRACT
    cpuData.userTime = usertime;
    cpuData.niceTime = nicetime;
    cpuData.systemTime = systemtime;
    cpuData.systemAllTime = systemalltime;
    cpuData.idleAllTime = idlealltime;
    cpuData.idleTime = idletime;
    cpuData.ioWaitTime = ioWait;
    cpuData.irqTime = irq;
    cpuData.softIrqTime = softIrq;
    cpuData.stealTime = steal;
    cpuData.guestTime = virtalltime;
    cpuData.totalTime = totaltime;

    if (cpuData.totalPeriod == 0)
        return;
    float total = (float)cpuData.totalPeriod;
    float v[4];
    v[0] = cpuData.nicePeriod * 100.0f / total;
    v[1] = cpuData.userPeriod * 100.0f / total;

    /* if not detailed */
    v[2] = cpuData.systemAllPeriod * 100.0f / total;
    v[3] = (cpuData.stealPeriod + cpuData.guestPeriod) * 100.0f / total;
    //cpuData.percent = std::clamp(v[0]+v[1]+v[2]+v[3], 0.0f, 100.0f);
    cpuData.percent = std::min(std::max(v[0]+v[1]+v[2]+v[3], 0.0f), 100.0f);
}

CPUStats::CPUStats()
{
}

CPUStats::~CPUStats()
{
   if (m_cpuTempFile) {
        fclose(m_cpuTempFile);
        m_cpuTempFile = nullptr;
    }
}

bool CPUStats::Init()
{
    if (m_inited)
        return true;

#if defined(__ANDROID__)
    // ANDROID: /proc/stat 안 쓴다. sysfs로 코어 enum.
    if (!android_enumerate_cpus(m_cpuData, m_cpuDataTotal)) {
        SPDLOG_ERROR("Android CPU: sysfs enumeration failed, disabling CPU stats");
        return false;
}

#ifndef TEST_ONLY
    if (get_params()->enabled[OVERLAY_PARAM_ENABLED_core_type])
        get_cpu_cores_types();
#endif

    m_inited = true;
    return UpdateCPUData();

#else

    unsigned long long int usertime, nicetime, systemtime, idletime;
    unsigned long long int ioWait, irq, softIrq, steal, guest, guestnice;
    int cpuid = -1;
    size_t cpu_count = 0;

    std::string line;
    std::ifstream file (PROCSTATFILE);
    bool ret = false;

    if (!file.is_open()) {
        SPDLOG_ERROR("Failed to opening " PROCSTATFILE);
        return false;
    }

    do {
        if (!std::getline(file, line)) {
            break;
        } else if (!ret && sscanf(line.c_str(), "cpu  %16llu %16llu %16llu %16llu %16llu %16llu %16llu %16llu %16llu %16llu",
            &usertime, &nicetime, &systemtime, &idletime, &ioWait, &irq, &softIrq, &steal, &guest, &guestnice) == 10) {
            ret = true;
            calculateCPUData(m_cpuDataTotal, usertime, nicetime, systemtime, idletime, ioWait, irq, softIrq, steal, guest, guestnice);
        } else if (sscanf(line.c_str(), "cpu%4d %16llu %16llu %16llu %16llu %16llu %16llu %16llu %16llu %16llu %16llu",
            &cpuid, &usertime, &nicetime, &systemtime, &idletime, &ioWait, &irq, &softIrq, &steal, &guest, &guestnice) == 11) {

            //SPDLOG_DEBUG("Parsing 'cpu{}' line:{}", cpuid, line);

            if (!ret) {
                SPDLOG_DEBUG("Failed to parse 'cpu' line:{}", line);
                return false;
            }

            if (cpuid < 0) {
                SPDLOG_DEBUG("Cpu id '{}' is out of bounds", cpuid);
                return false;
            }

            if (cpu_count + 1 > m_cpuData.size() || m_cpuData[cpu_count].cpu_id != cpuid) {
                SPDLOG_DEBUG("Cpu id '{}' is out of bounds or wrong index, reiniting", cpuid);
                return Reinit();
            }

            CPUData& cpuData = m_cpuData[cpu_count];
            calculateCPUData(cpuData, usertime, nicetime, systemtime, idletime, ioWait, irq, softIrq, steal, guest, guestnice);
            cpuid = -1;
            cpu_count++;

        } else {
            break;
        }
    } while(true);

    if (cpu_count < m_cpuData.size())
        m_cpuData.resize(cpu_count);

    m_cpuPeriod = (double)m_cpuData[0].totalPeriod / m_cpuData.size();
    m_updatedCPUs = true;
    return ret;
#endif
}

bool CPUStats::UpdateCoreMhz() {
    m_coreMhz.clear();
    FILE *fp;

#if defined(__ANDROID__)
    constexpr int ANDROID_MHZ_MIN_UPDATE_MS = 500;
    static auto last_ts = std::chrono::steady_clock::time_point{};
    auto now = std::chrono::steady_clock::now();

    if (last_ts.time_since_epoch().count() != 0) {
        auto delta_ms =
            std::chrono::duration_cast<std::chrono::milliseconds>(now - last_ts).count();
        if (delta_ms < ANDROID_MHZ_MIN_UPDATE_MS) {
            // 기존 값만 유지하고 빠르게 반환
            for (const auto &cpu : m_cpuData)
                m_coreMhz.push_back(cpu.mhz);
            return true;
        }
    }
    last_ts = now;

    // policy*/scaling_cur_freq 매핑 1회 초기화
    android_probe_cpufreq_policies();

    bool used_policy   = false;
    bool used_cpufreq  = false;

    for (auto& cpu : m_cpuData) {
        int mhz = 0;
        std::string path;

        // 1) policy 기반 scaling_cur_freq 우선
        auto pit = g_cpu_policy_scaling_path.find(cpu.cpu_id);
        if (pit != g_cpu_policy_scaling_path.end()) {
            path = pit->second;
            if ((fp = fopen(path.c_str(), "r"))) {
                int64_t temp = 0;
                if (fscanf(fp, "%" PRId64, &temp) != 1)
                    temp = 0;
                fclose(fp);
                mhz = static_cast<int>(temp / 1000);
                if (mhz > 0)
                    used_policy = true;
            }
        }

        // 2) policy에서 못 읽었으면 per-cpu 경로 폴백
        if (mhz == 0) {
            path = "/sys/devices/system/cpu/cpu" +
                   std::to_string(cpu.cpu_id) +
                   "/cpufreq/scaling_cur_freq";

            if ((fp = fopen(path.c_str(), "r"))) {
                int64_t temp = 0;
                if (fscanf(fp, "%" PRId64, &temp) != 1)
                    temp = 0;
                fclose(fp);
                mhz = static_cast<int>(temp / 1000);
                if (mhz > 0)
                    used_cpufreq = true;
            }
        }

        cpu.mhz = mhz;
        m_coreMhz.push_back(mhz);
    }

    // ===== 한 방씩만 찍는 INFO 로그 =====
    if (used_policy && !g_logged_policy_mhz) {
        SPDLOG_INFO("Android CPU MHz: using cpufreq policy*/scaling_cur_freq as primary source");
        g_logged_policy_mhz = true;
    }

    if (used_cpufreq && !g_logged_cpufreq_mhz) {
        SPDLOG_INFO("Android CPU MHz: using cpu*/cpufreq/scaling_cur_freq as fallback source");
        g_logged_cpufreq_mhz = true;
    }

    if (!used_policy && !used_cpufreq && !g_logged_mhz_missing) {
        SPDLOG_INFO("Android CPU MHz: no readable cpufreq scaling_cur_freq, core MHz will remain 0");
        g_logged_mhz_missing = true;
    }

#else
    static bool scaling_freq = true;

    if (scaling_freq) {
        for (auto& cpu : m_cpuData) {
            std::string path = "/sys/devices/system/cpu/cpu" +
                               std::to_string(cpu.cpu_id) +
                               "/cpufreq/scaling_cur_freq";

            int mhz = 0;
            if ((fp = fopen(path.c_str(), "r"))) {
                int64_t temp = 0;
                if (fscanf(fp, "%" PRId64, &temp) != 1)
                    temp = 0;
                fclose(fp);
                mhz = static_cast<int>(temp / 1000);
            } else {
                scaling_freq = false;
                break;
            }

            cpu.mhz = mhz;
            m_coreMhz.push_back(mhz);
        }
    }

    if (!scaling_freq) {
        std::ifstream cpuInfo(PROCCPUINFOFILE);
        std::string row;
        size_t i = 0;

        while (std::getline(cpuInfo, row) && i < m_cpuData.size()) {
            if (row.find("MHz") == std::string::npos)
                continue;

            row = std::regex_replace(row, std::regex(R"([^0-9.])"), "");
            if (!try_stoi(m_cpuData[i].mhz, row))
                m_cpuData[i].mhz = 0;

            m_coreMhz.push_back(m_cpuData[i].mhz);
            i++;
        }
    }
#endif

    m_cpuDataTotal.cpu_mhz = 0;
    for (const auto& data : m_cpuData)
        if (data.mhz > m_cpuDataTotal.cpu_mhz)
            m_cpuDataTotal.cpu_mhz = data.mhz;

    return true;
}

bool CPUStats::ReadcpuTempFile(int& temp) {
	if (!m_cpuTempFile)
		return false;

	rewind(m_cpuTempFile);
	fflush(m_cpuTempFile);
	bool ret = (fscanf(m_cpuTempFile, "%d", &temp) == 1);
	temp = temp / 1000;

	return ret;
}

bool CPUStats::UpdateCpuTemp() {
#if defined(__ANDROID__)
    constexpr int ANDROID_CPU_TEMP_MIN_UPDATE_MS = 500;

    static auto  last_ts  = std::chrono::steady_clock::time_point{};
    static bool  last_ret = false;

    auto now = std::chrono::steady_clock::now();

    if (last_ts.time_since_epoch().count() != 0) {
        auto delta_ms =
            std::chrono::duration_cast<std::chrono::milliseconds>(now - last_ts).count();

        if (delta_ms < ANDROID_CPU_TEMP_MIN_UPDATE_MS) {
            return last_ret;
        }
    }
    last_ts = now;
#endif

    if (gpus) {
        for (auto gpu : gpus->available_gpus) {
            if (gpu->is_apu()) {
                m_cpuDataTotal.temp = gpu->metrics.apu_cpu_temp;
#if defined(__ANDROID__)
                last_ret = true;
#endif
                return true;
            }
        }
    }

    int temp = 0;
    bool ret = ReadcpuTempFile(temp);
    m_cpuDataTotal.temp = temp;

#if defined(__ANDROID__)
    last_ret = ret;
#endif

    return ret;
}

static bool get_cpu_power_k10temp(CPUPowerData* cpuPowerData, float& power) {
    CPUPowerData_k10temp* powerData_k10temp = (CPUPowerData_k10temp*)cpuPowerData;

    if(powerData_k10temp->corePowerFile || powerData_k10temp->socPowerFile)
    {
        rewind(powerData_k10temp->corePowerFile);
        rewind(powerData_k10temp->socPowerFile);
        fflush(powerData_k10temp->corePowerFile);
        fflush(powerData_k10temp->socPowerFile);
        int corePower, socPower;
        if (fscanf(powerData_k10temp->corePowerFile, "%d", &corePower) != 1)
            goto voltagebased;
        if (fscanf(powerData_k10temp->socPowerFile, "%d", &socPower) != 1)
            goto voltagebased;
        power = (corePower + socPower) / 1000000;
        return true;
    }
    voltagebased:
    if (!powerData_k10temp->coreVoltageFile || !powerData_k10temp->coreCurrentFile || !powerData_k10temp->socVoltageFile || !powerData_k10temp->socCurrentFile)
        return false;
    rewind(powerData_k10temp->coreVoltageFile);
    rewind(powerData_k10temp->coreCurrentFile);
    rewind(powerData_k10temp->socVoltageFile);
    rewind(powerData_k10temp->socCurrentFile);

    fflush(powerData_k10temp->coreVoltageFile);
    fflush(powerData_k10temp->coreCurrentFile);
    fflush(powerData_k10temp->socVoltageFile);
    fflush(powerData_k10temp->socCurrentFile);

    int coreVoltage, coreCurrent;
    int socVoltage, socCurrent;

    if (fscanf(powerData_k10temp->coreVoltageFile, "%d", &coreVoltage) != 1)
        return false;
    if (fscanf(powerData_k10temp->coreCurrentFile, "%d", &coreCurrent) != 1)
        return false;
    if (fscanf(powerData_k10temp->socVoltageFile, "%d", &socVoltage) != 1)
        return false;
    if (fscanf(powerData_k10temp->socCurrentFile, "%d", &socCurrent) != 1)
        return false;

    power = (coreVoltage * coreCurrent + socVoltage * socCurrent) / 1000000;

    return true;
}

static bool get_cpu_power_zenpower(CPUPowerData* cpuPowerData, float& power) {
    CPUPowerData_zenpower* powerData_zenpower = (CPUPowerData_zenpower*)cpuPowerData;

    if (!powerData_zenpower->corePowerFile || !powerData_zenpower->socPowerFile)
        return false;

    rewind(powerData_zenpower->corePowerFile);
    rewind(powerData_zenpower->socPowerFile);

    fflush(powerData_zenpower->corePowerFile);
    fflush(powerData_zenpower->socPowerFile);

    int corePower, socPower;

    if (fscanf(powerData_zenpower->corePowerFile, "%d", &corePower) != 1)
        return false;
    if (fscanf(powerData_zenpower->socPowerFile, "%d", &socPower) != 1)
        return false;

    power = (corePower + socPower) / 1000000;

    return true;
}

static bool get_cpu_power_zenergy(CPUPowerData* cpuPowerData, float& power) {
    CPUPowerData_zenergy* powerData_zenergy = (CPUPowerData_zenergy*)cpuPowerData;
    if (!powerData_zenergy->energyCounterFile)
        return false;

    rewind(powerData_zenergy->energyCounterFile);
    fflush(powerData_zenergy->energyCounterFile);

    uint64_t energyCounterValue = 0;
    if (fscanf(powerData_zenergy->energyCounterFile, "%" SCNu64, &energyCounterValue) != 1)
        return false;

    Clock::time_point now = Clock::now();
    Clock::duration timeDiff = now - powerData_zenergy->lastCounterValueTime;
    int64_t timeDiffMicro = std::chrono::duration_cast<std::chrono::microseconds>(timeDiff).count();
    uint64_t energyCounterDiff = energyCounterValue - powerData_zenergy->lastCounterValue;


    if (powerData_zenergy->lastCounterValue > 0 && energyCounterValue > powerData_zenergy->lastCounterValue)
        power = (float) energyCounterDiff / (float) timeDiffMicro;

    powerData_zenergy->lastCounterValue = energyCounterValue;
    powerData_zenergy->lastCounterValueTime = now;

    return true;
}

static bool get_cpu_power_rapl(CPUPowerData* cpuPowerData, float& power) {
    CPUPowerData_rapl* powerData_rapl = (CPUPowerData_rapl*)cpuPowerData;

    if (!powerData_rapl->energyCounterFile)
        return false;

    rewind(powerData_rapl->energyCounterFile);
    fflush(powerData_rapl->energyCounterFile);

    uint64_t energyCounterValue = 0;
    if (fscanf(powerData_rapl->energyCounterFile, "%" SCNu64, &energyCounterValue) != 1)
        return false;

    Clock::time_point now = Clock::now();
    Clock::duration timeDiff = now - powerData_rapl->lastCounterValueTime;
    int64_t timeDiffMicro = std::chrono::duration_cast<std::chrono::microseconds>(timeDiff).count();
    uint64_t energyCounterDiff = energyCounterValue - powerData_rapl->lastCounterValue;

    if (powerData_rapl->lastCounterValue > 0 && energyCounterValue > powerData_rapl->lastCounterValue)
        power = energyCounterDiff / timeDiffMicro;

    powerData_rapl->lastCounterValue = energyCounterValue;
    powerData_rapl->lastCounterValueTime = now;

    return true;
}

static bool get_cpu_power_amdgpu(float& power) {
    if (gpus)
        for (auto gpu : gpus->available_gpus)
            if (gpu->is_apu()) {
                power = gpu->metrics.apu_cpu_power;
                return true;
            }

    return false;
}

static bool get_cpu_power_xgene(CPUPowerData* cpuPowerData, float& power) {
    CPUPowerData_xgene* powerData_xgene = (CPUPowerData_xgene*)cpuPowerData;
    if (!powerData_xgene->powerFile)
        return false;

    rewind(powerData_xgene->powerFile);
    fflush(powerData_xgene->powerFile);

    uint64_t powerValue = 0;
    if (fscanf(powerData_xgene->powerFile, "%" SCNu64, &powerValue) != 1)
        return false;

    power = (float) powerValue / 1000000.0f;

    return true;
}

bool CPUStats::UpdateCpuPower() {
#if defined(__ANDROID__)
    m_cpuDataTotal.power = 0.0f;
    return false;
#else
    InitCpuPowerData();

    if(!m_cpuPowerData)
        return false;

    float power = 0;

    switch(m_cpuPowerData->source) {
        case CPU_POWER_K10TEMP:
            if (!get_cpu_power_k10temp(m_cpuPowerData.get(), power)) return false;
            break;
        case CPU_POWER_ZENPOWER:
            if (!get_cpu_power_zenpower(m_cpuPowerData.get(), power)) return false;
            break;
        case CPU_POWER_ZENERGY:
            if (!get_cpu_power_zenergy(m_cpuPowerData.get(), power)) return false;
            break;
        case CPU_POWER_RAPL:
            if (!get_cpu_power_rapl(m_cpuPowerData.get(), power)) return false;
            break;
        case CPU_POWER_AMDGPU:
            if (!get_cpu_power_amdgpu(power)) return false;
            break;
        case CPU_POWER_XGENE:
            if (!get_cpu_power_xgene(m_cpuPowerData.get(), power)) return false;
            break;
        default:
            return false;
    }

    m_cpuDataTotal.power = power;

    return true;
#endif
}

static bool find_input(const std::string& path, const char* input_prefix, std::string& input, const std::string& name)
{
    auto files = ls(path.c_str(), input_prefix, LS_FILES);
    for (auto& file : files) {
        if (!ends_with(file, "_label"))
            continue;

        auto label = read_line(path + "/" + file);
        if (label != name)
            continue;

        auto uscore = file.find_first_of("_");
        if (uscore != std::string::npos) {
            file.erase(uscore, std::string::npos);
            input = path + "/" + file + "_input";
            //9 characters should not overflow the 32-bit int
            return std::stoi(read_line(input).substr(0, 9)) > 0;
        }
    }
    return false;
}

static bool find_fallback_input(const std::string& path, const char* input_prefix, std::string& input)
{
    auto files = ls(path.c_str(), input_prefix, LS_FILES);
    if (!files.size())
        return false;

    std::sort(files.begin(), files.end());
    for (auto& file : files) {
        if (!ends_with(file, "_input"))
            continue;
        input = path + "/" + file;
		SPDLOG_DEBUG("fallback cpu {} input: {}", input_prefix, input);
        return true;
    }
    return false;
}

// thermal_zone type이 CPU 계열인지 대충 판정
static bool is_cpu_thermal_type(const std::string& type_raw)
{
    if (type_raw.empty())
        return false;

    std::string type = type_raw;
    std::transform(type.begin(), type.end(), type.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });

    // 원래 쓰던 cpuss- 계열은 그대로 인정
    if (starts_with(type, "cpuss-"))
        return true;

    // 일반적인 "cpu", "cpu-thermal", "cpu-therm" 같은 놈들
    if (starts_with(type, "cpu"))
        return true;
    if (type.find("cpu-thermal") != std::string::npos)
        return true;
    if (type.find("cpu-therm") != std::string::npos)
        return true;

    return false;
}

static void check_thermal_zones(std::string& path, std::string& input)
{
    // 레퍼런스: /sys/devices/virtual/thermal → 안 되면 /sys/class/thermal
    const char* bases[] = {
        "/sys/devices/virtual/thermal/",
        "/sys/class/thermal/"
    };

    for (const char* base : bases) {
        if (!ghc::filesystem::exists(base))
            continue;

        for (auto& d : ghc::filesystem::directory_iterator(base)) {
            auto fname = d.path().filename().string();
            if (fname.rfind("thermal_zone", 0) != 0)
                continue;

            std::string type = read_line(d.path() / "type");
            if (!is_cpu_thermal_type(type))
                continue;

            path = d.path().string();

            // 1순위: .../temp
            std::string cand = (d.path() / "temp").string();
            if (file_exists(cand)) {
                input = cand;
                return;
            }

            // 2순위: .../temp1_input
            cand = (d.path() / "temp1_input").string();
            if (file_exists(cand)) {
                input = cand;
                return;
            }

            // 3순위: .../freq1_input (중국 빌드에서 쓰던 애 커버용)
            cand = (d.path() / "freq1_input").string();
            if (file_exists(cand)) {
                input = cand;
                return;
            }
        }

        // 이 base에서 못 찾았으면 다음 base(/sys/class/thermal)로 넘어감
    }
}

bool CPUStats::UpdateCPUData()
{
    if (!m_inited)
        return false;

#if defined(__ANDROID__)
    constexpr int ANDROID_CPU_MIN_UPDATE_MS = 500;

    static auto last_ts = std::chrono::steady_clock::time_point{};
    auto now = std::chrono::steady_clock::now();

    if (last_ts.time_since_epoch().count() != 0) {
        auto delta_ms =
            std::chrono::duration_cast<std::chrono::milliseconds>(now - last_ts).count();

        if (delta_ms < ANDROID_CPU_MIN_UPDATE_MS) {
            return m_updatedCPUs;
        }
    }
    last_ts = now;

    std::unordered_map<int, CoreCtlEntry> corectl;
    bool have_corectl = read_core_ctl_global_state(corectl);

    float total_percent = 0.0f;
    int   count_online  = 0;

    if (have_corectl) {
        for (auto &cpu : m_cpuData) {
            auto it = corectl.find(cpu.cpu_id);
            if (it == corectl.end()) {
                cpu.percent       = 0.0f;
                cpu.totalPeriod   = 0;
                cpu.userPeriod    = 0;
                cpu.idleAllPeriod = 0;
                continue;
            }

            const auto &e = it->second;
            bool online = !e.has_online || e.online != 0;

            if (!online || !e.has_busy) {
                cpu.percent       = 0.0f;
                cpu.totalPeriod   = 0;
                cpu.userPeriod    = 0;
                cpu.idleAllPeriod = 0;
                continue;
            }

            float p = std::clamp(static_cast<float>(e.busy), 0.0f, 100.0f);

            cpu.percent       = p;
            cpu.totalPeriod   = 100;
            cpu.userPeriod    = static_cast<unsigned long long>(p);
            cpu.idleAllPeriod = 100 - cpu.userPeriod;

            total_percent += p;
            count_online++;
        }
    }

    if (have_corectl && count_online > 0) {
        m_cpuDataTotal.percent     = total_percent / static_cast<float>(count_online);
        m_cpuDataTotal.totalPeriod = 100;
        m_cpuPeriod   = 1.0;
        m_updatedCPUs = true;

        if (!g_logged_corectl_summary) {
            SPDLOG_INFO(
                "Android CPU: using core_ctl Busy%% path (entries={}, online+Busy%% cores={} / total cores={})",
                corectl.size(), count_online, m_cpuData.size());
            g_logged_corectl_summary = true;
        }

        return true;
    }

    // 여기서부터는 hack fallback:
    // per-core hack(core_ctl)이 터졌으므로 코어별 %는 전부 죽이고,
    // total CPU만 /proc/self/stat 기반으로 근사한다.
    for (auto &cpu : m_cpuData) {
        cpu.percent       = 0.0f;
        cpu.totalPeriod   = 0;
        cpu.userPeriod    = 0;
        cpu.idleAllPeriod = 0;
    }

    if (android_update_total_cpu_fallback(m_cpuDataTotal, m_cpuData)) {
        m_cpuPeriod   = 1.0;
        m_updatedCPUs = true;

        if (!g_logged_fallback_switch) {
            SPDLOG_INFO("Android CPU: core_ctl unusable, using /proc/self/stat fallback (total-only)");
            g_logged_fallback_switch = true;
        }
        return true;
    }

    // 이건 아예 DEBUG로 내려서 조용히
    SPDLOG_DEBUG("Android CPU: no core_ctl and fallback failed, CPU stats disabled");
    m_cpuDataTotal.percent     = 0.0f;
    m_cpuDataTotal.totalPeriod = 0;
    m_cpuPeriod   = 0.0;
    m_updatedCPUs = false;
    return false;

#else
    // ===== 이하 기존 리눅스 /proc/stat 코드 =====
    unsigned long long int usertime, nicetime, systemtime, idletime;
    unsigned long long int ioWait, irq, softIrq, steal, guest, guestnice;
    int cpuid = -1;
    size_t cpu_count = 0;

    std::string line;
    std::ifstream file (PROCSTATFILE);
    bool ret = false;

    if (!file.is_open()) {
        SPDLOG_ERROR("Failed to opening " PROCSTATFILE);
        return false;
    }

    do {
        if (!std::getline(file, line)) {
            break;
        } else if (!ret && sscanf(line.c_str(), "cpu  %16llu %16llu %16llu %16llu %16llu %16llu %16llu %16llu %16llu %16llu",
            &usertime, &nicetime, &systemtime, &idletime, &ioWait, &irq, &softIrq, &steal, &guest, &guestnice) == 10) {
            ret = true;
            calculateCPUData(m_cpuDataTotal, usertime, nicetime, systemtime, idletime, ioWait, irq, softIrq, steal, guest, guestnice);
        } else if (sscanf(line.c_str(), "cpu%4d %16llu %16llu %16llu %16llu %16llu %16llu %16llu %16llu %16llu %16llu",
            &cpuid, &usertime, &nicetime, &systemtime, &idletime, &ioWait, &irq, &softIrq, &steal, &guest, &guestnice) == 11) {

            if (!ret) {
                SPDLOG_DEBUG("Failed to parse 'cpu' line:{}", line);
                return false;
            }

            if (cpuid < 0) {
                SPDLOG_DEBUG("Cpu id '{}' is out of bounds", cpuid);
                return false;
            }

            if (cpu_count + 1 > m_cpuData.size() || m_cpuData[cpu_count].cpu_id != cpuid) {
                SPDLOG_DEBUG("Cpu id '{}' is out of bounds or wrong index, reiniting", cpuid);
                return Reinit();
            }

            CPUData& cpuData = m_cpuData[cpu_count];
            calculateCPUData(cpuData, usertime, nicetime, systemtime, idletime, ioWait, irq, softIrq, steal, guest, guestnice);
            cpuid = -1;
            cpu_count++;

        } else {
            break;
        }
    } while(true);

    if (cpu_count < m_cpuData.size())
        m_cpuData.resize(cpu_count);

    m_cpuPeriod   = static_cast<double>(m_cpuData[0].totalPeriod) / m_cpuData.size();
    m_updatedCPUs = true;
    return ret;
#endif
}

bool CPUStats::GetCpuFile() {
    if (m_cpuTempFile)
        return true;

    std::string name, path, input;
    std::string hwmon = "/sys/class/hwmon/";
    std::smatch match;

    auto dirs = ls(hwmon.c_str());
    for (auto& dir : dirs) {
        path = hwmon + dir;
        name = read_line(path + "/name");
        SPDLOG_DEBUG("hwmon: sensor name: {}", name);

        if (name == "coretemp") {
            find_input(path, "temp", input, "Package id 0");
            break;
        }
        else if ((name == "zenpower" || name == "k10temp")) {
            if (!find_input(path, "temp", input, "Tdie"))
                find_input(path, "temp", input, "Tctl");
            break;
        } else if (name == "atk0110") {
            find_input(path, "temp", input, "CPU Temperature");
            break;
        } else if (name == "it8603") {
            find_input(path, "temp", input, "temp1");
            break;
        } else if (starts_with(name, "cpuss0_")) {
            find_fallback_input(path, "temp1", input);
            break;
        } else if (starts_with(name, "nct")) {
            // Only break if nct module has TSI0_TEMP node
            if (find_input(path, "temp", input, "TSI0_TEMP"))
                break;

        } else if (name == "asusec") {
            // Only break if module has CPU node
            if (find_input(path, "temp", input, "CPU"))
                break;
        } else if (name == "l_pcs") {
            // E2K (Elbrus 2000) CPU temperature module
            find_input(path, "temp", input, "Node 0 Max");
            break;
        } else if (std::regex_match(name, match, std::regex("cpu\\d*_thermal"))) {
            find_fallback_input(path, "temp1", input);
            break;
        } else if (name == "apm_xgene") {
            find_input(path, "temp", input, "SoC Temperature");
            break;
        } else {
            path.clear();
        }
    }

    if (path.empty()) {
        try {
            check_thermal_zones(path, input);
        } catch (ghc::filesystem::filesystem_error& ex) {
            SPDLOG_DEBUG("check_thermal_zones: {}", ex.what());
        }
    }

    if (path.empty() || (!file_exists(input) && !find_fallback_input(path, "temp", input))) {
        SPDLOG_ERROR("Could not find cpu temp sensor location");
        return false;
    } else {
        SPDLOG_DEBUG("hwmon: using input: {}", input);
        m_cpuTempFile = fopen(input.c_str(), "r");
    }
    return true;
}

static CPUPowerData_k10temp* init_cpu_power_data_k10temp(const std::string path) {
    auto powerData = std::make_unique<CPUPowerData_k10temp>();

    std::string coreVoltageInput, coreCurrentInput;
    std::string socVoltageInput, socCurrentInput;
    std::string socPowerInput, corePowerInput;

    if(find_input(path, "power", corePowerInput, "Pcore") && find_input(path, "power", socPowerInput, "Psoc")) {
        powerData->corePowerFile = fopen(corePowerInput.c_str(), "r");
        powerData->socPowerFile = fopen(socPowerInput.c_str(), "r");
        SPDLOG_DEBUG("hwmon: using input: {}", corePowerInput);
        SPDLOG_DEBUG("hwmon: using input: {}", socPowerInput);
        return powerData.release();
    }

    if(!find_input(path, "in", coreVoltageInput, "Vcore")) return nullptr;
    if(!find_input(path, "curr", coreCurrentInput, "Icore")) return nullptr;
    if(!find_input(path, "in", socVoltageInput, "Vsoc")) return nullptr;
    if(!find_input(path, "curr", socCurrentInput, "Isoc")) return nullptr;

    SPDLOG_DEBUG("hwmon: using input: {}", coreVoltageInput);
    SPDLOG_DEBUG("hwmon: using input: {}", coreCurrentInput);
    SPDLOG_DEBUG("hwmon: using input: {}", socVoltageInput);
    SPDLOG_DEBUG("hwmon: using input: {}", socCurrentInput);

    powerData->coreVoltageFile = fopen(coreVoltageInput.c_str(), "r");
    powerData->coreCurrentFile = fopen(coreCurrentInput.c_str(), "r");
    powerData->socVoltageFile = fopen(socVoltageInput.c_str(), "r");
    powerData->socCurrentFile = fopen(socCurrentInput.c_str(), "r");

    return powerData.release();
}

static CPUPowerData_zenpower* init_cpu_power_data_zenpower(const std::string path) {
    auto powerData = std::make_unique<CPUPowerData_zenpower>();

    std::string corePowerInput, socPowerInput;

    if(!find_input(path, "power", corePowerInput, "SVI2_P_Core")) return nullptr;
    if(!find_input(path, "power", socPowerInput, "SVI2_P_SoC")) return nullptr;

    SPDLOG_DEBUG("hwmon: using input: {}", corePowerInput);
    SPDLOG_DEBUG("hwmon: using input: {}", socPowerInput);

    powerData->corePowerFile = fopen(corePowerInput.c_str(), "r");
    powerData->socPowerFile = fopen(socPowerInput.c_str(), "r");

    return powerData.release();
}

static CPUPowerData_zenergy* init_cpu_power_data_zenergy(const std::string path) {
    auto powerData = std::make_unique<CPUPowerData_zenergy>();
    std::string energyCounterPath;

    if(!find_input(path, "energy", energyCounterPath, "Esocket0")) return nullptr;

    SPDLOG_DEBUG("hwmon: using input: {}", energyCounterPath);
    powerData->energyCounterFile = fopen(energyCounterPath.c_str(), "r");

    return powerData.release();
}

static CPUPowerData_rapl* init_cpu_power_data_rapl(const std::string path) {
    auto powerData = std::make_unique<CPUPowerData_rapl>();

    std::string energyCounterPath = path + "/energy_uj";
    if (!file_exists(energyCounterPath)) return nullptr;

    powerData->energyCounterFile = fopen(energyCounterPath.c_str(), "r");
    if (!powerData->energyCounterFile) {
        SPDLOG_DEBUG("Rapl: energy_uj is not accessible");
        powerData->energyCounterFile = nullptr;
        return nullptr;
    }

    return powerData.release();
}

static CPUPowerData_xgene* init_cpu_power_data_xgene(const std::string path) {
    auto powerData = std::make_unique<CPUPowerData_xgene>();
    std::string powerPath;

    if(!find_input(path, "power", powerPath, "CPU power")) return nullptr;

    SPDLOG_DEBUG("hwmon: using input: {}", powerPath);
    powerData->powerFile = fopen(powerPath.c_str(), "r");

    return powerData.release();
}

bool CPUStats::InitCpuPowerData() {
    if(m_cpuPowerData != nullptr)
        return true;

    // only try to find a valid method 5 times
    static int retries = 0;
    if (retries >= 5)
        return true;

    retries++;
    
    std::string name, path;
    std::string hwmon = "/sys/class/hwmon/";

    CPUPowerData* cpuPowerData = nullptr;

    auto dirs = ls(hwmon.c_str());
    for (auto& dir : dirs) {
        path = hwmon + dir;
        name = read_line(path + "/name");
        SPDLOG_DEBUG("hwmon: sensor name: {}", name);

        if (name == "k10temp") {
            cpuPowerData = (CPUPowerData*)init_cpu_power_data_k10temp(path);
        } else if (name == "zenpower") {
            cpuPowerData = (CPUPowerData*)init_cpu_power_data_zenpower(path);
            break;
        } else if (name == "zenergy") {
            cpuPowerData = (CPUPowerData*)init_cpu_power_data_zenergy(path);
            break;
        } else if (name == "apm_xgene") {
            cpuPowerData = (CPUPowerData*)init_cpu_power_data_xgene(path);
            break;
        }
    }

    if (!cpuPowerData) {
        if (gpus) {
            for (auto gpu : gpus->available_gpus) {
                if (gpu->vendor_id == 0x1002 && gpu->is_apu() && gpu->get_metrics().apu_cpu_power > 0) {
                    auto powerData = std::make_unique<CPUPowerData_amdgpu>();
                    cpuPowerData = (CPUPowerData*)powerData.release();
                }
            }
        }
    }

    if (!cpuPowerData) {
        std::string powercap = "/sys/class/powercap/";
        auto powercap_dirs = ls(powercap.c_str());
        for (auto& dir : powercap_dirs) {
            path = powercap + dir;
            name = read_line(path + "/name");
            SPDLOG_DEBUG("powercap: name: {}", name);
            if (name == "package-0") {
                cpuPowerData = (CPUPowerData*)init_cpu_power_data_rapl(path);
                break;
            }
        }
    }
    
    if(cpuPowerData == nullptr) {
        SPDLOG_ERROR("Failed to initialize CPU power data");
        return false;
    }

    m_cpuPowerData.reset(cpuPowerData);
    return true;
}

void CPUStats::get_cpu_cores_types() {
#if defined(__x86_64__) || defined(__i386__)
    std::ifstream cpuinfo(PROCCPUINFOFILE);

    if (!cpuinfo.is_open()) {
        SPDLOG_ERROR("failed to open {}", PROCCPUINFOFILE);
        return;
    }

    std::string vendor = "unknown";
    for (std::string line; std::getline(cpuinfo, line);) {
        if (line.empty() || line.find(":") + 1 == line.length())
            continue;

        std::string key = line.substr(0, line.find(":") - 1);
        std::string val = line.substr(key.length() + 3);

        if (key == "vendor_id") {
            vendor = val;
            break;
        }
    }

    SPDLOG_INFO("cpu vendor: {}", vendor);

    if (vendor == "GenuineIntel")
        get_cpu_cores_types_intel();
#endif

#if defined(__arm__) || defined(__aarch64__)
    get_cpu_cores_types_arm();
#endif
}

void CPUStats::get_cpu_cores_types_intel() {
    for (auto const& it : intel_cores) {
        auto key = it.first;
        auto file = it.second;

        std::ifstream core_file(file);

        if (!core_file.is_open()) {
            SPDLOG_ERROR("failed to open core info file");
            return;
        }

        std::string cpus;
        std::getline(core_file, cpus);

        std::regex rx("(\\d+)-(\\d+)");
        std::smatch matches;

        if (!std::regex_match(cpus, matches, rx) || matches.size() != 3)
            continue;

        int start = 0, end = 0;

        try {
            start = std::stoi(matches[1]);
            end = std::stoi(matches[2]) + 1;
        } catch (...) {
            SPDLOG_ERROR("error parsing cpus \"{}\"", cpus);
        }

        for (int i = start; i < end; i++) {
            for (size_t k = 0; k < m_cpuData.size(); k++) {
                if (m_cpuData[k].cpu_id != i)
                    continue;

                m_cpuData[k].label = key;
                break;
            }
        }
    }
}

void CPUStats::get_cpu_cores_types_arm() {
    std::ifstream cpuinfo(PROCCPUINFOFILE);

    if (!cpuinfo.is_open()) {
        SPDLOG_ERROR("failed to open {}", PROCCPUINFOFILE);
        return;
    }

    uint8_t cur_core = 0;
    bool detected_first_core = false;

    for (std::string line; std::getline(cpuinfo, line);) {
        if (line.empty() || line.find(":") + 1 == line.length())
            continue;

        auto key = line.substr(0, line.find(":") - 1);
        auto val = line.substr(key.length() + 3);

        if (key != "CPU part")
            continue;

        if (detected_first_core)
            cur_core += 1;
        else
            detected_first_core = true;

        std::string core_type;

        try {
            core_type = arm_cores.at(val);
            SPDLOG_INFO("found {} core", core_type);
        }
        catch(const std::out_of_range& ex) {
            SPDLOG_WARN("unknown cpu part {}", val);
            continue;
        }

        // just in case
        for (size_t i = 0; i < m_cpuData.size(); i++) {
            if (m_cpuData[i].cpu_id != cur_core)
                continue;

            m_cpuData[i].label = core_type;
        }
    }
}

CPUStats cpuStats;

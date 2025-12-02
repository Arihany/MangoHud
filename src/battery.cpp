#include <spdlog/spdlog.h>
#include <filesystem.h>
#include "battery.h"
#include <cstdio>
#include <cstring>
#include <cmath>

namespace fs = ghc::filesystem;
using namespace std;

static inline float readFloatFast(const char* path, float divisor, bool* ok) {
    if (ok)
        *ok = false;

    FILE* f = fopen(path, "r");
    if (!f)
        return 0.0f;

    float value = 0.0f;
    if (fscanf(f, "%f", &value) == 1) {
        if (ok)
            *ok = true;
    }

    fclose(f);
    return value / divisor;
}

static inline float readFloatFast(const char* path, float divisor = 1.0f) {
    return readFloatFast(path, divisor, nullptr);
}

static inline char readStatusChar(const char* path) {
    FILE* f = fopen(path, "r");
    if (!f)
        return 0;

    char c = 0;
    if (fscanf(f, " %c", &c) != 1)
        c = 0;

    fclose(f);
    return c;
}

// 경로 합성용 템플릿 (매크로 대체)
template <size_t N>
static inline void make_path(char (&buf)[N], const std::string& base, const char* file) {
    snprintf(buf, N, "%s/%s", base.c_str(), file);
}

void BatteryStats::numBattery() {
    batt_count = 0;
    batt_check = true;

    const fs::path path("/sys/class/power_supply/");

#if defined(__ANDROID__)
    static bool android_batt_info_logged = false;
#endif

    try {
        if (!fs::exists(path)) {
#if defined(__ANDROID__)
            if (!android_batt_info_logged) {
                SPDLOG_INFO("Battery: /sys/class/power_supply not accessible");
                android_batt_info_logged = true;
            }
#endif
            return;
        }

        int batteryCount = 0;

        for (const auto& p : fs::directory_iterator(path)) {
            const std::string fileName = p.path().filename().string();

#if defined(__ANDROID__)
            bool is_battery = false;

            if (fileName.find("battery") != std::string::npos ||
                fileName.find("bat")     != std::string::npos ||
                fileName.find("BAT")     != std::string::npos) {
                is_battery = true;
            }

            if (!is_battery)
                continue;
#else
            if (fileName.find("BAT") == std::string::npos)
                continue;
#endif

            if (batteryCount >= MAX_BATTERY_COUNT) {
                SPDLOG_WARN("Battery: MAX_BATTERY_COUNT ({}) exceeded", MAX_BATTERY_COUNT);
                break;
            }

            battPath[batteryCount] = p.path().string();
            batteryCount++;
        }

        batt_count = batteryCount;
    }
    catch (const fs::filesystem_error& e) {
        SPDLOG_ERROR("Battery: filesystem scan error: {}", e.what());
        batt_count = 0;
    }
}

void BatteryStats::update() {
    if (!batt_check) {
        numBattery();
    }

    if (batt_count <= 0) {
        current_watt    = 0.0f;
        current_percent = 0.0f;
        remaining_time  = 0.0f;
        return;
    }

    current_watt    = getPower();
    current_percent = getPercent();
    remaining_time  = getTimeRemaining();
}

float BatteryStats::getPercent() {
    if (batt_count <= 0)
        return 0.0f;

    float total_now  = 0.0f;
    float total_full = 0.0f;
    char pathBuf[256];

    for (int i = 0; i < batt_count; i++) {
        const string& basePath = battPath[i];

        bool  ok_now  = false;
        bool  ok_full = false;
        float now     = 0.0f;
        float full    = 0.0f;

        // 1순위: charge_*
        make_path(pathBuf, basePath, "charge_now");
        now = readFloatFast(pathBuf, 1000000.0f, &ok_now);

        if (ok_now) {
            make_path(pathBuf, basePath, "charge_full");
            full = readFloatFast(pathBuf, 1000000.0f, &ok_full);
        }

        // 2순위: energy_*
        if (!ok_now || !ok_full) {
            ok_now  = false;
            ok_full = false;

            make_path(pathBuf, basePath, "energy_now");
            now = readFloatFast(pathBuf, 1000000.0f, &ok_now);

            if (ok_now) {
                make_path(pathBuf, basePath, "energy_full");
                full = readFloatFast(pathBuf, 1000000.0f, &ok_full);
            }
        }

        // 3순위: capacity
        if (!ok_now || !ok_full) {
            ok_now  = false;
            ok_full = false;

            make_path(pathBuf, basePath, "capacity");
            now = readFloatFast(pathBuf, 100.0f, &ok_now); // 0.xx ~ 1.0

            if (ok_now) {
                full    = 1.0f;
                ok_full = true;
            }
        }

        if (ok_now && ok_full) {
            total_now  += now;
            total_full += full;
        }
    }

    if (total_full <= 0.0001f)
        return 0.0f;

    return (total_now / total_full) * 100.0f;
}

float BatteryStats::getPower() {
    if (batt_count <= 0)
        return 0.0f;

    float total_watts = 0.0f;
    char pathBuf[256];

    for (int i = 0; i < batt_count; i++) {
        const string& basePath = battPath[i];

        make_path(pathBuf, basePath, "status");
        char status = readStatusChar(pathBuf);

        // Charging(C) or Full(F) -> 방전량 0 처리
        if (status == 'C' || status == 'F')
            continue;

        float watts      = 0.0f;
        bool  ok_current = false;

        make_path(pathBuf, basePath, "current_now");
        float current = readFloatFast(pathBuf, 1000000.0f, &ok_current); // uA -> A

        if (ok_current) {
            make_path(pathBuf, basePath, "voltage_now");
            float voltage = readFloatFast(pathBuf, 1000000.0f);          // uV -> V
            watts = std::fabs(current) * std::fabs(voltage);
        } else {
            make_path(pathBuf, basePath, "power_now");
            float power = readFloatFast(pathBuf, 1000000.0f);            // uW -> W
            watts = std::fabs(power);
        }

        total_watts += watts;
    }

    return total_watts;
}

float BatteryStats::getTimeRemaining() {
    if (batt_count <= 0)
        return 0.0f;

    float current_sum_ah = 0.0f;
    float charge_sum_ah  = 0.0f;
    char pathBuf[256];

    for (int i = 0; i < batt_count; i++) {
        const string& basePath = battPath[i];

        // 1. 전류(A)
        bool  ok_current = false;
        make_path(pathBuf, basePath, "current_now");
        float current = readFloatFast(pathBuf, 1000000.0f, &ok_current); // uA -> A

        if (!ok_current) {
            make_path(pathBuf, basePath, "power_now");
            float power = readFloatFast(pathBuf, 1000000.0f);            // uW -> W

            make_path(pathBuf, basePath, "voltage_now");
            float volt  = readFloatFast(pathBuf, 1000000.0f);           // uV -> V

            if (volt > 0.0001f)
                current = power / volt;
        }

        current_sum_ah += std::fabs(current);

        // 2. 잔량(Ah)
        bool  ok_charge = false;
        make_path(pathBuf, basePath, "charge_now");
        float charge = readFloatFast(pathBuf, 1000000.0f, &ok_charge);  // uAh -> Ah

        if (!ok_charge) {
            make_path(pathBuf, basePath, "energy_now");
            float energy = readFloatFast(pathBuf, 1000000.0f);          // uWh -> Wh

            make_path(pathBuf, basePath, "voltage_now");
            float volt   = readFloatFast(pathBuf, 1000000.0f);         // uV -> V

            if (volt > 0.0001f)
                charge = energy / volt;
        }

        charge_sum_ah += charge;
    }

    // 전류가 흐를 때만 기록
    if (current_sum_ah > 0.0001f) {
        if (current_now_vec.size() >= 25) {
            current_now_vec.erase(current_now_vec.begin());
        }
        current_now_vec.push_back(current_sum_ah);
    }

    if (current_now_vec.empty() || charge_sum_ah <= 0.0001f)
        return 0.0f;

    float avg_current = 0.0f;
    for (float c : current_now_vec)
        avg_current += c;

    avg_current /= static_cast<float>(current_now_vec.size());

    if (avg_current <= 0.0001f)
        return 0.0f;

    // 시간(시간 단위) = 용량(Ah) / 전류(A)
    return charge_sum_ah / avg_current;
}

BatteryStats Battery_Stats;

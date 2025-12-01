#include <spdlog/spdlog.h>
#include <filesystem.h>
#include "battery.h"

namespace fs = ghc::filesystem;
using namespace std;

void BatteryStats::numBattery() {
    batt_count = 0;
    batt_check = true;

    const fs::path path("/sys/class/power_supply/");

#if defined(__ANDROID__)
    // 안드로이드: 배터리 후보가 전혀 없을 때 INFO 한 번만 찍기 위한 플래그
    static bool android_batt_info_logged = false;
#endif

    try {
        if (!fs::exists(path)) {
#if defined(__ANDROID__)
            if (!android_batt_info_logged) {
                SPDLOG_INFO(
                    "Battery: /sys/class/power_supply not accessible, disabling battery stats");
                android_batt_info_logged = true;
            }
#else
            SPDLOG_DEBUG("Battery: {} not present", path.string());
#endif
            return;
        }

        int batteryCount = 0;

        for (const auto& p : fs::directory_iterator(path)) {
            const std::string fileName = p.path().filename().string();

#if defined(__ANDROID__)
            // Android / rooted 환경:
            // 1) type == "Battery" 우선
            // 2) fallback: 이름이 battery / BAT* / bat* 인 것
            const fs::path syspath    = p.path();
            const fs::path type_path  = syspath / "type";
            bool is_battery = false;

            if (fs::exists(type_path)) {
                std::ifstream input(type_path.string());
                std::string type;
                if (std::getline(input, type)) {
                    if (type == "Battery")
                        is_battery = true;
                }
            }

            if (!is_battery) {
                if (fileName == "battery" ||
                    fileName.rfind("BAT", 0) == 0 ||
                    fileName.rfind("bat", 0) == 0) {
                    is_battery = true;
                }
            }

            if (!is_battery)
                continue;

            battPath[batteryCount] = syspath.string();
            batteryCount += 1;
#else
            // 비-안드로이드: 기존 BAT* 탐색 유지
            if (fileName.find("BAT") != std::string::npos) {
                battPath[batteryCount] = p.path();
                batteryCount += 1;
            }
#endif
        }

        batt_count = batteryCount;

#if defined(__ANDROID__)
        if (batt_count == 0 && !android_batt_info_logged) {
            SPDLOG_INFO(
                "Battery: no usable power_supply entries under {}, disabling battery stats",
                path.string());
            android_batt_info_logged = true;
        }
#endif
    }
    catch (const fs::filesystem_error& e) {
#if defined(__ANDROID__)
        // 안드로이드는 여기서도 INFO 한 번만
        static bool android_batt_fs_logged = false;
        if (!android_batt_fs_logged) {
            SPDLOG_INFO(
                "Battery: filesystem error while scanning {}: {}, disabling battery stats",
                path.string(), e.what());
            android_batt_fs_logged = true;
        }
#else
        SPDLOG_DEBUG("Battery: failed to scan {}: {}", path.string(), e.what());
#endif
        batt_count = 0;
    }
}

void BatteryStats::update() {
#if defined(__ANDROID__)
    // 안드로이드:
    //  - 첫 호출에서만 numBattery() 실행 (권한/경로 체크 + INFO 로그 가능)
    //  - batt_count == 0 이면 이후로는 조용히 0 리턴
    if (!batt_check) {
        numBattery();
    }

    if (batt_count <= 0) {
        current_watt    = 0.0f;
        current_percent = 0.0f;
        remaining_time  = 0.0f;
        return;
    }
#else
    // 일반 리눅스 쪽은 기존 로직 유지
    if (!batt_check) {
        numBattery();
        if (batt_count == 0) {
            SPDLOG_ERROR("No battery found");
        }
    }

    if (batt_count <= 0) {
        current_watt    = 0.0f;
        current_percent = 0.0f;
        remaining_time  = 0.0f;
        return;
    }
#endif

    // 여기까지 왔으면 batt_count > 0
    current_watt    = getPower();
    current_percent = getPercent();
    remaining_time  = getTimeRemaining();
}

float BatteryStats::getPercent()
{
    if (batt_count <= 0)
        return 0.0f;

    float charge_n = 0.0f;
    float charge_f = 0.0f;

    for (int i = 0; i < batt_count; i++) {
        const string syspath      = battPath[i];
        const string charge_now   = syspath + "/charge_now";
        const string charge_full  = syspath + "/charge_full";
        const string energy_now   = syspath + "/energy_now";
        const string energy_full  = syspath + "/energy_full";
        const string capacity     = syspath + "/capacity";

        if (fs::exists(charge_now)) {
            std::ifstream input_now(charge_now);
            std::ifstream input_full(charge_full);
            std::string line;

            if (std::getline(input_now, line)) {
                charge_n += (stof(line) / 1000000.0f);
            }
            if (std::getline(input_full, line)) {
                charge_f += (stof(line) / 1000000.0f);
            }
        }
        else if (fs::exists(energy_now)) {
            std::ifstream input_now(energy_now);
            std::ifstream input_full(energy_full);
            std::string line;

            if (std::getline(input_now, line)) {
                charge_n += (stof(line) / 1000000.0f);
            }
            if (std::getline(input_full, line)) {
                charge_f += (stof(line) / 1000000.0f);
            }
        }
        else {
            // /capacity만 있는 구형 케이스: 퍼센트 평균
            std::ifstream input(capacity);
            std::string line;
            if (std::getline(input, line)) {
                charge_n += stof(line) / 100.0f;
                charge_f = static_cast<float>(batt_count);
            }
        }
    }

    if (charge_f <= 0.0f)
        return 0.0f;

    return (charge_n / charge_f) * 100.0f;
}

float BatteryStats::getPower() {
    if (batt_count <= 0)
        return 0.0f;

    float current = 0.0f;
    float voltage = 0.0f;

    for (int i = 0; i < batt_count; i++) {
        const string syspath        = battPath[i];
        const string current_power  = syspath + "/current_now";
        const string current_voltage= syspath + "/voltage_now";
        const string power_now      = syspath + "/power_now";
        const string status         = syspath + "/status";

        std::ifstream input_status(status);
        std::string line;
        if (std::getline(input_status, line)) {
            current_status = line;
            state[i]       = current_status;
        }

        // 충전 중이거나 풀 상태면 방전 전력 0으로 간주
        if (state[i] == "Charging" || state[i] == "Unknown" || state[i] == "Full") {
            return 0.0f;
        }

        if (fs::exists(current_power)) {
            std::ifstream input_current(current_power);
            std::ifstream input_voltage(current_voltage);
            std::string line_c, line_v;

            if (std::getline(input_current, line_c)) {
                current += (stof(line_c) / 1000000.0f);
            }
            if (std::getline(input_voltage, line_v)) {
                voltage += (stof(line_v) / 1000000.0f);
            }
        } else {
            std::ifstream input_power(power_now);
            std::string line_p;
            if (std::getline(input_power, line_p)) {
                current += (stof(line_p) / 1000000.0f);
                voltage = 1.0f;
            }
        }
    }

    return current * voltage;
}

float BatteryStats::getTimeRemaining() {
    if (batt_count <= 0)
        return 0.0f;

    float current = 0.0f;
    float charge  = 0.0f;

    for (int i = 0; i < batt_count; i++) {
        const string syspath      = battPath[i];
        const string current_now  = syspath + "/current_now";
        const string charge_now   = syspath + "/charge_now";
        const string energy_now   = syspath + "/energy_now";
        const string voltage_now  = syspath + "/voltage_now";
        const string power_now    = syspath + "/power_now";

        // 현재 전류
        if (fs::exists(current_now)) {
            std::ifstream input(current_now);
            std::string line;
            if (std::getline(input, line)) {
                current_now_vec.push_back(stof(line));
            }
        } else if (fs::exists(power_now)) {
            float voltage = 0.0f;
            float power   = 0.0f;
            std::string line_v, line_p;

            std::ifstream input_power(power_now);
            std::ifstream input_voltage(voltage_now);

            if (std::getline(input_voltage, line_v)) {
                voltage = stof(line_v);
            }
            if (std::getline(input_power, line_p)) {
                power = stof(line_p);
            }

            if (voltage > 0.0f)
                current_now_vec.push_back(power / voltage);
        }

        // 잔여 용량
        if (fs::exists(charge_now)) {
            std::ifstream input(charge_now);
            std::string line;
            if (std::getline(input, line)) {
                charge += stof(line);
            }
        }
        else if (fs::exists(energy_now)) {
            float energy  = 0.0f;
            float voltage = 0.0f;
            std::string line_e, line_v;

            std::ifstream input_energy(energy_now);
            std::ifstream input_voltage(voltage_now);

            if (std::getline(input_energy, line_e)) {
                energy = stof(line_e);
            }
            if (std::getline(input_voltage, line_v)) {
                voltage = stof(line_v);
            }

            if (voltage > 0.0f)
                charge += energy / voltage;
        }

        if (current_now_vec.size() > 25)
            current_now_vec.erase(current_now_vec.begin());
    }

    if (current_now_vec.empty() || charge <= 0.0f)
        return 0.0f;

    for (auto& current_now : current_now_vec)
        current += current_now;

    current /= static_cast<float>(current_now_vec.size());

    if (current <= 0.0f)
        return 0.0f;

    return charge / current;
}

BatteryStats Battery_Stats;

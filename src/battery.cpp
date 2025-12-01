#include <spdlog/spdlog.h>
#include <filesystem.h>
#include "battery.h"

namespace fs = ghc::filesystem;
using namespace std;

// 헬퍼 함수: 파일에서 float 값 읽기 (실패 시 0.0f 반환)
static float readFloatFromFile(const std::string& path, float divisor = 1.0f) {
    if (!fs::exists(path)) return 0.0f;
    std::ifstream input(path);
    std::string line;
    if (std::getline(input, line)) {
        try {
            return std::stof(line) / divisor;
        } catch (...) {
            return 0.0f;
        }
    }
    return 0.0f;
}

// 헬퍼 함수: 파일에서 string 값 읽기
static std::string readStringFromFile(const std::string& path) {
    if (!fs::exists(path)) return "";
    std::ifstream input(path);
    std::string line;
    std::getline(input, line);
    return line;
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
                SPDLOG_INFO("Battery: /sys/class/power_supply not accessible, disabling battery stats");
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
            // 안전 장치: 배열 크기를 넘지 않도록 (배열 크기가 헤더에 정의되어 있다고 가정)
            // if (batteryCount >= MAX_BATTERY_COUNT) break; 

#if defined(__ANDROID__)
            const fs::path syspath   = p.path();
            const fs::path type_path = syspath / "type";
            bool is_battery = false;

            // 1. type 파일 확인
            std::string type = readStringFromFile(type_path.string());
            if (type == "Battery") {
                is_battery = true;
            }

            // 2. 이름으로 fallback 확인
            if (!is_battery) {
                if (fileName == "battery" ||
                    fileName.find("BAT") == 0 || 
                    fileName.find("bat") == 0) { // rfind(..., 0) == 0 은 starts_with 의미
                    is_battery = true;
                }
            }

            if (!is_battery) continue;

            battPath[batteryCount] = syspath.string();
            batteryCount++;
#else
            if (fileName.find("BAT") != std::string::npos) {
                battPath[batteryCount] = p.path().string(); // p.path()는 path객체이므로 string변환
                batteryCount++;
            }
#endif
        }

        batt_count = batteryCount;

#if defined(__ANDROID__)
        if (batt_count == 0 && !android_batt_info_logged) {
            SPDLOG_INFO("Battery: no usable power_supply entries under {}, disabling battery stats", path.string());
            android_batt_info_logged = true;
        }
#endif
    }
    catch (const fs::filesystem_error& e) {
#if defined(__ANDROID__)
        static bool android_batt_fs_logged = false;
        if (!android_batt_fs_logged) {
            SPDLOG_INFO("Battery: filesystem error while scanning {}: {}, disabling battery stats", path.string(), e.what());
            android_batt_fs_logged = true;
        }
#else
        SPDLOG_DEBUG("Battery: failed to scan {}: {}", path.string(), e.what());
#endif
        batt_count = 0;
    }
}

void BatteryStats::update() {
    if (!batt_check) {
        numBattery();
#if !defined(__ANDROID__)
        if (batt_count == 0) {
            SPDLOG_ERROR("No battery found");
        }
#endif
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
    if (batt_count <= 0) return 0.0f;

    float total_charge_now = 0.0f;
    float total_charge_full = 0.0f;

    for (int i = 0; i < batt_count; i++) {
        const string syspath = battPath[i];
        
        // 우선순위: charge_now (Ah) -> energy_now (Wh) -> capacity (%)
        // 단위를 통일하기 위해 단순히 값을 읽어서 비율만 계산합니다.
        
        if (fs::exists(syspath + "/charge_now")) {
            total_charge_now  += readFloatFromFile(syspath + "/charge_now", 1000000.0f);
            total_charge_full += readFloatFromFile(syspath + "/charge_full", 1000000.0f);
        } 
        else if (fs::exists(syspath + "/energy_now")) {
            total_charge_now  += readFloatFromFile(syspath + "/energy_now", 1000000.0f);
            total_charge_full += readFloatFromFile(syspath + "/energy_full", 1000000.0f);
        } 
        else {
            // 정보가 부족한 구형 기기: % 자체를 읽어서 평균을 내기 위해 누적
            total_charge_now  += readFloatFromFile(syspath + "/capacity", 100.0f); // 0.xx 형태로 변환
            total_charge_full += 1.0f; // 100% 기준
        }
    }

    if (total_charge_full <= 0.0f) return 0.0f;

    return (total_charge_now / total_charge_full) * 100.0f;
}

float BatteryStats::getPower() {
    if (batt_count <= 0) return 0.0f;

    float total_watts = 0.0f;

    for (int i = 0; i < batt_count; i++) {
        const string syspath = battPath[i];
        
        // 상태 확인
        string s = readStringFromFile(syspath + "/status");
        if (!s.empty()) {
            current_status = s;
            state[i] = s;
        }

        // 충전 중이거나 완충 상태면 방전 전력(소모량)은 0으로 간주
        if (state[i] == "Charging" || state[i] == "Unknown" || state[i] == "Full") {
            continue; // 이 배터리의 소모 전력은 0W
        }

        float inst_current = 0.0f;
        float inst_voltage = 0.0f;
        
        if (fs::exists(syspath + "/current_now")) {
            inst_current = readFloatFromFile(syspath + "/current_now", 1000000.0f);
            inst_voltage = readFloatFromFile(syspath + "/voltage_now", 1000000.0f);
        
            inst_current = std::fabs(inst_current);
            inst_voltage = std::fabs(inst_voltage);
            total_watts += (inst_current * inst_voltage);
        } else {
            float power = readFloatFromFile(syspath + "/power_now", 1000000.0f);
            total_watts += std::fabs(power);
        }

        // [중요 수정] 배터리별로 (V * I)를 계산 후 합산해야 정확함
        total_watts += (inst_current * inst_voltage);
    }

    return total_watts;
}

float BatteryStats::getTimeRemaining() {
    if (batt_count <= 0) return 0.0f;

    float system_current_sum = 0.0f; // 현재 시점의 모든 배터리 전류 합
    float total_charge = 0.0f;       // 현재 모든 배터리의 잔여 용량 합

    for (int i = 0; i < batt_count; i++) {
        const string syspath = battPath[i];

        // 1. 전류 읽기 (Ah 계산용 전류)
        float inst_current = 0.0f;
        
        if (fs::exists(syspath + "/current_now")) {
            inst_current = readFloatFromFile(syspath + "/current_now"); // 단위 보정 없이 raw값 읽음 (나중에 나눌 때 상쇄되거나, 일관성 유지)
        } else if (fs::exists(syspath + "/power_now")) {
            float p = readFloatFromFile(syspath + "/power_now");
            float v = readFloatFromFile(syspath + "/voltage_now");
            if (v > 0.0f) inst_current = p / v; // W / V = A
        }
        
        system_current_sum += inst_current;

        // 2. 잔여 용량 읽기
        if (fs::exists(syspath + "/charge_now")) {
            total_charge += readFloatFromFile(syspath + "/charge_now");
        } 
        else if (fs::exists(syspath + "/energy_now")) {
            // Wh -> Ah 변환 필요 (Ah = Wh / V)
            float e = readFloatFromFile(syspath + "/energy_now");
            float v = readFloatFromFile(syspath + "/voltage_now");
            if (v > 0.0f) {
                total_charge += (e / v); 
            }
        }
    }

    // [수정] 모든 배터리의 전류 합을 벡터에 한 번만 저장 (시스템 전체 부하)
    current_now_vec.push_back(system_current_sum);

    if (current_now_vec.size() > 25) {
        current_now_vec.erase(current_now_vec.begin());
    }

    if (current_now_vec.empty() || total_charge <= 0.0f) return 0.0f;

    // 평균 전류 계산
    float avg_current = 0.0f;
    for (float c : current_now_vec) {
        avg_current += c;
    }
    avg_current /= static_cast<float>(current_now_vec.size());

    if (avg_current <= 0.0f) return 0.0f;

    return total_charge / avg_current;
}

BatteryStats Battery_Stats;

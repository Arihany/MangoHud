#include "device.h"
#include <filesystem.h>
#include <iostream>
#include <algorithm>
#include <spdlog/spdlog.h>
#include <chrono>
#include <mutex>
#include <vector>
#include <fstream>

namespace fs = ghc::filesystem;
using namespace std;

// 전역 변수 정의
std::mutex device_lock;
std::vector<device_batt> device_data;
std::vector<std::string> list;
bool device_found = false;
bool check_gamepad = false;
bool check_mouse = false;
int  device_count = 0;

// 각 장치 카운터
int xbox_count = 0;
int ds4_count = 0;
int ds5_count = 0;
int switch_count = 0;
int bitdo_count = 0;
int shield_count = 0;

static const std::string xbox_paths[2] = {"gip", "xpadneo"};

static bool operator<(const device_batt& a, const device_batt& b) {
    return a.name < b.name;
}

void device_update(const struct overlay_params& params) {
#if defined(__ANDROID__)
    // 안드로이드: 주변기기 배터리 조회 기능 비활성화 (권한/호환성 문제)
    // 필요하다면 JNI를 통해 Android Java API(InputDevice)를 써야 함.
    return; 
#endif

    std::unique_lock<std::mutex> l(device_lock);

    // 1. 필요한지 체크 (Early Out)
    bool want_gamepad = std::find(params.device_battery.begin(), params.device_battery.end(), "gamepad") != params.device_battery.end();
    bool want_mouse   = std::find(params.device_battery.begin(), params.device_battery.end(), "mouse")   != params.device_battery.end();

    if (!want_gamepad && !want_mouse) {
        list.clear();
        device_found = false;
        device_count = 0;
        return;
    }

    // 2. 2초(2000ms)마다 갱신 (주변기기 연결은 자주 바뀌지 않음)
    static auto last_update = std::chrono::steady_clock::now();
    auto now = std::chrono::steady_clock::now();
    if (std::chrono::duration_cast<std::chrono::milliseconds>(now - last_update).count() < 2000) {
        return;
    }
    last_update = now;

    // 3. 리셋
    list.clear();
    device_found = false;
    check_gamepad = want_gamepad;
    check_mouse = want_mouse;

    xbox_count = ds4_count = ds5_count = switch_count = bitdo_count = shield_count = 0;

    const fs::path path("/sys/class/power_supply");

    // 4. 스캔
    try {
        if (!fs::exists(path)) return;

        for (const auto &entry : fs::directory_iterator(path)) {
            std::string fileName = entry.path().filename().string();
            std::string syspath  = entry.path().string();

            if (check_gamepad) {
                for (const auto &n : xbox_paths) {
                    if (fileName.find(n) != std::string::npos) {
                        list.push_back(syspath);
                        device_found = true;
                        xbox_count++;
                        goto next_entry; // 중복 카운팅 방지
                    }
                }
                if (fileName.find("sony_controller") != std::string::npos) {
                    list.push_back(syspath); device_found = true; ds4_count++;
                } else if (fileName.find("ps-controller") != std::string::npos) {
                    list.push_back(syspath); device_found = true; ds5_count++;
                } else if (fileName.find("nintendo_switch_controller") != std::string::npos) {
                    list.push_back(syspath); device_found = true; switch_count++;
                } else if (fileName.find("hid-e4") != std::string::npos) { // 8BitDo
                    list.push_back(syspath); device_found = true; bitdo_count++;
                } else if (fileName.find("thunderstrike") != std::string::npos) { // Shield
                    list.push_back(syspath); device_found = true; shield_count++;
                }
            }

            if (check_mouse && fileName.find("hidpp_battery") != std::string::npos) {
                list.push_back(syspath);
                device_found = true;
            }

            next_entry:;
        }
    } catch (const fs::filesystem_error &e) {
        // 권한 에러 등은 조용히 무시 (로그 스팸 방지)
    }
}

void device_info() {
#if defined(__ANDROID__)
    return;
#endif

    std::unique_lock<std::mutex> l(device_lock);
    
    device_data.clear();
    device_count = 0;

    if (list.empty()) return;

    int xc=0, d4c=0, d5c=0, swc=0, bdc=0, shc=0;

    for (const auto &path : list) {
        device_data.emplace_back();
        device_batt &dev = device_data.back();

        // 파일 열기 (실패 시 조용히 넘어감)
        std::ifstream input_cap(path + "/capacity");
        std::ifstream input_stat(path + "/status");
        
        // 이름 설정 로직 (간소화)
        if (path.find("gip") != string::npos || path.find("xpadneo") != string::npos)
            dev.name = (xbox_count==1) ? "XBOX" : "XBOX-" + to_string(++xc);
        else if (path.find("sony_controller") != string::npos)
            dev.name = (ds4_count==1) ? "DS4" : "DS4-" + to_string(++d4c);
        else if (path.find("ps-controller") != string::npos)
            dev.name = (ds5_count==1) ? "DS5" : "DS5-" + to_string(++d5c);
        else if (path.find("nintendo") != string::npos)
            dev.name = "SWITCH";
        else if (path.find("hidpp") != string::npos)
            dev.name = "MOUSE";
        else
            dev.name = "GAMEPAD";

        // 상태 읽기
        std::string line;
        if (std::getline(input_stat, line)) {
            if (line == "Charging" || line == "Full") dev.is_charging = true;
        }

        // 용량 읽기
        if (std::getline(input_cap, line)) {
            dev.report_percent = true;
            dev.battery_percent = line;
            try {
                int p = std::stoi(line);
                if (p <= 20) dev.battery = "Low";
                else if (p <= 50) dev.battery = "Med";
                else dev.battery = "High";
            } catch (...) {}
        }
    }
    
    device_count = device_data.size();
}

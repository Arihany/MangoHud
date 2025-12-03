#pragma once

#include <string>
#include <vector>

#ifndef MAX_BATTERY_COUNT
#define MAX_BATTERY_COUNT 4
#endif

class BatteryStats {
public:
    void numBattery();
    void update();

    float getPower();
    float getPercent();
    float getTimeRemaining();

    std::string battPath[MAX_BATTERY_COUNT];
    std::string state[MAX_BATTERY_COUNT];

    float current_watt    = 0.0f;
    float current_percent = 0.0f;
    float remaining_time  = 0.0f;

    std::string current_status;

    int  batt_count  = 0;
    bool batt_check  = false;

    std::vector<float> current_now_vec{};
};

extern BatteryStats Battery_Stats;

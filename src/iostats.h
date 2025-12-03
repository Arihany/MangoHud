#pragma once
#ifndef MANGOHUD_IOSTATS_H
#define MANGOHUD_IOSTATS_H

#include <inttypes.h>
#include "timing.hpp"

struct iostats {
    struct bytes {
        unsigned long long read_bytes;
        unsigned long long write_bytes;
    };

    struct rate {
        float read;
        float write;
    };

    bytes curr{};
    bytes prev{};
    rate  diff{};
    rate  per_second{};
    Clock::time_point last_update{};
};

extern iostats g_io_stats;
void getIoStats(iostats& io);

#endif // MANGOHUD_IOSTATS_H

#include "net.h"
#include "hud_elements.h"

namespace fs = ghc::filesystem;

namespace {

// 안전한 stoll 래퍼
static inline long long safe_stoll(const std::string& str, long long default_value = 0)
{
    if (str.empty()) {
        SPDLOG_DEBUG("net: tx/rx returned an empty string");
        return default_value;
    }

    try {
        return std::stoll(str);
    } catch (const std::invalid_argument&) {
        SPDLOG_DEBUG("net: stoll invalid argument for \"{}\"", str);
    } catch (const std::out_of_range&) {
        SPDLOG_DEBUG("net: stoll out of range for \"{}\"", str);
    }
    return default_value;
}

// throughput 계산 유틸
static inline uint64_t calculateThroughput(
    long long currentBytes,
    long long previousBytes,
    std::chrono::steady_clock::time_point previousTime,
    std::chrono::steady_clock::time_point currentTime)
{
    // 첫 호출 또는 초기값이면 그냥 0
    if (previousTime.time_since_epoch().count() == 0)
        return 0;

    auto elapsed = std::chrono::duration<double>(currentTime - previousTime).count();
    if (elapsed <= 0.0)
        return 0;

    long long delta = currentBytes - previousBytes;
    if (delta <= 0)
        return 0;

    double bps = static_cast<double>(delta) / elapsed;
    if (bps < 0.0)
        bps = 0.0;

    return static_cast<uint64_t>(bps);
}

} // anonymous namespace

Net::Net()
{
    auto params = get_params();
    should_reset = false;

#if defined(__ANDROID__)
    // Android / Winlator:
    // - /sys/class/net 읽는 것 자체가 큰 의미 없고, 프레임당 sysfs 접근만 늘어난다.
    // - 그냥 전역적으로 네트워크 지표 비활성화.
    SPDLOG_DEBUG("Network: disabled on Android (skipping {} enumeration)", NETDIR);
    return;
#else
    fs::path net_dir(NETDIR);

    try {
        if (fs::exists(net_dir) && fs::is_directory(net_dir)) {
            for (const auto& entry : fs::directory_iterator(net_dir)) {
                if (!entry.is_directory())
                    continue;

                auto val = entry.path().filename().string();
                if (val == "lo")
                    continue;

                // network=1  => 모든 인터페이스 자동 선택
                // network=... => 지정한 이름만 선택
                if (!params->network.empty() && params->network.front() == "1") {
                    interfaces.push_back({ val, 0, 0, 0, 0, {} });
                } else if (!params->network.empty()) {
                    auto it = std::find(params->network.begin(), params->network.end(), val);
                    if (it != params->network.end())
                        interfaces.push_back({ val, 0, 0, 0, 0, {} });
                }
            }
        }
    } catch (const fs::filesystem_error& e) {
        SPDLOG_DEBUG("Network: failed to scan {}: {}", net_dir.string(), e.what());
    }

    static bool logged_once = false;
    if (interfaces.empty() && !logged_once) {
        SPDLOG_DEBUG("Network: no usable interfaces found under {}", NETDIR);
        logged_once = true;
    }
#endif
}

void Net::update()
{
#if defined(__ANDROID__)
    static bool logged_once = false;
    if (!logged_once) {
        SPDLOG_DEBUG("Network: update() is no-op on Android");
        logged_once = true;
    }

    // 혹시라도 인터페이스 벡터를 쓰는 코드가 있어도 값은 항상 0으로 유지
    for (auto& iface : interfaces) {
        iface.txBps = 0;
        iface.rxBps = 0;
    }
    return;
#else
    if (interfaces.empty())
        return;

    auto now = std::chrono::steady_clock::now();

    for (auto& iface : interfaces) {
        // path to tx_bytes and rx_bytes
        const std::string txfile = NETDIR + iface.name + TXFILE;
        const std::string rxfile = NETDIR + iface.name + RXFILE;

        const uint64_t prevTx = iface.txBytes;
        const uint64_t prevRx = iface.rxBytes;

        iface.txBytes = static_cast<uint64_t>(safe_stoll(read_line(txfile)));
        iface.rxBytes = static_cast<uint64_t>(safe_stoll(read_line(rxfile)));

        iface.txBps = calculateThroughput(
            static_cast<long long>(iface.txBytes),
            static_cast<long long>(prevTx),
            iface.previousTime,
            now
        );

        iface.rxBps = calculateThroughput(
            static_cast<long long>(iface.rxBytes),
            static_cast<long long>(prevRx),
            iface.previousTime,
            now
        );

        iface.previousTime = now;
    }
#endif
}

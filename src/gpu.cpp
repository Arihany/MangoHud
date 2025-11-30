#include "gpu.h"
#include "file_utils.h"
#include "hud_elements.h"

namespace fs = ghc::filesystem;

GPUS::GPUS() {
    std::set<std::string> gpu_entries;

#if defined(__ANDROID__)
    // ANDROID: /sys/class/drm 존재/권한 확인 후 renderD* 나열
    const fs::path drm_root{"/sys/class/drm"};
    std::error_code ec;

    const bool exists = fs::exists(drm_root, ec);

    if (!exists || ec) {
        if (ec == std::errc::permission_denied) {
            SPDLOG_DEBUG(
                "Android: skipping {} enumeration (permission denied)",
                drm_root.string()
            );
        } else if (!exists && !ec) {
            SPDLOG_DEBUG(
                "Android: {} does not exist, skipping DRM GPU discovery",
                drm_root.string()
            );
        } else {
            SPDLOG_WARN(
                "Android: error probing {}: {}",
                drm_root.string(),
                ec ? ec.message() : "no error_code"
            );
        }
    } else {
        fs::directory_iterator it{drm_root, ec};
        fs::directory_iterator end{};

        for (; !ec && it != end; it.increment(ec)) {
            const auto& entry = *it;
            if (!entry.is_directory())
                continue;

            const std::string node_name = entry.path().filename().string();

            if (node_name.rfind("renderD", 0) != 0 || node_name.length() <= 7)
                continue;

            const std::string render_number = node_name.substr(7);
            if (std::all_of(render_number.begin(), render_number.end(), ::isdigit))
                gpu_entries.insert(node_name);
        }

        if (ec == std::errc::permission_denied) {
            SPDLOG_DEBUG(
                "Android: permission denied while iterating {} – treating as no DRM GPUs",
                drm_root.string()
            );
            gpu_entries.clear();
        } else if (ec) {
            SPDLOG_WARN(
                "Android: error while iterating {}: {}",
                drm_root.string(),
                ec.message()
            );
        }
    }

#else
    // 일반 리눅스: /sys/class/drm에서 renderD* 나열
    for (const auto& entry : fs::directory_iterator("/sys/class/drm")) {
        if (!entry.is_directory())
            continue;

        const std::string node_name = entry.path().filename().string();

        if (node_name.rfind("renderD", 0) != 0 || node_name.length() <= 7)
            continue;

        const std::string render_number = node_name.substr(7);
        if (std::all_of(render_number.begin(), render_number.end(), ::isdigit))
            gpu_entries.insert(node_name);
    }
#endif

    uint8_t idx = 0;
    uint8_t total_active = 0;

#if defined(__ANDROID__)
    // DRM renderD*가 하나도 안 보일 때: KGSL fallback
    if (gpu_entries.empty()) {
        const fs::path kgsl_root{"/sys/class/kgsl/kgsl-3d0"};
        std::error_code kgsl_ec;
        const bool kgsl_exists = fs::exists(kgsl_root, kgsl_ec);

        if (kgsl_exists && !kgsl_ec) {
            SPDLOG_INFO(
                "Android: no DRM render nodes visible, using KGSL fallback at {}",
                kgsl_root.string()
            );

            const std::string node_name = "kgsl-3d0";
            const std::string driver    = "msm_drm";
            const char* pci_dev         = "";

            uint32_t vendor_id = 0;
            uint32_t device_id = 0;

            auto ptr = std::make_shared<GPU>(
                node_name, vendor_id, device_id, pci_dev, driver
            );

            ptr->is_active = true;
            available_gpus.emplace_back(ptr);

            SPDLOG_DEBUG(
                "GPU Found (KGSL fallback): node_name: {}, driver: {}, "
                "vendor_id: {:x} device_id: {:x} pci_dev: {}",
                node_name, driver, vendor_id, device_id, pci_dev
            );

            total_active = 1;
        } else {
            if (kgsl_ec == std::errc::permission_denied) {
                SPDLOG_DEBUG(
                    "Android: KGSL node {} present but permission denied; skipping GPU discovery",
                    kgsl_root.string()
                );
            } else if (kgsl_ec) {
                SPDLOG_DEBUG(
                    "Android: error probing KGSL node {}: {}",
                    kgsl_root.string(),
                    kgsl_ec.message()
                );
            } else {
                SPDLOG_DEBUG(
                    "Android: KGSL node {} not present; no Android GPUs discovered",
                    kgsl_root.string()
                );
            }
        }
    }
#endif

    // DRM 기반 GPU들 처리
    for (const auto& node_name : gpu_entries) {
        const std::string driver = get_driver(node_name);

        if (driver.empty()) {
            SPDLOG_DEBUG("Failed to query driver name of node \"{}\"", node_name);
            continue;
        }

        {
            const std::string* d =
                std::find(std::begin(supported_drivers),
                          std::end(supported_drivers),
                          driver);

            if (d == std::end(supported_drivers)) {
                SPDLOG_WARN(
                    "node \"{}\" is using driver \"{}\" which is unsupported by MangoHud. Skipping...",
                    node_name, driver
                );
                continue;
            }
        }

        uint32_t vendor_id = 0;
        uint32_t device_id = 0;
        const char* pci_dev = "";

#if !defined(__ANDROID__)
        // 데스크탑/일반 리눅스: PCI 경로에서 vendor/device 조회
        const std::string path = "/sys/class/drm/" + node_name;
        const std::string device_address = get_pci_device_address(path);

        if (!device_address.empty()) {
            pci_dev = device_address.c_str();

            const std::string vendor_path =
                "/sys/bus/pci/devices/" + device_address + "/vendor";
            const std::string device_path =
                "/sys/bus/pci/devices/" + device_address + "/device";

            try {
                vendor_id = std::stoul(read_line(vendor_path), nullptr, 16);
            } catch (...) {
                SPDLOG_DEBUG("stoul failed on vendor path: {}", vendor_path);
            }

            try {
                device_id = std::stoul(read_line(device_path), nullptr, 16);
            } catch (...) {
                SPDLOG_DEBUG("stoul failed on device path: {}", device_path);
            }
        }
#else
        // ANDROID:
        // - 모바일 SoC는 일반적으로 PCI 버스를 안 쓰므로 vendor/device는 0으로 둔다.
        (void)node_name;
#endif

        auto ptr = std::make_shared<GPU>(node_name, vendor_id, device_id, pci_dev, driver);

        if (params()->gpu_list.size() == 1 && params()->gpu_list[0] == idx++)
            ptr->is_active = true;

        if (!params()->pci_dev.empty() && pci_dev == params()->pci_dev)
            ptr->is_active = true;

        available_gpus.emplace_back(ptr);

        SPDLOG_DEBUG(
            "GPU Found: node_name: {}, driver: {}, vendor_id: {:x} device_id: {:x} pci_dev: {}",
            node_name, driver, vendor_id, device_id, pci_dev
        );

        if (ptr->is_active) {
            SPDLOG_INFO(
                "Set {} as active GPU (driver={} id={:x}:{:x} pci_dev={})",
                node_name, driver, vendor_id, device_id, pci_dev
            );
            total_active++;
        }
    }

    if (total_active < 2)
        return;

    for (auto& gpu : available_gpus) {
        if (!gpu->is_active)
            continue;

        SPDLOG_WARN(
            "You have more than 1 active GPU, check if you use both pci_dev "
            "and gpu_list. If you use fps logging, MangoHud will log only "
            "this GPU: name = {}, driver = {}, vendor = {:x}, pci_dev = {}",
            gpu->drm_node, gpu->driver, gpu->vendor_id, gpu->pci_dev
        );
        break;
    }
}

std::string GPUS::get_driver(const std::string& node) {
    std::string path = "/sys/class/drm/" + node + "/device/driver";

    if (!fs::exists(path)) {
        SPDLOG_ERROR("{} doesn't exist", path);
        return "";
    }

    if (!fs::is_symlink(path)) {
        SPDLOG_ERROR("{} is not a symlink (it should be)", path);
        return "";
    }

    std::string driver = fs::read_symlink(path).string();
    driver = driver.substr(driver.rfind("/") + 1);

    return driver;
}

std::string GPUS::get_pci_device_address(const std::string& drm_card_path) {
#if defined(__ANDROID__)
    // ANDROID:
    // - 모바일 SoC는 일반적으로 PCI 버스로 GPU를 보지 않는다.
    // - 여기서 canonical/read_symlink를 돌려봤자 의미 없고, I/O + 예외 오버헤드만 생긴다.
    return {};
#else
    std::error_code ec;

    const auto subsystem_path = drm_card_path + "/device/subsystem";
    auto subsystem = fs::canonical(subsystem_path, ec);
    if (ec) {
        SPDLOG_DEBUG(
            "get_pci_device_address: canonical({}) failed: {}",
            subsystem_path, ec.message()
        );
        return {};
    }

    const auto subsystem_str = subsystem.string();
    auto idx = subsystem_str.rfind('/');
    if (idx == std::string::npos)
        return {};

    const auto bus_name = subsystem_str.substr(idx + 1);
    if (bus_name != "pci")
        return {};

    const auto device_symlink_path = drm_card_path + "/device";
    auto pci_symlink = fs::read_symlink(device_symlink_path, ec);
    if (ec) {
        SPDLOG_DEBUG(
            "get_pci_device_address: read_symlink({}) failed: {}",
            device_symlink_path, ec.message()
        );
        return {};
    }

    const auto pci_str = pci_symlink.string();
    idx = pci_str.rfind('/');
    if (idx == std::string::npos)
        return {};

    return pci_str.substr(idx + 1);
#endif
}

int GPU::index_in_selected_gpus() {
    auto selected_gpus = gpus->selected_gpus();
    auto it = std::find_if(selected_gpus.begin(), selected_gpus.end(),
                        [this](const std::shared_ptr<GPU>& gpu) {
                            return gpu.get() == this;
                        });
    if (it != selected_gpus.end()) {
        return std::distance(selected_gpus.begin(), it);
    }
    return -1;
}

std::string GPU::gpu_text() {
    std::string gpu_text;
    size_t index = this->index_in_selected_gpus();

    if (gpus->selected_gpus().size() == 1) {
        // When there's exactly one selected GPU, return "GPU" without index
        gpu_text = "GPU";
        if (gpus->params()->gpu_text.size() > 0) {
            gpu_text = gpus->params()->gpu_text[0];
        }
    } else if (gpus->selected_gpus().size() > 1) {
        // When there are multiple selected GPUs, use GPU+index or matching gpu_text
        gpu_text = "GPU" + std::to_string(index);
        if (gpus->params()->gpu_text.size() > index) {
            gpu_text = gpus->params()->gpu_text[index];
        }
    } else {
        // Default case for no selected GPUs
        gpu_text = "GPU";
    }

    return gpu_text;
}

std::string GPU::vram_text() {
    std::string vram_text;
    size_t index = this->index_in_selected_gpus();
    if (gpus->selected_gpus().size() > 1)
        vram_text = "VRAM" + std::to_string(index);
    else
        vram_text = "VRAM";
    return vram_text;
}

std::shared_ptr<const overlay_params> GPUS::params() {
    return get_params();
}

std::unique_ptr<GPUS> gpus = nullptr;

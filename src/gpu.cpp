#include "gpu.h"
#include "file_utils.h"
#include "hud_elements.h"

namespace fs = ghc::filesystem;

GPUS::GPUS() {

#if defined(__ANDROID__)

    // ANDROID:
    // - /sys/class/drm, KGSL, hwmon 전부 무시.
    // - Vulkan timestamp 백엔드에서 계산한 gpu_usage% / gpu_time_ms를
    //   HUD에 뿌려주기 위한 "논리 GPU" 슬롯 1개만 등록한다.
    //
    //   실제 값 주입 경로:
    //     vulkan.cpp::snapshot_swapchain_frame(...)
    //       → android_gpu_usage_get_metrics(...)
    //       → selected_gpus()[0]->metrics.load / gpu_time_ms 갱신.
    //
    //   여기서는 오직 "이름표"와 "슬롯"만 만든다.

    const std::string node_name = "android-vulkan";
    const std::string driver    = "vulkan_timestamp";
    const char*       pci_dev   = "";

    uint32_t vendor_id = 0;
    uint32_t device_id = 0;

    auto ptr = std::make_shared<GPU>(
        node_name,
        vendor_id,
        device_id,
        pci_dev,
        driver
    );

    // Android: 항상 이 하나만 활성 GPU로 취급
    ptr->is_active = true;
    available_gpus.emplace_back(ptr);

    SPDLOG_INFO(
        "Android: registered synthetic GPU node '{}' (driver={}) for Vulkan timestamp backend",
        node_name,
        driver
    );

    // Android에서는 항상 1개만, multi-GPU 경고 같은 것도 필요 없음.
    return;

#else
    // ===== 일반 리눅스: 기존 DRM 기반 GPU 열거 로직 유지 =====
    std::set<std::string> gpu_entries;

    // /sys/class/drm 에서 renderD* 노드 찾기
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

    uint8_t idx = 0;
    uint8_t total_active = 0;

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
                    node_name,
                    driver
                );
                continue;
            }
        }

        uint32_t    vendor_id = 0;
        uint32_t    device_id = 0;
        const char* pci_dev   = "";

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

        auto ptr = std::make_shared<GPU>(node_name, vendor_id, device_id, pci_dev, driver);

        if (params()->gpu_list.size() == 1 && params()->gpu_list[0] == idx++)
            ptr->is_active = true;

        if (!params()->pci_dev.empty() && pci_dev == params()->pci_dev)
            ptr->is_active = true;

        available_gpus.emplace_back(ptr);

        SPDLOG_DEBUG(
            "GPU Found: node_name: {}, driver: {}, vendor_id: {:x} device_id: {:x} pci_dev: {}",
            node_name,
            driver,
            vendor_id,
            device_id,
            pci_dev
        );

        if (ptr->is_active) {
            SPDLOG_INFO(
                "Set {} as active GPU (driver={} id={:x}:{:x} pci_dev={})",
                node_name,
                driver,
                vendor_id,
                device_id,
                pci_dev
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
            gpu->drm_node,
            gpu->driver,
            gpu->vendor_id,
            gpu->pci_dev
        );
        break;
    }
#endif
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

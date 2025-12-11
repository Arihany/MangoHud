#pragma once
#ifndef MANGOHUD_GPU_H
#define MANGOHUD_GPU_H

#include <cstdio>
#include <cstdint>
#include "overlay_params.h"
#include <thread>
#include <mutex>
#include <atomic>
#include <chrono>
#include <regex>
#include <iostream>
#include <array>
#include "amdgpu.h"
#include "nvidia.h"
#include "gpu_metrics_util.h"
#include "gpu_fdinfo.h"
#include <cstdlib>

class GPU {
    public:
        gpu_metrics metrics;
        std::string drm_node;
        std::unique_ptr<NVIDIA> nvidia = nullptr;
        std::unique_ptr<AMDGPU> amdgpu = nullptr;
        std::unique_ptr<GPU_fdinfo> fdinfo = nullptr;
        bool is_active = false;
        std::string pci_dev;
        uint32_t vendor_id = 0;
        uint32_t device_id = 0;
        const std::string driver;

GPU(
    std::string drm_node, uint32_t vendor_id, uint32_t device_id, const char* pci_dev,
    std::string driver
)
    : drm_node(drm_node), pci_dev(pci_dev), vendor_id(vendor_id), device_id(device_id),
      driver(driver)
{
    if (vendor_id == 0x10de)
        nvidia = std::make_unique<NVIDIA>(pci_dev);

    if (vendor_id == 0x1002)
        amdgpu = std::make_unique<AMDGPU>(pci_dev, device_id, vendor_id);

#if defined(__ANDROID__)
    // ANDROID:
    // - GPUS::GPUS()에서 synthetic node:
    //     drm_node = "android-vulkan"
    //     driver   = "vulkan_timestamp"
    //
    // - VKP_DISABLE=0  : Vulkan timestamp backend 전용 (fdinfo/KGSL 사용 안 함)
    // - VKP_DISABLE!=0 : Vulkan backend 비활성, fdinfo + KGSL fallback 활성
    if (driver == "vulkan_timestamp") {
        const char* env = std::getenv("VKP_DISABLE");
        bool vkp_disabled =
            env && env[0] != '\0' && env[0] != '0';

        if (vkp_disabled) {
            // GPU_fdinfo 쪽에는 "msm_drm" 모듈로 넘겨서:
            // - find_fd(): Android 브랜치에서 module 무시하고 fdinfo 스캔
            // - init_kgsl(): /sys/class/kgsl/kgsl-3d0 기반 gpubusy 폴백 활성
            fdinfo = std::make_unique<GPU_fdinfo>("msm_drm", "", drm_node);
        }
    } else
#endif
    if (
        driver == "i915" || driver == "xe" || driver == "panfrost" ||
        driver == "msm_dpu" || driver == "msm_drm"
    ) {
        fdinfo = std::make_unique<GPU_fdinfo>(driver, pci_dev, drm_node);
    }
}

        gpu_metrics get_metrics() {
            if (nvidia) {
                metrics = nvidia->copy_metrics();
            } else if (amdgpu) {
                metrics = amdgpu->copy_metrics();
            } else if (fdinfo) {
                metrics = fdinfo->copy_metrics();
            }
            return metrics;
        };

        std::vector<int> nvidia_pids() {
#ifdef HAVE_NVML
            if (nvidia)
                return nvidia->pids();
#endif
            return std::vector<int>();
        }

        void pause() {
            if (nvidia)
                nvidia->pause();

            if (amdgpu)
                amdgpu->pause();

            if (fdinfo)
                fdinfo->pause();
        }

        void resume() {
            if (nvidia)
                nvidia->resume();

            if (amdgpu)
                amdgpu->resume();

            if (fdinfo)
                fdinfo->resume();
        }

        bool is_apu() {
            if (amdgpu)
                return amdgpu->is_apu;
            else
                return false;
        }

        std::shared_ptr<Throttling> throttling() {
            if (nvidia)
                return nvidia->throttling;

            if (amdgpu)
                return amdgpu->throttling;

            return nullptr;
        }

        std::string gpu_text();
        std::string vram_text();

    private:
        std::thread thread;

        int index_in_selected_gpus();
};

class GPUS {
    public:
        std::vector<std::shared_ptr<GPU>> available_gpus;
        std::mutex mutex;
        overlay_params* const* params_pointer;

        GPUS();

        std::shared_ptr<const overlay_params> params();

        void pause() {
            for (auto& gpu : available_gpus)
                gpu->pause();
        }

        void resume() {
            for (auto& gpu : available_gpus)
                gpu->resume();
        }

        std::shared_ptr<GPU> active_gpu() {
            if (available_gpus.empty())
                return nullptr;

            for (auto gpu : available_gpus) {
                if (gpu->is_active)
                    return gpu;
            }

            // if no GPU is marked as active, just set it to the last one
            // because integrated gpus are usually first
            return available_gpus.back();
        }

        void update_throttling() {
            for (auto gpu : available_gpus)
                if (gpu->throttling())
                    gpu->throttling()->update();
        }

        void get_metrics() {
            std::lock_guard<std::mutex> lock(mutex);
            for (auto gpu : available_gpus)
                gpu->get_metrics();
        }

        std::vector<std::shared_ptr<GPU>> selected_gpus() {
            std::lock_guard<std::mutex> lock(mutex);
            std::vector<std::shared_ptr<GPU>> vec;

            if (params()->gpu_list.empty() && params()->pci_dev.empty())
                return available_gpus;

            if (!params()->gpu_list.empty()) {
                for (unsigned index : params()->gpu_list) {
                    if (index < available_gpus.size()) {
                        if (available_gpus[index])
                            vec.push_back(available_gpus[index]);
                    }
                }

                return vec;
            }

            if (!params()->pci_dev.empty()) {
                for (auto &gpu : available_gpus) {
                    if (gpu->pci_dev == params()->pci_dev) {
                        vec.push_back(gpu);
                        return vec;
                    }
                }

                return vec;
            }

            return vec;
        }

    private:
        std::string get_pci_device_address(const std::string& drm_card_path);
        std::string get_driver(const std::string& node);

        const std::array<std::string, 7> supported_drivers = {
            "amdgpu", "nvidia", "i915", "xe", "panfrost", "msm_dpu", "msm_drm"
        };
};

extern std::unique_ptr<GPUS> gpus;

void getNvidiaGpuInfo(const struct overlay_params& params);
void getAmdGpuInfo(void);
void getIntelGpuInfo();
bool checkNvidia(const char *pci_dev);
extern void nvapi_util();
extern bool checkNVAPI();
#endif //MANGOHUD_GPU_H

#include <sstream>
#include <iomanip>
#include <array>
#include <algorithm>
#include <vector>
#include <thread>
#include <chrono>
#include <ctime>
#include <cstdlib>
#include <fstream>

#include <spdlog/spdlog.h>
#include "logging.h"
#include "overlay.h"
#include "config.h"
#include "file_utils.h"
#include "string_utils.h"
#include "version.h"
#include "fps_metrics.h"

std::string os, cpu, gpu, ram, kernel, driver, cpusched;
bool sysInfoFetched = false;
double fps;
float frametime;
logData currentLogData = {};
std::unique_ptr<Logger> logger;
std::ofstream output_file;
std::thread log_thread;

#if !defined(__ANDROID__)
// ======================= 데스크톱 / 일반 리눅스용 =======================

std::string exec(std::string command) {
#ifndef _WIN32
    command = "unset LD_PRELOAD; " + command;
#endif
    std::array<char, 128> buffer{};
    std::string result;
    auto deleter = [](FILE* ptr){ pclose(ptr); };
    std::unique_ptr<FILE, decltype(deleter)> pipe(popen(command.c_str(), "r"), deleter);
    if (!pipe) {
        return "popen failed!";
    }
    while (fgets(buffer.data(), buffer.size(), pipe.get()) != nullptr) {
        result += buffer.data();
    }
    return result;
}

static void upload_file(const std::string& logFile) {
    std::string command =
        "curl --include --request POST https://flightlessmango.com/logs "
        "-F 'log[game_id]=26506' -F 'log[user_id]=176' -F 'attachment=true' "
        "-A 'mangohud' ";
    command += " -F 'log[uploads][]=@" + logFile + "'";
    command += " | grep Location | cut -c11-";

    std::string url = exec(command);
    std::cout << "upload url: " << url;
    exec("xdg-open " + url);
}

static void upload_files(const std::vector<std::string>& logFiles) {
    std::string command =
        "curl --include --request POST https://flightlessmango.com/logs "
        "-F 'log[game_id]=26506' -F 'log[user_id]=176' -F 'attachment=true' "
        "-A 'mangohud' ";
    for (const auto& file : logFiles)
        command += " -F 'log[uploads][]=@" + file + "'";

    command += " | grep Location | cut -c11-";
    std::string url = exec(command);
    std::cout << "upload url: " << url;
    exec("xdg-open " + url);
}

#else
// ======================= ANDROID용: 전부 스텁 처리 =======================

std::string exec(std::string) {
    // Android에서는 popen / curl / xdg-open 경로 자체를 안 씀
    return {};
}

static void upload_file(const std::string&) {
    // no-op
}

static void upload_files(const std::vector<std::string>&) {
    // no-op
}

#endif // __ANDROID__

static bool compareByFps(const logData &a, const logData &b)
{
    return a.fps < b.fps;
}

static void writeSummary(std::string filename){
    auto& logArray = logger->get_log_data();
    // if the log is stopped/started too fast we might end up with an empty vector.
    // in that case, just bail.
    if (logArray.empty()){
        logger->stop_logging();
        return;
    }

    filename = filename.substr(0, filename.size() - 4);
    filename += "_summary.csv";
    SPDLOG_INFO("{}", filename);
    SPDLOG_DEBUG("Writing summary log file [{}]", filename);

    std::ofstream out(filename, std::ios::out | std::ios::app);
    if (out){
        out << "0.1% Min FPS," << "1% Min FPS," << "97% Percentile FPS," 
            << "Average FPS," << "GPU Load," << "CPU Load," << "Average Frame Time,"
            << "Average GPU Temp," << "Average CPU Temp," << "Average VRAM Used,"
            << "Average RAM Used," << "Average Swap Used," << "Peak GPU Load,"
            << "Peak CPU Load," << "Peak GPU Temp," << "Peak CPU Temp,"
            << "Peak VRAM Used," << "Peak RAM Used," << "Peak Swap Used" << "\n";

        std::vector<logData> sorted = logArray;
        std::sort(sorted.begin(), sorted.end(), compareByFps);

        float total          = 0.0f;
        float total_gpu      = 0.0f;
        float total_cpu      = 0.0f;
        int   total_gpu_temp = 0;
        int   total_cpu_temp = 0;
        float total_vram     = 0.0f;
        float total_ram      = 0.0f;
        float total_swap     = 0.0f;

        int   peak_gpu       = 0;
        float peak_cpu       = 0.0f;
        int   peak_gpu_temp  = 0;
        int   peak_cpu_temp  = 0;
        float peak_vram      = 0.0f;
        float peak_ram       = 0.0f;
        float peak_swap      = 0.0f;

        std::vector<float> fps_values;
        fps_values.reserve(sorted.size());
        for (auto& data : sorted)
            fps_values.push_back(data.frametime);

        std::vector<std::string> metrics {"0.001", "0.01", "0.97"};
        auto fpsmetrics = std::make_unique<fpsMetrics>(metrics, fps_values);
        auto metrics_copy = fpsmetrics->copy_metrics();
        for (auto& metric : metrics_copy)
            out << metric.value << ",";

        fpsmetrics.reset();

        for (const auto& input : sorted){
            total          += input.frametime;
            total_gpu      += input.gpu_load;
            total_cpu      += input.cpu_load;
            total_gpu_temp += input.gpu_temp;
            total_cpu_temp += input.cpu_temp;
            total_vram     += input.gpu_vram_used;
            total_ram      += input.ram_used;
            total_swap     += input.swap_used;

            peak_gpu       = std::max(peak_gpu, input.gpu_load);
            peak_cpu       = std::max(peak_cpu, input.cpu_load);
            peak_gpu_temp  = std::max(peak_gpu_temp, input.gpu_temp);
            peak_cpu_temp  = std::max(peak_cpu_temp, input.cpu_temp);
            peak_vram      = std::max(peak_vram, input.gpu_vram_used);
            peak_ram       = std::max(peak_ram, input.ram_used);
            peak_swap      = std::max(peak_swap, input.swap_used);
        }

        // Average FPS
        float result = 1000.0f / (total / static_cast<float>(sorted.size()));
        out << std::fixed << std::setprecision(1) << result << ",";
        // GPU Load (Average)
        result = total_gpu / static_cast<float>(sorted.size());
        out << result << ",";
        // CPU Load (Average)
        result = total_cpu / static_cast<float>(sorted.size());
        out << result << ",";
        // Average Frame Time
        result = total / static_cast<float>(sorted.size());
        out << result << ",";
        // Average GPU Temp
        result = static_cast<float>(total_gpu_temp) / static_cast<float>(sorted.size());
        out << result << ",";
        // Average CPU Temp
        result = static_cast<float>(total_cpu_temp) / static_cast<float>(sorted.size());
        out << result << ",";
        // Average VRAM Used
        result = total_vram / static_cast<float>(sorted.size());
        out << result << ",";
        // Average RAM Used
        result = total_ram / static_cast<float>(sorted.size());
        out << result << ",";
        // Average Swap Used
        result = total_swap / static_cast<float>(sorted.size());
        out << result << ",";
        // Peak GPU Load
        out << peak_gpu << ",";
        // Peak CPU Load
        out << peak_cpu << ",";
        // Peak GPU Temp
        out << peak_gpu_temp << ",";
        // Peak CPU Temp
        out << peak_cpu_temp << ",";
        // Peak VRAM Used
        out << peak_vram << ",";
        // Peak RAM Used
        out << peak_ram << ",";
        // Peak Swap Used
        out << peak_swap;
    } else {
        SPDLOG_ERROR("Failed to write log file");
    }
    out.close();
}

static void writeFileHeaders(std::ofstream& out){
    auto params = get_params();  
    if (params->enabled[OVERLAY_PARAM_ENABLED_log_versioning]){
        out << "v1" << std::endl;
        out << MANGOHUD_VERSION << std::endl;
        out << "---------------------SYSTEM INFO---------------------" << std::endl;
    }

    out << "os," << "cpu," << "gpu," << "ram," << "kernel," << "driver," << "cpuscheduler" << std::endl;
    out << os << "," << cpu << "," << gpu << "," << ram << "," << kernel << "," << driver << "," << cpusched << std::endl;

    if (params->enabled[OVERLAY_PARAM_ENABLED_log_versioning])
        out << "--------------------FRAME METRICS--------------------" << std::endl;

    out << "fps," << "frametime," << "cpu_load," << "cpu_power," << "gpu_load,"
        << "cpu_temp," << "gpu_temp," << "gpu_core_clock," << "gpu_mem_clock,"
        << "gpu_vram_used," << "gpu_power," << "ram_used," << "swap_used,"
        << "process_rss," << "cpu_mhz," << "elapsed" << std::endl;
}

void Logger::writeToFile()
{
    // 방어용: 이론상 비면 안 되지만, 안전하게
    if (m_log_files.empty())
        return;

    if (!output_file.is_open()) {
        // [안드로이드 포함 공통] 파일 오픈 시도
        // std::ofstream::open()은 기본적으로 예외 안 던지지만,
        // 외부에서 exceptions() 설정했을 가능성 있기에 try/catch는 유지.
        try {
            output_file.open(m_log_files.back(), std::ios::out | std::ios::app);
        } catch (...) {
            static bool log_fail_once = false;
            if (!log_fail_once) {
                SPDLOG_ERROR("Logger: failed to open log file (exception): {}", m_log_files.back());
                log_fail_once = true;
            }
            return;
        }

        if (!output_file.good()) {
            static bool log_fail_once2 = false;
            if (!log_fail_once2) {
                SPDLOG_ERROR("Logger: failed to open log file (bad stream): {}", m_log_files.back());
                log_fail_once2 = true;
            }
            return;
        }

        // 여기까지 왔으면 정상 오픈
        writeFileHeaders(output_file);
    }

    auto& logArray = logger->get_log_data();
    if (!output_file.good() || logArray.empty())
        return;

    const auto& back = logArray.back();

    // per-frame flush 제거된 버전
    output_file << back.fps << ","
                << back.frametime << ","
                << back.cpu_load << ","
                << back.cpu_power << ","
                << back.gpu_load << ","
                << back.cpu_temp << ","
                << back.gpu_temp << ","
                << back.gpu_core_clock << ","
                << back.gpu_mem_clock << ","
                << back.gpu_vram_used << ","
                << back.gpu_power << ","
                << back.ram_used << ","
                << back.swap_used << ","
                << back.process_rss << ","
                << back.cpu_mhz << ","
                << std::chrono::duration_cast<std::chrono::nanoseconds>(back.previous).count()
                << "\n";
    // flush 없음: 안드로이드 I/O 목 조르던 쓰레기 호출 제거
}

static std::string get_log_suffix(){
    std::time_t now_log = std::time(nullptr);
    std::tm *log_time = std::localtime(&now_log);
    std::ostringstream buffer;
    buffer << std::put_time(log_time, "%Y-%m-%d_%H-%M-%S") << ".csv";
    return buffer.str();
}

Logger::Logger(const overlay_params* in_params)
  : output_folder(in_params->output_folder),
    log_interval(in_params->log_interval),
    log_duration(in_params->log_duration),
    m_logging_on(false),
    m_values_valid(false)
{
    if (output_folder.empty()) {
        if (const char* home = std::getenv("HOME"))
            output_folder = home;
        else
            output_folder = ".";
    }
    m_log_end = Clock::now() - std::chrono::seconds(15);
    SPDLOG_DEBUG("Logger constructed!");
}

void Logger::start_logging() {
    if (m_logging_on) return;
    m_values_valid = false;
    m_logging_on   = true;
    m_log_start    = Clock::now();

    std::string program = get_wine_exe_name();
    if (program.empty())
        program = get_program_name();

    m_log_files.emplace_back(output_folder + "/" + program + "_" + get_log_suffix());

    if (log_interval != 0) {
        // 이전 로그 스레드 남아있으면 정리
        if (log_thread.joinable())
            log_thread.join();

        log_thread = std::thread(&Logger::logging, this);
#if !defined(__APPLE__) && !defined(_WIN32)
        pthread_setname_np(log_thread.native_handle(), "mangohud-log");
#endif
    }
}

void Logger::stop_logging()
{
    if (!m_logging_on)
        return;

    m_logging_on = false;
    m_log_end    = Clock::now();

    // log_interval == 0이면 별도 스레드 없을 수 있음
    if (log_interval != 0 &&
        log_thread.joinable() &&
        std::this_thread::get_id() != log_thread.get_id())
    {
        log_thread.join();
    }

    calculate_benchmark_data();

    // [변경] 종료 시에만 flush + close
    try {
        if (output_file.is_open()) {
            output_file.flush();
            output_file.close();
        }
    } catch (...) {
        SPDLOG_INFO("Logger: something went wrong when closing output_file");
    }

    if (!m_log_files.empty())
        writeSummary(m_log_files.back());
    else
        SPDLOG_INFO("Logger: can't write summary because m_log_files is empty");

    clear_log_data();

#if defined(__linux__) && !defined(__ANDROID__)
    // 안드로이드에선 control 클라이언트도 의미 없음
    control_client_check(get_params()->control, global_control_client, gpu.c_str());
    const char * cmd = "LoggingFinished";
    control_send(global_control_client, cmd, std::strlen(cmd), 0, 0);
#endif
}

void Logger::logging(){
    wait_until_data_valid();
    while (is_active()){
        try_log();
        std::this_thread::sleep_for(std::chrono::milliseconds(log_interval));
    }
}

void Logger::try_log() {
    if (!is_active()) return;
    if (!m_values_valid) return;

    auto now = Clock::now();
    auto elapsedLog = now - m_log_start;

    currentLogData.previous   = elapsedLog;
    currentLogData.fps        = fps;
    currentLogData.frametime  = frametime;
    m_log_array.push_back(currentLogData);
    writeToFile();

    if (log_duration && (elapsedLog >= std::chrono::seconds(log_duration))){
        stop_logging();
    }
}

void Logger::wait_until_data_valid() {
    std::unique_lock<std::mutex> lck(m_values_valid_mtx);
    while (!m_values_valid)
        m_values_valid_cv.wait(lck);
}

void Logger::notify_data_valid() {
    std::unique_lock<std::mutex> lck(m_values_valid_mtx);
    m_values_valid = true;
    m_values_valid_cv.notify_all();
}

#if !defined(__ANDROID__)
void Logger::upload_last_log() {
    if (m_log_files.empty()) return;
    std::thread(upload_file, m_log_files.back()).detach();
}

void Logger::upload_last_logs() {
    if (m_log_files.empty()) return;
    std::thread(upload_files, m_log_files).detach();
}
#else
// Android: 업로드 관련 기능 완전 무시
void Logger::upload_last_log()  {}
void Logger::upload_last_logs() {}
#endif

void autostart_log(int sleep) {
    // os_time_sleep() causes freezes with zink + autologging :frog_donut:
    std::this_thread::sleep_for(std::chrono::seconds(sleep));
    logger->start_logging();
}

void Logger::calculate_benchmark_data(){
    std::vector<float> fps_values;
    fps_values.reserve(m_log_array.size());
    for (auto& point : m_log_array)
        fps_values.push_back(point.frametime);

    benchmark.percentile_data.clear();

    std::vector<std::string> metrics {"0.97", "avg", "0.01", "0.001"};
    auto params = get_params();
    if (!params->fps_metrics.empty())
        metrics = params->fps_metrics;
    
    auto fpsmetrics = std::make_unique<fpsMetrics>(metrics, fps_values);
    auto metrics_copy = fpsmetrics->copy_metrics();
    for (auto& metric : metrics_copy)
        benchmark.percentile_data.push_back({metric.display_name, metric.value});

    fpsmetrics.reset();
}

#include "file_utils.h"
#include "string_utils.h"
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <dirent.h>
#include <limits.h>
#include <fstream>
#include <cstring>
#include <string>
#include <algorithm>
#include <regex>
#include <cctype>
#include <cerrno>
#include <spdlog/spdlog.h>

#ifndef PROCDIR
#define PROCDIR "/proc"
#endif

std::string read_line(const std::string& filename)
{
    std::string line;
    std::ifstream file(filename);
    if (file.fail()){
        return line;
    }
    std::getline(file, line);
    return line;
}

std::string get_basename(const std::string& path)
{
    auto npos = path.find_last_of("/\\");
    if (npos == std::string::npos)
        return path;

    if (npos < path.size() - 1)
        return path.substr(npos + 1);
    return path;
}

#ifdef __linux__

std::vector<std::string> ls(const char* root, const char* prefix, LS_FLAGS flags)
{
    std::vector<std::string> list;
    struct dirent* dp;

    DIR* dirp = opendir(root);
    if (!dirp) {
        int err = errno;

        if (err == EACCES || err == EPERM) {
            SPDLOG_DEBUG("Skipping directory '{}' due to permissions: {}", root, strerror(err));
            return list;
        }
        if (err == ENOENT || err == ENOTDIR) {
            SPDLOG_DEBUG("Directory '{}' not present: {}", root, strerror(err));
            return list;
        }

        SPDLOG_ERROR("Error opening directory '{}': {}", root, strerror(err));
        return list;
    }

    while ((dp = readdir(dirp))) {
        if ((prefix && !starts_with(dp->d_name, prefix))
            || !strcmp(dp->d_name, ".")
            || !strcmp(dp->d_name, ".."))
            continue;

        switch (dp->d_type) {
        case DT_LNK: {
            struct stat s;
            std::string path(root);
            if (path.back() != '/')
                path += "/";
            path += dp->d_name;

            if (stat(path.c_str(), &s))
                continue;

            if (((flags & LS_DIRS) && S_ISDIR(s.st_mode))
                || ((flags & LS_FILES) && S_ISREG(s.st_mode))) {
                list.push_back(dp->d_name);
            }
            break;
        }
        case DT_DIR:
            if (flags & LS_DIRS)
                list.push_back(dp->d_name);
            break;
        case DT_REG:
            if (flags & LS_FILES)
                list.push_back(dp->d_name);
            break;
        default:
            break;
        }
    }

    closedir(dirp);
    return list;
}

bool file_exists(const std::string& path)
{
    struct stat s;
    return !stat(path.c_str(), &s) && !S_ISDIR(s.st_mode);
}

bool dir_exists(const std::string& path)
{
    struct stat s;
    return !stat(path.c_str(), &s) && S_ISDIR(s.st_mode);
}

std::string read_symlink(const char* link)
{
    char result[PATH_MAX] {};
    ssize_t count = readlink(link, result, PATH_MAX);
    if (count <= 0)
        return {};
    return std::string(result, static_cast<size_t>(count));
}

std::string read_symlink(const std::string& link)
{
    return read_symlink(link.c_str());
}

std::string get_exe_path()
{
    return read_symlink(PROCDIR "/self/exe");
}

std::string get_wine_exe_name(bool keep_ext)
{
    const std::string exe_path = get_exe_path();
    if (!ends_with(exe_path, "wine-preloader") && !ends_with(exe_path, "wine64-preloader")) {
        return std::string();
    }

    std::string line = read_line(PROCDIR "/self/comm"); // max 16 characters though
    if (ends_with(line, ".exe", true))
    {
        auto dot = keep_ext ? std::string::npos : line.find_last_of('.');
        return line.substr(0, dot);
    }

    std::ifstream cmdline(PROCDIR "/self/cmdline");
    // Iterate over arguments (separated by NUL byte).
    while (std::getline(cmdline, line, '\0')) {
        auto n = std::string::npos;
        if (!line.empty()
            && ((n = line.find_last_of("/\\")) != std::string::npos)
            && n < line.size() - 1) // have at least one character
        {
            auto dot = keep_ext ? std::string::npos : line.find_last_of('.');
            if (dot < n)
                dot = line.size();
            return line.substr(n + 1, dot - n - 1);
        }
        else if (ends_with(line, ".exe", true))
        {
            auto dot = keep_ext ? std::string::npos : line.find_last_of('.');
            return line.substr(0, dot);
        }
    }
    return std::string();
}

std::string get_home_dir()
{
    std::string path;
    const char* p = getenv("HOME");

    if (p)
        path = p;
    return path;
}

std::string get_data_dir()
{
    const char* p = getenv("XDG_DATA_HOME");
    if (p)
        return p;

    std::string path = get_home_dir();
    if (!path.empty())
        path += "/.local/share";
    return path;
}

std::string get_config_dir()
{
    const char* p = getenv("XDG_CONFIG_HOME");
    if (p)
        return p;

    std::string path = get_home_dir();
    if (!path.empty())
        path += "/.config";
    return path;
}

bool lib_loaded(const std::string& lib, pid_t pid)
{
    // 검색할 패턴은 미리 lowercase
    const std::string needle = to_lower(lib);
    std::string who;

#if defined(__ANDROID__)
    // Android: "self"만 허용, 나머지는 바로 포기
    pid_t self = getpid();
    if (pid != -1 && pid != self) {
        SPDLOG_DEBUG("lib_loaded: skipping scan for pid={} on Android (self only)", pid);
        return false;
    }

    who = std::to_string(self);
    fs::path base = fs::path("/proc") / who;

    // Android: /proc/<pid>/fd 만 체크 (map_files는 대개 권한/의미 둘 다 애매)
    auto paths = { base / "fd" };

#else
    // 일반 리눅스: 요청 pid 또는 self 사용
    if (pid == -1)
        who = "self";
    else
        who = std::to_string(pid);

    fs::path base = fs::path("/proc") / who;
    auto paths = { base / "map_files", base / "fd" };
#endif

    for (auto& path : paths) {
        const auto path_str = path.string();

        if (!dir_exists(path_str)) {
            SPDLOG_DEBUG("lib_loaded: tried to access path that doesn't exist {}", path_str);
            continue;
        }

        try {
            for (auto& p : fs::directory_iterator(path)) {
                const auto file = p.path().string();
                const auto sym  = read_symlink(file.c_str());
                if (sym.empty())
                    continue;

                if (to_lower(sym).find(needle) != std::string::npos) {
                    return true;
                }
            }
        }
        catch (const fs::filesystem_error& e) {
            SPDLOG_DEBUG("lib_loaded: cannot scan '{}': {}", path_str, e.what());
            continue;
        }
    }

    return false;
}

std::string remove_parentheses(const std::string& text) {
    // Remove parentheses and text between them
    std::regex pattern("\\([^)]*\\)");
    return std::regex_replace(text, pattern, "");
}

std::string to_lower(const std::string& str) {
    std::string lowered = str;
    std::transform(lowered.begin(), lowered.end(), lowered.begin(),
                   [](unsigned char c) { return std::tolower(c); });
    return lowered;
}

#endif // __linux__

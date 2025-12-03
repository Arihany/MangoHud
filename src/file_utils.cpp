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
    if (!file.is_open())
        return line;

    std::getline(file, line);
    return line;
}

std::string get_basename(const std::string& path)
{
    const auto npos = path.find_last_of("/\\");
    if (npos == std::string::npos)
        return path;

    if (npos + 1 < path.size())
        return path.substr(npos + 1);

    return path;
}

#ifdef __linux__

std::vector<std::string> ls(const char* root, const char* prefix, LS_FLAGS flags)
{
    std::vector<std::string> list;

    DIR* dirp = opendir(root);
    if (!dirp) {
        const int err = errno;

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

    struct dirent* dp = nullptr;
    while ((dp = readdir(dirp))) {
        const char* name = dp->d_name;

        if ((prefix && !starts_with(name, prefix)) ||
            std::strcmp(name, ".")  == 0 ||
            std::strcmp(name, "..") == 0)
            continue;

        switch (dp->d_type) {
        case DT_LNK: {
            struct stat s;
            std::string path(root);
            if (!path.empty() && path.back() != '/')
                path += '/';
            path += name;

            if (stat(path.c_str(), &s) != 0)
                continue;

            if (((flags & LS_DIRS)  && S_ISDIR(s.st_mode)) ||
                ((flags & LS_FILES) && S_ISREG(s.st_mode))) {
                list.emplace_back(name);
            }
            break;
        }
        case DT_DIR:
            if (flags & LS_DIRS)
                list.emplace_back(name);
            break;
        case DT_REG:
            if (flags & LS_FILES)
                list.emplace_back(name);
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
    return ::stat(path.c_str(), &s) == 0 && !S_ISDIR(s.st_mode);
}

bool dir_exists(const std::string& path)
{
    struct stat s;
    return ::stat(path.c_str(), &s) == 0 && S_ISDIR(s.st_mode);
}

std::string read_symlink(const char* link)
{
    char result[PATH_MAX] = {};
    const ssize_t count = ::readlink(link, result, sizeof(result));
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
    if (!ends_with(exe_path, "wine-preloader") &&
        !ends_with(exe_path, "wine64-preloader")) {
        return {};
    }

    // 먼저 comm 확인 (16자 제한)
    std::string line = read_line(PROCDIR "/self/comm");
    if (ends_with(line, ".exe", true)) {
        const auto dot = keep_ext ? std::string::npos : line.find_last_of('.');
        return line.substr(0, dot);
    }

    // cmdline 인자들에서 exe 후보 찾기
    std::ifstream cmdline(PROCDIR "/self/cmdline");
    while (std::getline(cmdline, line, '\0')) {
        std::size_t n = std::string::npos;

        if (!line.empty() &&
            (n = line.find_last_of("/\\")) != std::string::npos &&
            n + 1 < line.size()) {

            auto dot = keep_ext ? std::string::npos : line.find_last_of('.');
            if (dot != std::string::npos && dot < n)
                dot = line.size();

            const std::size_t start = n + 1;
            const std::size_t len   = (dot == std::string::npos)
                                      ? std::string::npos
                                      : (dot - start);
            return line.substr(start, len);
        }

        if (ends_with(line, ".exe", true)) {
            const auto dot = keep_ext ? std::string::npos : line.find_last_of('.');
            return line.substr(0, dot);
        }
    }

    return {};
}

std::string get_home_dir()
{
    std::string path;
    if (const char* p = std::getenv("HOME"))
        path = p;
    return path;
}

std::string get_data_dir()
{
    if (const char* p = std::getenv("XDG_DATA_HOME"))
        return p;

    std::string path = get_home_dir();
    if (!path.empty())
        path += "/.local/share";
    return path;
}

std::string get_config_dir()
{
    if (const char* p = std::getenv("XDG_CONFIG_HOME"))
        return p;

    std::string path = get_home_dir();
    if (!path.empty())
        path += "/.config";
    return path;
}

bool lib_loaded(const std::string& lib, pid_t pid)
{
    // 검색 대상은 한 번만 lowercase
    const std::string needle = to_lower(lib);
    std::string who;

#if defined(__ANDROID__)
    // Android: self만 허용
    const pid_t self = ::getpid();
    if (pid != -1 && pid != self) {
        SPDLOG_DEBUG("lib_loaded: skipping scan for pid={} on Android (self only)", pid);
        return false;
    }

    who = std::to_string(self);
    std::string base = std::string(PROCDIR) + "/" + who;
    std::string paths[1] = { base + "/fd" };

#else
    // 일반 리눅스: 요청 pid 또는 self 사용
    if (pid == -1)
        who = "self";
    else
        who = std::to_string(pid);

    std::string base = std::string(PROCDIR) + "/" + who;
    std::string paths[2] = {
        base + "/map_files",
        base + "/fd"
    };
#endif

    for (const auto& path : paths) {
        if (path.empty())
            continue;

        if (!dir_exists(path)) {
            SPDLOG_DEBUG("lib_loaded: tried to access path that doesn't exist {}", path);
            continue;
        }

        DIR* dirp = ::opendir(path.c_str());
        if (!dirp) {
            SPDLOG_DEBUG("lib_loaded: cannot open '{}': {}", path, std::strerror(errno));
            continue;
        }

        struct dirent* dp = nullptr;

        // 엔트리 경로 조립용 버퍼 (재사용)
        std::string entry;
        entry.reserve(path.size() + 2 + NAME_MAX);

        while ((dp = ::readdir(dirp))) {
            // skip . ..
            if (std::strcmp(dp->d_name, ".") == 0 ||
                std::strcmp(dp->d_name, "..") == 0)
                continue;

            entry.assign(path);
            if (!entry.empty() && entry.back() != '/')
                entry.push_back('/');
            entry.append(dp->d_name);

            const std::string target = read_symlink(entry.c_str());
            if (target.empty())
                continue;

            // case-insensitive substring 검사 (target 안에 needle 들어있는지)
            auto it = std::search(
                target.begin(), target.end(),
                needle.begin(), needle.end(),
                [](unsigned char ch1, unsigned char ch2) {
                    return static_cast<char>(std::tolower(ch1)) == ch2;
                }
            );

            if (it != target.end()) {
                ::closedir(dirp);
                return true;
            }
        }

        ::closedir(dirp);
    }

    return false;
}

std::string remove_parentheses(const std::string& text)
{
    // 괄호 안의 내용 전체 제거 (중첩 포함)
    std::string out;
    out.reserve(text.size());

    int depth = 0;
    for (char ch : text) {
        if (ch == '(') {
            ++depth;
        } else if (ch == ')') {
            if (depth > 0)
                --depth;
        } else if (depth == 0) {
            out.push_back(ch);
        }
    }

    return out;
}

std::string to_lower(const std::string& str)
{
    std::string lowered = str;
    std::transform(lowered.begin(), lowered.end(), lowered.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return lowered;
}

#endif // __linux__

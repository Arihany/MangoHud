#pragma once
#ifndef MANGOHUD_FILE_UTILS_H
#define MANGOHUD_FILE_UTILS_H

#include <string>
#include <vector>
#include <regex>
#include <array>
#include <filesystem.h>
namespace fs = ghc::filesystem;

enum LS_FLAGS
{
    LS_DIRS  = 0x01,
    LS_FILES = 0x02,
};

std::string read_line(const std::string& filename);

std::vector<std::string> ls(const char* root,
                            const char* prefix = nullptr,
                            LS_FLAGS flags = LS_DIRS);

bool file_exists(const std::string& path);
bool dir_exists(const std::string& path);

std::string read_symlink(const char* link);
// ★ rvalue 말고 ref로 통일
std::string read_symlink(const std::string& link);

// ★ 이것도 ref로 통일
std::string get_basename(const std::string& path); // name만 basename과 안 겹치게 한 거지, rvalue랑 아무 상관 없음

std::string get_exe_path();
std::string get_wine_exe_name(bool keep_ext = false);
std::string get_home_dir();
std::string get_data_dir();
std::string get_config_dir();
bool lib_loaded(const std::string& lib, pid_t pid);
std::string remove_parentheses(const std::string&);
std::string to_lower(const std::string& str);

#endif // MANGOHUD_FILE_UTILS_H

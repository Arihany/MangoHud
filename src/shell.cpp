#include "shell.h"
#include <spdlog/spdlog.h>

#if defined(__ANDROID__)

// ============================
// Android stub implementation
// ============================

Shell::Shell() {
    success = false;
    runtime = false;
    // 파이프/프로세스 안 만든다
    SPDLOG_DEBUG("Shell: disabled on Android (no external shell exec)");
}

Shell::~Shell() {
    // 아무것도 안 함
}

std::string Shell::exec(std::string cmd) {
    (void)cmd;
    // 안드로이드에선 그냥 사용 불가
    return {};
}

void Shell::writeCommand(std::string command) {
    (void)command;
    // no-op
}

#else

// ============================
// 기존 리눅스 구현 (조금 손질)
// ============================

#include <thread>
#include <iostream>
#include <sys/wait.h>
#include <array>
#include <fcntl.h>
#include <unistd.h>
#include <errno.h>
#include <string.h>
#include "string_utils.h"

std::string Shell::readOutput() {
    std::this_thread::sleep_for(std::chrono::milliseconds(50));

    std::array<char, 128> buffer;
    std::string result;
    ssize_t count;
    while ((count = ::read(from_shell[0], buffer.data(), buffer.size())) > 0) {
        result.append(buffer.data(), count);
    }

    std::istringstream stream(result);
    std::string line;
    std::string last_line;
    while (std::getline(stream, line)) {
        last_line = line;
    }

    SPDLOG_DEBUG("Shell: recieved output: {}", last_line);
    return last_line;
}

Shell::Shell() {
    success = false;
    runtime = false;
    shell_pid = -1;
    to_shell[0] = to_shell[1] = -1;
    from_shell[0] = from_shell[1] = -1;

    if (stat("/run/pressure-vessel", &stat_buffer) == 0)
        runtime = true;

    bool failed = false;

    if (pipe(to_shell) == -1) {
        SPDLOG_ERROR("Failed to create to_shell pipe: {}", strerror(errno));
        failed = true;
    }

    if (pipe(from_shell) == -1) {
        SPDLOG_ERROR("Failed to create from_shell pipe: {}", strerror(errno));
        failed = true;
    }

    if (failed) {
        SPDLOG_ERROR("Shell has failed, will not be able to use exec");
        return;
    }

    shell_pid = fork();

    if (shell_pid == 0) { // Child process
        close(to_shell[1]);
        close(from_shell[0]);

        dup2(to_shell[0], STDIN_FILENO);
        dup2(from_shell[1], STDOUT_FILENO);
        dup2(from_shell[1], STDERR_FILENO);
        execl("/bin/sh", "sh", "-c", "unset LD_PRELOAD; exec /bin/sh", nullptr);
        _exit(1);
    } else if (shell_pid > 0) {
        close(to_shell[0]);
        close(from_shell[1]);
        setNonBlocking(from_shell[0]);
        success = true;
    } else {
        SPDLOG_ERROR("fork() for Shell failed: {}", strerror(errno));
    }
}

std::string Shell::exec(std::string cmd) {
    if (!success)
        return {};
    writeCommand(cmd);
    return readOutput();
}

void Shell::writeCommand(std::string command) {
    if (write(to_shell[1], command.c_str(), command.length()) == -1)
        SPDLOG_ERROR("Failed to write to shell");

    if (runtime)
        command = "steam-runtime-launch-client --alongside-steam --host -- " + command;

    trim(command);
    SPDLOG_DEBUG("Shell: wrote command: {}", command);
}

Shell::~Shell() {
    if (to_shell[1] != -1)
        close(to_shell[1]);
    if (from_shell[0] != -1)
        close(from_shell[0]);

    if (shell_pid > 0)
        waitpid(shell_pid, nullptr, 0);
}

#endif

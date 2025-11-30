#include <chrono>
#include <unistd.h>
#include <fcntl.h>
#include <sys/types.h>
#include <sys/inotify.h>
#include <spdlog/spdlog.h>
#include "config.h"
#include "notify.h"

#define EVENT_SIZE   (sizeof(struct inotify_event))
#define EVENT_BUF_LEN (1024 * (EVENT_SIZE + 16))

static void fileChanged(notify_thread *nt) {
    char buffer[EVENT_BUF_LEN];
    overlay_params local_params = *nt->params;

    while (!nt->quit) {
        // 논블로킹 read 결과 체크
        int length = read(nt->fd, buffer, EVENT_BUF_LEN);
        if (length <= 0) {
            // EAGAIN 등: 이벤트 없음 -> 500ms 슬립 후 재시도
            std::this_thread::sleep_for(std::chrono::milliseconds(500));
            continue;
        }

        int i = 0;
        while (i < length) {
            auto *event = reinterpret_cast<struct inotify_event *>(&buffer[i]);
            i += EVENT_SIZE + event->len;

            if (event->mask & (IN_MODIFY | IN_DELETE_SELF)) {
                // 파일 덮어쓰기 여유 시간
                std::this_thread::sleep_for(std::chrono::milliseconds(100));

                parse_overlay_config(&local_params,
                                     getenv("MANGOHUD_CONFIG"),
                                     false);

                // 파일이 삭제 후 새 파일로 교체되었거나, 경로가 바뀐 경우 watch 교체
                if ((event->mask & IN_DELETE_SELF) ||
                    (nt->params->config_file_path != local_params.config_file_path)) {

                    SPDLOG_DEBUG("Watching config file: {}",
                                 local_params.config_file_path.c_str());

                    inotify_rm_watch(nt->fd, nt->wd);
                    nt->wd = inotify_add_watch(
                        nt->fd,
                        local_params.config_file_path.c_str(),
                        IN_MODIFY | IN_DELETE_SELF
                    );
                }

                {
                    std::lock_guard<std::mutex> lk(nt->mutex);
                    *nt->params = local_params;
                }
            }
        }

        // 이벤트 한 번 털었으면 최소 500ms 쉰다
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
    }
}

bool start_notifier(notify_thread& nt)
{
    nt.fd = inotify_init1(IN_NONBLOCK);
    if (nt.fd < 0) {
        SPDLOG_ERROR("inotify_init1 failed: {}", strerror(errno));
        return false;
    }

    nt.wd = inotify_add_watch(nt.fd, nt.params->config_file_path.c_str(), IN_MODIFY | IN_DELETE_SELF);
    if (nt.wd < 0) {
        close(nt.fd);
        nt.fd = -1;
        return false;
    }

    if (nt.thread.joinable())
        nt.thread.join();
    nt.thread = std::thread(fileChanged, &nt);
    return true;
}

void stop_notifier(notify_thread& nt)
{
    if (nt.fd < 0)
        return;

    nt.quit = true;
    if (nt.thread.joinable())
        nt.thread.join();
    inotify_rm_watch(nt.fd, nt.wd);
    close(nt.fd);
    nt.fd = -1;
}

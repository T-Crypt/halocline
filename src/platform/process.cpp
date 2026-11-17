// src/platform/process.cpp - see include/halo/platform/process.hpp.
#include "halo/platform/process.hpp"

#include "halo/platform/file.hpp"

#include <cerrno>
#include <chrono>
#include <csignal>
#include <cstdlib>
#include <string>
#include <vector>

#include <fcntl.h>
#include <poll.h>
#include <spawn.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

namespace halo::platform {

std::string quote_arg(const std::string& s) {
    // An argument list is one string here, so an argument that holds a space or a quote has to be quoted and its
    // backslashes kept meaningful: they escape quotes, so a run of them before a quote - or before the closing
    // quote - has to be doubled.
    if (s.empty()) return "\"\"";
    if (s.find_first_of(" \t\"") == std::string::npos) return s;
    std::string out = "\"";
    size_t backslashes = 0;
    for (char c : s) {
        if (c == '\\') { ++backslashes; out += c; continue; }
        if (c == '"') { out.append(backslashes + 1, '\\'); out += '"'; backslashes = 0; continue; }
        backslashes = 0;
        out += c;
    }
    out.append(backslashes, '\\');
    out += '"';
    return out;
}

namespace {
/// argv as the exec and spawn calls want it: writable elements, terminated by a null.
std::vector<char*> c_argv(std::vector<std::string>& args) {
    std::vector<char*> av;
    av.reserve(args.size() + 1);
    for (std::string& s : args) av.push_back(s.data());
    av.push_back(nullptr);
    return av;
}

int exited(int status) {
    if (WIFEXITED(status)) return WEXITSTATUS(status);
    if (WIFSIGNALED(status)) return 128 + WTERMSIG(status);
    return -1;
}
}  // namespace

bool spawn_piped(const std::vector<std::string>& argv, const std::string& stderr_path, PipedChild& out,
                 std::string& err) {
    out = PipedChild{};
    if (argv.empty()) { err = "no program to start"; return false; }
    int in[2], outp[2];
    if (pipe(in) != 0 || pipe(outp) != 0) { err = "pipe failed"; return false; }
    std::vector<std::string> store = argv;
    std::vector<char*> av = c_argv(store);
    const pid_t pid = fork();
    if (pid < 0) { err = "fork failed"; return false; }
    if (pid == 0) {
        dup2(in[0], 0);
        dup2(outp[1], 1);
        const int lf = open(stderr_path.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
        if (lf >= 0) dup2(lf, 2);
        close(in[0]); close(in[1]); close(outp[0]); close(outp[1]);
        execv(store[0].c_str(), av.data());
        _exit(127);
    }
    close(in[0]);
    close(outp[1]);
    out.in_fd = in[1];
    out.out_fd = outp[0];
    out.child.id = (int64_t) pid;
    return true;
}

bool pipe_write_all(int fd, const void* data, size_t n) {
    const char* p = (const char*) data;
    size_t off = 0;
    while (off < n) {
        const ssize_t w = ::write(fd, p + off, n - off);
        if (w < 0) {
            if (errno == EINTR) continue;
            return false;
        }
        off += (size_t) w;
    }
    return true;
}

int64_t pipe_read(int fd, void* dst, size_t n) {
    for (;;) {
        const ssize_t r = ::read(fd, dst, n);
        if (r < 0 && errno == EINTR) continue;
        return (int64_t) r;
    }
}

bool pipe_wait_readable(int fd, int timeout_ms) {
    pollfd p{fd, POLLIN, 0};
    const int r = ::poll(&p, 1, timeout_ms);
    return r != 0;   // an interrupted or failed poll lets the read report it
}

int child_wait(Child& c, int timeout_ms) {
    if (c.id <= 0) return -1;
    const bool block = timeout_ms < 0;
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
    for (;;) {
        int status = 0;
        const pid_t r = waitpid((pid_t) c.id, &status, block ? 0 : WNOHANG);
        if (r == (pid_t) c.id) {
            c.id = -1;   // reaped: a second wait would report the child as gone
            return exited(status);
        }
        if (r == 0) {   // still running, and the call was not told to wait for it
            if (std::chrono::steady_clock::now() >= deadline) return -1;
            const timespec every{0, 20 * 1000 * 1000};   // a look every 20 ms; a signal interrupts it, and it is looked at again
            (void) nanosleep(&every, nullptr);
            continue;
        }
        if (r < 0 && errno == EINTR) continue;
        return -1;   // ECHILD: there is nothing left to wait for
    }
}

void child_terminate(Child& c) {
    if (c.id > 0) (void) kill((pid_t) c.id, SIGTERM);
}

void child_close(Child& c) {
    c.handle = nullptr;
    c.id = -1;
}

namespace {
Child g_stop_child;   // the child a stop signal ends: the run's engine, while it is running

void on_stop_signal(int) noexcept {
    if (g_stop_child.id > 0 || g_stop_child.handle != nullptr) child_terminate(g_stop_child);
    // The child has been told; this process does not wait for it to finish writing.
    ::_exit(130);
}
}  // namespace

void set_stop_child(const Child& c) { g_stop_child = c; }
void clear_stop_child() { g_stop_child = Child{}; }

void install_child_stop_signals() {
    (void) std::signal(SIGINT, on_stop_signal);
    (void) std::signal(SIGTERM, on_stop_signal);
}

void ignore_broken_pipe() {
    (void) std::signal(SIGPIPE, SIG_IGN);
}

void set_env(const std::string& name, const std::string& value) {
    setenv(name.c_str(), value.c_str(), 1);
}

int spawn_and_wait(const std::vector<std::string>& argv) {
    if (argv.empty()) { errno = EINVAL; return -1; }
    // The child's own exec failure has to reach the caller, and a spawn reports it the way the platform does:
    // posix_spawnp answers with the error the exec would have returned.
    std::vector<std::string> args = argv;
    std::vector<char*> av = c_argv(args);
    pid_t pid = -1;
    const int e = posix_spawnp(&pid, args[0].c_str(), nullptr, nullptr, av.data(), environ);
    if (e != 0) { errno = e; return -1; }
    Child c;
    c.id = (int64_t) pid;
    return child_wait(c, -1);
}

int exec_replacing(const std::vector<std::string>& argv) {
    if (argv.empty()) { errno = EINVAL; return -1; }
    std::vector<std::string> args = argv;
    std::vector<char*> av = c_argv(args);
    execvp(args[0].c_str(), av.data());
    return -1;   // errno says why, and it is the caller's message
}

}  // namespace halo::platform

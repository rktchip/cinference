#include "product/logging/logging.h"
#include "product/logging/startup_log.h"
#include "serve/generation_service.h"
#include "serve/http_server.h"
#include "serve/serve_options.h"

#include <spdlog/logger.h>

#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdlib>
#include <exception>
#include <fstream>
#include <iostream>
#include <memory>
#include <vector>
#include <stdexcept>
#include <string>
#include <utility>
#ifdef __linux__
#include <execinfo.h>
#include <signal.h>
#include <unistd.h>
#endif

namespace {

std::atomic<ninfer::serve::HttpServer*> g_server{nullptr};

void handle_signal(int) {
    ninfer::serve::HttpServer* server = g_server.load();
    if (server != nullptr) { server->stop(); }
}

#ifdef __linux__
// Crash handler (durability, see scripts/harness_notes.md): a segfault
// kills buffered logs with it, so flush stderr and print a backtrace
// before dying. Async-signal-safe subset only (write/_exit); backtrace()+
// backtrace_symbols_fd() are the standard glibc practice here.
void handle_crash(int sig) {
    const char msg[] = "ninfer-serve: fatal signal, backtrace:\n";
    (void)::write(STDERR_FILENO, msg, sizeof(msg) - 1);
    void* frames[64];
    const int depth = ::backtrace(frames, 64);
    ::backtrace_symbols_fd(frames, depth, STDERR_FILENO);
    (void)::fsync(STDERR_FILENO);
    ::signal(sig, SIG_DFL);
    ::raise(sig);
}
// Alternate stack: a stack-overflow segfault leaves no stack for the
// handler itself, which would explain a missing trace. 64KB, installed
// once at startup; SA_ONSTACK on both crash signals.
void install_crash_altstack() {
    static std::vector<char> altstack(SIGSTKSZ);
    stack_t ss;
    ss.ss_sp    = altstack.data();
    ss.ss_size  = altstack.size();
    ss.ss_flags = 0;
    if (::sigaltstack(&ss, nullptr) != 0) { return; }
}
void install_crash_handler() {
    install_crash_altstack();
    struct sigaction sa;
    sa.sa_handler = handle_crash;
    ::sigemptyset(&sa.sa_mask);
    sa.sa_flags = SA_RESETHAND | SA_NODEFER | SA_ONSTACK;
    ::sigaction(SIGSEGV, &sa, nullptr);
    ::sigaction(SIGABRT, &sa, nullptr);
}
#else
void install_crash_handler() {}
#endif

} // namespace

int main(int argc, char** argv) {
    install_crash_handler();
    ninfer::serve::ServeOptions options;
    try {
        options = ninfer::serve::parse_serve_options(argc, argv);
    } catch (const std::invalid_argument& exception) {
        std::cerr << "ninfer-serve: " << exception.what() << '\n';
        std::cerr << ninfer::serve::serve_usage_text(argv[0]);
        return 1;
    } catch (const std::exception& exception) {
        std::cerr << "ninfer-serve: " << exception.what() << '\n';
        return 1;
    }
    if (options.help_requested) {
        std::cout << ninfer::serve::serve_usage_text(argv[0]);
        return 0;
    }

    // Fail-closed forced-token boot check (adjudication harness): a set but
    // unloadable NINFER_FORCE_TOKENS used to fall back to normal generation
    // silently, wasting the run. Refuse to start instead (non-zero exit).
    if (const char* force_path = std::getenv("NINFER_FORCE_TOKENS")) {
        std::size_t force_n = 0;
        if (std::ifstream force_in(force_path); force_in) {
            long force_tok = 0;
            while (force_in >> force_tok) { ++force_n; }
        }
        if (force_n == 0) {
            std::cerr << "ninfer-serve: FATAL NINFER_FORCE_TOKENS=" << force_path
                      << " unreadable or empty: refusing to start (fail-closed)\n";
            return 1;
        }
        std::cerr << "FORCE: " << force_n << " tokens loaded from " << force_path << '\n';
    }

    ninfer::product::LoggingRuntime logging(
        {.logger_name  = "ninfer-serve",
         .level        = options.log_level,
         .presentation = ninfer::product::LogPresentation::Service});
    const std::shared_ptr<spdlog::logger> logger = logging.logger();
    ninfer::product::StartupLogRenderer startup_log(logging);
    ninfer::serve::OperationalLog operational_log(logger);
    bool serving = false;

    try {
        ninfer::serve::HttpServer server(options, logger);
        if (!server.bind()) {
            operational_log.bind_failure(options.host, options.port);
            return 1;
        }

        ninfer::serve::GenerationService service(options, startup_log.observer());
        startup_log.engine_ready(service.load_summary());
        operational_log.engine_capacity(service);

        using Clock                            = std::chrono::steady_clock;
        const Clock::time_point warmup_started = Clock::now();
        operational_log.warmup_started();
        try {
            service.warmup();
        } catch (const std::exception& exception) {
            const double seconds =
                std::chrono::duration<double>(Clock::now() - warmup_started).count();
            operational_log.warmup_failure(seconds, exception.what());
            return 1;
        }
        operational_log.warmup_complete(
            std::chrono::duration<double>(Clock::now() - warmup_started).count());
        server.attach(service);

        g_server.store(&server);
        std::signal(SIGINT, handle_signal);
        std::signal(SIGTERM, handle_signal);

        serving = true;
        operational_log.server_ready(options.host, options.port, server.public_model_id(),
                                     !options.api_key.empty());

        const bool ok = server.listen();
        g_server.store(nullptr);
        if (!ok) {
            operational_log.listen_failure(options.host, options.port);
            return 1;
        }
        operational_log.server_stopped();
        return 0;
    } catch (const std::exception& exception) {
        g_server.store(nullptr);
        operational_log.server_failure(serving, exception.what());
        return 1;
    }
}

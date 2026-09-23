#pragma once

// Product-side adapter from one protocol-neutral generation request to the public Engine. Wire
// adapters normalize before this layer and render IDs, usage, and response events after it.

#include "ninfer/engine.h"
#include "serve/request.h"
#include "serve/serve_options.h"

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <condition_variable>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

namespace ninfer::serve {

struct RequestLifetime;
struct RequestCapacity;
struct ServeRequestState;
class ServeHookLoop;

struct GenerationMetrics {
    double prepare_seconds         = 0.0;
    double ttft_seconds            = 0.0;
    double vision_seconds          = 0.0;
    double prefill_seconds         = 0.0;
    double decode_seconds          = 0.0;
    double prompt_wall_seconds     = 0.0;
    double generation_wall_seconds = 0.0;
    double total_seconds           = 0.0;
    ninfer::GenerationEngineTiming engine_timing;

    SpeculativeBackend speculative_backend    = SpeculativeBackend::None;
    std::uint32_t speculative_draft_window    = 0;
    std::uint64_t speculative_rounds          = 0;
    std::uint64_t speculative_draft_tokens    = 0;
    std::uint64_t speculative_accepted_tokens = 0;
    std::uint64_t speculative_fallback_steps  = 0;
    std::vector<std::uint64_t> speculative_accepted_per_position;
    std::uint32_t prefix_cache_hit_tokens     = 0;
    ninfer::PrefixReusePath prefix_reuse_path = ninfer::PrefixReusePath::Root;
    ninfer::MaterializationDiagnostics materialization;
};

struct GenerationOutcome {
    std::string text;
    std::string reasoning;
    std::vector<ninfer::GeneratedToolCall> tool_calls;
    ninfer::ToolCallParseDiagnostics tool_call_parse;
    int prompt_tokens     = 0;
    int completion_tokens = 0;
    int reasoning_tokens  = 0;
    ninfer::ThinkingBudgetStats thinking;
    ninfer::FinishReason finish_reason = ninfer::FinishReason::OutputLimit;
    std::optional<std::string> matched_stop_string;
    GenerationMetrics metrics;
};

struct StreamSink {
    std::function<void(const ninfer::GenerationStart& start)> on_start;
    std::function<void(const ninfer::PromptProgress& progress)> on_progress;
    std::function<void(const ninfer::GenerationTimingObservation& timing)> on_timing;
    std::function<void(const std::string& delta_text)> on_content;
    std::function<void(const std::string& delta_text)> on_reasoning;
    std::function<bool()> is_cancelled;
};

enum class GenerationConsumerMode : std::uint8_t {
    Aggregate,
    Streaming,
};

// Translate Engine request failures into the shared protocol-neutral HTTP error contract.
ApiError request_error_to_api_error(const ninfer::RequestError& exception);

// Preparation ends by synchronously submitting the owning prompt to the Engine FIFO. The returned
// request keeps its ingress/response lifetime reservation until the HTTP response is released and
// is consumed exactly once by run().
struct PreparedRequest {
    // Committee A+B single-scheduler admission (no Engine queue). The owning
    // prompt reaches the Engine ONLY through engine_->prepare (tokenize, no
    // enqueue); scheduler membership comes from hook_loop submit_inbox (ONE
    // RequestScheduler), and run() pumps schedule -> dispatch ->
    // Engine::run_batch_step -> on_step_done. No GenerationHandle exists on
    // this path (serve-mode Engine::submit throws). Single-use: consumed
    // exactly once by run().
    std::shared_ptr<ServeRequestState> stream_state;
    std::uint64_t req_id         = 0;
    std::uint32_t max_new_tokens = 0;
    ninfer::ResolvedSamplingParameters sampling;
    double prepare_seconds     = 0.0;
    double acquisition_seconds = 0.0;
    PromptPreparationStats preparation;
    int prompt_tokens    = 0;
    bool enable_thinking = true;
    std::optional<std::uint32_t> thinking_budget;
    std::optional<ninfer::ReasoningEffort> reasoning_effort;
    std::optional<bool> preserve_thinking;
    std::shared_ptr<RequestLifetime> lifetime;
};

class GenerationService {
public:
    explicit GenerationService(ServeOptions options, StartupObserver startup_observer = {});
    // Defined in generation_service.cpp: ServeHookLoop is complete there.
    ~GenerationService();

    GenerationService(const GenerationService&)            = delete;
    GenerationService& operator=(const GenerationService&) = delete;

    [[nodiscard]] const ServeOptions& options() const noexcept { return options_; }

    // Process-lifetime S1 loop binding (one RequestScheduler plus its
    // EngineHooks, sized from ServeOptions). Shared by every HTTP request on
    // this process: prepare() mutates no scheduler state and run() consumes
    // only its own single-use PreparedRequest, so two different prompts
    // admitted here stream through the one loop.
    [[nodiscard]] ServeHookLoop& hook_loop() noexcept { return *hook_loop_; }
    [[nodiscard]] const ServeHookLoop& hook_loop() const noexcept { return *hook_loop_; }

    // Engine owns the once-normalized startup configuration. Serving diagnostics must use this
    // value instead of reinterpreting optional defaults from ServeOptions.
    [[nodiscard]] const ninfer::EngineOptions& engine_options() const { return engine_->options(); }

    [[nodiscard]] ninfer::LoadSummary load_summary() const { return engine_->load_summary(); }

    [[nodiscard]] ninfer::MemorySummary memory_summary() const { return engine_->memory_summary(); }

    [[nodiscard]] ninfer::RuntimeStats runtime_stats() const { return engine_->runtime_stats(); }

    [[nodiscard]] bool is_available() const { return engine_->is_available(); }

    [[nodiscard]] ninfer::MediaCacheSummary media_cache_summary() const {
        return engine_->media_cache_summary();
    }

    [[nodiscard]] ninfer::ModelSamplingDefaults sampling_defaults() const {
        return engine_->sampling_defaults();
    }

    // Tokenize and handle media only; mutates no scheduler state. Admission
    // passes through hook_loop() first (throw_if_pages_exhausted, Overloaded
    // -> HTTP 429) and then submits to the Engine; the returned request keeps
    // its ingress/response lifetime reservation until the HTTP response is
    // released and is consumed exactly once by run().
    [[nodiscard]] PreparedRequest prepare(const GenerationRequest& req,
                                          GenerationConsumerMode consumer_mode,
                                          ninfer::GenerationObservationOptions observation = {},
                                          std::function<bool()> is_cancelled               = {},
                                          ContextCacheHints context_cache = {}) const;
    [[nodiscard]] int count_prompt_tokens(const GenerationRequest& req,
                                          std::function<bool()> is_cancelled = {}) const;

    // Consumes prepared.stream_state. A PreparedRequest is single-use.
    // S1 run path: the owning prompt passed the hook_loop admission gate at
    // submit time; run() pumps schedule_step -> dispatch_step -> on_step_done
    // itself and publishes decoded tokens per seq_id to the sink with no
    // full-answer buffer. The enqueue-only handle is released unconsumed so
    // this pump is the single driver.
    GenerationOutcome run(PreparedRequest& prepared, const StreamSink* sink,
                          std::function<bool()> is_cancelled = {});

    // One warmup pass per process lifetime (EXL3 weights already loaded once
    // by Engine construction). Re-entry is a no-op via ensure_warmed_once.
    void warmup();

private:
    enum class CacheParticipation : std::uint8_t {
        Disabled,
        ReadWrite,
    };

    enum class DeadlinePolicy : std::uint8_t {
        ClientPendingTimeout,
        UnboundedStartup,
    };

    [[nodiscard]] PreparedRequest
    prepare_impl(const GenerationRequest& req, GenerationConsumerMode consumer_mode,
                 ninfer::GenerationObservationOptions observation,
                 std::function<bool()> is_cancelled, ContextCacheHints context_cache,
                 CacheParticipation cache_participation, DeadlinePolicy deadline_policy) const;
    [[nodiscard]] std::shared_ptr<RequestLifetime>
    acquire_request_lifetime(DeadlinePolicy deadline_policy) const;

    ServeOptions options_;
    std::unique_ptr<ninfer::Engine> engine_;
    std::shared_ptr<RequestCapacity> request_capacity_;
    // Process-lifetime S1 loop binding. Built once in the constructor next to
    // the EXL3-loaded Engine; never rebuilt per request.
    std::unique_ptr<ServeHookLoop> hook_loop_;
    // Slot A pump mutex: serializes the run() schedule_step -> dispatch_step
    // -> on_step_done triplet so two HTTP threads sharing this scheduler
    // cannot double-schedule one step. Dedicated to the pump; never
    // warmup_mutex_ (that one guards only the once-gate in ServeHookLoop).
    std::mutex pump_mutex_;
    // Slot A idle backoff: signaled by the prepare/submit path after every
    // engine_->submit so an idle pump wakes the moment new work arrives.
    // Mutable: prepare_impl() notifies from a const path.
    mutable std::mutex idle_mutex_;
    mutable std::condition_variable idle_cv_;
};

} // namespace ninfer::serve

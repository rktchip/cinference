#include "serve/generation_service.h"

#include "batch/batch.h"
#include "batch/cinference_hooks.h"
#include "batch/scheduler.h"
#include "product/media_acquire/acquire.h"
#include "serve/hook_loop.h"
#include "serve/translate.h"

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <utility>

namespace ninfer::exl3 {
// Host-opaque view of the process-lifetime EXL3 side-car store + v3
// workspace. The full device types live behind __CUDACC__ in
// ops/linear/exl3/exl3_bind.h (device upload is a Linux-deployment path);
// serve never needs their layout, only the reserve entry, so the store and
// workspace are named here opaquely. Signatures must match exl3_bind.h.
struct Exl3EngineStore;
struct Exl3EngineWorkspace;
void exl3_engine_reserve_workspace(const Exl3EngineStore& store, std::int32_t m_max,
                                   Exl3EngineWorkspace& ws);
Exl3EngineStore& exl3_process_store() noexcept;
Exl3EngineWorkspace& exl3_process_workspace() noexcept;
} // namespace ninfer::exl3

namespace ninfer::serve {

struct RequestCapacity {
    explicit RequestCapacity(std::size_t limit) : maximum(limit) {}

    std::mutex mutex;
    std::size_t active = 0;
    const std::size_t maximum;
};

struct RequestLifetime {
    RequestLifetime(std::shared_ptr<RequestCapacity> owner,
                    std::chrono::steady_clock::time_point begin,
                    std::chrono::steady_clock::time_point limit)
        : capacity(std::move(owner)), started(begin), deadline(limit) {}

    ~RequestLifetime() {
        std::lock_guard lock(capacity->mutex);
        --capacity->active;
    }

    std::shared_ptr<RequestCapacity> capacity;
    std::chrono::steady_clock::time_point started;
    std::chrono::steady_clock::time_point deadline;
};

ApiError request_error_to_api_error(const ninfer::RequestError& exception) {
    ApiError error;
    error.param   = "messages";
    error.message = exception.what();
    switch (exception.kind()) {
    case ninfer::RequestErrorKind::ContextLengthExceeded:
        error.status = 400;
        error.code   = "context_length_exceeded";
        break;
    case ninfer::RequestErrorKind::ThinkingBudgetCapacityInsufficient:
        error.param.clear();
        error.status = 400;
        error.code   = "thinking_budget_capacity_insufficient";
        break;
    case ninfer::RequestErrorKind::MediaBudgetExceeded:
        error.status = 400;
        error.code   = "media_budget_exceeded";
        break;
    case ninfer::RequestErrorKind::InvalidMedia:
        error.status = 400;
        error.code   = "invalid_media";
        break;
    case ninfer::RequestErrorKind::Overloaded:
        error.param.clear();
        error.status = 429;
        error.type   = "rate_limit_error";
        error.code   = "server_overloaded";
        break;
    case ninfer::RequestErrorKind::QueueTimeout:
        error.param.clear();
        error.status = 503;
        error.type   = "server_error";
        error.code   = "request_queue_timeout";
        break;
    case ninfer::RequestErrorKind::Cancelled:
        error.param.clear();
        error.status = 499;
        error.type   = "request_cancelled";
        error.code   = "client_disconnected";
        break;
    case ninfer::RequestErrorKind::Unavailable:
        error.param.clear();
        error.status = 503;
        error.type   = "server_error";
        error.code   = "service_unavailable";
        break;
    }
    return error;
}

namespace {

using Clock = std::chrono::steady_clock;

[[noreturn]] void throw_preparation_cancelled();

[[noreturn]] void throw_media_error(const ninfer::product::media_acquire::Error& exception) {
    ApiError error;
    error.param   = "messages";
    error.message = exception.what();
    switch (exception.kind()) {
    case ninfer::product::media_acquire::ErrorKind::BudgetExceeded:
        error.status = 400;
        error.code   = "media_budget_exceeded";
        break;
    case ninfer::product::media_acquire::ErrorKind::RemoteUnavailable:
        error.status = 502;
        error.type   = "server_error";
        error.code   = "media_fetch_failed";
        break;
    case ninfer::product::media_acquire::ErrorKind::RemoteTimeout:
        error.status = 504;
        error.type   = "server_error";
        error.code   = "media_fetch_timeout";
        break;
    case ninfer::product::media_acquire::ErrorKind::DeadlineExceeded:
        error.param.clear();
        error.status = 503;
        error.type   = "server_error";
        error.code   = "request_queue_timeout";
        break;
    case ninfer::product::media_acquire::ErrorKind::Cancelled:
        throw_preparation_cancelled();
    }
    throw ApiException(std::move(error));
}

[[noreturn]] void throw_invalid_input(const std::exception& exception, const char* code) {
    ApiError error;
    error.status  = 400;
    error.param   = "messages";
    error.code    = code;
    error.message = exception.what();
    throw ApiException(std::move(error));
}

[[noreturn]] void throw_preparation_cancelled() {
    ApiError error;
    error.status  = 499;
    error.type    = "request_cancelled";
    error.code    = "client_disconnected";
    error.message = "client disconnected during media preparation";
    throw ApiException(std::move(error));
}

ninfer::OwnedMedia acquire_media(const ContentPart& part, Clock::time_point deadline,
                                 const std::function<bool()>& is_cancelled,
                                 std::size_t& remaining_bytes) {
    if (remaining_bytes == 0) {
        throw_media_error(ninfer::product::media_acquire::Error(
            ninfer::product::media_acquire::ErrorKind::BudgetExceeded,
            "request media exceeds aggregate byte limit"));
    }
    ninfer::product::media_acquire::Policy policy;
    policy.max_bytes    = std::min(policy.max_bytes, remaining_bytes);
    policy.deadline     = deadline;
    policy.is_cancelled = is_cancelled;
    std::vector<std::uint8_t> source_bytes;
    try {
        source_bytes = ninfer::product::media_acquire::acquire_bytes(part.source, policy);
    } catch (const ninfer::product::media_acquire::Error& exception) {
        throw_media_error(exception);
    } catch (const std::invalid_argument& exception) {
        throw_invalid_input(exception, "invalid_media");
    }

    remaining_bytes -= source_bytes.size();
    ninfer::OwnedMedia media;
    media.kind =
        part.kind == ContentKind::Image ? ninfer::MediaKind::Image : ninfer::MediaKind::Video;
    media.media_type = part.source.media_type;
    switch (part.source.kind) {
    case ninfer::product::media_acquire::SourceKind::Path:
    case ninfer::product::media_acquire::SourceKind::Url:
        media.source_name = part.source.value;
        break;
    case ninfer::product::media_acquire::SourceKind::Data:
        media.source_name = "inline-data";
        break;
    case ninfer::product::media_acquire::SourceKind::Bytes:
        media.source_name = "inline-bytes";
        break;
    }
    media.bytes               = std::move(source_bytes);
    media.image_resize_policy = part.image_resize_policy;
    return media;
}

[[noreturn]] void throw_request_error(const ninfer::RequestError& exception) {
    throw ApiException(request_error_to_api_error(exception));
}

void check_preparation_control(Clock::time_point deadline,
                               const std::function<bool()>& is_cancelled) {
    if (is_cancelled && is_cancelled()) { throw_preparation_cancelled(); }
    if (Clock::now() >= deadline) {
        throw_request_error(ninfer::RequestError(RequestErrorKind::QueueTimeout,
                                                 "inference request expired during preparation"));
    }
}

class ServiceOutputSink final : public ninfer::OutputSink {
public:
    explicit ServiceOutputSink(const StreamSink& sink) : sink_(&sink) {}

    void start(ninfer::GenerationStart start) override {
        if (sink_->on_start) { sink_->on_start(start); }
    }

    void progress(ninfer::PromptProgress progress) override {
        if (sink_->on_progress) { sink_->on_progress(progress); }
    }

    void timing(ninfer::GenerationTimingObservation timing) override {
        if (sink_->on_timing) { sink_->on_timing(timing); }
    }

    void publish(ninfer::OutputDelta delta) override {
        if (delta.text.empty()) { return; }
        if (delta.channel == ninfer::OutputChannel::Reasoning) {
            if (sink_->on_reasoning) { sink_->on_reasoning(delta.text); }
        } else {
            if (sink_->on_content) { sink_->on_content(delta.text); }
        }
    }

private:
    const StreamSink* sink_ = nullptr;
};

} // namespace

GenerationService::GenerationService(ServeOptions options, StartupObserver startup_observer)
    : options_(std::move(options)) {
    ninfer::EngineOptions engine_options;
    engine_options.artifact_path            = options_.artifact_path;
    engine_options.chat_template_path       = options_.chat_template_path;
    engine_options.device                   = options_.device;
    engine_options.max_context              = options_.max_context;
    engine_options.kv_capacity              = options_.kv_capacity;
    engine_options.max_concurrency          = options_.max_concurrency;
    engine_options.max_pending_requests     = options_.max_pending_requests;
    engine_options.pending_timeout_ms       = options_.pending_timeout_ms;
    engine_options.prefill_chunk            = options_.prefill_chunk;
    engine_options.kv_cache                 = options_.kv_cache;
    engine_options.enable_vision            = options_.enable_vision;
    engine_options.use_cuda_graph           = options_.use_cuda_graph;
    engine_options.speculative              = options_.speculative;
    engine_options.context_cache            = options_.context_cache;
    engine_options.context_cost.preset_path = options_.context_cost_presets;
    engine_options.media_cache_bytes        = options_.media_cache_bytes;
    engine_options.media_live_bytes         = options_.media_live_bytes;
    engine_options.media_preprocess_threads = options_.media_preprocess_threads;
    engine_options.startup_observer         = std::move(startup_observer);
    engine_           = std::make_unique<ninfer::Engine>(std::move(engine_options));
    // Committee A+B: serve owns ONE scheduler (hook_loop) and steps via
    // Engine::run_batch_step. Serve-mode submit() throws so per-client
    // generate loops cannot reappear on this path; CLI never sets it.
    engine_->set_serve_mode(true);
    request_capacity_ = std::make_shared<RequestCapacity>(
        static_cast<std::size_t>(options_.max_concurrency) + options_.max_pending_requests);
    // S2(a): one process-lifetime S1 loop binding beside the EXL3-loaded
    // Engine. Weights load once inside the Engine above; this scheduler is
    // sized from the same options and is never rebuilt per request.
    hook_loop_ = std::make_unique<ServeHookLoop>(make_hook_loop_config(
        options_.max_concurrency, options_.max_context,
        options_.kv_capacity.mode == ninfer::KvCapacityMode::Explicit
            ? options_.kv_capacity.explicit_tokens
            : 0,
        options_.prefill_chunk));
    // A2(c): serve must never run degraded on host-only buffers. This throw
    // is fatal at startup (the service fails to construct, never a
    // per-request fallback).
    ninfer::batch::RequireDeviceBuffersForServe(*hook_loop_->device_buffers(),
                                                hook_loop_->scheduler()->pool());
    // Slot A: reserve the process-lifetime EXL3 v3 workspace once, next to
    // the device-buffer guard above. One step carries at most a full prefill
    // slice plus one decode token per row, so prefill_chunk +
    // kMaximumConcurrency bounds every M. An empty side-car store (non-EXL3
    // artifact) means nothing to reserve and throws invalid_argument, which
    // is skipped here; a CUDA failure throws runtime_error and stays fatal
    // like the guard above.
    const std::int32_t ws_m_max =
        static_cast<std::int32_t>(options_.prefill_chunk + ninfer::kMaximumConcurrency);
    try {
        ninfer::exl3::exl3_engine_reserve_workspace(ninfer::exl3::exl3_process_store(),
                                                   ws_m_max,
                                                   ninfer::exl3::exl3_process_workspace());
    } catch (const std::invalid_argument&) {
        // No EXL3 store loaded: nothing to reserve.
    }
}

GenerationService::~GenerationService() = default;

std::shared_ptr<RequestLifetime>
GenerationService::acquire_request_lifetime(DeadlinePolicy deadline_policy) const {
    const auto started = Clock::now();
    {
        std::lock_guard lock(request_capacity_->mutex);
        if (request_capacity_->active >= request_capacity_->maximum) {
            throw_request_error(ninfer::RequestError(RequestErrorKind::Overloaded,
                                                     "inference request queue is full"));
        }
        ++request_capacity_->active;
    }
    try {
        const Clock::time_point deadline =
            deadline_policy == DeadlinePolicy::UnboundedStartup
                ? Clock::time_point::max()
                : started + std::chrono::milliseconds(options_.pending_timeout_ms);
        return std::make_shared<RequestLifetime>(request_capacity_, started, deadline);
    } catch (...) {
        std::lock_guard lock(request_capacity_->mutex);
        --request_capacity_->active;
        throw;
    }
}

PreparedRequest GenerationService::prepare(const GenerationRequest& request,
                                           GenerationConsumerMode consumer_mode,
                                           ninfer::GenerationObservationOptions observation,
                                           std::function<bool()> is_cancelled,
                                           ContextCacheHints context_cache) const {
    return prepare_impl(
        request, consumer_mode, observation, std::move(is_cancelled), std::move(context_cache),
        options_.allow_prefix_reuse ? CacheParticipation::ReadWrite : CacheParticipation::Disabled,
        DeadlinePolicy::ClientPendingTimeout);
}

PreparedRequest GenerationService::prepare_impl(const GenerationRequest& request,
                                                GenerationConsumerMode consumer_mode,
                                                ninfer::GenerationObservationOptions observation,
                                                std::function<bool()> is_cancelled,
                                                ContextCacheHints context_cache,
                                                CacheParticipation cache_participation,
                                                DeadlinePolicy deadline_policy) const {
    PreparedRequest prepared;
    const ResolvedPromptSemantics semantics = resolve_prompt_semantics(request, options_);
    ninfer::RequestOptions request_options  = to_request_options(
        request, options_, semantics, cache_participation == CacheParticipation::ReadWrite);
    prepared.thinking_budget     = request_options.execution.thinking.budget;
    prepared.reasoning_effort    = semantics.reasoning_effort;
    prepared.preserve_thinking   = semantics.preserve_thinking;
    const bool request_has_media = request.media_item_count() != 0;
    if (request_has_media && !options_.enable_vision) {
        const std::invalid_argument error("Vision is disabled for this server");
        throw_invalid_input(error, "vision_disabled");
    }
    prepared.lifetime = acquire_request_lifetime(deadline_policy);

    try {
        const auto acquisition_started = Clock::now();
        std::size_t remaining_media_bytes =
            std::min(options_.max_request_bytes, ninfer::kMaximumPromptMediaBytes);
        ninfer::PromptInput input =
            to_prompt_input(request, semantics, [&](const ContentPart& part) {
                return acquire_media(part, prepared.lifetime->deadline, is_cancelled,
                                     remaining_media_bytes);
            });
        std::vector<PromptCacheMarker> protocol_markers = std::move(input.context_cache.markers);
        const bool protocol_allows_engine_automatic =
            input.context_cache.allow_engine_automatic_shared_prefixes;
        input.context_cache = std::move(context_cache);
        input.context_cache.markers.insert(input.context_cache.markers.end(),
                                           std::make_move_iterator(protocol_markers.begin()),
                                           std::make_move_iterator(protocol_markers.end()));
        input.context_cache.allow_engine_automatic_shared_prefixes =
            input.context_cache.allow_engine_automatic_shared_prefixes &&
            protocol_allows_engine_automatic;
        prepared.acquisition_seconds =
            std::chrono::duration<double>(Clock::now() - acquisition_started).count();
        check_preparation_control(prepared.lifetime->deadline, is_cancelled);
        const PreparationControl control{
            .deadline     = prepared.lifetime->deadline,
            .cancellation = CancellationView(is_cancelled),
        };
        ninfer::PreparedPrompt prompt = engine_->prepare(std::move(input), control);
        check_preparation_control(prepared.lifetime->deadline, is_cancelled);
        prepared.enable_thinking = prompt.summary().starts_in_reasoning;
        if (!prepared.enable_thinking) {
            request_options.execution.thinking.budget.reset();
            prepared.thinking_budget.reset();
        }
        prepared.prompt_tokens = static_cast<int>(prompt.summary().prompt_tokens);
        prepared.preparation   = prompt.preparation_stats();
        prepared.prepare_seconds =
            std::chrono::duration<double>(Clock::now() - prepared.lifetime->started).count();
        // S2b: hook_loop is the admission oracle on this path. No
        // request reaches the Engine enqueue below without passing this gate;
        // page exhaustion throws Overloaded here (HTTP 429 via
        // request_error_to_api_error). QueueTimeout stays 503.
        // A2 execute-path role: this engine enqueue call only establishes
        // Engine FIFO membership and returns a handle. The fused single-lane
        // Engine::generate path is banned in batch mode: serve always uses
        // this prepare/enqueue split and run() pumps the hook loop itself.
        hook_loop_->throw_if_pages_exhausted();
        // Committee A+B admission (ONE scheduler): tokenize/media above, then
        // admit via the hook_loop inbox. The per-request stream state is
        // registered BEFORE admission (submit_inbox mints req/seq ids into
        // the state first), so no token can miss its stream. Serve never
        // calls Engine::submit (serve mode throws); Engine::generate stays
        // for CLI. Page exhaustion threw Overloaded above (HTTP 429).
        (void)consumer_mode;
        (void)observation;
        std::vector<TokenId> prompt_ids = engine_->prompt_token_ids(prompt);
        prepared.sampling               = engine_->resolved_sampling(prompt, request_options);
        const std::uint32_t prompt_len  = static_cast<std::uint32_t>(prompt_ids.size());
        const std::uint32_t max_context = engine_->options().max_context;
        const std::uint64_t room =
            prompt_len <= max_context ? static_cast<std::uint64_t>(max_context - prompt_len) + 1U
                                      : 0U;
        const std::uint32_t max_new = static_cast<std::uint32_t>(
            std::min<std::uint64_t>(request_options.execution.requested_output_tokens, room));
        auto stream_state = std::make_shared<ServeRequestState>();
        batch::Request batch_request(0, std::move(prompt_ids), max_new);
        // Step-1 EOS stop: model defaults (+ caller token stops) travel with
        // the admitted request; scheduler finishes on a hit, the pump drops
        // the stop token and anything after it from the emit.
        {
            std::vector<TokenId> stops = engine_->default_stop_token_ids();
            for (const TokenId stop : request_options.stop.token_ids) {
                if (std::find(stops.begin(), stops.end(), stop) == stops.end()) {
                    stops.push_back(stop);
                }
            }
            batch_request.stop_token_ids = std::move(stops);
        }
        prepared.req_id         = hook_loop_->submit_inbox(std::move(batch_request), stream_state);
        prepared.stream_state   = std::move(stream_state);
        prepared.max_new_tokens = max_new;
        // Slot A: wake any idle pump. run() waits on idle_cv_ when its plan
        // is empty; this kick (plus its timeout fallback) bounds wakeup
        // latency without busy-spinning.
        idle_cv_.notify_all();
    } catch (const ApiException&) { throw; } catch (const ninfer::RequestError& exception) {
        throw_request_error(exception);
    } catch (const std::invalid_argument& exception) {
        throw_invalid_input(exception, "invalid_prompt");
    }
    return prepared;
}

int GenerationService::count_prompt_tokens(const GenerationRequest& request,
                                           std::function<bool()> is_cancelled) const {
    const bool request_has_media = request.media_item_count() != 0;
    if (request_has_media && !options_.enable_vision) {
        const std::invalid_argument error("Vision is disabled for this server");
        throw_invalid_input(error, "vision_disabled");
    }
    const Clock::time_point deadline =
        Clock::now() + std::chrono::milliseconds(options_.pending_timeout_ms);
    const ResolvedPromptSemantics semantics = resolve_prompt_semantics(request, options_);
    try {
        std::size_t remaining_media_bytes =
            std::min(options_.max_request_bytes, ninfer::kMaximumPromptMediaBytes);
        ninfer::PromptInput input =
            to_prompt_input(request, semantics, [&](const ContentPart& part) {
                return acquire_media(part, deadline, is_cancelled, remaining_media_bytes);
            });
        check_preparation_control(deadline, is_cancelled);
        const PreparationControl control{
            .deadline     = deadline,
            .cancellation = CancellationView(is_cancelled),
        };
        const int prompt_tokens =
            static_cast<int>(engine_->count_tokens(std::move(input), control));
        check_preparation_control(deadline, is_cancelled);
        return prompt_tokens;
    } catch (const ApiException&) { throw; } catch (const ninfer::RequestError& exception) {
        throw_request_error(exception);
    } catch (const std::invalid_argument& exception) {
        throw_invalid_input(exception, "invalid_prompt");
    }
}

GenerationOutcome GenerationService::run(PreparedRequest& prepared, const StreamSink* sink,
                                         std::function<bool()> is_cancelled) {
    // Committee A+B pump: the single driver for serve generations.
    // prepare_impl admitted via the hook_loop inbox (ONE RequestScheduler);
    // this loop pumps schedule_step -> dispatch_step -> Engine::
    // run_batch_step -> on_step_done in order, then publishes decoded
    // (seq_id, token) pairs by seq/req id to the registered per-request
    // sink. No Engine::submit/wait appears here (serve mode throws) and no
    // second generate loop exists. Page exhaustion stays admission pressure
    // (RequestErrorKind::Overloaded, 429) via throw_request_error, never a
    // process abort.
    std::unique_ptr<ServiceOutputSink> output_sink;
    if (sink != nullptr) { output_sink = std::make_unique<ServiceOutputSink>(*sink); }
    ninfer::OutputSink* public_sink = output_sink.get();
    ninfer::CancellationView cancellation;
    if (is_cancelled || (sink != nullptr && sink->is_cancelled)) {
        cancellation = ninfer::CancellationView([external = std::move(is_cancelled), sink]() {
            return (external && external()) ||
                   (sink != nullptr && sink->is_cancelled && sink->is_cancelled());
        });
    }
    const std::shared_ptr<ServeRequestState> state = prepared.stream_state;
    if (state == nullptr) { throw std::logic_error("PreparedRequest has no stream state"); }
    const std::uint64_t own_req = prepared.req_id;
    // TTFT clock: first published content token for the owning request.
    // Stamped at the first non-empty content publish (or first decoded
    // content piece when no streaming sink is attached), so
    // outcome.metrics.ttft_seconds and the req-done log line are real.
    Clock::time_point first_token_at{};
    bool have_first_token = false;
    // Register the live sink BEFORE draining the inbox (the drain below may
    // admit this request). Decoded text produced before registration is
    // buffered in the state and flushed here in order, so no token misses
    // its stream.
    {
        std::vector<std::string> flush;
        {
            std::lock_guard state_lock(state->mutex);
            state->sink = public_sink;
            flush.swap(state->pending);
        }
        for (const std::string& piece : flush) {
            if (!piece.empty() && public_sink != nullptr) {
                ninfer::OutputDelta delta;
                delta.channel = ninfer::OutputChannel::Content;
                delta.text    = piece;
                public_sink->publish(std::move(delta));
                if (!have_first_token) {
                    first_token_at = Clock::now();
                    have_first_token = true;
                }
            }
        }
    }

    for (;;) {
        if (cancellation.requested()) {
            throw_request_error(ninfer::RequestError(ninfer::RequestErrorKind::Cancelled,
                                                     "inference request cancelled"));
        }
        if (Clock::now() >= prepared.lifetime->deadline) {
            throw_request_error(ninfer::RequestError(ninfer::RequestErrorKind::QueueTimeout,
                                                     "inference request expired during generation"));
        }
        // Slot A pump: HTTP serves on the httplib ThreadPool, so run()
        // threads share this scheduler. Inbox drain + schedule_step ->
        // dispatch_step -> run_batch_step -> on_step_done is one pump_mutex_
        // critical section, so two pumps cannot double-schedule one step and
        // corrupt computed_len. Publish happens outside the lock below
        // (non-blocking fan-out per stream state).
        batch::StepPlan plan;
        std::vector<std::pair<std::uint64_t, TokenId>> decoded;
        std::vector<std::uint64_t> finished;
        bool own_done = false;
        std::vector<std::pair<std::shared_ptr<ServeRequestState>, TokenId>> publish;
        {
            std::lock_guard pump_lock(pump_mutex_);
            hook_loop_->drain_inbox_to_scheduler();
            plan = hook_loop_->schedule_step();
            if (plan.empty()) {
                // No step to run: this pump is done only when its own request
                // is finished or gone (aborted and reclaimed). Any other
                // empty plan is transient backoff, never a drain break.
                own_done = hook_loop_->scheduler()->find_request(own_req) == nullptr;
            } else {
                const batch::StepDispatch dispatch = hook_loop_->dispatch_step(plan);
                decoded                            = engine_->run_batch_step(plan, dispatch);
                finished                           = hook_loop_->on_step_done(plan, decoded);
                own_done = std::find(finished.begin(), finished.end(), own_req) !=
                               finished.end() ||
                           hook_loop_->scheduler()->find_request(own_req) == nullptr;
                publish.reserve(decoded.size());
                // Step-1 EOS stop: per-step set of seqs already stopped. The
                // stop token itself is never emitted (matches
                // publish_stop_token=false); anything after it in the same
                // step (spec-on commit tail past EOS) is dropped.
                std::vector<std::uint64_t> stopped_seqs;
                for (const auto& [seq_id, token] : decoded) {
                    if (auto target = hook_loop_->find_state_by_seq(seq_id)) {
                        std::lock_guard state_lock(target->mutex);
                        const bool already_stopped =
                            std::find(stopped_seqs.begin(), stopped_seqs.end(), seq_id) !=
                            stopped_seqs.end();
                        const bool is_stop =
                            std::find(target->stop_token_ids.begin(),
                                      target->stop_token_ids.end(), token) !=
                            target->stop_token_ids.end();
                        if (already_stopped) { continue; }
                        if (is_stop) {
                            stopped_seqs.push_back(seq_id);
                            target->stopped_on_token = true;
                            target->finished          = true;
                            continue;
                        }
                        publish.emplace_back(std::move(target), token);
                    }
                }
                hook_loop_->erase_states_for_reqs(finished);
            }
        }
        for (const auto& [target, token] : publish) {
            const std::vector<TokenId> one{token};
            std::string piece;
            try {
                piece = engine_->decode_tokens(one);
            } catch (...) {
                piece.clear();
            }
            std::string deposit;
            ninfer::OutputSink* live = nullptr;
            {
                std::lock_guard state_lock(target->mutex);
                target->generated_ids.push_back(token);
                target->text += piece;
                if (target->sink != nullptr) {
                    live    = target->sink;
                    deposit = piece;
                } else {
                    target->pending.push_back(piece);
                }
                if (std::find(finished.begin(), finished.end(), target->req_id) != finished.end()) {
                    target->finished = true;
                }
            }
            if (live != nullptr && !deposit.empty()) {
                ninfer::OutputDelta delta;
                delta.channel = ninfer::OutputChannel::Content;
                delta.text    = std::move(deposit);
                live->publish(std::move(delta));
            }
            if (!have_first_token && target->req_id == own_req && !piece.empty()) {
                first_token_at = Clock::now();
                have_first_token = true;
            }
        }
        if (!plan.empty()) {
            // A prefill-only step decodes nothing while the prompt advances:
            // keep pumping and never break here. Only the owning request
            // finishing (or going away) ends this pump.
            if (own_done) { break; }
            continue;
        }
        {
            std::lock_guard state_lock(state->mutex);
            if (state->finished || own_done) { break; }
        }
        // Slot A idle backoff: an empty plan is transient (another
        // request's step may be in flight), so never busy-spin. Wait on
        // idle_cv_, kicked by the prepare path after every inbox submit,
        // with a 1 ms timeout fallback so a missed kick only delays one step.
        std::unique_lock idle_lock(idle_mutex_);
        idle_cv_.wait_for(idle_lock, std::chrono::milliseconds(1));
    }

    GenerationOutcome outcome;
    bool stopped_on_token = false;
    {
        std::lock_guard state_lock(state->mutex);
        outcome.text              = state->text;
        outcome.completion_tokens = static_cast<int>(state->generated_ids.size());
        stopped_on_token          = state->stopped_on_token;
    }
    outcome.prompt_tokens = prepared.prompt_tokens;
    // Step 1: the pump observes a stop-token terminal event (StopToken); the
    // max-tokens path still reports None (pre-existing behavior).
    outcome.finish_reason = stopped_on_token ? ninfer::FinishReason::StopToken
                                             : ninfer::FinishReason::None;
    outcome.metrics.prepare_seconds = prepared.prepare_seconds;
    if (have_first_token) {
        outcome.metrics.ttft_seconds = std::chrono::duration<double>(first_token_at -
            prepared.lifetime->started).count();
    } else {
        outcome.metrics.ttft_seconds = 0.0;
    }
    outcome.metrics.total_seconds   = prepared.prepare_seconds;
    return outcome;
}
void GenerationService::warmup() {
    // S2(e): exactly one warmup pass per process. EXL3 weights load once at
    // Engine construction; this warms the execution path a single time and
    // never per request. main() already calls warmup() once at startup; the
    // once flag makes re-entry structurally free.
    hook_loop_->ensure_warmed_once([this] {
        GenerationRequest request;
        ChatTurn turn;
        turn.role = ChatRole::User;
        ContentPart content;
        content.kind     = ContentKind::Text;
        content.text     = "hi";
        content.type_raw = "text";
        turn.content.push_back(std::move(content));
        request.messages.push_back(std::move(turn));
        request.max_tokens = 4;
        PreparedRequest prepared =
            prepare_impl(request, GenerationConsumerMode::Aggregate, {}, {}, {},
                         CacheParticipation::Disabled, DeadlinePolicy::UnboundedStartup);
        run(prepared, nullptr);
    });
}

} // namespace ninfer::serve

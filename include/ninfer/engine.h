#pragma once

#include "ninfer/types.h"

#include <chrono>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace ninfer {

namespace batch {
struct StepPlan;
struct StepDispatch;
} // namespace batch

class PreparedPrompt {
public:
    PreparedPrompt() noexcept;
    ~PreparedPrompt();

    PreparedPrompt(PreparedPrompt&&) noexcept;
    PreparedPrompt& operator=(PreparedPrompt&&) noexcept;

    PreparedPrompt(const PreparedPrompt&)            = delete;
    PreparedPrompt& operator=(const PreparedPrompt&) = delete;

    [[nodiscard]] const PromptSummary& summary() const noexcept;
    [[nodiscard]] const PromptPreparationStats& preparation_stats() const noexcept;
    [[nodiscard]] explicit operator bool() const noexcept;

private:
    class Impl;
    explicit PreparedPrompt(std::unique_ptr<Impl> impl) noexcept;
    std::unique_ptr<Impl> impl_;

    friend class Engine;
};

class GenerationHandle {
public:
    GenerationHandle() noexcept;
    ~GenerationHandle();

    GenerationHandle(GenerationHandle&&) noexcept;
    GenerationHandle& operator=(GenerationHandle&&) noexcept;

    GenerationHandle(const GenerationHandle&)            = delete;
    GenerationHandle& operator=(const GenerationHandle&) = delete;

    [[nodiscard]] explicit operator bool() const noexcept;
    [[nodiscard]] const ResolvedSamplingParameters& resolved_sampling() const noexcept;

    GenerationResult wait(OutputSink* sink = nullptr, const CancellationView& cancellation = {});

private:
    class Impl;
    explicit GenerationHandle(std::unique_ptr<Impl> impl) noexcept;
    std::unique_ptr<Impl> impl_;

    friend class Engine;
};

class Engine {
public:
    explicit Engine(EngineOptions options);
    ~Engine();

    Engine(Engine&&) noexcept;
    Engine& operator=(Engine&&) noexcept;

    Engine(const Engine&)            = delete;
    Engine& operator=(const Engine&) = delete;

    [[nodiscard]] PreparedPrompt prepare(PromptInput input,
                                         const PreparationControl& control = {}) const;

    // Raw token input is retained for repeatable correctness and performance measurement.
    [[nodiscard]] PreparedPrompt prepare_tokens(std::vector<TokenId> token_ids,
                                                bool allow_prefix_identity = true) const;

    // Artifact-tokenizer raw-text encoding. No chat template or implicit special token is added.
    [[nodiscard]] std::vector<TokenId> tokenize_text(std::string_view text) const;

    // Returns log p(tokens[i] | tokens[0..i)) for i in [first_target,tokens.size()).
    [[nodiscard]] std::vector<float> score_tokens(std::vector<TokenId> tokens,
                                                  std::uint32_t first_target);

    [[nodiscard]] std::uint32_t count_tokens(PromptInput input,
                                             const PreparationControl& control = {}) const;
    [[nodiscard]] ModelSamplingDefaults sampling_defaults() const;

    // Establishes queue membership synchronously with a fixed output consumer mode. Destroying an
    // unconsumed handle cancels its request; wait() owns result consumption and may run
    // independently from GPU execution. Streaming mode requires a non-null sink in wait() and
    // publishes one exact GenerationStart before output deltas; Aggregate mode requires a null
    // sink. Observation options request protocol-neutral publication facts without changing the
    // execution request.
    [[nodiscard]] GenerationHandle
    submit(PreparedPrompt prompt, RequestOptions options,
           OutputConsumerMode consumer_mode                       = OutputConsumerMode::Aggregate,
           GenerationObservationOptions observation               = {},
           std::chrono::steady_clock::time_point pending_deadline = {});

    GenerationResult generate(PreparedPrompt prompt, RequestOptions options,
                              OutputSink* sink                     = nullptr,
                              const CancellationView& cancellation = {});

    // Serve batch step (Committee A+B, single scheduler): one StepPlan ->
    // one production forward over the dispatch ragged batch (embed at
    // m == tokens.size(), causal_softmax_attention_ragged over the batch
    // device view, EXL3 linear per layer, sample_decode_rows for decode rows
    // only) through the Engine-owned single-slot forward context. Hook loop
    // owns no TextContext/KV/GDN bytes; the scheduler owns page ids; Engine
    // reads dispatch only.
    [[nodiscard]] std::vector<std::pair<std::uint64_t, TokenId>>
    run_batch_step(const batch::StepPlan& plan, const batch::StepDispatch& dispatch);

    // Serve helpers: prompt ids for scheduler admission (no Engine queue),
    // plus host-side detokenization for pump-published deltas. The full
    // thinking/reasoning/stop-string/tool-call policy stays in OutputSession
    // (EngineCore path); this is the minimal pump text path.
    [[nodiscard]] std::vector<TokenId> prompt_token_ids(const PreparedPrompt& prompt) const;
    [[nodiscard]] std::string decode_tokens(std::span<const TokenId> ids) const;
    // Serve stop set: model-default stop-token ids (EOS et al) for the pump's
    // EOS finish. Caller stop policy merges at prepare; the full
    // thinking/reasoning/stop-string policy stays in OutputSession.
    [[nodiscard]] std::vector<TokenId> default_stop_token_ids() const;
    [[nodiscard]] ResolvedSamplingParameters resolved_sampling(const PreparedPrompt& prompt,
                                                               const RequestOptions& options) const;

    // Serve mode guard: GenerationService sets this at construction. While
    // set, submit() throws (serve admits via hook_loop EngineHooks::
    // on_new_request and steps via run_batch_step; per-client generate
    // loops are banned on the serve path). CLI never sets it, so generate()
    // keeps working there.
    void set_serve_mode(bool serve);

    [[nodiscard]] const EngineOptions& options() const;
    [[nodiscard]] LoadSummary load_summary() const;
    [[nodiscard]] MemorySummary memory_summary() const;
    [[nodiscard]] RuntimeStats runtime_stats() const;
    [[nodiscard]] MediaCacheSummary media_cache_summary() const;
    [[nodiscard]] bool is_available() const;

    void reset_memory_peaks() noexcept;

private:
    class Impl;
    std::shared_ptr<Impl> impl_;
};

} // namespace ninfer

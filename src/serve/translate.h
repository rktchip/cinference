#pragma once

// Adapter between the wire-independent protocol request and the public engine API.

#include "ninfer/types.h"
#include "serve/request.h"
#include "serve/serve_options.h"

#include <functional>

#include <string>
#include <string_view>
#include <vector>

namespace ninfer::serve {

// Media acquisition is a product-layer concern. Translation preserves part order
// and asks the caller to turn each wire source into owning bytes before the
// target frontend sees it.
using MediaAcquirer = std::function<ninfer::OwnedMedia(const ContentPart&)>;

struct ResolvedPromptSemantics {
    std::optional<bool> enable_thinking;
    std::optional<ninfer::ReasoningEffort> reasoning_effort;
    std::optional<bool> preserve_thinking;
    std::string chat_template_kwargs_json;
};

ResolvedPromptSemantics resolve_prompt_semantics(const GenerationRequest& req,
                                                 const ServeOptions& server);

ninfer::PromptInput to_prompt_input(const GenerationRequest& req,
                                    const ResolvedPromptSemantics& semantics,
                                    const MediaAcquirer& acquire_media);

// Build public request options (output budget, thinking, stop policy, sampler). The
// sampler is resolved from the request's SamplingParams over the server defaults;
// --greedy on the server forces exact argmax regardless of the request. Prefix-reuse
// participation is resolved by GenerationService and supplied explicitly because
// operational requests do not inherit the external-traffic policy.
ninfer::RequestOptions to_request_options(const GenerationRequest& req, const ServeOptions& server,
                                          const ResolvedPromptSemantics& semantics,
                                          bool allow_prefix_reuse);

struct GenerationOutcome;

// Response formatting: route model thinking spans out of content.
//
// The serve pump decodes raw tokens, so Qwen thinking markers ("<think>" and
// "</think>") can reach the formatters inside the content text. The helpers
// below split them back into the reasoning channel. They are pure string
// scans: deterministic for a fixed input, with no sampler or forward-path
// state.
struct SplitThinkingResult {
    std::string reasoning;
    std::string content;
};

// One-shot split of a complete raw text. starts_in_reasoning mirrors the
// prompt summary and the engine decoder: a reasoning-first text splits at
// the first "</think>" (leading text with no opening marker is reasoning;
// without a close it is all reasoning), and everything after that close is
// literal content. Content-first texts pass through untouched.
SplitThinkingResult split_thinking_spans(std::string_view text, bool starts_in_reasoning);

// Rewrite a terminal outcome in place: thinking spans leave text and join
// reasoning. Content-first texts pass through untouched; a reasoning-first
// text with no close marker is all reasoning, matching the engine decoder.
void apply_thinking_split(GenerationOutcome& outcome, bool starts_in_reasoning);

// Incremental form of split_thinking_spans for streaming deltas. Feeding the
// raw chunks in order and then calling finish() yields the same reasoning and
// content bytes as the one-shot split of the concatenated text.
class ThinkingSpanSplitter {
public:
    struct Segment {
        ninfer::OutputChannel channel = ninfer::OutputChannel::Content;
        std::string text;
    };

    explicit ThinkingSpanSplitter(bool starts_in_reasoning);

    std::vector<Segment> feed(std::string_view chunk);
    std::vector<Segment> finish();

private:
    void emit_reasoning(std::vector<Segment>& out, std::string_view bytes);
    void emit_content(std::vector<Segment>& out, std::string_view bytes);

    bool in_reasoning_;
    const bool seeded_reasoning_first_;
    bool first_       = true;
    bool strip_ws_    = false;
    bool closed_once_ = false;
    std::string pending_;
};

} // namespace ninfer::serve

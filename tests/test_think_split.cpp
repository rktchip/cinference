// Host-level formatter test: model thinking spans must route to the
// reasoning channel and stay out of HTTP content. Exercises the shared
// translate.h splitter (one-shot + incremental equivalence) and the OpenAI
// chat non-streaming/streaming response builders. No engine, no GPU.
#include "serve/generation_service.h"
#include "serve/openai_chat.h"
#include "serve/translate.h"

#include <nlohmann/json.hpp>

#include <cstddef>
#include <iostream>
#include <string>
#include <string_view>
#include <vector>

namespace {

using namespace ninfer::serve;
using Json = nlohmann::json;

int failures = 0;

void check(bool condition, const std::string& label) {
    if (condition) { return; }
    std::cerr << "FAIL: " << label << '\n';
    ++failures;
}

bool contains(std::string_view haystack, std::string_view needle) {
    return haystack.find(needle) != std::string_view::npos;
}

void expect_split(std::string_view raw, bool starts_in_reasoning, std::string_view reasoning,
                  std::string_view content, const std::string& label) {
    const SplitThinkingResult split = split_thinking_spans(raw, starts_in_reasoning);
    check(split.reasoning == reasoning, label + " reasoning=<" + split.reasoning + ">");
    check(split.content == content, label + " content=<" + split.content + ">");
}

void accumulate(const std::vector<ThinkingSpanSplitter::Segment>& segments, std::string& reasoning,
                std::string& content) {
    for (const auto& segment : segments) {
        if (segment.channel == ninfer::OutputChannel::Reasoning) {
            reasoning += segment.text;
        } else {
            content += segment.text;
        }
    }
}

// Incremental feed+finish must equal the one-shot split for every 2-chunk
// cut and for byte-at-a-time delivery.
void expect_stream_equivalent(std::string_view raw, bool starts_in_reasoning,
                              const std::string& label) {
    const SplitThinkingResult want = split_thinking_spans(raw, starts_in_reasoning);
    for (std::size_t cut = 0; cut <= raw.size(); ++cut) {
        ThinkingSpanSplitter splitter(starts_in_reasoning);
        std::string reasoning;
        std::string content;
        accumulate(splitter.feed(raw.substr(0, cut)), reasoning, content);
        accumulate(splitter.feed(raw.substr(cut)), reasoning, content);
        accumulate(splitter.finish(), reasoning, content);
        check(reasoning == want.reasoning && content == want.content,
              label + " 2-chunk cut=" + std::to_string(cut));
    }
    {
        ThinkingSpanSplitter splitter(starts_in_reasoning);
        std::string reasoning;
        std::string content;
        for (char byte : raw) { accumulate(splitter.feed(std::string_view(&byte, 1)), reasoning, content); }
        accumulate(splitter.finish(), reasoning, content);
        check(reasoning == want.reasoning && content == want.content, label + " byte-at-a-time");
    }
}

int test_one_shot() {
    expect_split("Considering the options.</think>\n\nParis", true, "Considering the options.",
                 "Paris", "qwen-prefill");
    expect_split("<think>abc</think>def", true, "abc", "def", "explicit-open-sir");
    expect_split("<think>abc</think>def", false, "", "<think>abc</think>def", "content-first-literal");
    expect_split("Just Paris", true, "Just Paris", "", "untagged-sir-is-reasoning");
    expect_split("Just Paris", false, "", "Just Paris", "untagged-content-first-unchanged");
    expect_split("half thought", true, "half thought", "", "unclosed-sir");
    expect_split("half thought", false, "", "half thought", "unclosed-content-first");
    expect_split("pre<think>post", false, "", "pre<think>post", "content-first-literal-span");
    expect_split("pre<think>post", true, "pre<think>post", "", "unclosed-sir-keeps-markers");
    expect_split("a</think>b", false, "", "a</think>b", "stray-close-literal");
    expect_split("a</think>b", true, "a", "b", "stray-close-sir-boundary");
    expect_split("a<think>b</think>c<think>d</think>e", true, "a<think>b", "c<think>d</think>e",
                 "sir-single-transition");
    expect_split("", true, "", "", "empty-sir");
    expect_split("", false, "", "", "empty-content-first");
    expect_split("<think></think>X", true, "", "X", "empty-think");
    expect_split("r1</think>c1</think>c2", true, "r1", "c1</think>c2", "second-close-literal");
    expect_split("</think>Paris", true, "", "Paris", "empty-leading-reasoning");
    return failures;
}

int test_stream_equivalence() {
    const std::vector<std::string> corpus = {
        "Considering the options.</think>\n\nParis",
        "<think>abc</think>def",
        "Just Paris",
        "half thought",
        "pre<think>post",
        "a</think>b",
        "a<think>b</think>c<think>d</think>e",
        "",
        "<think></think>X",
        "r1</think>c1</think>c2",
        "</think>Paris",
        "x<thi",
        "<think",
        "ab</th",
    };
    for (const std::string& raw : corpus) {
        expect_stream_equivalent(raw, true, "sir:<" + raw + ">");
        expect_stream_equivalent(raw, false, "cf:<" + raw + ">");
    }
    return failures;
}

Json parse_event(const std::string& event) {
    constexpr std::string_view prefix = "data: ";
    check(event.starts_with(prefix) && event.ends_with("\n\n"), "sse framing");
    return Json::parse(event.substr(prefix.size(), event.size() - prefix.size() - 2));
}

int test_chat_nonstreaming() {
    GenerationOutcome outcome;
    outcome.text      = "Considering the options.</think>\n\nParis";
    outcome.reasoning = "";
    apply_thinking_split(outcome, true);
    check(outcome.text == "Paris", "nonstream content");
    check(outcome.reasoning == "Considering the options.", "nonstream reasoning");

    const OpenAIChatResponseIdentity identity{.id = "chatcmpl-test", .model = "qwen", .created = 1};
    const Json body = Json::parse(make_chat_completion_response(identity, outcome));
    const Json& message = body["choices"][0]["message"];
    check(message["content"] == "Paris", "wire content");
    check(message["reasoning_content"] == "Considering the options.", "wire reasoning_content");
    check(!contains(body.dump(), "<think>") && !contains(body.dump(), "</think>"),
          "wire has no think markers");

    // Untagged content-first text passes through byte-identical.
    GenerationOutcome plain;
    plain.text = "Paris";
    apply_thinking_split(plain, false);
    check(plain.text == "Paris" && plain.reasoning.empty(), "plain passthrough");
    return failures;
}

int test_chat_streaming() {
    const OpenAIChatResponseIdentity identity{.id = "chatcmpl-test", .model = "qwen", .created = 1};
    OpenAIChatStream encoder(identity, false);
    encoder.start();
    ninfer::GenerationStart start;
    start.prompt.prompt_tokens   = 8;
    start.reused_prompt_tokens   = 0;
    encoder.note_start(start);

    // Raw pump text with awkward chunk boundaries across both markers.
    const std::string raw = "Considering the options.</think>\n\nParis";
    const std::vector<std::string> chunks = {"Conside", "ring the opt", "ions.</th", "ink>\n\nPa", "ris"};
    ThinkingSpanSplitter splitter(true);
    std::string streamed_reasoning;
    std::string streamed_content;
    bool content_started = false;
    for (const std::string& chunk : chunks) {
        for (ThinkingSpanSplitter::Segment segment : splitter.feed(chunk)) {
            if (segment.text.empty()) { continue; }
            if (segment.channel == ninfer::OutputChannel::Reasoning && !content_started) {
                streamed_reasoning += segment.text;
                const Json delta = parse_event(encoder.reasoning_delta(segment.text));
                check(!delta["choices"][0]["delta"].contains("content"), "reasoning delta has no content");
            } else {
                content_started = true;
                streamed_content += segment.text;
                const Json delta = parse_event(encoder.content_delta(segment.text));
                check(!delta["choices"][0]["delta"].contains("reasoning_content"),
                      "content delta has no reasoning");
            }
        }
    }
    for (ThinkingSpanSplitter::Segment segment : splitter.finish()) {
        if (segment.text.empty()) { continue; }
        content_started = true;
        streamed_content += segment.text;
        parse_event(encoder.content_delta(segment.text));
    }
    check(streamed_reasoning == "Considering the options.", "streamed reasoning total");
    check(streamed_content == "Paris", "streamed content total");
    check(!contains(streamed_content, "think") && !contains(streamed_content, "Considering"),
          "think text out of streamed content");

    GenerationOutcome outcome;
    outcome.text = raw;
    GenerationOutcome display = outcome;
    const SplitThinkingResult split = split_thinking_spans(display.text, true);
    const std::string canonical     = display.reasoning + split.reasoning;
    if (canonical.starts_with(streamed_reasoning) && split.content.starts_with(streamed_content)) {
        display.reasoning = canonical;
        display.text      = std::move(split.content);
    } else {
        display.reasoning = streamed_reasoning;
        display.text      = streamed_content;
    }
    const std::vector<std::string> terminal = encoder.finish(display);
    check(!terminal.empty() && terminal.back() == "data: [DONE]\n\n", "stream terminates");
    return failures;
}

} // namespace

int main() {
    test_one_shot();
    test_stream_equivalence();
    test_chat_nonstreaming();
    test_chat_streaming();
    if (failures == 0) { std::cout << "think-split: all checks passed\n"; }
    return failures == 0 ? 0 : 1;
}

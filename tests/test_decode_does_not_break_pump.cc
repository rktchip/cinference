// tests/test_decode_does_not_break_pump.cc
// Slot D seam test (a): GenerationService::run() must not break the pump on
// decode rows.
//
// TARGET contract (slot A removes the decode break concurrently): run()
// drives schedule_step -> dispatch_step -> on_step_done for EVERY non-empty
// plan, including plans that carry decode_seq_ids. No decode gate with a
// break, and no "until that runner lands" holding comment, may remain.
//
// Fixture form: this TU reads src/serve/generation_service.cpp at runtime
// (NINFER_SOURCE_DIR under ctest, else derived from __FILE__), extracts the
// GenerationService::run body by brace matching on the comment-stripped
// source, and asserts the TARGET shape:
//   - the run body contains no "decode_seq_ids.empty()" gate,
//   - no "decode_seq_ids" occurrence in the run body is followed by a break,
//   - the raw file contains no "until that runner lands" comment.
// Standalone TU: STL only, no library dependencies, no device handle.
//
// NOTE: written against the TARGET contract, so it FAILS on the pre-A tree
// (run breaks when decode_seq_ids is non-empty) and passes once A lands.
// GPU: GATED (no execution).
#include <iostream>
#include <string>

#ifndef NINFER_SOURCE_DIR
#define NINFER_SOURCE_DIR ""
#endif

namespace {

int failures = 0;

#define CHECK(cond)                                                                  \
    do {                                                                             \
        if (!(cond)) {                                                               \
            ++failures;                                                              \
            std::cout << "FAIL " << __LINE__ << ": " #cond "\n";                     \
        }                                                                            \
    } while (0)

std::string normalize_seps(std::string path) {
    for (char& c : path) {
        if (c == '\\') {
            c = '/';
        }
    }
    return path;
}

// Derive the repo root from this TU's own path:
// <root>/tests/test_decode_does_not_break_pump.cc -> <root>.
std::string repo_root_from_here(const char* here) {
    std::string path = normalize_seps(here);
    const std::string suffix = "tests/test_decode_does_not_break_pump.cc";
    if (path.size() > suffix.size() &&
        path.compare(path.size() - suffix.size(), suffix.size(), suffix) == 0) {
        path.erase(path.size() - suffix.size());
        while (!path.empty() && path.back() == '/') {
            path.pop_back();
        }
        return path;
    }
    return "";
}

std::string read_file(const std::string& path) {
    std::string out;
    FILE* file = nullptr;
#if defined(_MSC_VER)
    fopen_s(&file, path.c_str(), "rb");
#else
    file = std::fopen(path.c_str(), "rb");
#endif
    if (file == nullptr) {
        return out;
    }
    char chunk[4096];
    std::size_t n = 0;
    while ((n = std::fread(chunk, 1, sizeof(chunk), file)) > 0) {
        out.append(chunk, n);
    }
    std::fclose(file);
    return out;
}

// Strip // line comments and /* */ block comments, respecting string and
// character literals (comment text must not satisfy or hide the checks;
// comment braces must not disturb brace matching).
std::string strip_comments(const std::string& source) {
    std::string out;
    out.reserve(source.size());
    bool in_line = false;
    bool in_block = false;
    bool in_string = false;
    bool in_char = false;
    for (std::size_t i = 0; i < source.size(); ++i) {
        const char c = source[i];
        const char next = (i + 1 < source.size()) ? source[i + 1] : '\0';
        if (in_line) {
            if (c == '\n') {
                in_line = false;
                out.push_back(c);
            }
            continue;
        }
        if (in_block) {
            if (c == '*' && next == '/') {
                in_block = false;
                ++i;
            }
            continue;
        }
        if (in_string) {
            out.push_back(c);
            if (c == '\\' && i + 1 < source.size()) {
                out.push_back(source[++i]);
            } else if (c == '"') {
                in_string = false;
            }
            continue;
        }
        if (in_char) {
            out.push_back(c);
            if (c == '\\' && i + 1 < source.size()) {
                out.push_back(source[++i]);
            } else if (c == '\'') {
                in_char = false;
            }
            continue;
        }
        if (c == '/' && next == '/') {
            in_line = true;
            ++i;
            continue;
        }
        if (c == '/' && next == '*') {
            in_block = true;
            ++i;
            continue;
        }
        if (c == '"') {
            in_string = true;
        } else if (c == '\'') {
            in_char = true;
        }
        out.push_back(c);
    }
    return out;
}

// Extract the body of GenerationService::run (without the outer braces).
// Returns false when the definition or its brace match is not found.
bool extract_run_body(const std::string& source, std::string& body) {
    const std::string anchor = "GenerationService::run(";
    const std::size_t at = source.find(anchor);
    if (at == std::string::npos) {
        return false;
    }
    std::size_t open = source.find('{', at);
    if (open == std::string::npos) {
        return false;
    }
    int depth = 0;
    for (std::size_t i = open; i < source.size(); ++i) {
        if (source[i] == '{') {
            ++depth;
        } else if (source[i] == '}') {
            --depth;
            if (depth == 0) {
                body = source.substr(open + 1, i - open - 1);
                return true;
            }
        }
    }
    return false;
}

bool contains(const std::string& haystack, const std::string& needle) {
    return haystack.find(needle) != std::string::npos;
}

// True when a "break" follows a decode_seq_ids mention within window chars
// (the pre-A gate keeps its break inside the same if body, so any window
// covering the block body catches it; comment stripping already shrank the
// gap to the code tokens).
bool decode_gate_breaks(const std::string& run_body) {
    const std::string needle = "decode_seq_ids";
    const std::size_t window = 400;
    std::size_t at = 0;
    while ((at = run_body.find(needle, at)) != std::string::npos) {
        const std::string tail = run_body.substr(at, window);
        if (tail.find("break") != std::string::npos) {
            return true;
        }
        at += needle.size();
    }
    return false;
}

} // namespace

int main() {
    std::string root = NINFER_SOURCE_DIR;
    if (root.empty()) {
        root = repo_root_from_here(__FILE__);
    }
    CHECK(!root.empty());
    const std::string path = normalize_seps(root) + "/src/serve/generation_service.cpp";
    const std::string source = read_file(path);
    CHECK(!source.empty());
    if (source.empty()) {
        std::cout << "decode_no_break: FAILURES (cannot open " << path << ")\n";
        return 1;
    }

    // Sanity: we scanned the right file and the right function.
    CHECK(contains(source, "GenerationService::run("));
    CHECK(contains(source, "schedule_step("));

    // The holding comment lives in comments, so check the raw source: the
    // TARGET tree carries no "until that runner lands" text anywhere.
    CHECK(!contains(source, "until that runner lands"));

    // Comment text must not satisfy or hide the gate check, so match braces
    // on the comment-stripped source and inspect the run body only. The
    // legitimate plan.empty() -> break stays; only a decode-gated break is
    // banned.
    const std::string code = strip_comments(source);
    std::string run_body;
    CHECK(extract_run_body(code, run_body));
    if (!run_body.empty()) {
        // TARGET: no decode gate in run() at all ...
        CHECK(!contains(run_body, "decode_seq_ids.empty()"));
        // ... and specifically no decode mention followed by a break.
        CHECK(!decode_gate_breaks(run_body));
        // The empty-plan break is the only sanctioned break in the pump.
        CHECK(contains(run_body, "plan.empty()"));
    }

    std::cout << "decode_no_break GPU section: GATED (source fixture only, no execution)\n";
    if (failures == 0) {
        std::cout << "decode_no_break: PASS (run() pumps decode rows, no decode break)\n";
    } else {
        std::cout << "decode_no_break: " << failures
                  << " FAILURES (decode break still present -- awaits slot A)\n";
    }
    return failures == 0 ? 0 : 1;
}

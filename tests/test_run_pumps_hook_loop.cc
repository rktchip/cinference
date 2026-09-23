// tests/test_run_pumps_hook_loop.cc
// Swarm G2 seam test (a): GenerationService::run() pumps the hook loop.
//
// TARGET contract (swarm A rewiring in flight): run() drives the S1 pump --
// schedule_step appears inside GenerationService::run -- and no
// wait-on-submit-generation remains (the Engine GenerationHandle .wait() the
// pre-A2 run() used while the worker loop drove the token loop elsewhere).
//
// Fixture form: this TU reads src/serve/generation_service.cpp at runtime
// (NINFER_SOURCE_DIR under ctest, else derived from __FILE__), strips
// comments, extracts the GenerationService::run body by brace matching, and
// asserts the TARGET shape: schedule_step( present, .wait( absent,
// generation.wait absent, and no legacy generate(/execute( entry.
// Standalone TU: STL only, no library dependencies, no device handle.
//
// NOTE: written against the TARGET contract, so it FAILS on the pre-A2 tree
// (run waits on prepared.generation, no schedule_step pump yet) and passes
// once A2 lands. GPU: GATED (no execution).
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
// <root>/tests/test_run_pumps_hook_loop.cc -> <root>.
std::string repo_root_from_here(const char* here) {
    std::string path = normalize_seps(here);
    const std::string suffix = "tests/test_run_pumps_hook_loop.cc";
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
        std::cout << "run_pump: FAILURES (cannot open " << path << ")\n";
        return 1;
    }

    // Sanity: we scanned the right file (hook-loop admission gate present).
    CHECK(contains(source, "throw_if_pages_exhausted"));

    // Comment text (e.g. "drives schedule_step -> ...") must not satisfy the
    // pump check, so match braces on the comment-stripped source.
    const std::string code = strip_comments(source);
    std::string run_body;
    CHECK(extract_run_body(code, run_body));
    if (!run_body.empty()) {
        // TARGET pump: run() drives the hook loop via schedule_step.
        CHECK(contains(run_body, "schedule_step("));
        // TARGET: no wait-on-submit-generation remains in run().
        CHECK(!contains(run_body, ".wait("));
        CHECK(!contains(run_body, "generation.wait"));
        // Legacy execute entries must not appear in run().
        CHECK(!contains(run_body, "generate("));
        CHECK(!contains(run_body, "execute("));
        CHECK(!contains(run_body, "engine_->generate"));
        CHECK(!contains(run_body, "engine_->submit"));
    }

    std::cout << "run_pump GPU section: GATED (source fixture only, no execution)\n";
    if (failures == 0) {
        std::cout << "run_pump: PASS (run() pumps schedule_step, no submit-generation wait)\n";
    } else {
        std::cout << "run_pump: " << failures
                  << " FAILURES (TARGET contract not met -- awaits A2 rewire)\n";
    }
    return failures == 0 ? 0 : 1;
}

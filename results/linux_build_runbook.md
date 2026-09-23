# Linux build runbook (post-WSL2) -- ninfer-serve + ctest two-sessions

Status: WSL2 NOT INSTALLED on this box (re-verified 2026-09-22 UTC:
`wsl --status` prints "The Windows Subsystem for Linux is not
installed..." and exits 50). No install attempted: `wsl --install`
needs admin plus a reboot, which is a user action. This file is the
exact sequence to run once WSL2/Ubuntu is present. Nothing below has
been executed; there is no linked Linux ninfer-serve yet.

Why Linux is required: full configure on Windows stops at the first
fatal error `Could NOT find PkgConfig (missing:
PKG_CONFIG_EXECUTABLE)` (cmake exit 1). PkgConfig gates the FFmpeg
check (libavformat libavcodec libavutil libswscale) and the libcurl
check (libcurl>=7.85) in cmake/Dependencies.cmake, so configure never
reaches apps/. pkg-config, FFmpeg dev, and libcurl dev are all absent
on the Windows side. The single authoritative platform statement lives
at apps/CMakeLists.txt:1-5: full serve is Linux-only until WSL2.

## 0. One-time Windows setup (user action, admin shell, then reboot)

    wsl --install -d Ubuntu

Reboot when prompted, create the Ubuntu user, then verify from
Windows:

    wsl --status
    wsl -l -v

## 1. Inside Ubuntu: toolchain + dev packages

CUDA note: WSL2 reuses the Windows NVIDIA driver; still install a CUDA
toolkit that supports sm_120a (the repo pins
CMAKE_CUDA_ARCHITECTURES=120a and fatals otherwise; the Windows box
uses toolkit 13.3). Either the NVIDIA WSL-Ubuntu apt repo
(cuda-toolkit package) or a matching runfile install works. A CUDA
GPU must be visible (`nvidia-smi`) to run the served engine; without
one, build + host tests still work but the serve test cannot run.

    sudo apt-get update
    sudo apt-get install -y pkg-config cmake ninja-build g++ python3 curl \
      libavformat-dev libavcodec-dev libavutil-dev libswscale-dev \
      libcurl4-openssl-dev

Package-to-check mapping (cmake/Dependencies.cmake):
pkg-config -> PkgConfig; libavformat-dev libavcodec-dev libavutil-dev
libswscale-dev -> FFMPEG modules; libcurl4-openssl-dev -> LIBCURL
(needs >= 7.85 for CURLOPT_PROTOCOLS_STR); python3 -> tests
(find_package Python3 Interpreter); cmake/ninja-build/g++ -> build.

## 2. Get the source (either works; never build in-source)

    git clone <repo-url> ~/cinference        # fresh clone, or
    ls /mnt/c/src/cinference/CMakeLists.txt  # bind-mount of this box

## 3. Configure (explicit -S and -B, out-of-source)

    cmake -S ~/cinference -B ~/cinference-build -G Ninja \
      -DCMAKE_BUILD_TYPE=Release \
      -DNINFER_BUILD_APPS=ON -DBUILD_TESTING=ON

BUILD_TESTING=ON is required: it enables testing and adds tests/,
which registers the serve session test. NINFER_BUILD_APPS=ON builds
apps/ (ninfer, ninfer-serve, ninfer-perplexity).

## 4. Build the serve target

    cmake --build ~/cinference-build --target ninfer-serve -j"$(nproc)"

Expected artifact (no RUNTIME_OUTPUT_DIRECTORY override, so the
target lands under apps/): ~/cinference-build/apps/ninfer-serve.
CLI contract (apps/serve/main.cpp + src/serve/serve_options.cpp):

    ninfer-serve <model.ninfer> [--host H] [--port N] [--greedy]

## 5. Run the session test

Registered in tests/serve/tests.cmake as ninfer_two_sessions_test
(non-WIN32 only), command `bash test_two_sessions.sh`, SKIP_RETURN_CODE
77, TIMEOUT 300. The script starts one server, waits on /health, fires
two concurrent POST /v1/chat/completions clients with distinct prompts
(ALPHA / BETA), and requires HTTP 200 plus non-empty choice text from
both. It exits 77 (SKIP, not FAIL) when not on Linux, when the binary
is missing, or when NINFER_TEST_MODEL is unset.

Targeted run (needs a .ninfer model artifact):

    NINFER_SERVE_BIN=~/cinference-build/apps/ninfer-serve \
    NINFER_TEST_MODEL=<path-to-model.ninfer> \
    NINFER_TEST_HOST=127.0.0.1 NINFER_TEST_PORT=18080 \
    ctest --test-dir ~/cinference-build \
      -R ninfer_two_sessions_test --output-on-failure

Full suite afterwards:

    ctest --test-dir ~/cinference-build --output-on-failure

## 6. Report back

EXIT is met when either (a) branch holds: a linked ninfer-serve plus
ctest output, or (b) branch holds (current state): this runbook plus a
current Windows TU-gate table (results/tu_gate_s6b_windows.md, 21/21).
On success report: `wsl -l -v` output, cmake configure log tail, the
linked binary path, and the ctest log (PASS / SKIP-77 with reason /
FAIL with server log excerpt).

# PLATFORM GATE (single authoritative statement -- do not restate elsewhere):
# full serve (ninfer-serve plus media decode and libcurl acquire) is
# Linux-only until WSL2 is installed on this box. Windows gates cover TU
# compiles only; no Windows link of serve binaries is claimed.
message(STATUS "cinference: full serve is Linux-only until WSL2 (Windows gates cover TU compiles only)")

add_executable(ninfer
  cli/main.cpp
  cli/options.cpp)
ninfer_internal_includes(ninfer)
target_link_libraries(ninfer PRIVATE
  ninfer_engine
  ninfer_product_logging
  ninfer_product_prompt_input)

add_executable(ninfer-serve serve/main.cpp)
ninfer_internal_includes(ninfer-serve)
target_link_libraries(ninfer-serve PRIVATE ninfer_serve ninfer_product_logging)

add_executable(ninfer-perplexity
  perplexity/main.cpp
  perplexity/corpus.cpp
  perplexity/evaluation.cpp)
ninfer_internal_includes(ninfer-perplexity)
target_link_libraries(ninfer-perplexity PRIVATE
  ninfer_engine ninfer_product_logging ninfer::json)

# Linux-ready binary names: cinference (CLI) and cinference-serve share the
# one engine entry (ninfer_engine) via alias; they configure and link on
# Linux. No Windows link is claimed.
add_executable(cinference ALIAS ninfer)
add_executable(cinference-serve ALIAS ninfer-serve)

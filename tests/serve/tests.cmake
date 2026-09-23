# Serve-level session tests (Linux-only; the served engine needs a device).
# The two-sessions script skips itself (exit 77) when NINFER_TEST_MODEL is
# unset or the platform is not Linux, so wiring it here never breaks host
# or Windows runs.
if(NOT WIN32)
  add_test(NAME ninfer_two_sessions_test
    COMMAND bash "${CMAKE_CURRENT_LIST_DIR}/test_two_sessions.sh")
  set_tests_properties(ninfer_two_sessions_test PROPERTIES
    SKIP_RETURN_CODE 77
    TIMEOUT 300)
endif()

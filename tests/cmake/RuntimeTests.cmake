ninfer_add_test(ninfer_admission_policy_test SOURCES "${CMAKE_CURRENT_LIST_DIR}/../test_admission_policy.cpp"
  LIBRARIES ninfer_runtime_support ninfer_batch)

ninfer_add_test(ninfer_context_cost_test SOURCES "${CMAKE_CURRENT_LIST_DIR}/../test_context_cost.cpp"
  LIBRARIES ninfer_runtime_support ninfer::json)

ninfer_add_test(ninfer_resource_manager_test SOURCES "${CMAKE_CURRENT_LIST_DIR}/../test_resource_manager.cpp"
  LIBRARIES ninfer_runtime_support)

ninfer_add_test(ninfer_kv_capacity_test SOURCES "${CMAKE_CURRENT_LIST_DIR}/../test_kv_capacity.cpp"
  LIBRARIES ninfer_runtime_support)

ninfer_add_test(ninfer_sampling_defaults_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/../test_sampling_defaults.cpp"
  LIBRARIES ninfer_engine ninfer_core)

ninfer_add_test(ninfer_continuous_batch_test SOURCES "${CMAKE_CURRENT_LIST_DIR}/../test_continuous_batch.cc"
  LIBRARIES ninfer_batch)

ninfer_add_test(ninfer_mixed_step_test SOURCES "${CMAKE_CURRENT_LIST_DIR}/../test_mixed_step.cc"
  LIBRARIES ninfer_batch)

ninfer_add_test(ninfer_step_forward_m_test SOURCES "${CMAKE_CURRENT_LIST_DIR}/../test_step_forward_m.cc"
  LIBRARIES ninfer_batch)

# Swarm G proof deltas: mixed-step M + fused-qkv counter (host-only, real
# scheduler, GPU GATED).
ninfer_add_test(ninfer_mixed_step_m_test SOURCES "${CMAKE_CURRENT_LIST_DIR}/../test_mixed_step_m.cc"
  LIBRARIES ninfer_batch)

# Swarm G2 seam tests: run-pump fixture (post-A2: run() pumps the hook loop;
# host-only source grep, GPU GATED) and the ragged
# token->(seq,page) map over the real batch resolvers (host-only, GPU GATED).
ninfer_add_test(ninfer_run_pump_test SOURCES "${CMAKE_CURRENT_LIST_DIR}/../test_run_pumps_hook_loop.cc"
  NEEDS_SOURCE_DIR)

ninfer_add_test(ninfer_ragged_map_test SOURCES "${CMAKE_CURRENT_LIST_DIR}/../test_ragged_attention_map.cc"
  LIBRARIES ninfer_batch)
target_include_directories(ninfer_ragged_map_test PRIVATE
  ${PROJECT_SOURCE_DIR}/src ${PROJECT_SOURCE_DIR}/include)

# Slot D: decode-break removal target fixture (host-only source grep, GPU
# GATED) and mixed-step decode-emit over the real scheduler with a mocked
# sample step (host-only, GPU GATED).
ninfer_add_test(ninfer_decode_no_break_test SOURCES "${CMAKE_CURRENT_LIST_DIR}/../test_decode_does_not_break_pump.cc"
  NEEDS_SOURCE_DIR)

ninfer_add_test(ninfer_mixed_step_decode_emit_test SOURCES "${CMAKE_CURRENT_LIST_DIR}/../test_mixed_step_decode_emit.cc"
  LIBRARIES ninfer_batch)

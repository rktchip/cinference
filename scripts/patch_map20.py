import json

P = "/mnt/c/src/cinference/docs/forward-map.json"
d = json.load(open(P))
e = d["engine_now"]

e["bucket_anatomy_2026_09_26"] = {
    "status": "CORRECTED full-name attribution, graph25 = 120897 kernels "
              "/ 902ms / 64 toks = 1889/tok; graphs 19/22/25 = 3 prompts, "
              "11/13 = prefill, 16 = 39ms unknown",
    "correction": "gemm 'm3/m4/m5' labels were BITS (kernel template "
                  "<BITS,CB,BM,BN,BK,NW,ST,SPLIT,...>), NOT batch M. "
                  "Spec-off decode IS pure M=1 (gemm gridY=1 everywhere).",
    "routing_actual": "Unfused M=1 K3/K4 -> gemv_plain (219 launches/tok); "
                      "fused multi-group (gate_up grid272 n=34816 UNSHARDED "
                      "both shards one launch, qkv grid112 n=14336) -> "
                      "gemm_m SPLIT + separate epilogue (80 launches/tok); "
                      "lm_head-class BITS5 grid40 split13 ~1/tok + "
                      "had_in grid6 + epilogue grid5 companions.",
    "had_gridY_is_k_over_128": "40->5120, 48->6144, 136->17408; "
                               "had_in_kernel grid5 n=5040 (78.75/tok) = all "
                               "fused layers.",
    "counts_balance": "gate_up 45.3 BITS3 + 17.7 BITS4 = 63 ~= 64 layers; "
                      "qkv 15.75 ~= 16 full layers; down had 63 ~= 64.",
    "open_shape_questions": "BITS5 identity (head_bits=6, mtp=4, bulk 3.5: "
                            "what is 5? down/o_proj sensitive-layer bits?); "
                            "n=14336 qkv composition; K3-grid160 vs "
                            "K4-grid160 same-grid 37us vs 20us (cfg/k differ "
                            "or K3 path slower). Step-4 diagnostics, read "
                            "from quant files not names."}

e["step4_prediction_2026_09_26"] = {
    "status": "PAPER PREDICTION WRITTEN, code not started",
    "target": "spec-off 13.3 -> ~11 ms/tok via EXL3 linear efficiency "
              "(bucket #1 trigger: linears ~9.6 vs 6.0 floor)",
    "mechanism": ["Extra launches: 4/linear (had_in+gemv+had_out+cast) + "
                  "fused gemm pays separate epilogue; ~900 extra nodes x "
                  "~1.5us in-graph cost ~= 1.0-1.4ms",
                  "Wave quantization: occ_blocks=2 fallback caps at 296 "
                  "blocks; grid320 kernels run 2 waves with 8% tail "
                  "(~0.5ms); grid272 split2 similar",
                  "bf16 strays: 2 full passes/layer, traffic-small but "
                  "2 nodes x 300 linears"],
    "predicted_delta": "1.5-2.5 ms/tok (medium-low)",
    "kill_line": "<0.5ms measured -> revert, record, move to step 5",
    "floor_uncertainty": "If BITS5 = down/o_proj at higher bits, floor "
                         "10.643GB is UNDERSTATED and headroom shrinks; "
                         "verify from quant files during implementation.",
    "first_cut": "overhead bundle (occupancy cap + epilogue fuse + cast "
                 "hoist, all low-risk same mechanism) predict 0.8-1.2; "
                 "coop fusion separately predict 0.6-1.0."}

json.dump(d, open(P, "w"), indent=1)
print("MAP20-OK")

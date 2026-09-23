# scripts/selfcheck.ps1
# Phase 1 slot E selfcheck (Windows primary; run via powershell or pwsh).
# Source-grep checks only: no build, no test binaries, no GPU, no tree edits.
# Prints PHASE <id> PASS|FAIL lines plus INFO lines, then a final STATE= line.
# STATE=LINUX_ONLY means P1-P6 plus P8-P11 all pass (P7 report-only pre-Phase-4);
# STATE=CODE means a real tree problem.
# ASCII only.
$ErrorActionPreference = 'Continue'
$RepoRoot = Split-Path -Parent $PSScriptRoot
$script:Failed = @()

function RPath([string]$Rel) { return (Join-Path $RepoRoot $Rel) }

function File-Lines([string]$Rel) {
  $p = RPath $Rel
  if (-not (Test-Path $p)) { return $null }
  return @(Get-Content -Encoding Ascii $p)
}

function File-Text([string]$Rel) {
  $p = RPath $Rel
  if (-not (Test-Path $p)) { return $null }
  return (Get-Content -Raw -Encoding Ascii $p)
}

function Count-Matches([string]$Rel, [string]$Pattern) {
  $p = RPath $Rel
  if (-not (Test-Path $p)) { return -1 }
  $m = Select-String -Path $p -Pattern $Pattern
  if ($null -eq $m) { return 0 }
  return @($m).Count
}

function Grep-Count([string[]]$RelDirs, [string]$Pattern) {
  $total = 0
  foreach ($d in $RelDirs) {
    $p = RPath $d
    if (-not (Test-Path $p)) { continue }
    $files = Get-ChildItem -Path $p -Recurse -File -Include '*.cpp','*.h','*.cc'
    foreach ($f in $files) {
      $m = Select-String -Path $f.FullName -Pattern $Pattern
      if ($null -ne $m) { $total += @($m).Count }
    }
  }
  return $total
}

function Phase([string]$Name, [bool]$Ok, [string]$Detail) {
  if ($Ok) { Write-Output ("PHASE " + $Name + " PASS " + $Detail) }
  else { Write-Output ("PHASE " + $Name + " FAIL " + $Detail); $script:Failed += $Name }
}

function First-Index([string[]]$Lines, [string]$Pattern) {
  return (Next-Index $Lines $Pattern -1)
}

function Next-Index([string[]]$Lines, [string]$Pattern, [int]$After) {
  if ($null -eq $Lines) { return -1 }
  for ($i = $After + 1; $i -lt $Lines.Count; $i++) {
    if ($Lines[$i] -match $Pattern) { return $i }
  }
  return -1
}

Write-Output ("INFO root=" + $RepoRoot)

# P1_PUMP: run() drives schedule_step -> dispatch_step -> on_step_done,
# no .wait( inside run(), exactly one submit( in src/serve + apps/serve.
$gs = 'src/serve/generation_service.cpp'
$gsLines = File-Lines $gs
$runStart = First-Index $gsLines 'GenerationService::run'
$runEnd = -1
if ($runStart -ge 0) {
  for ($i = $runStart + 1; $i -lt $gsLines.Count; $i++) {
    if ($gsLines[$i] -match '^void GenerationService::warmup') { $runEnd = $i; break }
  }
  if ($runEnd -lt 0) { $runEnd = $gsLines.Count }
}
$body = ''
if ($runStart -ge 0) { $body = ($gsLines[$runStart..($runEnd - 1)] -join "`n") }
$p1sched = $body -match 'schedule_step'
$p1disp = $body -match 'dispatch_step'
$p1done = $body -match 'on_step_done'
$p1wait = $body -match '\.wait\s*\('
$p1submits = Grep-Count @('src/serve','apps/serve') 'submit\s*\('
$p1ok = ($runStart -ge 0) -and $p1sched -and $p1disp -and $p1done -and (-not $p1wait) -and ($p1submits -eq 1)
Phase 'P1_PUMP' $p1ok ("sched=" + $p1sched + " disp=" + $p1disp + " done=" + $p1done + " nowait=" + (-not $p1wait) + " submits=" + $p1submits)

# P2_LANE: no set_prefill_lane call on the batch path (serve/batch/apps);
# any remaining call must live in engine_core.h behind the legacy flag.
$callPat = '[.>]set_prefill_lane\s*\('
$p2batch = Grep-Count @('src/serve','src/batch','apps/serve') $callPat
$p2src = Grep-Count @('src') $callPat
$p2core = Count-Matches 'src/runtime/engine/engine_core.h' $callPat
$p2legacy = (File-Text 'src/runtime/engine/engine_core.h') -match 'legacy_single_lane_'
if ($p2core -eq 0 -and $p2src -eq 0) { $p2state = 'no-call-sites' }
elseif ($p2legacy) { $p2state = 'legacy-gated' }
else { $p2state = 'unguarded' }
$p2ok = ($p2batch -eq 0) -and ($p2src -eq $p2core) -and (($p2core -eq 0) -or $p2legacy)
Phase 'P2_LANE' $p2ok ("batchpath_calls=" + $p2batch + " core_calls=" + $p2core + " state=" + $p2state)

# P3_M: ragged M pinned to tokens.size() in both step sites.
$p3a = Count-Matches 'src/runtime/engine/step_forward.h' 'tokens\.size\s*\(\s*\)'
$p3b = Count-Matches 'src/batch/cinference_hooks.cc' 'tokens\.size\s*\(\s*\)'
$p3ok = ($p3a -gt 0) -and ($p3b -gt 0)
Phase 'P3_M' $p3ok ("step_forward_h=" + $p3a + " hooks_cc=" + $p3b)

# P4_ATTN: ragged branch precedes uniform [W,B] in the serve text path;
# the ragged kernel must be referenced. Scoped: the uniform else-if must
# follow the ragged if (an earlier uniform-only MTP helper is expected).
$txLines = File-Lines 'src/models/qwen3_5/execution/text.cpp'
$p4rag = First-Index $txLines 'if \(active_ragged_batch_ != nullptr\)'
$p4uni = Next-Index $txLines 'active_sequence_batch_ != 0' $p4rag
$p4kern = (Count-Matches 'src/models/qwen3_5/execution/text.cpp' 'causal_softmax_attention_ragged') -gt 0
$p4ok = ($p4rag -ge 0) -and ($p4uni -gt $p4rag) -and $p4kern
Phase 'P4_ATTN' $p4ok ("ragged_ln=" + ($p4rag + 1) + " uniform_ln=" + ($p4uni + 1) + " kernel=" + $p4kern)

# P5_GUARD: serve refuses host-only buffers at construction.
$gsText = File-Text $gs
$p5guard = $gsText -match 'RequireDeviceBuffersForServe'
$p5dtor = $gsText -match 'GenerationService::~GenerationService'
$p5ok = $p5guard -and $p5dtor
Phase 'P5_GUARD' $p5ok ("ctor_guard=" + $p5guard)

# P6_TESTS: hook-loop pump + ragged map + m47 tests exist and are registered.
$t1 = Test-Path (RPath 'tests/test_run_pumps_hook_loop.cc')
$t2 = Test-Path (RPath 'tests/test_ragged_attention_map.cc')
$t3 = Test-Path (RPath 'tests/ops/test_exl3_launch_m47.cc')
$r1 = (File-Text 'tests/cmake/RuntimeTests.cmake') -match 'test_run_pumps_hook_loop'
$r2 = (File-Text 'tests/cmake/RuntimeTests.cmake') -match 'test_ragged_attention_map'
$r3 = (File-Text 'tests/ops/tests.cmake') -match 'test_exl3_launch_m47'
$p6ok = $t1 -and $t2 -and $t3 -and $r1 -and $r2 -and $r3
Phase 'P6_TESTS' $p6ok ("files=" + $t1 + "," + $t2 + "," + $t3 + " cmake=" + $r1 + "," + $r2 + "," + $r3)
# test_serve_path.cc is slot D's job: report only, never edit, never gate.
$spp = RPath 'tests/test_serve_path.cc'
if (-not (Test-Path $spp)) { Write-Output 'INFO test_serve_path.cc gone' }
elseif ((File-Text 'tests/test_serve_path.cc') -match 'hook_loop-only') { Write-Output 'INFO test_serve_path.cc present-hookloop-rewrite' }
elseif ((File-Text 'tests/test_serve_path.cc') -match '\.wait') { Write-Output 'INFO test_serve_path.cc present-legacy' }
else { Write-Output 'INFO test_serve_path.cc present-other' }

# P7_MTP: spec off when plan.has_prefill(). Report only, non-blocking pre-Phase-4.
$p7gate = (File-Text 'src/batch/scheduler.h') -match '!plan\.has_prefill\(\)'
if ($p7gate) { Write-Output 'PHASE P7_MTP PASS spec-off-on-prefill' }
else { Write-Output 'PHASE P7_MTP FAIL no-prefill-gate' }

# P8_DECODE: run() must not bail on decode rows (slot A owns the runner seam).
# Brace-extract run(), strip //- and /* */-comments, then FAIL if
# decode_seq_ids.empty() is followed within ~6 lines by break, or if the
# 'until that runner lands' marker is present. PASS otherwise.
$p8runLines = @()
if ($runStart -ge 0) {
  $p8depth = 0; $p8started = $false
  for ($i = $runStart; $i -lt $gsLines.Count; $i++) {
    $p8runLines += $gsLines[$i]
    foreach ($ch in $gsLines[$i].ToCharArray()) {
      if ($ch -eq '{') { $p8depth++; $p8started = $true }
      elseif ($ch -eq '}') { $p8depth-- }
    }
    if ($p8started -and $p8depth -le 0) { break }
  }
}
$p8raw = ($p8runLines -join "`n")
$p8stripped = @()
$p8inBlock = $false
foreach ($ln in $p8runLines) {
  $s = $ln
  if ($p8inBlock) {
    $e = $s.IndexOf('*/')
    if ($e -ge 0) { $s = $s.Substring($e + 2); $p8inBlock = $false } else { $s = '' }
  }
  while (-not $p8inBlock) {
    $b = $s.IndexOf('/*'); $c = $s.IndexOf('//')
    if ($b -ge 0 -and ($c -lt 0 -or $b -lt $c)) {
      $e = $s.IndexOf('*/', $b + 2)
      if ($e -ge 0) { $s = $s.Substring(0, $b) + $s.Substring($e + 2) }
      else { $s = $s.Substring(0, $b); $p8inBlock = $true }
    }
    else { break }
  }
  if (-not $p8inBlock) {
    $c = $s.IndexOf('//')
    if ($c -ge 0) { $s = $s.Substring(0, $c) }
  }
  $p8stripped += $s
}
$p8break = $false
for ($i = 0; $i -lt $p8stripped.Count; $i++) {
  if ($p8stripped[$i] -match 'decode_seq_ids\.empty\s*\(\s*\)') {
    $hi = [Math]::Min($i + 6, $p8stripped.Count - 1)
    for ($j = $i; $j -le $hi; $j++) {
      if ($p8stripped[$j] -match '(^|[^_A-Za-z])break\s*;') { $p8break = $true; break }
    }
  }
  if ($p8break) { break }
}
$p8marker = $p8raw -match 'until that runner lands'
$p8ok = ($runStart -ge 0) -and (-not $p8break) -and (-not $p8marker)
Phase 'P8_DECODE' $p8ok ("decode_break=" + $p8break + " marker=" + $p8marker)

# P9_PUMP_LOCK: run() must hold a lock_guard (or unique_lock) over a mutex
# whose scope covers schedule_step AND dispatch_step AND on_step_done.
# Brace-extracted run() (see P8), comments stripped; accept pump_mutex_ or
# equivalent *mutex* name, report the name. FAIL until slot A lands.
$p9lockIdx = -1; $p9mutex = 'none'
$p9sched = -1; $p9disp = -1; $p9done = -1
if ($runStart -ge 0 -and $null -ne $p8stripped -and $p8stripped.Count -gt 0) {
  for ($i = 0; $i -lt $p8stripped.Count; $i++) {
    $ln = $p8stripped[$i]
    if ($p9lockIdx -lt 0 -and $ln -match '(lock_guard|unique_lock)') {
      $p9lockIdx = $i
      $hits = [regex]::Matches($ln, '[A-Za-z_][A-Za-z0-9_]*mutex[A-Za-z0-9_]*')
      if ($hits.Count -gt 0) {
        $pick = ''
        foreach ($h in $hits) { if ($h.Value -match '_') { $pick = $h.Value } }
        if ($pick -eq '') { $pick = $hits[$hits.Count - 1].Value }
        $p9mutex = $pick
      }
      else { $p9mutex = 'unknown' }
    }
    if ($p9sched -lt 0 -and $ln -match 'schedule_step') { $p9sched = $i }
    if ($p9disp -lt 0 -and $ln -match 'dispatch_step') { $p9disp = $i }
    if ($p9done -lt 0 -and $ln -match 'on_step_done') { $p9done = $i }
  }
}
$p9ok = ($p9lockIdx -ge 0) -and ($p9sched -gt $p9lockIdx) -and ($p9disp -gt $p9lockIdx) -and ($p9done -gt $p9lockIdx)
Phase 'P9_PUMP_LOCK' $p9ok ("mutex=" + $p9mutex + " lock_ln=" + $p9lockIdx + " sched=" + $p9sched + " disp=" + $p9disp + " done=" + $p9done)

# P10_WORKSPACE: ctor must call exl3_engine_reserve_workspace (slot A).
$p10n = Count-Matches $gs 'exl3_engine_reserve_workspace'
$p10ok = $p10n -gt 0
Phase 'P10_WORKSPACE' $p10ok ("calls=" + $p10n)

# P11_CONCURRENCY: serve_options.h max_concurrency default must read 8.
$p11val = 'none'
$p11text = File-Text 'src/serve/serve_options.h'
if ($null -ne $p11text) {
  $m11 = [regex]::Match($p11text, 'max_concurrency\s*=\s*(\d+)')
  if ($m11.Success) { $p11val = $m11.Groups[1].Value }
}
$p11ok = ($p11val -eq '8')
Phase 'P11_CONCURRENCY' $p11ok ("default=" + $p11val)

if ($script:Failed.Count -eq 0) { Write-Output 'STATE=LINUX_ONLY' }
else { Write-Output 'STATE=CODE' }
exit 0

#!/usr/bin/env bash
# =============================================================================
# retrain_all_from_raw.sh
#
# Single command that RETRAINS every ANNITIA slot1 component FROM RAW DATA and
# assembles the final slot1 prediction, then verifies it against
# frozen/slot1_prediction.csv.
#
#   bash pipeline_full/runnable/retrain_all_from_raw.sh
#
# It does NOT use cached_intermediates/ as model inputs, and does NOT read any
# old OOF / test / submission prediction CSV as a model input. Raw data is taken
# from review_repo/data/raw/. All work happens under retrain_work/ and all final
# outputs/reports under retrain_outputs/. frozen/slot1_prediction.csv and
# cached_intermediates/ are never written.
# =============================================================================
set -Eeuo pipefail

# --- determinism / thread caps (REQUIRED: avoid RSF n_jobs=-1 thread thrash) ---
export PYTHONHASHSEED=0
export LOKY_MAX_CPU_COUNT=8
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
export NUMEXPR_NUM_THREADS=2

# --- paths ---
RUNNABLE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$RUNNABLE/../.." && pwd)"          # review_repo
REF="$RUNNABLE/../reference_upstream"
RAW="$REPO/data/raw"
FROZEN="$REPO/frozen/slot1_prediction.csv"
WORK="$RUNNABLE/retrain_work"
OUT="$RUNNABLE/retrain_outputs"
LOGS="$OUT/logs"
LIB="$RUNNABLE/retrain_lib"
PY="${PYTHON:-python}"

GPT="$WORK/gpt"
CLA="$WORK/claude"
MRG="$WORK/merged"

mkdir -p "$LOGS"
MASTER="$LOGS/STAGES.log"
: > "$MASTER"

log()  { echo "$@" | tee -a "$MASTER"; }
die()  { echo "FATAL: $*" | tee -a "$MASTER" >&2; exit 1; }

# run_stage <NN_name> <logfile> -- <cmd...>
run_stage() {
  local name="$1"; shift
  local lf="$LOGS/$1.log"; shift
  [ "$1" = "--" ] && shift
  log ""
  log "### STAGE $name"
  log "    cmd: $*"
  local t0 ec
  t0=$(date +%s)
  log "    start: $(date -Is)"
  set +e
  ( "$@" ) >"$lf" 2>&1
  ec=$?
  set -e
  local t1; t1=$(date +%s)
  log "    end:   $(date -Is)   exit=$ec   runtime=$((t1 - t0))s"
  if [ $ec -ne 0 ]; then
    log "    ---- last 25 log lines ($lf) ----"
    tail -n 25 "$lf" | sed 's/^/    /' | tee -a "$MASTER"
    die "stage $name failed (exit $ec). See $lf"
  fi
}

require_file() { [ -f "$1" ] || die "expected freshly-generated artifact missing: $1"; }

log "=== retrain_all_from_raw.sh ==="
log "repo:     $REPO"
log "raw:      $RAW"
log "work:     $WORK"
log "outputs:  $OUT"
log "started:  $(date -Is)"
[ -f "$RAW/train.csv" ] && [ -f "$RAW/test.csv" ] || die "raw data not found under $RAW"
FROZEN_MD5_BEFORE=$(md5sum "$FROZEN" | awk '{print $1}')

# clean working tree (outputs are regenerated every run)
rm -rf "$WORK"
mkdir -p "$WORK" "$OUT"

# =============================================================================
# TRACK B (GPT) — 10-step horizon-blend chain from raw
# =============================================================================
log ""
log "########## TRACK B (GPT) ##########"
cp -a "$REF/gpt_track" "$GPT"
# clean any vendored old outputs so nothing stale is read as an input
rm -rf "$GPT/experiments/outputs" "$GPT/submissions" "$GPT/reports" "$GPT/data/processed"
mkdir -p "$GPT/experiments/outputs" "$GPT/submissions" "$GPT/reports" "$GPT/data/raw"
# stage raw with the gpt-track filenames (content identical to review_repo raw)
cp "$RAW/train.csv"                  "$GPT/data/raw/DB-1773398340961.csv"
cp "$RAW/test.csv"                   "$GPT/data/raw/DB-1773398340961-test.csv"
cp "$RAW/dictionary.csv"             "$GPT/data/raw/Dictionary-1775552221412.csv"
cp "$RAW/hello_world_submission.csv" "$GPT/data/raw/hello_world_submission-1773575379610.csv"

gpt_run() { ( cd "$GPT" && PYTHONPATH="$GPT" "$PY" "$@" ); }

run_stage "01_phase3_current_state_v2"             "01_b_p3csv2"  -- bash -c "cd '$GPT' && PYTHONPATH='$GPT' '$PY' -m src.run_experiment --config experiments/configs/phase3_current_state_v2.yaml"
run_stage "02_phase2_NIT_plus_scores_longitudinal" "02_b_p2nit"   -- bash -c "cd '$GPT' && PYTHONPATH='$GPT' '$PY' -m src.run_experiment --config experiments/configs/phase2_NIT_plus_scores_longitudinal.yaml"
run_stage "03_phase3_6_no_visit_history"           "03_b_nvh"     -- bash -c "cd '$GPT' && PYTHONPATH='$GPT' '$PY' -m src.run_experiment --config experiments/configs/phase3_6_no_visit_history.yaml"
run_stage "04_phase3_6_csv2_extra_seeds"           "04_b_seeds"   -- bash -c "cd '$GPT' && PYTHONPATH='$GPT' '$PY' -m src.run_experiment --config experiments/configs/phase3_6_csv2_extra_seeds.yaml"
run_stage "05_phase3_6_hepatic_aug"                "05_b_hepaug"  -- bash -c "cd '$GPT' && PYTHONPATH='$GPT' '$PY' -m src.run_experiment --config experiments/configs/phase3_6_hepatic_aug.yaml"
# 06: prerequisite that writes the phase3_current_state_v2 JSON sidecar (omitted by the raw chain list)
run_stage "06_build_phase3_submissions"            "06_b_p3sub"   -- bash -c "cd '$GPT' && PYTHONPATH='$GPT' '$PY' -m src.build_phase3_submissions"
run_stage "07_build_phase3_5_candidates"           "07_b_p35"     -- bash -c "cd '$GPT' && PYTHONPATH='$GPT' '$PY' -m src.build_phase3_5_candidates"
run_stage "08_build_phase3_6_candidates"           "08_b_p36"     -- bash -c "cd '$GPT' && PYTHONPATH='$GPT' '$PY' -m src.build_phase3_6_candidates"
run_stage "09_run_phase3_9_horizon"                "09_b_p39"     -- bash -c "cd '$GPT' && PYTHONPATH='$GPT' '$PY' -m src.run_phase3_9_horizon"
run_stage "10_run_phase3_10_horizon"               "10_b_p310"    -- bash -c "cd '$GPT' && PYTHONPATH='$GPT' '$PY' -m src.run_phase3_10_horizon"

GPT_ANCHOR=$(ls -t "$GPT"/submissions/*phase3_10_horizon_blend_v2.csv 2>/dev/null | head -1 || true)
[ -n "${GPT_ANCHOR:-}" ] || die "Track B anchor (phase3_10_horizon_blend_v2.csv) was not produced"
require_file "$GPT_ANCHOR"
log "    Track B anchor: $GPT_ANCHOR"

# =============================================================================
# TRACK A (Claude) — 2-way OOF-stacked blend from raw
# =============================================================================
log ""
log "########## TRACK A (Claude) ##########"
cp -a "$REF/claude_track" "$CLA"
rm -rf "$CLA/submissions" "$CLA/reports" "$CLA/data/processed"
mkdir -p "$CLA/submissions" "$CLA/reports" "$CLA/data/raw"
cp "$RAW/train.csv"                  "$CLA/data/raw/train.csv"
cp "$RAW/test.csv"                   "$CLA/data/raw/test.csv"
cp "$RAW/dictionary.csv"             "$CLA/data/raw/dictionary.csv"
cp "$RAW/hello_world_submission.csv" "$CLA/data/raw/hello_world_submission.csv"

run_stage "11_phase2_stack_2way" "11_a_stack2way" -- bash -c "cd '$CLA' && ANNITIA_DATA_ROOT='$CLA/data/raw' PYTHONPATH='$CLA' '$PY' experiments/phase2_stack_2way.py"

CL_ANCHOR="$CLA/submissions/phase2_blend_2way_optimal.csv"
require_file "$CL_ANCHOR"
log "    Track A anchor: $CL_ANCHOR"

# =============================================================================
# DEATH — CWGBSA / GBSA tree-survival from raw (task3, I/O redirected, no cached writes)
# =============================================================================
log ""
log "########## DEATH (CWGBSA / GBSA) ##########"
mkdir -p "$MRG"
run_stage "12_task3_tree_survival" "12_death_task3" -- bash -c "ANNITIA_DATA_ROOT='$RAW' '$PY' '$LIB/run_death_task3.py' --merged-root '$MRG' --gpt-anchor '$GPT_ANCHOR' --cl-anchor '$CL_ANCHOR' --runnable '$RUNNABLE'"

CWGBS="$MRG/model_zoo_sprint/predictions/test__survtree__dea__longitudinal_summary__cwgbs_300_lr05.csv"
GBSA="$MRG/model_zoo_sprint/predictions/test__survtree__dea__longitudinal_summary__gbsa_200_lr05_d3.csv"
require_file "$CWGBS"; require_file "$GBSA"

# =============================================================================
# FINAL ASSEMBLY — slot1 from regenerated components (build_slot1_only.py verbatim)
# =============================================================================
log ""
log "########## FINAL SLOT1 ASSEMBLY ##########"
FINAL="$OUT/final_retrain_prediction.csv"
run_stage "13_assemble_slot1" "13_assemble" -- bash -c "ANNITIA_DATA_ROOT='$RAW' '$PY' '$LIB/assemble_slot1.py' --gpt-anchor '$GPT_ANCHOR' --cl-anchor '$CL_ANCHOR' --cwgbs '$CWGBS' --gbsa '$GBSA' --raw-root '$RAW' --runnable '$RUNNABLE' --work '$WORK/slot1_assembly' --out '$FINAL'"
require_file "$FINAL"

# =============================================================================
# VALIDATION vs frozen
# =============================================================================
log ""
log "########## VALIDATION vs frozen/slot1_prediction.csv ##########"
run_stage "14_validate_against_frozen" "14_validate" -- bash -c "'$PY' '$LIB/validate_against_frozen.py' --pred '$FINAL' --frozen '$FROZEN'"
cp "$LOGS/14_validate.log" "$OUT/comparison_to_frozen.txt"

# --- integrity: frozen must be unchanged ---
FROZEN_MD5_AFTER=$(md5sum "$FROZEN" | awk '{print $1}')
[ "$FROZEN_MD5_BEFORE" = "$FROZEN_MD5_AFTER" ] || die "frozen/slot1_prediction.csv was modified (md5 changed)!"
log ""
log "frozen md5 unchanged: $FROZEN_MD5_AFTER"
log "=== DONE: $(date -Is) ==="
log "final prediction: $FINAL"
echo ""
echo "SUCCESS: see $OUT/comparison_to_frozen.txt and $LOGS/STAGES.log"

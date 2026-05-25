#!/bin/bash
# ===========================================================================
# Cross-dataset baselines for EvoXXLTraffic main result tables (PEMS03–PEMS12).
# Single entry point for every baseline that is NOT covered by the per-district
# full pipeline (pemsXX_run.sh).
#
# ─── Method groups (select via GROUP env var) ─────────────────────────────
#   GROUP=core    (default)   Retrain × {STGNN, ASTGNN, DCRNN, TGCN}
#                             + PECPM + STRAP            default seeds 47-51
#   GROUP=extra               GWN / STID / ITransformer / DLinear
#                             (STBP-paper extras)        default seeds 42-46
#   GROUP=sttc                ST-TTC (NeurIPS'25)        default seeds 42-46
#   GROUP=all                 runs core → extra → sttc in sequence
#
# ─── Env vars (all optional, all forwarded to every group) ────────────────
#   GROUP       core | extra | sttc | all
#   DATASETS    space-sep list (default: all 9 PEMS districts)
#   METHODS     space-sep list, restricted by GROUP
#   SEEDS       space-sep list (default depends on GROUP — see above)
#   GPU         GPU id, default 0
#   NOHUP=1     fork to background, log to run_logs/run_all_<group>_<ts>.log
#
# ─── Usage examples ────────────────────────────────────────────────────────
#   cd eac/
#   bash scripts/run_all_baselines.sh                                    # GROUP=core
#   GROUP=all   bash scripts/run_all_baselines.sh                        # everything
#   GROUP=core  METHODS="strap pecpm"     bash scripts/run_all_baselines.sh
#   GROUP=extra DATASETS="PEMS03 PEMS04"  bash scripts/run_all_baselines.sh
#   GROUP=sttc  DATASETS="pems05 pems06"  bash scripts/run_all_baselines.sh
#   NOHUP=1 GROUP=all                     bash scripts/run_all_baselines.sh
#
# Seed defaults are chosen to align with the per-district pemsXX_run.sh seed
# pools so columns can be compared 1-to-1 with the main result tables.
# ===========================================================================

set -euo pipefail
cd "$(dirname "$0")/.."

GROUP=${GROUP:-core}

# ---------------------------------------------------------------------------
# NOHUP wrapper
# ---------------------------------------------------------------------------
if [[ "${NOHUP:-0}" == "1" && -z "${EAC_BG:-}" ]]; then
    mkdir -p run_logs
    LOG_FILE="run_logs/run_all_${GROUP}_$(date +%Y%m%d_%H%M%S).log"
    echo "[nohup] backgrounding to $LOG_FILE"
    EAC_BG=1 nohup bash "$0" "$@" > "$LOG_FILE" 2>&1 &
    echo "[nohup] PID=$!"
    echo "[nohup] tail -f $LOG_FILE"
    exit 0
fi

GPU=${GPU:-0}

# ---------------------------------------------------------------------------
# Helpers shared by `core` group (PECPM / STRAP need first-year STGNN weight)
# ---------------------------------------------------------------------------
first_year_of() {
    local ds="$1" low
    low=$(echo "$ds" | tr 'A-Z' 'a-z')
    python - <<PY
import json
print(json.load(open("conf/$ds/retrain_st_${low}.json"))["begin_year"])
PY
}

first_year_pkl_for() {
    local ds="$1" seed="$2" first_year="$3" low
    low=$(echo "$ds" | tr 'A-Z' 'a-z')
    local dir="log/${ds}/retrain_stgnn_${low}-${seed}/${first_year}"
    ls "${dir}"/*.pkl 2>/dev/null | head -n 1
}

link_pkl_to_logname() {
    # Mirror pemsXX_run.sh's AutoLink: utils.common_tools.load_test_best_model
    # listdirs log/<DS>/<logname>-<seed>/<year>/ and ignores --first_year_model_path,
    # so symlink the retrain_stgnn first-year pkl under the target logname.
    local ds="$1" pkl="$2" target_logname="$3" seed="$4" first_year="$5"
    local dst_dir="log/${ds}/${target_logname}-${seed}/${first_year}"
    mkdir -p "$dst_dir"
    ln -sf "$(readlink -f "$pkl")" "$dst_dir/$(basename "$pkl")"
}

# ===========================================================================
# GROUP: core — retrain × 4 backbones + PECPM + STRAP
# ===========================================================================
run_core_group() {
    local DSS=${DATASETS:-"PEMS03 PEMS04 PEMS05 PEMS06 PEMS07 PEMS08 PEMS10 PEMS11 PEMS12"}
    local SDS=${SEEDS:-"47 48 49 50 51"}
    local MTS=${METHODS:-"retrain_stgnn retrain_astgnn retrain_dcrnn retrain_tgcn pecpm strap"}

    echo "[core] DATASETS = $DSS"
    echo "[core] METHODS  = $MTS"
    echo "[core] SEEDS    = $SDS   GPU=$GPU"

    for ds in $DSS; do
        local low; low=$(echo "$ds" | tr 'A-Z' 'a-z')
        local FY;  FY=$(first_year_of "$ds")
        echo ""
        echo "############# [core] $ds  (begin_year=$FY) #############"
        for m in $MTS; do
            case "$m" in
                retrain_stgnn|retrain_astgnn|retrain_dcrnn|retrain_tgcn)
                    local bk=${m#retrain_}
                    echo "---------- [$ds] retrain backbone=$bk ----------"
                    for seed in $SDS; do
                        python main.py --conf "conf/${ds}/retrain_${bk}_${low}.json" \
                            --gpuid "$GPU" --seed "$seed"
                    done
                    ;;
                pecpm|strap)
                    echo "---------- [$ds] $m (STGNN backbone) ----------"
                    for seed in $SDS; do
                        local pkl; pkl=$(first_year_pkl_for "$ds" "$seed" "$FY")
                        if [[ -z "$pkl" ]]; then
                            echo "  [skip] $m seed=$seed: missing retrain_stgnn first-year pkl for $ds"
                            continue
                        fi
                        link_pkl_to_logname "$ds" "$pkl" "${m}_${low}" "$seed" "$FY"
                        python main.py --conf "conf/${ds}/${m}_${low}.json" \
                            --load_first_year 1 --first_year_model_path "$pkl" \
                            --gpuid "$GPU" --seed "$seed"
                    done
                    ;;
                *)
                    echo "[error] unknown core method '$m'" >&2; exit 1 ;;
            esac
        done
    done
}

# ===========================================================================
# GROUP: extra — GWN / STID / ITransformer / DLinear (STBP extras)
# ===========================================================================
run_extra_group() {
    local DSS=${DATASETS:-"PEMS03 PEMS04 PEMS05 PEMS06 PEMS07 PEMS08 PEMS10 PEMS11 PEMS12"}
    local SDS=${SEEDS:-"42 43 44 45 46"}
    local MTS=${METHODS:-"gwn stid itransformer dlinear"}

    echo "[extra] DATASETS = $DSS"
    echo "[extra] METHODS  = $MTS"
    echo "[extra] SEEDS    = $SDS   GPU=$GPU"

    for ds in $DSS; do
        local low; low=$(echo "$ds" | tr 'A-Z' 'a-z')
        echo ""
        echo "############# [extra] $ds #############"
        for m in $MTS; do
            local conf="conf/${ds}/retrain_${m}_${low}.json"
            [[ -f "$conf" ]] || { echo "  [skip] missing config: $conf"; continue; }
            echo "---------- [$ds] retrain backbone=$m ----------"
            for seed in $SDS; do
                python main.py --conf "$conf" --gpuid "$GPU" --seed "$seed"
            done
        done
    done
}

# ===========================================================================
# GROUP: sttc — ST-TTC (NeurIPS'25 spectral calibrator + streaming memory)
# ===========================================================================
run_sttc_group() {
    # ST-TTC conf naming convention uses lowercase district name
    local raw=${DATASETS:-"pems03 pems04 pems05 pems06 pems07 pems08 pems10 pems11 pems12"}
    local DSS_LOWER; DSS_LOWER=$(echo "$raw" | tr 'A-Z' 'a-z')
    local SDS=${SEEDS:-"42 43 44 45 46"}

    echo "[sttc] DATASETS = $DSS_LOWER"
    echo "[sttc] SEEDS    = $SDS   GPU=$GPU"

    for ds in $DSS_LOWER; do
        local DS_UP; DS_UP=$(echo "$ds" | tr 'a-z' 'A-Z')
        local conf="conf/${DS_UP}/sttc_${ds}.json"
        [[ -f "$conf" ]] || { echo "  [skip] missing config: $conf"; continue; }
        echo ""
        echo "############# [sttc] $DS_UP #############"
        for seed in $SDS; do
            echo "-------- seed=$seed --------"
            python main.py --conf "$conf" --gpuid "$GPU" --seed "$seed"
        done
    done
}

# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------
case "$GROUP" in
    core)  run_core_group  ;;
    extra) run_extra_group ;;
    sttc)  run_sttc_group  ;;
    all)
        run_core_group
        run_extra_group
        run_sttc_group
        ;;
    *)
        echo "[error] Unknown GROUP='$GROUP' (valid: core | extra | sttc | all)" >&2
        exit 1
        ;;
esac

echo ""
echo "==================== run_all_baselines DONE (GROUP=$GROUP) ===================="

#!/usr/bin/env bash

set -uo pipefail
shopt -s nullglob

TIMESTAMP="$(date +"%Y%m%d_%H%M%S")"
ROOT_EVAL_LOG_DIR="eval-logs"
ROOT_FAILURE_LOG="${ROOT_EVAL_LOG_DIR}/failures_${TIMESTAMP}.log"
mkdir -p "${ROOT_EVAL_LOG_DIR}"

log_message() {
    printf '[%s] %s\n' "$(date +"%Y-%m-%d %H:%M:%S")" "$1"
}

run_with_logs() {
    local logs_dir="$1"
    local eval_name="$2"
    local output_path="$3"
    shift 3

    local model_eval_log_dir="${logs_dir}/eval-logs"
    local per_eval_log="${output_path}/run_${TIMESTAMP}.log"
    local per_model_log="${model_eval_log_dir}/${eval_name}_${TIMESTAMP}.log"
    local per_model_failure_log="${model_eval_log_dir}/failures_${TIMESTAMP}.log"
    local root_eval_log="${ROOT_EVAL_LOG_DIR}/${eval_name}_${TIMESTAMP}.log"

    mkdir -p "${output_path}" "${model_eval_log_dir}"

    log_message "START ${eval_name} for ${logs_dir}" | tee -a "${per_eval_log}" "${per_model_log}" "${root_eval_log}"
    log_message "COMMAND $*" | tee -a "${per_eval_log}" "${per_model_log}" "${root_eval_log}"

    if "$@" > >(tee -a "${per_eval_log}" "${per_model_log}" "${root_eval_log}") 2>&1; then
        log_message "SUCCESS ${eval_name} for ${logs_dir}" | tee -a "${per_eval_log}" "${per_model_log}" "${root_eval_log}"
        return 0
    else
        local status=$?
        local failure_message="FAIL ${eval_name} for ${logs_dir} (exit ${status}). See ${per_eval_log}"
        log_message "${failure_message}" | tee -a "${per_eval_log}" "${per_model_log}" "${root_eval_log}" "${per_model_failure_log}" "${ROOT_FAILURE_LOG}"
        return "${status}"
    fi
}

SETUP_LOG="${ROOT_EVAL_LOG_DIR}/setup_${TIMESTAMP}.log"
if uv add lm-eval accelerate wonderwords nltk jieba fuzzywuzzy rouge > >(tee -a "${SETUP_LOG}") 2>&1; then
    :
else
    status=$?
    log_message "FAIL dependency setup (exit ${status}). See ${SETUP_LOG}" | tee -a "${SETUP_LOG}" "${ROOT_FAILURE_LOG}"
    exit "${status}"
fi

had_failures=0
found_logs_dir=0

for LOGS in "$@"; do
    [[ -d "${LOGS}" ]] || continue

    name="$(basename "${LOGS%/}")"
    case "$name" in
        eval-logs|evals-*) continue ;;
    esac
        
    found_logs_dir=1

    echo "Evaluating ${LOGS} on 32K NIAH"
    if ! run_with_logs \
        "${LOGS}" \
        "niah_32k" \
        "${LOGS}/evals-long-niah" \
        uv run accelerate launch -m wrap_lmeval \
        --model hf \
        --model_args "pretrained=${LOGS},trust_remote_code=True" \
        --batch_size 8 \
        --tasks niah_single_1 niah_single_2 niah_single_3 \
        --metadata '{"max_seq_lengths":[32768]}' \
        --output_path "${LOGS}/evals-long-niah"; then
        had_failures=1
    fi

    echo "Evaluating ${LOGS} on 4K, 8K, 16K NIAH"
    if ! run_with_logs \
        "${LOGS}" \
        "niah_4k_8k_16k" \
        "${LOGS}/evals-niah" \
        uv run accelerate launch -m wrap_lmeval \
        --model hf \
        --model_args "pretrained=${LOGS},trust_remote_code=True" \
        --batch_size 32 \
        --tasks niah_single_1 niah_single_2 niah_single_3 \
        --metadata '{"max_seq_lengths":[4096, 8192, 16384]}' \
        --output_path "${LOGS}/evals-niah"; then
        had_failures=1
    fi

    echo "Evaluating ${LOGS} on short evals"
    if ! run_with_logs \
        "${LOGS}" \
        "short_evals" \
        "${LOGS}/evals-short" \
        uv run accelerate launch -m wrap_lmeval \
        --model hf \
        --model_args "pretrained=${LOGS},trust_remote_code=True" \
        --batch_size 32 \
        --tasks lambada_openai piqa winogrande arc_easy arc_challenge hellaswag \
        --output_path "${LOGS}/evals-short"; then
        had_failures=1
    fi

    echo "Evaluating ${LOGS} on ruler 4k"
    if ! run_with_logs \
        "${LOGS}" \
        "ruler_4k" \
        "${LOGS}/evals-ruler" \
        uv run accelerate launch -m wrap_lmeval \
        --model hf \
        --model_args "pretrained=${LOGS},trust_remote_code=True" \
        --batch_size 32 \
        --tasks ruler \
        --output_path "${LOGS}/evals-ruler"; then
        had_failures=1
    fi

    echo "Evaluating ${LOGS} on longbench fewshot"
    if ! run_with_logs \
        "${LOGS}" \
        "longbench_fewshot" \
        "${LOGS}/evals-longbench-fewshot" \
        uv run accelerate launch -m wrap_lmeval \
        --model hf \
        --model_args "pretrained=${LOGS},trust_remote_code=True" \
        --batch_size 32 \
        --tasks longbench_fewshot \
        --output_path "${LOGS}/evals-longbench-fewshot"; then
        had_failures=1
    fi
done

if (( ! found_logs_dir )); then
    log_message "No checkpoint directories found under new_logs/*" | tee -a "${ROOT_FAILURE_LOG}"
    exit 1
fi

if (( had_failures )); then
    log_message "Completed with failures. Summary: ${ROOT_FAILURE_LOG}" | tee -a "${ROOT_FAILURE_LOG}"
    exit 1
fi

log_message "All evals completed successfully." | tee -a "${ROOT_EVAL_LOG_DIR}/run_${TIMESTAMP}.log"

# uv add lm-eval accelerate wonderwords nltk jieba fuzzywuzzy rouge

# for LOGS in new_logs/*; do

#     echo Evaluating "${LOGS}" on 32K NIAH

#     uv run accelerate launch -m wrap_lmeval --model hf --model_args pretrained="${LOGS}",trust_remote_code=True --batch_size 16 --tasks niah_single_1 niah_single_2 niah_single_3 --metadata='{"max_seq_lengths":[32768]}' --output_path "${LOGS}"/evals-long-niah
    
#     echo Evaluating "${LOGS}" on 4K, 8K, 16K NIAH

#     uv run accelerate launch -m wrap_lmeval --model hf --model_args pretrained="${LOGS}",trust_remote_code=True --batch_size 32 --tasks niah_single_1 niah_single_2 niah_single_3 --metadata='{"max_seq_lengths":[4096, 8192, 16384]}' --output_path "${LOGS}"/evals-niah
    
#     echo Evaluating "${LOGS}" on short evals

#     uv run accelerate launch -m wrap_lmeval --model hf --model_args pretrained="${LOGS}",trust_remote_code=True --batch_size 32 --tasks lambada_openai piqa winogrande arc_easy arc_challenge hellaswag --output_path "${LOGS}"/evals-short
    
#     echo Evaluating "${LOGS}" on longbench fewshot

#     uv run accelerate launch -m wrap_lmeval --model hf --model_args pretrained="${LOGS}",trust_remote_code=True --batch_size 32 --tasks longbench_fewshot --output_path "${LOGS}"/evals-longbench-fewshot

# done

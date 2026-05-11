#!/bin/bash

maindir="/root/kvm_paper/paper_logs_v2/prolong"
dataset="princeton-nlp/TextbookChapters"
text_column="chapter"
split="train"

nickname="textbook_chapters"

context_length=32768

# List of model sizes
sizes=("120M" "350M")
#sizes=("350M")

# List of subdirectories
models=("rwkv7_ffn4" "gptalphaprope_swa256" "kvm256" "kvmsqrt16" "kvmsat_swa256" "gptalpha" "bswa" "ovqsat_swa256")
#models=("gptalphaprope_swa256" "gptalpha_swa256" "gptalpha_swa256_3Gtok")

# Loop over sizes
for size in "${sizes[@]}"; do
    # Loop over subdirs
    for subdir in "${models[@]}"; do
        uv run torchrun --standalone --nproc_per_node=8 -m loss_plot.eval --eval.nickname $nickname --eval.context_length $context_length --eval.dataset $dataset --eval.split $split --eval.text_column $text_column --logs_path ${maindir}/${subdir}_${size}/
    done
done

# Run the plotting script
uv run loss_plot/results/plot_combined.py --logs-root ${maindir} --nickname $nickname --context_length $context_length --sizes "${sizes[@]}"

#!/bin/bash

# ABLATIONS

#bash run.sh -c configs/prolong/base.yaml -c configs/prolong/tokens_3B.yaml -c configs/prolong/120M/ablation_kvm_no_head_temps.yaml
#bash run.sh -c configs/prolong/base.yaml -c configs/prolong/tokens_3B.yaml -c configs/prolong/120M/ablation_kvm_no_merge_gate.yaml
#bash run.sh -c configs/prolong/base.yaml -c configs/prolong/tokens_3B.yaml -c configs/prolong/120M/ablation_kvm_no_sink.yaml
#bash run.sh -c configs/prolong/base.yaml -c configs/prolong/tokens_3B.yaml -c configs/prolong/120M/ablation_kvm_no_vlens.yaml


# 120M

#bash run.sh -c configs/prolong/base.yaml -c configs/prolong/tokens_3B.yaml -c configs/prolong/120M/gptalpha_bswa_wd.yaml
#bash run.sh -c configs/prolong/base.yaml -c configs/prolong/tokens_3B.yaml -c configs/prolong/120M/gptalpha_wd.yaml
#bash run.sh -c configs/prolong/base.yaml -c configs/prolong/tokens_3B.yaml -c configs/prolong/120M/gptalpha_nope_swa_rope_wd.yaml
#bash run.sh -c configs/prolong/base.yaml -c configs/prolong/tokens_3B.yaml -c configs/prolong/120M/gptalpha_halfrope_swa_rope_wd.yaml
#bash run.sh -c configs/prolong/base.yaml -c configs/prolong/tokens_3B.yaml -c configs/prolong/120M/kvm_wd.yaml
#bash run.sh -c configs/prolong/base.yaml -c configs/prolong/tokens_3B.yaml -c configs/prolong/120M/kvm_sqrt16_wd.yaml
#bash run.sh -c configs/prolong/base.yaml -c configs/prolong/tokens_3B.yaml -c configs/prolong/120M/kvm_swa_saturate1024_wd.yaml
#bash run.sh -c configs/prolong/base.yaml -c configs/prolong/tokens_3B.yaml -c configs/prolong/120M/rwkv_wd.yaml
#bash run.sh -c configs/prolong/base.yaml -c configs/prolong/tokens_3B.yaml -c configs/prolong/120M/ovq_swa_ts_c256_s1024_wd.yaml

#bash run.sh -c configs/prolong/base.yaml -c configs/prolong/tokens_3B.yaml -c configs/prolong/120M/kvm_sqrt16_no_merge_gate_wd.yaml
#bash run.sh -c configs/prolong/base.yaml -c configs/prolong/tokens_3B.yaml -c configs/prolong/120M/kvm_swa_saturate1024_no_merge_gate_wd.yaml


# 350M

#bash run.sh -c configs/prolong/base.yaml -c configs/prolong/tokens_7p5B.yaml -c configs/prolong/350M/gptalpha_bswa_wd.yaml
#bash run.sh -c configs/prolong/base.yaml -c configs/prolong/tokens_7p5B.yaml -c configs/prolong/350M/gptalpha_wd.yaml
#bash run.sh -c configs/prolong/base.yaml -c configs/prolong/tokens_7p5B.yaml -c configs/prolong/350M/gptalpha_nope_swa_rope_wd.yaml
#bash run.sh -c configs/prolong/base.yaml -c configs/prolong/tokens_7p5B.yaml -c configs/prolong/350M/gptalphahalfrope_swa_rope_wd.yaml
#bash run.sh -c configs/prolong/base.yaml -c configs/prolong/tokens_7p5B.yaml -c configs/prolong/350M/kvm_wd.yaml
#bash run.sh -c configs/prolong/base.yaml -c configs/prolong/tokens_7p5B.yaml -c configs/prolong/350M/kvm_sqrt16_wd.yaml
#bash run.sh -c configs/prolong/base.yaml -c configs/prolong/tokens_7p5B.yaml -c configs/prolong/350M/kvm_swa_saturate1024_wd.yaml
#bash run.sh -c configs/prolong/base.yaml -c configs/prolong/tokens_7p5B.yaml -c configs/prolong/350M/rwkv_wd.yaml
#bash run.sh -c configs/prolong/base.yaml -c configs/prolong/tokens_7p5B.yaml -c configs/prolong/350M/ovq_swa_ts_c256_s1024_wd.yaml

#bash run.sh -c configs/prolong/base.yaml -c configs/prolong/tokens_7p5B.yaml -c configs/prolong/350M/kvm_no_merge_gate_wd.yaml

bash run.sh -c configs/prolong/base.yaml -c configs/prolong/tokens_3B.yaml -c configs/prolong/350M/gptalpha_nope_swa_rope_wd.yaml

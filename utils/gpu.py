import torch


def torch_runtime_label():
    hip_version = getattr(torch.version, "hip", None)
    if hip_version:
        return f"HIP {hip_version}"

    cuda_version = getattr(torch.version, "cuda", None)
    if cuda_version:
        return f"CUDA {cuda_version}"

    return "an unknown accelerator runtime"


def collect_accelerator_smi_output():
    import subprocess

    candidates = (
        ["amd-smi", "rocm-smi"]
        if getattr(torch.version, "hip", None)
        else ["nvidia-smi"]
    )
    for smi_name in candidates:
        try:
            result = subprocess.run(
                [smi_name],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
            )
        except FileNotFoundError:
            continue

        smi_output = result.stdout.strip()
        if not smi_output:
            smi_output = result.stderr.strip()
        if result.returncode != 0:
            smi_output = f"[exit code {result.returncode}]\n{smi_output}".strip()
        return smi_name, smi_output or "(no output)"

    return candidates[0], "(command not available)"

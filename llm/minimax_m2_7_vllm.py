import modal

vllm_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.12"
    )
    .run_commands("apt-get update")
    .run_commands(
        "apt-get install -y bash "
        "build-essential "
        "git "
        "git-lfs "
        "curl "
        "ca-certificates"
    )
    .run_commands("curl -LsSf https://astral.sh/uv/install.sh | sh")
    .env({"PATH": "/root/.local/bin:$PATH"})
    .run_commands("uv pip install --system -U 'vllm>=0.19.0' --torch-backend=cu128")
)

MODELS_DIR = "/llama_models"
MODEL_NAME = "MiniMaxAI/MiniMax-M2.7"

vllm_cache_vol = modal.Volume.from_name("vllm-cache", create_if_missing=True)
volume = modal.Volume.from_name("llama_models", create_if_missing=True)

app = modal.App("minimax-m2-7-vllm-openai-compatible")

N_GPU = 4
MINUTES = 60
HOURS = 60 * MINUTES
VLLM_PORT = 8000


@app.function(
    image=vllm_image,
    gpu=f"H100:{N_GPU}",
    max_containers=1,
    scaledown_window=30 * MINUTES,
    timeout=24 * HOURS,
    volumes={
        MODELS_DIR: volume,
        "/root/.cache/vllm": vllm_cache_vol,
    },
)
@modal.concurrent(max_inputs=100)
@modal.web_server(port=VLLM_PORT, startup_timeout=30 * MINUTES)
def serve():
    import subprocess
    import os

    os.environ["SAFETENSORS_FAST_GPU"] = "1"

    cmd = [
        "vllm",
        "serve",
        MODELS_DIR + "/" + MODEL_NAME,
        "--served-model-name",
        MODEL_NAME,
        "--trust-remote-code",
        "--port",
        str(VLLM_PORT),
        "--host",
        "0.0.0.0",
        "--tensor-parallel-size",
        str(N_GPU),
        "--max-model-len",
        "131072",
        "--enable-auto-tool-choice",
        "--tool-call-parser",
        "minimax_m2",
        "--reasoning-parser",
        "minimax_m2_append_think",
    ]

    print(cmd)
    subprocess.Popen(" ".join(cmd), shell=True)

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
MODEL_NAME = "Qwen/Qwen3.6-35B-A3B"

vllm_cache_vol = modal.Volume.from_name("vllm-cache", create_if_missing=True)
volume = modal.Volume.from_name("llama_models", create_if_missing=True)

app = modal.App("qwen36-35b-a3b-vllm-openai-compatible")

N_GPU = 2
MINUTES = 60
HOURS = 60 * MINUTES
VLLM_PORT = 8000


@app.function(
    image=vllm_image,
    gpu=f"A100-80GB:{N_GPU}",
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

    cmd = [
        "vllm",
        "serve",
        MODELS_DIR + "/" + MODEL_NAME,
        "--served-model-name",
        MODEL_NAME,
        "--port",
        str(VLLM_PORT),
        "--host",
        "0.0.0.0",
        "--tensor-parallel-size",
        str(N_GPU),
        "--max-model-len",
        "262144",
        "--reasoning-parser",
        "qwen3",
        "--enable-auto-tool-choice",
        "--tool-call-parser",
        "qwen3_coder",
    ]

    print(cmd)
    subprocess.Popen(" ".join(cmd), shell=True)

# GPUtw deployment

GPUtw guarantees persistence for files under `/vault`. Keep the repository,
input data, output artifacts, Python environment, and model caches there. Files
elsewhere in the container may disappear after an instance is stopped or
deleted.

## 1. Create and connect to an instance

Choose the **PyTorch 2.x + JupyterLab** template for the easiest start, or a
CUDA template when using SSH only. The platform supplies the NVIDIA host driver;
do not attempt to replace it from inside the container.

Add an SSH public key in the GPUtw dashboard when using SSH, deploy the instance,
and copy the connection command shown by the dashboard.

## 2. Put the project in the Vault

Clone the repository directly below `/vault`:

```bash
cd /vault
git clone <repository-url> net-packages
cd /vault/net-packages
```

Alternatively, upload the complete project to `/vault/net-packages` with
JupyterLab, SFTP, or `rsync`. Do not put credentials or a GPUtw dashboard token
in the repository.

Upload training inputs to:

```text
/vault/net-packages/Year=2022/
```

The existing configs use paths relative to the repository root. Because the
whole repository is under `/vault`, `Year=2022/`, `outputs/`, `.venv/`, prepared
embeddings, checkpoints, and reports all survive instance replacement.

## 3. Build and verify the environment

Run the idempotent setup script from the repository root:

```bash
bash scripts/setup_gputw.sh
```

The script:

- refuses to proceed if the project is outside the persistent Vault;
- verifies the NVIDIA driver and selected GPU with `nvidia-smi`;
- installs the `btop` system monitor and `tmux` terminal multiplexer through
  the system package manager;
- installs pinned `uv` 0.11.32 and managed Python 3.12 in
  `/vault/.net-packages`;
- installs the exact `uv.lock` environment plus all optional project extras;
- installs and verifies the lockfile-managed `gdown` command;
- keeps uv, Hugging Face, Sentence Transformers, Torch, and Matplotlib caches in
  `/vault/.net-packages/cache`;
- creates persistent input and output directories; and
- verifies CUDA with PyTorch and a real matrix multiplication on `cuda:0`.

The script is safe to run again. It reuses downloads and synchronizes the
environment with `uv.lock`.

For a nonstandard Vault mount, set it explicitly:

```bash
GPUTW_VAULT_DIR=/your/persistent/mount bash scripts/setup_gputw.sh
```

`GPUTW_UV_VERSION` can override the pinned uv version if the lockfile format is
updated in the future.

## 4. Train

Source the generated environment file once in each new terminal:

```bash
source /vault/net-packages/.gputw-env
```

Then run the desired pipeline. The default configs already use
`training.device: auto`, which selects CUDA when the verification above passes.

AE/CVAE tactic pipeline:

```bash
uv run python experiments/ae_cvae_tactic/run_experiment.py \
  --config experiments/ae_cvae_tactic/configs/default.yaml \
  --run all
```

Contrastive CVAE tactic pipeline:

```bash
uv run python experiments/ae_cvae_tactic/run_contrastive_experiment.py \
  --config experiments/ae_cvae_tactic/configs/contrastive.yaml
```

Step1 disentangled CVAE pipeline:

```bash
uv run python experiments/disentangled_cvae_step1/run_experiment.py \
  --config experiments/disentangled_cvae_step1/configs/default.yaml \
  --stage all
```

Center-augmented Step1 pipeline:

```bash
uv run python experiments/center_augmented_cvae_step1/run_experiment.py \
  --config experiments/center_augmented_cvae_step1/configs/default.yaml \
  --stage all
```

Golden-oracle Step2 pipeline:

```bash
uv run python experiments/golden_oracle_cvae_step2/run_experiment.py \
  --config experiments/golden_oracle_cvae_step2/configs/default.yaml \
  --stage all
```

Use `tmux` for long SSH jobs so training is not tied to the connection. Logs,
checkpoints, prepared embeddings, and final reports are written under
`/vault/net-packages/outputs/`.

## 5. Stop compute safely

Before stopping the instance, confirm the latest artifacts exist:

```bash
find outputs -maxdepth 3 -type f -printf '%TY-%Tm-%Td %TH:%TM %p\n' | sort | tail
df -h /vault
```

Once the process has ended and files are visible under `/vault`, stop the
instance in the dashboard to stop GPU compute billing. Persistent storage is not
a backup, so copy irreplaceable curated data and final checkpoints to another
location as well.

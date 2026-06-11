# gloss

Minimal [PyTorch Lightning](https://lightning.ai/) + [Hydra](https://hydra.cc/) +
[Weights & Biases](https://wandb.ai/) training template with multi-GPU (DDP) support.

Structure follows the ImmunoFoundation convention: a root `train.py` with an
`Experiment` class, nested plain-YAML configs (no `_target_` instantiation), and
config sections passed straight into module constructors.

## Layout

```
configs/train.yaml          # nested config: data / model / experiment
train.py                    # Experiment class + @hydra.main entrypoint
gloss/
  data/MNISTDataModule.py   # LightningDataModule (takes data_cfg)
  models/MNISTModule.py     # LightningModule   (takes model_cfg)
  models/components/        # plain nn.Modules, selected via a registry dict
  utils.py                  # rank-zero logger + config flattening for W&B
scripts/train.sh            # SLURM + torchrun multi-GPU launcher
tests/                      # CPU smoke tests
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

## Train

Single process (W&B off — no run name set):

```bash
python train.py
```

Enable W&B and override any config value from the CLI (Hydra dotlist):

```bash
python train.py experiment.wandb.name=my-run experiment.trainer.max_epochs=20
```

Multi-GPU via DDP (4 GPUs on one node):

```bash
torchrun --nproc_per_node=4 train.py experiment.wandb.name=my-run
# or submit the SLURM script:
sbatch scripts/train.sh
```

`experiment.num_devices` caps the GPU count; the actual number used is
`min(num_devices, visible GPUs)`.

## Test

```bash
pytest tests/
```

## Adapting it

- New dataset → add a `LightningDataModule` under `gloss/data/`, swap it in `train.py`.
- New architecture → add an `nn.Module` under `gloss/models/components/` and register
  it in the `NETS` dict in `gloss/models/MNISTModule.py`.
- New hyperparameters → add fields to `configs/train.yaml`; they're available as
  `cfg.<section>.<field>` with no code changes.

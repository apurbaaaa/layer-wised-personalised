# RunPod Commands — ISIC 2019 Federated SwinV2

All commands assume an **A40 (48 GB)** RunPod instance with CUDA.

---

## 1. Initial Setup (one-time)

```bash
# Clone repo
git clone <your-repo-url> && cd Def-vit

# Install deps in venv
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Or use the existing setup script (installs globally, downloads data, then runs centralized training):

```bash
bash setup_runpod.sh
```

---

## 2. Download Dataset (if not using setup_runpod.sh)

```bash
mkdir -p data && cd data

# CSVs
wget -q https://isic-archive.s3.amazonaws.com/challenges/2019/ISIC_2019_Training_GroundTruth.csv
wget -q https://isic-archive.s3.amazonaws.com/challenges/2019/ISIC_2019_Training_Metadata.csv
wget -q https://isic-archive.s3.amazonaws.com/challenges/2019/ISIC_2019_Test_GroundTruth.csv
wget -q https://isic-archive.s3.amazonaws.com/challenges/2019/ISIC_2019_Test_Metadata.csv

# Training images (~9 GB)
wget -q --show-progress https://isic-archive.s3.amazonaws.com/challenges/2019/ISIC_2019_Training_Input.zip
unzip -q ISIC_2019_Training_Input.zip && rm ISIC_2019_Training_Input.zip

# Test images (~3 GB)
wget -q --show-progress https://isic-archive.s3.amazonaws.com/challenges/2019/ISIC_2019_Test_Input.zip
unzip -q ISIC_2019_Test_Input.zip && rm ISIC_2019_Test_Input.zip

cd ..
```

---

## 3. Centralized Baseline Training

```bash
source .venv/bin/activate

python train.py \
    --device cuda:0 \
    --batch_size 48 \
    --accum_steps 1 \
    --workers 12 \
    --epochs 25 \
    --lr 4e-4 \
    --compile \
    --mixup
```

---

## 4. Federated Training (Structure-Aware Personalized)

### Default run (5 clients, drift-aware, non-IID α=0.5)

```bash
python federated_train.py \
    --device cuda:0 \
    --num_clients 5 \
    --rounds 50 \
    --local_epochs 3 \
    --client_frac 1.0 \
    --dirichlet_alpha 0.5 \
    --drift_weighting True \
    --batch_size 32 \
    --lr 4e-4 \
    --workers 8
```

### More clients, partial participation

```bash
python federated_train.py \
    --device cuda:0 \
    --num_clients 10 \
    --rounds 100 \
    --local_epochs 3 \
    --client_frac 0.5 \
    --dirichlet_alpha 0.3 \
    --drift_weighting True \
    --batch_size 32 \
    --lr 4e-4 \
    --workers 8
```

### Highly non-IID (α=0.1)

```bash
python federated_train.py \
    --device cuda:0 \
    --num_clients 5 \
    --rounds 80 \
    --local_epochs 5 \
    --client_frac 1.0 \
    --dirichlet_alpha 0.1 \
    --drift_weighting True \
    --batch_size 32 \
    --lr 4e-4 \
    --workers 8
```

### Without drift weighting (plain FedAvg on Group B for ablation)

```bash
python federated_train.py \
    --device cuda:0 \
    --num_clients 5 \
    --rounds 50 \
    --local_epochs 3 \
    --client_frac 1.0 \
    --dirichlet_alpha 0.5 \
    --drift_weighting False \
    --batch_size 32 \
    --lr 4e-4 \
    --workers 8
```

### Resume from checkpoint

```bash
python federated_train.py \
    --device cuda:0 \
    --num_clients 5 \
    --rounds 50 \
    --local_epochs 3 \
    --dirichlet_alpha 0.5 \
    --drift_weighting True \
    --batch_size 32 \
    --lr 4e-4 \
    --workers 8 \
    --resume checkpoints/federated/last_federated.pt
```

---

## 5. Evaluate

### Centralized best model

```bash
python evaluate.py --checkpoint checkpoints/best.pt --device cuda:0 --batch_size 64 --workers 8
```

### Centralized best model with TTA

```bash
python evaluate.py --checkpoint checkpoints/best.pt --device cuda:0 --batch_size 64 --workers 8 --tta
```

### Evaluate on validation split

```bash
python evaluate.py --checkpoint checkpoints/best.pt --split val --device cuda:0
```

---

## 6. Quick Smoke Test (verify everything works)

```bash
python test_federated.py
```

---

## CLI Flag Reference

| Flag | Default | Description |
|------|---------|-------------|
| `--num_clients` | 5 | Number of federated clients |
| `--rounds` | 50 | Global communication rounds |
| `--local_epochs` | 3 | Local training epochs per client per round |
| `--client_frac` | 1.0 | Fraction of clients selected each round |
| `--dirichlet_alpha` | 0.5 | Dirichlet α (lower = more non-IID) |
| `--drift_weighting` | True | Drift-aware aggregation for Group B |
| `--batch_size` | 32 | Per-client batch size |
| `--lr` | 4e-4 | Learning rate |
| `--workers` | 8 | DataLoader workers |
| `--resume` | None | Path to federated checkpoint |
| `--device` | auto | `cuda:0`, `mps`, or `cpu` |

---

## Checkpoints

All federated checkpoints are saved to `checkpoints/federated/`:

| File | Contents |
|------|----------|
| `global_round_{r}.pt` | Full state after round r |
| `best_federated.pt` | Best global balanced accuracy |
| `last_federated.pt` | Latest round (for resume) |
| `client_{k}_final.pt` | Personalized model for client k |

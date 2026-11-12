# Benchmark Leakage Trap

📄 [Paper](https://arxiv.org/abs/2602.13626) · 📘 [Usage Guide](docs/usage.md)

CoLLM-MF and CoLLM-DIN experiments on **MovieLens-1M** and **Amazon-Book**, comparing Clean and Dirty LLM backbones under controlled data leakage.

## 🛠️ Installation

Use Python 3.10/3.11 and a CUDA-compatible PyTorch/torchvision installation. The pipeline uses one GPU.

```bash
pip install -r requirements.txt
```

## 📂 Data & Configuration

Prepare the target dataset splits (`train_ood2.pkl`, `valid_ood2.pkl`, `test_ood2.pkl`), six OOD JSONL files, and a local **Vicuna-7B** checkpoint following the [usage guide](docs/usage.md). Data and weights are not included.

Set `base_model`, `data_dir`, and `ood_sources` in [experiment_ml1m.json](configs/experiment_ml1m.json) or [experiment_amazon_book.json](configs/experiment_amazon_book.json). The default exposure is **ID 10% + OOD 60%**.

## 🚀 Run

Run from the repository root; each command evaluates both CoLLM methods:

```bash
# MovieLens-1M
python run_leakage_impact_analysis.py --config configs/experiment_ml1m.json --output-dir outputs/ml1m_mixed

# Amazon-Book
python run_leakage_impact_analysis.py --config configs/experiment_amazon_book.json --output-dir outputs/book_mixed
```

## 📊 Results

Results are saved to **`<output-dir>/final_results.csv`**, including Clean/Dirty **AUC**, **UAUC**, and relative changes. Per-model folders contain predictions and checkpoints.

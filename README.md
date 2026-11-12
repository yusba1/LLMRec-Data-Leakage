# Benchmark Leakage Trap: Can We Trust LLM-based Recommendation?

📄 [Paper](https://arxiv.org/abs/2602.13626) · 📘 [Detailed Guide](docs/usage.md)

Study the impact of upstream LLM data leakage on **CoLLM-MF** and **CoLLM-DIN**, using **MovieLens-1M** and **Amazon-Book**.

**Data sampling → LoRA injection → Clean/Dirty recommendation → AUC & UAUC**

The default setting mixes **ID 10% + OOD 60%** from six external sources. Clean and Dirty use the same downstream data, architecture, and hyperparameters.

## 📂 Code Structure

```text
├── configs/                        # Experiment settings
├── dataset/                        # Data and source links
├── leakage/                        # Data, training, and metrics
├── minigpt4/                       # CoLLM model components
├── prompts/                        # Recommendation prompts
├── train_configs/                  # MF/DIN model configurations
├── preprocess_data.py              # Target data preparation
├── prepare_ood_raw.py               # External data preparation
├── build_leakage_corpus.py          # Controlled ID/OOD sampling
├── train_upstream_lora.py           # Upstream leakage injection
├── train_rec_encoder.py             # Collaborative encoder training
├── train_downstream_rec.py          # Downstream recommendation
└── run_leakage_impact_analysis.py    # Full experiment pipeline
```

## 🚀 Quick Start

**1. Install.** Use Python 3.10/3.11 and a compatible CUDA-enabled PyTorch/torchvision installation. The pipeline uses one GPU.

```bash
pip install -r requirements.txt
```

**2. Prepare data and weights.** Download [MovieLens-1M](https://grouplens.org/datasets/movielens/1m/) or [Amazon-Book](https://jmcauley.ucsd.edu/data/amazon/), the [six OOD sources](dataset/download_ood.sh), and a local Vicuna-7B checkpoint. Data and model weights are not bundled.

Prepare `train_ood2.pkl`, `valid_ood2.pkl`, and `test_ood2.pkl` with a **7:2:1** split, plus OOD JSONL files. See the [preparation commands and input formats](docs/usage.md#target-datasets).

**3. Configure and run.** Set `base_model`, `data_dir`, and `ood_sources` in the corresponding [experiment configuration](configs/), then run from the repository root:

```bash
# MovieLens-1M: CoLLM-MF + CoLLM-DIN
python run_leakage_impact_analysis.py --config configs/experiment_ml1m.json --output-dir outputs/ml1m_mixed

# Amazon-Book: CoLLM-MF + CoLLM-DIN
python run_leakage_impact_analysis.py --config configs/experiment_amazon_book.json --output-dir outputs/book_mixed
```

## 📊 Results

Each run saves **`final_results.csv`** with Clean/Dirty AUC, UAUC, and relative changes. Per-model folders contain test predictions, metrics, and checkpoints.

## 🛠️ Configuration

- **Training settings:** [MovieLens-1M](configs/experiment_ml1m.json) · [Amazon-Book](configs/experiment_amazon_book.json).
- **Exposure settings:** `exposure.id_ratio` and `exposure.ood_ratio`.
- **Optional encoder weights:** set `pretrained_encoder`; otherwise the pipeline trains a shared MF/DIN encoder.
- **Stage commands and output details:** [Detailed Guide](docs/usage.md).

## 🙏 Acknowledgments

Built on CoLLM and MiniGPT-4/LAVIS-derived components. Original source attributions are retained.

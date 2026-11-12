# Detailed Usage Guide

[Back to README](../README.md)

Code for studying how controlled upstream data exposure affects CoLLM-MF and CoLLM-DIN recommendation. Paper: [Benchmark Leakage Trap: Can We Trust LLM-based Recommendation?](https://arxiv.org/abs/2602.13626).

## Pipeline

1. Prepare the target benchmark with train/validation/test partitions of 7:2:1.
2. Sample ID interactions from all three partitions, preserving their 7:2:1 composition. Sample external OOD sources independently.
3. Train upstream LoRA adapters on serialized context/target records, masking context tokens from the loss. The original LLM weights are frozen.
4. Pretrain a collaborative encoder using the target training partition and select it using validation AUC. Reuse this encoder for both paired runs.
5. Train CoLLM with the original backbone (Clean), and separately with the original backbone plus the upstream adapter (Dirty). Both runs use identical downstream data, configuration, and random seed.
6. Select downstream checkpoints by validation AUC and evaluate each selected model on the same test partition. Save continuous scores, AUC, UAUC, and relative changes.

The default mixed corpus contains ID 10% plus OOD 60%, relative to the target dataset size. The six OOD sources contribute equally. Sampling uses replacement only when the requested sample count exceeds the available source records. Upstream adapters are merged into the Dirty backbone in memory before initializing a separate downstream adapter. The merged 7B checkpoint does not need to be written to disk.

## Environment

Use Python 3.10 or 3.11. Install a PyTorch/torchvision build compatible with the GPU and CUDA environment first, then install the remaining dependencies:

```bash
python -m pip install -r requirements.txt
```

The entry points below use one visible CUDA GPU. Select the GPU through `CUDA_VISIBLE_DEVICES` when necessary. The library versions for the LLM interface are Transformers 4.39.3 and PEFT 0.10.0. Model files, datasets, and outputs are external assets and are excluded by `.gitignore`.

## Target datasets

Raw sources: [MovieLens-1M](https://grouplens.org/datasets/movielens/1m/) and [Amazon Reviews](https://jmcauley.ucsd.edu/data/amazon/). Use the review subset and corresponding metadata for the selected benchmark.

MovieLens requires both `ratings.dat` and `movies.dat`:

```bash
python preprocess_data.py --dataset ml-1m --raw-path dataset/raw/ml-1m --output-dir dataset/ml-1m
```

For Amazon-Book, pass review JSONL (optionally gzip-compressed) and matching item metadata:

```bash
python preprocess_data.py --dataset amazon-book --raw-path dataset/raw/Books_5.json.gz --metadata-path dataset/raw/meta_Books.json.gz --output-dir dataset/amazon-book
```

Amazon input can also be a CSV with `uid,iid,timestamp,title,rating` or `uid,iid,timestamp,title,label`. `label` must already be binary. Ratings are binarized at 4 for MovieLens and 5 for Amazon by default; `--positive-threshold` sets another threshold. Optional `--min-user-interactions` and `--min-item-interactions` apply iterative interaction-count filtering before splitting. Timestamps determine the global chronological split. Each context contains prior positive interactions and excludes the current target.

Outputs are `train_ood2.pkl`, `valid_ood2.pkl`, `test_ood2.pkl`, a `valid_small_ood2.pkl` compatibility copy of the full validation split, `id_mapping.json`, and `dataset_manifest.json`. User and target-item IDs start at 1; ID 0 is reserved for padding. The manifest records preprocessing options, counts, and file hashes.

Existing processed data can be used directly. Each split must contain:

| Field | Meaning |
| --- | --- |
| `uid`, `iid` | Positive integer user and target-item IDs, consistent across splits and encoder checkpoints |
| `title` | Target-item title |
| `his`, `his_title` | Equal-length lists of historical item IDs and titles |
| `label` | Binary recommendation label, 0 or 1 |
| `record_id` | Optional unique interaction ID; otherwise generated from split and row index |

Input `.pkl` files must come from a trusted source. Full validation data are used by the public training entry points; the compatibility copy is not used to select a smaller validation subset.

## External sources and serialization

Source links are listed in `dataset/download_ood.sh`. That script prints download locations; it does not download or preprocess data.

`prepare_ood_raw.py` converts raw source tables directly into injection JSONL:

```bash
python prepare_ood_raw.py --source lastfm --input dataset/raw/hetrec2011-lastfm-2k --output dataset/ood/lastfm.jsonl
python prepare_ood_raw.py --source mind --input dataset/raw/MINDlarge_train --output dataset/ood/mind.jsonl
python prepare_ood_raw.py --source amazon-sports --input dataset/raw/Sports_and_Outdoors_5.json.gz --metadata dataset/raw/meta_Sports_and_Outdoors.json.gz --output dataset/ood/amazon-sports.jsonl
python prepare_ood_raw.py --source amazon-beauty --input dataset/raw/Beauty_5.json.gz --metadata dataset/raw/meta_Beauty.json.gz --output dataset/ood/amazon-beauty.jsonl
python prepare_ood_raw.py --source gowalla --input dataset/raw/Gowalla_totalCheckins.txt.gz --output dataset/ood/gowalla.jsonl
python prepare_ood_raw.py --source epinions --input dataset/raw/epinions.csv --metadata dataset/raw/epinions_items.csv --positive-threshold 4 --output dataset/ood/epinions.jsonl
```

Epinions input has `uid,iid,timestamp` and either `label` or `rating`; item metadata has `iid,title`. A `title` column in the interaction table can replace the metadata file. Last.fm uses the HetRec artist/user/tag tables and user-artist interactions, excluding tags attached to the target artist from that record's context. MIND uses `news.tsv` and `behaviors.tsv`, with clicked impressions as targets and historical news categories/titles as micro-behaviors. Amazon product contexts use prior highly rated products and their metadata. Gowalla uses timestamped check-ins and location coordinates. Amazon/Gowalla records with no historical context are omitted. Select the source subset before conversion and adjust history lengths and positive-rating thresholds through command-line arguments.

`prepare_ood.py` accepts source records in JSONL, CSV, or a pandas pickle. It applies the source-specific serialization templates:

| Source | Context fields | Target field |
| --- | --- | --- |
| `epinions` | `his_title` or `history`: list of prior item titles | `target` or `title` |
| `lastfm`, `mind` | `tags`: list of prior interest tags or micro-behaviors | `target` or `title` |
| `amazon-sports`, `amazon-beauty` | `profiles`: list of historical item objects with `brand`, `category`, `attributes` | `target` or `title` |
| `gowalla` | `locations`: list of prior check-ins | `target` or `title` |

List/object fields in CSV must be JSON-encoded. A source record can alternatively contain an explicit `prompt` and `target`; this preserves an existing serialization. Timestamped event sources use context preceding the target event. The Last.fm adapter uses user tags associated with other artists as a preference profile; it does not impose an event chronology on the user-artist table.

For example:

```bash
python prepare_ood.py --source epinions --input dataset/raw/epinions_records.jsonl --output dataset/ood/epinions.jsonl
python prepare_ood.py --source lastfm --input dataset/raw/lastfm_records.jsonl --output dataset/ood/lastfm.jsonl
python prepare_ood.py --source mind --input dataset/raw/mind_records.jsonl --output dataset/ood/mind.jsonl
python prepare_ood.py --source amazon-sports --input dataset/raw/sports_records.jsonl --output dataset/ood/amazon-sports.jsonl
python prepare_ood.py --source amazon-beauty --input dataset/raw/beauty_records.jsonl --output dataset/ood/amazon-beauty.jsonl
python prepare_ood.py --source gowalla --input dataset/raw/gowalla_records.jsonl --output dataset/ood/gowalla.jsonl
```

All six paths are configured under `ood_sources`. Each output JSONL line contains `source`, `record_id`, `prompt`, and `target`. Large source corpora are indexed by byte offset during sampling, avoiding loading all serialized texts into memory at once.

## Configuration and model weights

The two experiment configurations are:

- `configs/experiment_ml1m.json`
- `configs/experiment_amazon_book.json`

Set `base_model` to the local Vicuna-7B checkpoint directory containing model configuration, weights, and tokenizer files. Set `data_dir` and `ood_sources` to the prepared data locations. Relative paths are resolved from this repository root.

For each method, `pretrained_encoder` can point to an existing MF/DIN state dictionary compatible with the processed user/item ID mapping. If it is `null`, the pipeline trains the collaborative encoder once and shares the resulting checkpoint between Clean and Dirty. Encoder dimensions, alignment, downstream LoRA training, and upstream injection budgets are configured independently.

The LoRA rank is 8, alpha is 16, and dropout is 0.05. Upstream and downstream LLM training use a batch size of 2, gradient accumulation of 64, Adam with weight decay 0.01, and cosine learning-rate decay from 0.0002 to 0.00001. Edit the epoch counts, sequence length, and stage settings in the experiment configuration to control the training budget.

## Paired experiments

From the repository root:

```bash
python run_leakage_impact_analysis.py --config configs/experiment_ml1m.json --methods CoLLM-MF CoLLM-DIN --output-dir outputs/ml1m_mixed
python run_leakage_impact_analysis.py --config configs/experiment_amazon_book.json --methods CoLLM-MF CoLLM-DIN --output-dir outputs/book_mixed
```

Each command trains one exposed upstream backbone and evaluates both methods against it. Every method has a paired Clean run. The downstream training data are never replaced with the injection corpus.

ID-only and OOD-only settings:

```bash
python run_leakage_impact_analysis.py --dataset ml-1m --id-ratio 0.1 --ood-ratio 0 --output-dir outputs/ml1m_id
python run_leakage_impact_analysis.py --dataset ml-1m --id-ratio 0 --ood-ratio 0.6 --output-dir outputs/ml1m_ood
```

To study one OOD source, retain that source in the configuration's `ood_sources` mapping. The requested OOD total is allocated equally among the listed sources.

## Individual stages

The stages can also be called independently:

```bash
python build_leakage_corpus.py --config configs/experiment_ml1m.json --output-dir outputs/corpus
python train_upstream_lora.py --config configs/experiment_ml1m.json --corpus outputs/corpus/corpus.jsonl --output-dir outputs/adapter
python train_rec_encoder.py --config configs/experiment_ml1m.json --method CoLLM-MF --output-dir outputs/mf_encoder
python train_downstream_rec.py --config configs/experiment_ml1m.json --method CoLLM-MF --backbone clean --rec-checkpoint outputs/mf_encoder/encoder.pt --exposed-id outputs/corpus/exposed_id.json --output-dir outputs/mf_clean
python train_downstream_rec.py --config configs/experiment_ml1m.json --method CoLLM-MF --backbone dirty --rec-checkpoint outputs/mf_encoder/encoder.pt --upstream-adapter outputs/adapter --exposed-id outputs/corpus/exposed_id.json --output-dir outputs/mf_dirty
```

Choose a fresh output directory for each invocation. To reuse an upstream adapter in the paired pipeline, pass `--upstream-adapter PATH`; its base checkpoint and corpus manifest must match the new run.

## Outputs

- `injection/corpus.jsonl`: sampled and serialized injection records.
- `injection/manifest.json`: source counts, replacement flags, and file hashes.
- `injection/exposed_id.json`: sampled target interactions and user/item pairs.
- `upstream_adapter/`: upstream LoRA weights and the injection manifest.
- `<method>/encoder/encoder.pt`: shared collaborative encoder, when trained by the pipeline.
- `<method>/<clean|dirty>/alignment/` and `finetuning/`: validation history and selected stage checkpoints.
- `<method>/<clean|dirty>/test_predictions.csv`: test record IDs, users, items, labels, and continuous scores.
- `<method>/<clean|dirty>/test_metrics.json`: test AUC/UAUC, eligible user counts, and optional Seen/Unseen metrics.
- `final_results.csv`: paired test metrics and relative percentage changes.

UAUC averages per-user AUC over users with both positive and negative labels. An undefined subset metric is stored as `null`. Missing outputs and failed runs raise errors; they are never converted to zero-valued scores. Seen/Unseen membership uses user/item pairs present in the injection corpus.

To regenerate the summary from completed paired runs:

```bash
python run_leakage_impact_analysis.py --config configs/experiment_ml1m.json --output-dir outputs/ml1m_mixed --summarize-only
```

## Code organization

`leakage/` contains the shared data contract, metrics, and training utilities. `minigpt4/` contains the collaborative recommendation modules. The four MF/DIN configurations in `train_configs/` define model construction and prompts; active training settings are read from `configs/experiment_*.json`. The existing preprocessing notebooks remain available in `dataset/`.

## Acknowledgments

The collaborative recommendation implementation builds on CoLLM and includes MiniGPT-4/LAVIS-derived components. Original source headers and attributions are retained.

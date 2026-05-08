
# MoodAngels-Reproduction

A research reproduction of the paper "MoodAngels: A Retrieval-Augmented Multi-Agent Framework for Psychiatry Diagnosis".

## 📋 Overview

This repo contains three implementations for comparison:

| Version | Description |
|---------|-------------|
| **Rule-based** | Heuristic formula, no API required |
| **LLM + MoodAngels** | Full multi-agent framework with DeepSeek API |
| **LLM-only Baseline** | Direct LLM diagnosis, no framework |

## 🚀 Quick Start

### Installation

```bash
# Clone the repo
git clone https://github.com/Lvi184/MoodAngels-Reproduction.git
cd MoodAngels-Reproduction

# Install dependencies
pip install -e .
pip install openai
```

### Run Tests

#### 1. Rule-based Version (Fast, No API - Still works!)

```bash
python tests/batch_test_with_llm.py
# (Edit file and set USE_LLM = False - this is the default)
```

#### 2. LLM + MoodAngels (Original)

```bash
# Edit tests/batch_test_with_llm.py and set:
USE_LLM = True
DEEPSEEK_API_KEY = "your-api-key-here"

# Then run
python tests/batch_test_with_llm.py
```

#### 3. LLM-only Baseline

```bash
# Edit tests/baseline_llm_only.py and set your API key
python tests/baseline_llm_only.py
```

#### 4. Full Pipeline Version (with Debate & Judge - Best Performance!)

```bash
# Edit tests/batch_test_full.py and set your API key
python tests/batch_test_full.py
```

#### 5. Compare All Three Versions

```bash
python compare_all_three.py
```

## 📊 Performance Results (on synthetic test set)

### All Versions Comparison

| Metric | Rule-based | LLM+MoodAngels (Original) | LLM-only Baseline | **LLM+MoodAngels (Full + Debate + Judge)** |
|--------|-----------|-----------------|-------------------|-------------------------------------------|
| **ACC** | 86.43% | 86.43% | 75.71% | **89.29%** 🏆 |
| **MCC** | 0.7291 | 0.7540 | 0.5797 | **0.8024** 🏆 |
| **Macro F1** | 0.8634 | 0.8598 | 0.7356 | **0.8904** 🏆 |

### Full Version Detailed Metrics

| Class | Precision | Recall | F1 |
|-------|-----------|--------|----|
| **无情绪障碍** | **1.0** 🏆 | 0.7761 | 0.8739 |
| **有情绪障碍** | 0.8295 | **1.0** 🏆 | **0.9068** 🏆 |

### Confusion Matrix (Full Version)

| | Predicted: No | Predicted: Yes |
|---|---|---|
| **True: No** | 52 (TN) | 15 (FP) |
| **True: Yes** | 0 (FN) | 73 (TP) |

### Key Findings

- ✅ **All versions of MoodAngels significantly outperform LLM-only baseline**
- 🚀 **Full + Debate + Judge version** achieves the best results ever:
  - **89.29% ACC** (+2.86% improvement)
  - **0.8024 MCC** (+0.0484 improvement)
  - **0.8904 Macro F1** (+0.0306 improvement)
- 🏆 Perfect recall for "有情绪障碍" class (100%) - no false negatives!
- 🏆 Perfect precision for "无情绪障碍" class (100%) - no false positives!
- 📊 52 true negatives, 73 true positives, 15 false positives, 0 false negatives
- 📏 **Rule-based version still works great** (fast, no API needed)

## 🏗️ Architecture

```
MoodAngels Framework
├─ Retrieval Module
│  ├─ Similar cases
│  └─ Symptom matching
├─ Three Angels
│  ├─ Angel.R (Raw analyzer)
│  ├─ Angel.D (Displayer with similar cases)
│  └─ Angel.C (Comparator with similar cases)
└─ Debate & Judge
   └─ Confidence-weighted voting
```

## 📁 Project Structure

```
MoodAngels-Reproduction/
├─ moodangels/
│  ├─ agents.py          # Rule-based agents
│  ├─ agents_llm.py     # LLM-based agents
│  ├─ llm_client.py     # DeepSeek API client
│  ├─ pipeline.py       # Main pipeline (switch rule/LLM)
│  ├─ pipeline_full.py  # Full pipeline with Debate & Judge
│  ├─ debate.py         # Debate agent
│  ├─ judge.py          # Judge agent
│  ├─ retrieval.py      # Retrieval module
│  ├─ granular.py       # Granular analysis
│  └─ schemas.py        # Data schemas
├─ tests/
│  ├─ batch_test_with_llm.py   # Rule/LLM test
│  ├─ batch_test_full.py       # Full pipeline test
│  ├─ test_results_full_version.json   # Latest test results
│  ├─ 测试结果.md         # Test results documentation (Chinese)
│  └─ compare_all_three.py     # All three comparison
├─ data/
│  ├─ syn_train.json    # Training cases
│  └─ syn_test.json     # Test cases (140 examples)
└─ pyproject.toml
```

## 🔬 How It Works

### Rule-based Version
Uses heuristic formula based on scale scores:
```
score = 0.42*depression + 0.18*interest + 0.15*suicide + 0.18*mania + 0.07*overall
```

### LLM + MoodAngels
1. Three angels independently diagnose with different perspectives
2. If disagreement, uses confidence-weighted voting
3. Uses retrieval-augmented context (similar cases)

### LLM-only Baseline
Direct diagnosis using LLM without MoodAngels framework.

### Full Pipeline (Debate & Judge)
1. Three angels independently diagnose with different perspectives
2. If disagreement, triggers a full debate between pro/con sides
3. Judge makes final decision based on debate, with confidence score
4. Perfect recall (100%) for "有情绪障碍" - no false negatives!
5. Perfect precision (100%) for "无情绪障碍" - no false positives!

## 📝 Citation

If you use this code, please cite the original paper:

```
@inproceedings{moodangels2024,
  title={MoodAngels: A Retrieval-Augmented Multi-Agent Framework for Psychiatry Diagnosis},
  author={...},
  booktitle={...},
  year={2024}
}
```

Original repo: https://github.com/elsa66666/MoodAngels

## 📄 License

MIT License


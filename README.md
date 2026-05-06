# MoodAngels Complete (research reproduction)

This package is a runnable, self-contained reconstruction of the public MoodAngels repository plus missing project glue: data loading, granular scale analysis, DSM-style symptom matching, similar-case retrieval, Angel.R / Angel.D / Angel.C, a judge/debate layer, CLI, optional API, and tests.

It is **not** the authors' unpublished clinical system and must **not** be used as a medical diagnostic device. It is for research/prototyping on MoodSyn-style synthetic data.

## What was completed

The public repository currently exposes only a small README, `data_sample.csv`, and six Python files under `code/`. This version turns the idea into a runnable project:

- `moodangels/granular.py`: item-level symptom grouping for depression, interest/energy, suicide, anxiety, insomnia, mania/bipolar.
- `moodangels/retrieval.py`: lightweight local vector retrieval for criteria and similar synthetic cases.
- `moodangels/agents.py`: Angel.R, Angel.D, Angel.C and multi-Angels judge/debate flow.
- `moodangels/cli.py`: `predict` and `evaluate` commands.
- `moodangels/api.py`: optional FastAPI service.
- `data/syn_train.json`: included synthetic training data you provided.

## Install

```bash
cd MoodAngels_complete
python -m pip install -e .
```

## Run a single case

```bash
moodangels predict --case-json examples/example_case.json --retrieval-data data/syn_train.json --agent multi
```

## Evaluate quickly

```bash
moodangels evaluate --data data/syn_train.json --agent multi --limit 100
```

## Optional API

```bash
python -m pip install -e '.[api]'
uvicorn moodangels.api:app --reload
```

Then POST a MoodSyn-style JSON object to `/diagnose?agent=multi`.

## Notes on fidelity

The paper describes a retrieval-augmented multi-agent framework using granular scale analysis, DSM-5 criteria retrieval, historical case retrieval, three agents with different reliance on cases, and debate/judge synthesis. Since the public repository does not release the full clinical data store, exact DSM-5 knowledge base, prompts, or API-specific LLM orchestration, this implementation uses:

- local deterministic scoring for reproducibility;
- a small DSM-style criteria seed list rather than copyrighted DSM-5 text;
- simple TF-IDF-like retrieval with no heavyweight model downloads;
- optional extension points for LLM integration.

## Citation / provenance

- MoodAngels paper: `MoodAngels: A Retrieval-augmented Multi-agent Framework for Psychiatry Diagnosis`.
- Public repository: <https://github.com/elsa66666/MoodAngels>

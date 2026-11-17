# CS336 Spring 2025 Assignment 1: Basics

For a full description of the assignment, see the assignment handout at
[cs336_assignment1_basics.pdf](./cs336_assignment1_basics.pdf)

If you see any issues with the assignment handout or code, please feel free to
raise a GitHub issue or open a pull request with a fix.

## Setup

### Environment
We manage our environments with `uv` to ensure reproducibility, portability, and ease of use.
Install `uv` [here](https://github.com/astral-sh/uv#installation) (recommended), or run `pip install uv`/`brew install uv`.
We recommend reading a bit about managing projects in `uv` [here](https://docs.astral.sh/uv/guides/projects/#managing-dependencies) (you will not regret it!).

You can now run any code in the repo using
```sh
uv run <python_file_path>
```
and the environment will be automatically solved and activated when necessary.

### Run unit tests


```sh
uv run pytest
```

Initially, all tests should fail with `NotImplementedError`s.
To connect your implementation to the tests, complete the
functions in [./tests/adapters.py](./tests/adapters.py).

### Download data
Download the TinyStories data and a subsample of OpenWebText

``` sh
mkdir -p data
cd data

wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt
wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt

wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_train.txt.gz
gunzip owt_train.txt.gz
wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_valid.txt.gz
gunzip owt_valid.txt.gz

cd ..
```

---

## My Progress (Student Work)

> **Authorship:** All implementation code in this repository (`cs336_basics/`, notebooks, and the
> wiring in `tests/adapters.py`) is my own work as a CS336 student. An AI assistant was used only
> as a teaching aid, as allowed by [CLAUDE.md](./CLAUDE.md): it reviewed code, explained profiler
> output, and suggested what to investigate. It did not write or edit assignment code.

### Status

| Problem | Status |
|---|---|
| `train_bpe` — BPE tokenizer training | ✅ `tests/test_train_bpe.py` passing; optimization in progress |
| `train_bpe_tinystories` | 🔲 Not started |
| `train_bpe_expts_owt` | 🔲 Not started |

### Files

| Path | Purpose |
|---|---|
| [`cs336_basics/pretokenization.py`](./cs336_basics/pretokenization.py) | Splits the corpus into chunks on special-token boundaries and counts pre-tokens in parallel (`multiprocessing.Pool`) |
| [`cs336_basics/bpe_train.py`](./cs336_basics/bpe_train.py) | BPE training loop: builds byte-level vocab, counts pairs, applies merges |
| [`cs336_basics/test.ipynb`](./cs336_basics/test.ipynb) | Scratch notebook for running and profiling `train_bpe` |
| [`tasks/todo.md`](./tasks/todo.md) | Optimization task checklist and profiling notes |

### Run

```sh
# Unit tests for BPE training
uv run pytest tests/test_train_bpe.py

# Profile BPE training (results saved to bpe.prof)
uv run python -m cProfile -o bpe.prof -m pytest tests/test_train_bpe.py::test_train_bpe
uv run python -m pstats bpe.prof        # then: sort tottime / stats cs336_basics 15
```

### Profiling notes

| Dataset | vocab_size | Finding |
|---|---|---|
| `tests/fixtures/corpus.en` | 500 | `_merge_word` ≈ 83% of training time |
| `TinyStoriesV2-GPT4-valid.txt` | 1000 | `_merge_word` ≈ 92%; ~9.7M calls (13,111 unique words × 743 merges) |

Next steps are tracked in [`tasks/todo.md`](./tasks/todo.md).


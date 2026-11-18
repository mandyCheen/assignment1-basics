# BPE Training Optimization Tasks

> **Note on authorship:** All implementation code in this repository (`cs336_basics/bpe_train.py`,
> `cs336_basics/pretokenization.py`, notebooks, etc.) is the **student's own work**.
> An AI assistant was used only as a teaching aid, following `CLAUDE.md`: it reviewed code,
> explained profiler output, and suggested what to investigate. It did not write or edit any
> assignment code. This file is a task checklist, not a solution.

---

## ✅ Done

- [x] **T0-1** Profile baseline
  - `corpus.en`: `_merge_word` ≈ 83% of `train_bpe` time.
  - TinyStories valid (vocab 1000): `_merge_word` ≈ 92%; ~9.7M calls = 13,111 words × 743 merges.
- [x] Fix special-token handling in `re.split` (special tokens no longer get pretokenized).
- [x] Fix pair-count update after a merge (subtract all old pairs, add all new pairs).

## Phase 0 — Baseline numbers

- [x] **T0-3** Time the valid set (vocab 1000) **without** the profiler (`%%time`) and record the real runtime.
  - **Baseline (2026-11-17)** — `TinyStoriesV2-GPT4-valid.txt`, `vocab_size=1000`, `%%time` in `test.ipynb`, single run:
    - `Wall time: 5.16 s` ← compare against this
    - `CPU times: user 4.02 s, sys: 121 ms, total: 4.14 s` (main process only; pretokenize workers not counted)
- [x] **T0-2** Confirm `uv run pytest tests/test_train_bpe.py` passes. Re-run it after every task below.

## Phase 1 — Correctness / robustness

- [x] **T1-1** `pretokenization.py:89-90`: the loop overwrites `boundaries` on every pass, and `boundaries` is undefined when `special_tokens=[]`.
  - Fixed: `find_chunk_boundaries` now takes a list of special tokens and cuts at the earliest match; `pretokenize` calls it once.
- [x] **T1-2** `pretokenization.py:25`: with `special_tokens=[]`, the split pattern becomes an empty string.
- [x] **T1-3** Remove pairs whose count drops to 0 from `pair_counter`, so the `if not pair_counter` check on line 48 works.

## Phase 2 — Merge loop (⭐ top priority, backed by profile data)

- [x] **T2-1** Build a "pair → words containing that pair" index so each merge only touches affected words.
  - Done when: tests pass and `_merge_word` ncalls drops well below ~9.7M. ✅ `pytest tests/test_train_bpe.py` passes and the invariant check (recompute pair counts / index from scratch after each merge) matches — both run by the student.
  - **Implementation (2026-11-18):** `pair_to_word_dict: pair → set[word]`. Each merge takes a snapshot `list(...)` of the merged pair's word set. For every word in it: subtract all old pairs from `pair_counter` and `discard` the word from each old pair's set (deleting emptied sets), run `_merge_word`, then add all new pairs back to the counter and the index. The index therefore always matches reality: a word is in a pair's set iff it currently contains that pair (no duplicates, no stale entries).
  - **Result** — `TinyStoriesV2-GPT4-valid.txt`, `vocab_size=1000`, `test_notebook.ipynb`, single run each:

    | Metric | Baseline (T0-1 / T0-3) | After T2-1 | Change |
    |---|---|---|---|
    | `%%time` Wall time | 5.16 s | **1.05 s** | ≈ 4.9× faster |
    | `%%time` CPU time (total, main process) | 4.14 s | 0.40 s | ≈ 10× less |
    | `_merge_word` ncalls (`%%prun`) | 9,741,473 | **44,850** | ≈ 217× fewer (~60 words per merge) |

  - **Where time goes now** (`%%prun`, 1.31 s total, profiler-inflated; ratios only, varies run to run): `max` incl. its `lambda` 0.48 s (~36%); waiting on pretokenize workers (`posix.read`) 0.73 s cumulative; `train_bpe` self time 0.20 s; `_merge_word` 0.08 s. (`kqueue` / `select` rows are the Jupyter event loop, ignore.)
  - Note: switching to the exact set-based index cut `_merge_word` calls by ~24% vs the previous list-based version (59,062 → 44,850), but wall time did not visibly improve (single-run noise is about the same size). `_merge_word` is now a tiny share of the total, so the next real gains are `max` (T2-2) and pretokenize (Phase 3).

- [ ] **T2-3** Small cleanups: the duplicate `b"".join` (lines 56 / 58), the `new_tokens == tokens` check that may become unnecessary after T2-1, and the repeated `len` calls inside `_merge_word`.
- [ ] **T2-2** Re-profile after T2-1. **Only if** `max` becomes the main bottleneck: switch to a structure that can get the max quickly while keeping the `(count, pair)` tie-break.

## Phase 3 — Pretokenization memory / speed (before running the full train set)

- [ ] **T3-0** Run `vocab_size=1000` on the full `TinyStoriesV2-GPT4-train.txt` and check whether pretokenize's share of the time grows noticeably.
- [ ] **T3-1** Decouple the number of chunks from the number of processes (many small chunks handed to the pool); compare peak memory with `/usr/bin/time -l`.
- [ ] **T3-2** Set the process count from `os.cpu_count()` or make it a parameter.

## Phase 4 — Structure refactor (OOP; after the performance work)

- [ ] **T4-1** List the separate jobs `train_bpe` currently does and decide which should be split out.
- [ ] **T4-2** Decide whether the training state (word counts, pair counts, index) belongs in one object, based on whether these must always be updated together to stay consistent.
- [ ] **T4-3** Design vocab/merges serialization so the later `Tokenizer` (`from_files`) can reuse it.
- [ ] **T4-4** Avoid over-design: every extracted unit should be testable on its own with one small test.

## Phase 5 — Experiments (graded write-up answers)

- [ ] **T5-1** `train_bpe_tinystories` (vocab 10,000): record runtime, peak memory, and the longest token; serialize vocab/merges; answer (b) about which part takes the most time.
- [ ] **T5-2** `train_bpe_expts_owt` (vocab 32,000): run in the background with `caffeinate`; record the longest token; compare with TinyStories.

## Backlog (optional optimizations, do only if profiling says it's worth it)

- [ ] **B-1** Only update the pairs that actually changed in `train_bpe`'s per-word update. Currently every processed word has *all* its old pairs removed and *all* new pairs added back, including pairs the merge did not touch: their counts are subtracted then re-added, and a pair's index set that contains only this word is deleted then re-created. Instead, compare the word's old pairs with its new pairs (set difference): remove the word / subtract counts only for pairs that disappeared, add only for pairs that appeared. This is an optimization, not a correctness fix. Decide after re-profiling: if `train_bpe`'s own time is small compared with `max` and pretokenize, skip it.
- [ ] **B-2** Remove the redundant `if most_frequent_pair in pair_to_word_dict: del ...` at the end of each merge: the merged pair's set is already emptied and deleted while its words are processed. (Fold into T2-3.)

---

**Suggested order:** T0-3 → T0-2 → T1 → **T2-1** → re-profile → T2-2 or T3 depending on the data → T5-1 → T3 → T5-2 → T4

(T4-3 can be moved before T5-1, since T5 needs serialization.)

## Review

_(To be filled in by the student after finishing: what changed, before/after numbers.)_

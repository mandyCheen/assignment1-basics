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

- [ ] **T1-1** `pretokenization.py:89-90`: the loop overwrites `boundaries` on every pass, and `boundaries` is undefined when `special_tokens=[]`.
- [ ] **T1-2** `pretokenization.py:25`: with `special_tokens=[]`, the split pattern becomes an empty string.
- [ ] **T1-3** Remove pairs whose count drops to 0 from `pair_counter`, so the `if not pair_counter` check on line 48 works.

## Phase 2 — Merge loop (⭐ top priority, backed by profile data)

- [ ] **T2-1** Build a "pair → words containing that pair" index so each merge only touches affected words.
  - First work out on paper: after a merge, which index entries are added and which are removed? Use `[A, B, C]` merging `(A, B)` as the example.
  - Done when: tests pass and `_merge_word` ncalls drops well below ~9.7M.
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

---

**Suggested order:** T0-3 → T0-2 → T1 → **T2-1** → re-profile → T2-2 or T3 depending on the data → T5-1 → T3 → T5-2 → T4

(T4-3 can be moved before T5-1, since T5 needs serialization.)

## Review

_(To be filled in by the student after finishing: what changed, before/after numbers.)_

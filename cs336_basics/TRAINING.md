# BPE Tokenizer 訓練流程文件

> 範圍說明：目前 `cs336_basics/` 裡的「training」指的是 **byte-level BPE tokenizer 的訓練**（還沒有 Transformer 模型的 training loop）。本文件只描述 `pretokenization.py` 與 `bpe_train.py` 現有的實作。

## 1. 檔案與職責

| 檔案 | 職責 |
|---|---|
| `pretokenization.py` | 把 corpus 切成多個 chunk，平行做 pre-tokenization，回傳「每個 pre-token 出現幾次」的 `Counter` |
| `bpe_train.py` | 用 `Counter` 跑 BPE merge loop，產出 `vocab` 與 `merges`；也提供存檔 / 讀檔與 CLI |
| `pretokenization_example.py` | 範例檔，不在訓練流程內 |
| `test_notebook.ipynb` | 實驗用 notebook，不在訓練流程內 |

## 2. 名詞定義

- **corpus**：訓練用的原始文字檔（以 bytes 讀取、UTF-8 解碼）。
- **special token**：不參與 merge 的特殊字串，預設是 `<|endoftext|>`。
- **pre-token**：用 regex `PAT` 把文字切出來的片段（例如 `" hello"`、`" 123"`、`"'s"`）。BPE 的 merge 只會發生在單一 pre-token 內，不會跨 pre-token。
- **word_counter**：`Counter`，key 是 pre-token 字串，value 是它在整個 corpus 出現的次數。
- **token**：BPE 過程中的最小單位，型別是 `bytes`。一開始每個 token 是單一 byte。
- **pair**：相鄰兩個 token 組成的 `tuple[bytes, bytes]`。
- **vocab**：`dict[int, bytes]`，token id 對應 token 內容。
- **merges**：`list[tuple[bytes, bytes]]`，依照被合併的先後順序記錄每一次 merge 的 pair。

## 3. 整體流程（`main()` 的三個階段）

```
corpus ──[1/3 Pretokenize]──> word_counter ──[2/3 Merge loop]──> (vocab, merges) ──[3/3 Save + 驗證]──> out/bpe.pkl
```

CLI 參數：`--input`（必填）、`--vocab-size`（必填）、`--special-tokens`（預設 `<|endoftext|>`）、`--output`（預設 `out/bpe.pkl`）、`--num-processes`（預設 4）、`--num-chunks`（預設 8）。

## 4. 階段 1：Pre-tokenization（`pretokenization.py`）

入口函式是 `pretokenize`。

1. ✅ **找 chunk 邊界**（`find_chunk_boundaries`）：先把檔案大小平均分成 `num_chunks` 份，作為初始邊界。對每個中間邊界，從該位置往後每次讀 4096 bytes，找最近的 special token，把邊界移到 special token 的開頭。這樣每個 chunk 都在 document 邊界切開，不會把一個 document 切成兩半。若一路讀到檔尾都沒找到，邊界就設為檔尾。最後用 `set` 去重，所以實際 chunk 數可能少於 `num_chunks`。
2. ✅ **平行處理**：用 `multiprocessing.Pool(num_processes)` 對每個 `(start, end)` 呼叫 `_process_chunk`。chunk 數與 process 數是分開的兩個參數（commit `b6c3899` 解耦）。
3. ✅ **單一 chunk 的處理**（`_process_chunk`）：
   1. 以 `seek` + `read` 只讀自己負責的 byte 範圍，UTF-8 解碼成字串。
   2. 以 special token 當分隔符 `re.split`，special token 依長度由長到短排序（避免短的 token 先吃掉長 token 的一部分）。special token 本身會被丟掉，因此不會進入 `word_counter`。
   3. 對每個 segment 用 `PAT` 做 `finditer`，每個 match 在 `Counter` 裡 +1。
4. ✅ **合併結果**：主 process 把所有 chunk 的 `Counter` 用 `update` 加總成一個 `word_counter`。

## 5. 階段 2：BPE merge loop（`train_bpe_from_counts`）

### 5.1 初始化

1. ✅ **word → token 序列**：每個 pre-token 轉成 UTF-8 bytes，再拆成「每個 byte 一個 token」的 list（`word_tokens`）。
2. ✅ **統計所有 pair**：對每個 word 的相鄰 token 計算 pair，把 pair 的次數加上該 word 的出現次數（不是只加 1），存在 `pair_counter`。同時建立反向索引 `pair_to_word_dict`：pair → 含有這個 pair 的 word 集合。
3. ✅ **初始 vocab**：先放 256 個單一 byte（id 0–255），再依序追加 special tokens。`merges` 從空 list 開始。

### 5.2 每一輪 merge

迴圈次數為 `vocab_size - len(vocab)`（`vocab_size` 扣掉初始 vocab 大小）。每一輪：

1. ✅ **選 pair**：取 `pair_counter` 中次數最大者；次數相同時，用 pair 本身（bytes tuple）的字典序較大者。這個 tie-break 規則是為了讓結果可重現，並符合作業規定。若 `pair_counter` 已空就提前結束。
2. ✅ **建立新 token**：把 pair 的兩個 bytes 接起來，以 `len(vocab)` 作為新 id 加入 `vocab`。
3. ✅ **只更新受影響的 word**：透過 `pair_to_word_dict` 找出含有該 pair 的 word，不用掃整個 word_counter。對每個 word：
   1. 把這個 word「舊的」所有 pair 從 `pair_counter` 扣掉（扣的量是 word 的出現次數；歸零就刪除 key），並從 `pair_to_word_dict` 移除這個 word。
   2. 用 `_merge_word` 由左到右掃描，遇到目標 pair 就換成新 token（一次跨兩格），否則保留。
   3. 把「新的」所有 pair 加回 `pair_counter` 與 `pair_to_word_dict`。
   4. 更新 `word_tokens[word]`。
4. ✅ **記錄**：把這個 pair 附加到 `merges`。

迴圈結束回傳 `(vocab, merges)`。

## 6. 階段 3：存檔與驗證（`save_bpe` / `load_bpe`）

1. ✅ `save_bpe`：必要時建立輸出目錄，用 `pickle` 把 `{"vocab": ..., "merges": ...}` 寫成檔案。
2. ✅ `main()` 存完後立刻 `load_bpe` 讀回來，用 `assert` 確認 `vocab`、`merges` 與記憶體中的完全相同（round-trip 檢查）。
3. ✅ 印出各階段耗時（pretokenize、merge loop、total）與最終 vocab 大小、merge 數。

## 7. 執行方式

```bash
python -m cs336_basics.bpe_train --input <corpus路徑> --vocab-size 10000 --output out/bpe.pkl
```

可另外指定 `--num-processes`、`--num-chunks`、`--special-tokens`。程式也可以直接呼叫 `train_bpe(input_path, vocab_size, special_tokens)`，它內部就是「`pretokenize` → `train_bpe_from_counts`」兩步（用預設的 process / chunk 數）。

## 8. 設計重點整理

- **為什麼先統計 pre-token 次數**：同一個 pre-token 出現很多次，只需處理一次，pair 的次數乘上 word 次數即可，merge loop 的規模取決於「不重複 pre-token 數」而不是 corpus 大小。
- **為什麼用 `pair_to_word_dict`**：每輪只重算含目標 pair 的 word，避免每輪全部重掃。
- **為什麼 chunk 邊界對齊 special token**：保證平行處理的各 chunk 之間不會把同一個 document 切開，結果與單 process 一致。
- **special token 不被 merge**：它們在 pre-tokenization 時就被切掉，只在 vocab 裡佔固定 id。

## 9. 已確定 vs 尚未驗證

**已確定（直接由程式碼讀出）**
- 流程、參數預設值、tie-break 規則、vocab 初始順序（256 bytes → special tokens → merges）。
- chunk 數可能少於設定值（`set` 去重）。

**尚未驗證（本文件沒有實測數據）**
- 各階段實際耗時與瓶頸（專案內有 `bpe.prof`，可用 `python -m pstats` 或 snakeviz 查看；merge loop 裡每輪的 `max(pair_counter.items(), ...)` 是否為主要成本，需看 profile 才能確定）。
- 與官方測試（`tests/`）的通過狀況，本文件撰寫時沒有執行。

最便宜的驗證方式：對一個小 corpus 跑 CLI，看 summary 裡 pretokenize 與 merge loop 的秒數佔比，再對照 `bpe.prof`。

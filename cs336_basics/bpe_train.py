import os
import pickle
import argparse
import time

from cs336_basics.pretokenization import pretokenize
from collections import Counter


def save_bpe(vocab, merges, out_path) -> None:
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump({"vocab": vocab, "merges": merges}, f)


def load_bpe(in_path):
    with open(in_path, "rb") as f:
        data = pickle.load(f)
    return data["vocab"], data["merges"]


def _merge_word(tokens, pair, new_token):
    out, i = [], 0
    while i < len(tokens):
        if i < len(tokens) - 1 and tokens[i] == pair[0] and tokens[i + 1] == pair[1]:
            out.append(new_token)   
            i += 2                  
        else:
            out.append(tokens[i])   
            i += 1                 
    return out


def train_bpe_from_counts(word_counter, vocab_size, special_tokens):

    # Convert word to token
    word_tokens = {
        w: [bytes([b]) for b in w.encode("utf-8")]
        for w in word_counter
    }


    pair_counter = Counter()
    pair_to_word_dict: dict[tuple[bytes, bytes], set] = {}

    for word, tokens in word_tokens.items():
        for token_pair in zip(tokens, tokens[1:]):
            pair_counter[token_pair] += word_counter[word]
            pair_to_word_dict.setdefault(token_pair, set()).add(word)

    vocab = {i: bytes([i]) for i in range(256)}  # Initialize vocab with single-byte tokens
    for token in special_tokens:
        vocab[len(vocab)] = token.encode("utf-8")  # Add special tokens to vocab

    merges = list()

    # Merge process
    for i in range(0, vocab_size-len(vocab)):

        if not pair_counter:
            break  # No more pairs to merge

        # Find the most frequent pair
        most_frequent_pair = max(pair_counter.items(), key=lambda x: (x[1], x[0]))[0]

        new_token = b"".join(most_frequent_pair)

        # Update vocab and counter
        vocab[len(vocab)] = new_token

        words_to_update = list(pair_to_word_dict.get(most_frequent_pair, set()))

        for word in words_to_update:
            tokens = word_tokens[word]

            for token_pair in zip(tokens, tokens[1:]): 
                pair_counter[token_pair] -= word_counter[word]
                if pair_counter[token_pair] == 0:
                    del pair_counter[token_pair]
                if token_pair in pair_to_word_dict:
                    pair_to_word_dict[token_pair].discard(word)
                    if not pair_to_word_dict[token_pair]:
                        del pair_to_word_dict[token_pair]

            new_tokens = _merge_word(tokens, most_frequent_pair, new_token)

            for token_pair in zip(new_tokens, new_tokens[1:]):
                pair_counter[token_pair] += word_counter[word]
                pair_to_word_dict.setdefault(token_pair, set()).add(word)
                
            word_tokens[word] = new_tokens  

        merges.append(most_frequent_pair)

    return vocab, merges

def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str]
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    
    word_counter = pretokenize(input_path, special_tokens)

    return train_bpe_from_counts(word_counter, vocab_size, special_tokens)
        

def main() -> int:
    parser = argparse.ArgumentParser(description="Train a byte-level BPE tokenizer.")
    parser.add_argument("--input", required=True, help="Path to the training corpus")
    parser.add_argument("--vocab-size", type=int, required=True)
    parser.add_argument("--special-tokens", nargs="*", default=["<|endoftext|>"])
    parser.add_argument("--output", default="out/bpe.pkl", help="Output file path, e.g. out/bpe.pkl")
    parser.add_argument("--num-processes", type=int, default=4)
    parser.add_argument("--num-chunks", type=int, default=8)
    args = parser.parse_args()

    t0 = time.perf_counter()
    word_counter = pretokenize(args.input, args.special_tokens, args.num_processes, args.num_chunks)
    t1 = time.perf_counter()

    vocab, merges = train_bpe_from_counts(
        word_counter, args.vocab_size, args.special_tokens
    )
    t2 = time.perf_counter()

    save_bpe(vocab, merges, args.output)
    vocab2, merges2 = load_bpe(args.output)
    assert vocab == vocab2, "vocab roundtrip mismatch"
    assert merges == merges2, "merges roundtrip mismatch"

    print(f"pretokenize : {t1 - t0:.2f}s")
    print(f"merge loop  : {t2 - t1:.2f}s")
    print(f"total       : {t2 - t0:.2f}s")
    print(f"vocab size  : {len(vocab)}, merges: {len(merges)}")
    print(f"saved to    : {args.output} (roundtrip OK)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
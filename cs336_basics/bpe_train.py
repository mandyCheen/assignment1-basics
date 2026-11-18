import os

from .pretokenization import pretokenize
from collections import Counter


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

def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str]
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    
    word_counter = pretokenize(input_path, special_tokens)

    # Convert word to token
    word_tokens = {
        w: [bytes([b]) for b in w.encode("utf-8")]
        for w in word_counter
    }


    pair_counter = Counter()

    for word, tokens in word_tokens.items():
        for token_pair in zip(tokens, tokens[1:]):
            pair_counter[token_pair] += word_counter[word]

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


        # Update vocab and counter
        vocab[len(vocab)] = b"".join(most_frequent_pair)

        new_token = b"".join(most_frequent_pair)

        for word, tokens in word_tokens.items():
            new_tokens = _merge_word(tokens, most_frequent_pair, new_token)

            if new_tokens == tokens:
                continue

            for token_pair in zip(tokens, tokens[1:]):
                pair_counter[token_pair] -= word_counter[word]
                if pair_counter[token_pair] == 0:
                    del pair_counter[token_pair]

            for token_pair in zip(new_tokens, new_tokens[1:]):
                pair_counter[token_pair] += word_counter[word]
                
            word_tokens[word] = new_tokens  

        merges.append(most_frequent_pair)
    
    return (vocab, merges)
        






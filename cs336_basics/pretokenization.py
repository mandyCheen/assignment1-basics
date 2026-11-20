import os
import multiprocessing

import regex as re

from collections import Counter
from typing import BinaryIO


PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

def _process_chunk(
    input_path: str | os.PathLike,
    start: int,
    end: int,
    special_tokens: list[str]
) -> Counter:

    with open(input_path, "rb") as f:
        f.seek(start)
        chunk = f.read(end - start)

    counter = Counter()

    text = chunk.decode("utf-8")

    if special_tokens:
        sorted_tokens = sorted(special_tokens, key=len, reverse=True)
        pattern = "|".join(re.escape(t) for t in sorted_tokens)
        segments = re.split(pattern, text)
    else:
        segments = [text]

    for seg in segments:
        for m in re.finditer(PAT, seg):
            counter[m.group(0)] += 1

    return counter

def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_tokens: list[bytes],
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_tokens, list) and all( isinstance(t, bytes) for t in split_special_tokens), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_positions = [ pos for token in split_special_tokens if (pos := mini_chunk.find(token)) != -1]
            found_at = min(found_positions) if found_positions else -1
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))


def pretokenize(
    input_path: str | os.PathLike,
    special_tokens: list[str],
    num_processes: int = 4,
    num_chunks: int = 8,
) -> Counter:

    special_tokens_bytes = [t.encode("utf-8") for t in special_tokens]

    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, num_chunks, special_tokens_bytes)

    # Create a pool of worker processes
    with multiprocessing.Pool(processes=num_processes) as pool:
        # Prepare the arguments for each process
        args = [
            (input_path, boundaries[i], boundaries[i + 1], special_tokens)
            for i in range(len(boundaries) - 1)
        ]

        # Map the function to the arguments and collect results
        results = pool.starmap(_process_chunk, args)

    # Combine the results from all processes
    combined_counter = Counter()
    for result in results:
        combined_counter.update(result)

    return combined_counter

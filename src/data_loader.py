"""
data_loader.py
--------------
Downloads Google Books Ngrams 2-gram files and extracts co-occurrence
counts for a set of target words, aggregated by decade.

Only bigrams of the form (target_word, context_word) are collected,
i.e., we look at the files indexed by each target word's first two
letters — the same index Google Books uses for the first word of a
bigram.  This keeps the download footprint manageable while still
capturing rich forward-context information.

Usage
-----
    from src.data_loader import load_cooccurrences, TARGET_WORDS

    cooc = load_cooccurrences()
    # cooc["computer"][1980]["program"] -> 4271
"""

import gzip
import os
import pickle
import re
import urllib.request
from collections import defaultdict
from typing import Dict, List, Set

# ── Target words ──────────────────────────────────────────────────────────────

TARGET_WORDS: List[str] = [
    "computer", "virus",  "cloud",  "tablet",
    "mouse",    "spam",   "cell",   "web",
    "gay",      "tweet",  "crash",  "stream",
]

# ── Config ────────────────────────────────────────────────────────────────────

DECADE_START: int = 1850
DECADE_END:   int = 2009
MIN_COUNT:    int = 20      # drop bigrams rarer than this per year

BASE_URL = (
    "https://storage.googleapis.com/books/ngrams/books/"
    "googlebooks-eng-all-2gram-20120701-{prefix}.gz"
)

RAW_DIR        = os.path.join("data", "raw")
PROCESSED_DIR  = os.path.join("data", "processed")

# POS tags appended by Google Books (e.g. computer_NOUN) — strip them.
_POS_RE = re.compile(r"_[A-Z]+$")

# ── Public API ────────────────────────────────────────────────────────────────

def load_cooccurrences(
    target_words: List[str] = None,
    decade_start: int       = DECADE_START,
    decade_end:   int       = DECADE_END,
    min_count:    int       = MIN_COUNT,
    raw_dir:      str       = RAW_DIR,
    processed_dir:str       = PROCESSED_DIR,
    force:        bool      = False,
) -> Dict[str, Dict[int, Dict[str, int]]]:
    """
    Return co-occurrence counts from Google Books 2-grams.

    Parameters
    ----------
    target_words  : words to analyse (defaults to module-level TARGET_WORDS)
    decade_start  : first decade to include (e.g. 1850)
    decade_end    : last year to include   (e.g. 2009)
    min_count     : ignore bigrams whose annual count is below this
    raw_dir       : where to store downloaded .gz files
    processed_dir : where to cache the parsed result
    force         : re-download and re-parse even if a cache exists

    Returns
    -------
    cooc : dict
        ``cooc[word][decade][context_word]`` = total co-occurrence count
    """
    if target_words is None:
        target_words = TARGET_WORDS

    cache_path = os.path.join(processed_dir, "cooccurrences.pkl")
    if not force and os.path.exists(cache_path):
        print(f"[data_loader] Loading cache: {cache_path}")
        with open(cache_path, "rb") as fh:
            return pickle.load(fh)

    # Group words by the ngram-file they appear in
    prefix_to_words: Dict[str, Set[str]] = defaultdict(set)
    for w in target_words:
        prefix_to_words[_file_prefix(w)].add(w)

    # Accumulate co-occurrence counts
    cooc: Dict[str, Dict[int, Dict[str, int]]] = {
        w: defaultdict(lambda: defaultdict(int)) for w in target_words
    }

    for prefix, words in sorted(prefix_to_words.items()):
        gz_path = os.path.join(raw_dir, f"2gram_{prefix}.gz")
        if not os.path.exists(gz_path):
            _download(BASE_URL.format(prefix=prefix), gz_path)
        print(f"[data_loader] Parsing '{gz_path}' for: {sorted(words)}")
        _parse_gz(gz_path, words, cooc, decade_start, decade_end, min_count)

    # Convert inner defaultdicts to plain dicts before pickling
    result = {
        w: {dec: dict(ctx_counts) for dec, ctx_counts in dec_map.items()}
        for w, dec_map in cooc.items()
    }

    os.makedirs(processed_dir, exist_ok=True)
    with open(cache_path, "wb") as fh:
        pickle.dump(result, fh)
    print(f"[data_loader] Saved cache: {cache_path}")
    return result


# ── Internals ─────────────────────────────────────────────────────────────────

def _file_prefix(word: str) -> str:
    """Google Books indexes 2-gram files by first two chars of the first word."""
    return word[:2].lower()


def _to_decade(year: int) -> int:
    return (year // 10) * 10


def _download(url: str, dest: str) -> None:
    """Download *url* to *dest* with a simple progress hook."""
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    print(f"[data_loader] Downloading {url}")

    def _hook(blocks_done, block_size, total_size):
        mb_done = blocks_done * block_size / 1e6
        if total_size > 0:
            pct = min(100, blocks_done * block_size * 100 // total_size)
            print(f"\r  {pct:3d}%  {mb_done:.1f} MB", end="", flush=True)

    urllib.request.urlretrieve(url, dest, reporthook=_hook)
    print()  # newline after progress bar


def _strip_pos(token: str) -> str:
    """Remove POS tag suffix, e.g. 'computer_NOUN' → 'computer'."""
    return _POS_RE.sub("", token)


def _parse_gz(
    gz_path:      str,
    target_words: Set[str],
    cooc:         Dict[str, Dict[int, Dict[str, int]]],
    decade_start: int,
    decade_end:   int,
    min_count:    int,
) -> None:
    """
    Stream through one compressed 2-gram file.

    Each line has the format:
        ngram TAB year TAB match_count TAB volume_count

    We accumulate counts only for bigrams whose first token is a target word.
    """
    with gzip.open(gz_path, "rt", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) != 4:
                continue

            ngram, year_s, count_s, _ = parts

            # Fast pre-filter before parsing integers
            if not any(ngram.startswith(w) for w in target_words):
                continue

            try:
                year  = int(year_s)
                count = int(count_s)
            except ValueError:
                continue

            if count < min_count:
                continue
            if not (decade_start <= year <= decade_end):
                continue

            tokens = ngram.lower().split()
            if len(tokens) != 2:
                continue

            w1 = _strip_pos(tokens[0])
            w2 = _strip_pos(tokens[1])

            if w1 not in target_words:
                continue

            # Skip context words that are purely punctuation or digits
            if not w2.isalpha():
                continue

            decade = _to_decade(year)
            cooc[w1][decade][w2] += count

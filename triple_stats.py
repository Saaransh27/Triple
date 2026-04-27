"""
triple_stats.py — reads test_cases.txt and prints triple statistics.

Usage:
    python triple_stats.py
    python triple_stats.py --file path/to/other_file.txt
    python triple_stats.py --list             # list all distinct triples
    python triple_stats.py --breakdown        # show predicate frequency
"""

import re
import sys
from collections import Counter
from pathlib import Path

# ── config ─────────────────────────────────────────────────────────────────────

DEFAULT_FILE = "test_cases.txt"

# matches:  (User, CLICKS, Dev)  or  (TC_04, IS_IN, Test Studio)
TRIPLE_RE = re.compile(r"^\s*\((.+?),\s*([A-Z_]+)\s*,\s*(.+?)\)\s*$")

# ── parse ───────────────────────────────────────────────────────────────────────

def parse_triples(filepath: Path) -> list[tuple[str, str, str]]:
    """
    Returns list of (subject, predicate, object) for every triple line.
    """
    triples = []
    for line in filepath.read_text(encoding="utf-8").splitlines():
        m = TRIPLE_RE.match(line)
        if m:
            triples.append((m.group(1).strip(), m.group(2).strip(), m.group(3).strip()))
    return triples

# ── stats ───────────────────────────────────────────────────────────────────────

def print_stats(filepath: Path, show_breakdown: bool, list_triples: bool):
    if not filepath.exists():
        print(f"[error] file not found: {filepath}")
        sys.exit(1)

    triples = parse_triples(filepath)

    if not triples:
        print("No triples found in file.")
        sys.exit(0)

    total          = len(triples)
    full_triples   = [(s, p, o) for s, p, o in triples]
    distinct       = set(full_triples)
    counter        = Counter(full_triples)
    repeated       = {t: c for t, c in counter.items() if c > 1}
    pred_counter   = Counter(p for _, p, _ in triples)

    sep = "─" * 52

    print()
    print(sep)
    print(f"  Triple Statistics — {filepath.name}")
    print(sep)
    print(f"  Total triples       : {total}")
    print(f"  Distinct triples    : {len(distinct)}")
    print(f"  Repeated triples    : {len(repeated)}  (appear more than once)")
    print(f"  Distinct predicates : {len(pred_counter)}")
    print(sep)

    if show_breakdown:
        print("  Predicate frequency:")
        for pred, count in pred_counter.most_common():
            print(f"    {pred:<35} ×{count}")
        print(sep)

    if repeated:
        print("  Repeated triples:")
        for (s, p, o), count in counter.most_common(15):
            if count > 1:
                print(f"    ({s}, {p}, {o})")
                print(f"    {'':>4}↳ appears ×{count}")
        print(sep)

    if list_triples:
        print("  All distinct triples:")
        for s, p, o in sorted(distinct, key=lambda t: (t[0].lower(), t[1], t[2].lower())):
            print(f"    ({s}, {p}, {o})")
        print(sep)

    print()

# ── main ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    args = sys.argv[1:]

    filepath       = Path(DEFAULT_FILE)
    show_breakdown = "--breakdown" in args
    list_triples   = "--list" in args

    if "--file" in args:
        idx = args.index("--file")
        try:
            filepath = Path(args[idx + 1])
        except IndexError:
            print("[error] --file requires a path argument")
            sys.exit(1)

    print_stats(filepath, show_breakdown, list_triples)
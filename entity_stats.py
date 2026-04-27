"""
entity_stats.py — reads test_cases.txt and prints entity statistics
extracted from triple lines: (Subject, PREDICATE, Object)

Usage:
    python entity_stats.py
    python entity_stats.py --file path/to/other_file.txt
    python entity_stats.py --list             # list all distinct entities
    python entity_stats.py --breakdown        # show top subjects vs objects
"""

import re
import sys
from collections import Counter
from pathlib import Path

# ── config ─────────────────────────────────────────────────────────────────────

DEFAULT_FILE = "test_cases.txt"

# matches:  (User, CLICKS, Dev)  or  (TC_04, IS_IN, Test Studio)
TRIPLE_RE = re.compile(r"^\s*\((.+?),\s*[A-Z_]+\s*,\s*(.+?)\)\s*$")

# ── parse ───────────────────────────────────────────────────────────────────────

def parse_entities(filepath: Path) -> tuple[list[str], list[str]]:
    """
    Returns (subjects, objects) parsed from every triple line.
    """
    subjects = []
    objects  = []

    for line in filepath.read_text(encoding="utf-8").splitlines():
        m = TRIPLE_RE.match(line)
        if m:
            subjects.append(m.group(1).strip())
            objects.append(m.group(2).strip())

    return subjects, objects

# ── stats ───────────────────────────────────────────────────────────────────────

def print_stats(filepath: Path, show_breakdown: bool, list_entities: bool):
    if not filepath.exists():
        print(f"[error] file not found: {filepath}")
        sys.exit(1)

    subjects, objects = parse_entities(filepath)

    if not subjects:
        print("No triples found in file.")
        sys.exit(0)

    all_entities = subjects + objects
    total        = len(all_entities)
    distinct     = set(all_entities)
    counter      = Counter(all_entities)
    repeated     = {e: c for e, c in counter.items() if c > 1}

    subj_counter = Counter(subjects)
    obj_counter  = Counter(objects)

    sep = "─" * 52

    print()
    print(sep)
    print(f"  Entity Statistics — {filepath.name}")
    print(f"  (from triple subjects + objects)")
    print(sep)
    print(f"  Total triples       : {len(subjects)}")
    print(f"  Total entity refs   : {total}  (subjects + objects combined)")
    print(f"  Distinct entities   : {len(distinct)}")
    print(f"  Repeated entities   : {len(repeated)}  (appear more than once)")
    print(sep)

    if show_breakdown:
        print("  Top subjects (actors):")
        for name, count in subj_counter.most_common(10):
            print(f"    {name:<35} ×{count}")
        print()
        print("  Top objects (targets):")
        for name, count in obj_counter.most_common(10):
            print(f"    {name:<35} ×{count}")
        print(sep)

    if repeated:
        print("  Most frequent entities (any role):")
        for name, count in counter.most_common(15):
            if count > 1:
                print(f"    {name:<35} ×{count}")
        print(sep)

    if list_entities:
        print("  All distinct entities:")
        for name in sorted(distinct, key=str.lower):
            print(f"    {name}")
        print(sep)

    print()

# ── main ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    args = sys.argv[1:]

    filepath       = Path(DEFAULT_FILE)
    show_breakdown = "--breakdown" in args
    list_entities  = "--list" in args

    if "--file" in args:
        idx = args.index("--file")
        try:
            filepath = Path(args[idx + 1])
        except IndexError:
            print("[error] --file requires a path argument")
            sys.exit(1)

    print_stats(filepath, show_breakdown, list_entities)
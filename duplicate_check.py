"""
duplicate_check.py — finds duplicate requirements across test cases.

Two requirements are considered identical if they have the exact same set of
triples AND annotations (order does not matter).

Usage:
    python duplicate_check.py
    python duplicate_check.py --file path/to/other_file.txt
    python duplicate_check.py --show-content     # print the full triples/annotations of each duplicate group
"""

import re
import sys
from collections import defaultdict
from pathlib import Path

# ── config ─────────────────────────────────────────────────────────────────────

DEFAULT_FILE = "test_cases.txt"

TRIPLE_RE     = re.compile(r"^\s*\(.+?,\s*[A-Z_]+\s*,.+?\)\s*$")
ANNOTATION_RE = re.compile(r"^\s*\[.+?::.+?\]\s*$")
TC_RE         = re.compile(r"^\s*──\s*Test Case\s+(.+?)\s*─+")

# ── parse ───────────────────────────────────────────────────────────────────────

def parse_file(filepath: Path) -> list[dict]:
    """
    Returns a list of requirement blocks, each as:
    {
        "test_case" : "TC-001",
        "requirement": "Click the Dev button...",
        "triples"    : ["(User, CLICKS, Dev)", ...],
        "annotations": ["[Dev :: Button]", ...],
        "fingerprint": frozenset of all triples + annotations (normalized)
    }
    """
    blocks = []
    current_tc   = "unknown"
    current_req  = None
    current_triples     = []
    current_annotations = []

    def flush():
        if current_req is not None:
            fp = frozenset(
                line.strip().lower()
                for line in current_triples + current_annotations
            )
            blocks.append({
                "test_case"  : current_tc,
                "requirement": current_req,
                "triples"    : list(current_triples),
                "annotations": list(current_annotations),
                "fingerprint": fp,
            })

    for line in filepath.read_text(encoding="utf-8").splitlines():
        # new test case header
        tc_match = TC_RE.match(line)
        if tc_match:
            flush()
            current_tc          = tc_match.group(1).strip()
            current_req         = None
            current_triples     = []
            current_annotations = []
            continue

        stripped = line.strip()

        if not stripped:
            # blank line — if we were mid-requirement, flush it
            if current_req and (current_triples or current_annotations):
                flush()
                current_req         = None
                current_triples     = []
                current_annotations = []
            continue

        if TRIPLE_RE.match(stripped):
            current_triples.append(stripped)
        elif ANNOTATION_RE.match(stripped):
            current_annotations.append(stripped)
        else:
            # it's a requirement text line
            # if we already have an open requirement, flush it first
            if current_req and (current_triples or current_annotations):
                flush()
                current_triples     = []
                current_annotations = []
            current_req = stripped

    # flush final block
    flush()

    return blocks


# ── duplicate detection ─────────────────────────────────────────────────────────

def find_duplicates(blocks: list[dict]) -> dict:
    """
    Groups blocks by fingerprint. Returns only groups with 2+ members.
    """
    groups = defaultdict(list)
    for block in blocks:
        if block["fingerprint"]:   # skip empty blocks
            groups[block["fingerprint"]].append(block)

    return {fp: members for fp, members in groups.items() if len(members) > 1}


# ── display ─────────────────────────────────────────────────────────────────────

def print_stats(filepath: Path, show_content: bool):
    if not filepath.exists():
        print(f"[error] file not found: {filepath}")
        sys.exit(1)

    blocks     = parse_file(filepath)
    duplicates = find_duplicates(blocks)

    sep  = "─" * 52
    sep2 = "·" * 52

    print()
    print(sep)
    print(f"  Duplicate Requirement Check — {filepath.name}")
    print(sep)
    print(f"  Total requirements parsed : {len(blocks)}")
    print(f"  Duplicate groups found    : {len(duplicates)}")
    affected = sum(len(m) for m in duplicates.values())
    print(f"  Requirements affected     : {affected}")
    print(sep)

    if not duplicates:
        print("  ✓ No duplicate requirements found.")
        print(sep)
        print()
        return

    for i, (_, members) in enumerate(duplicates.items(), 1):
        print(f"  Duplicate group #{i}  —  {len(members)} occurrences")
        print()
        for member in members:
            print(f"    Test Case  : {member['test_case']}")
            print(f"    Requirement: {member['requirement']}")
        if show_content:
            print()
            print(f"    Triples & annotations:")
            for t in members[0]["triples"]:
                print(f"      {t}")
            for a in members[0]["annotations"]:
                print(f"      {a}")
        print()
        if i < len(duplicates):
            print(sep2)
            print()

    print(sep)
    print()


# ── main ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    args = sys.argv[1:]

    filepath     = Path(DEFAULT_FILE)
    show_content = "--show-content" in args

    if "--file" in args:
        idx = args.index("--file")
        try:
            filepath = Path(args[idx + 1])
        except IndexError:
            print("[error] --file requires a path argument")
            sys.exit(1)

    print_stats(filepath, show_content)
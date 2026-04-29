"""
req_extractor.py

Reads any document (PDF, DOCX, TXT, MD),
finds requirements, outputs them as clean sentences
ready to feed into the triple extractor (main.py).

Usage:
    python req_extractor.py --file path/to/document.pdf
    python req_extractor.py --file path/to/document.pdf --out requirements.txt
"""

import argparse
import asyncio
import io
import json
import os
import re
import sys
from pathlib import Path

import httpx
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_URL     = os.getenv(
    "GEMINI_URL",
    "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent"
)

# ── Prompt ─────────────────────────────────────────────────────────────────────

PROMPT = """You are an expert requirements analyst.

Read the document chunk below and extract ONLY functional requirements.

## What counts as a requirement:
- Statements about what a system must, shall, should, can, or will do
- User actions the system supports or enables
- Acceptance criteria or testable behaviours
- Validation rules, constraints, permissions
- Implicit behavioural statements: "Clicking X does Y", "The screen shows Z"

## What is NOT a requirement:
- Document titles, headings, section numbers
- Introduction, background, purpose, scope, objectives
- Stakeholder lists, project history, timelines, dates
- Glossary definitions, assumptions, references
- Sentences with no testable or actionable behaviour

## Rules for output:
- Each requirement must be ONE clear, atomic, self-contained sentence
- If a source sentence contains multiple requirements, split them
- Keep original meaning — do not paraphrase beyond clarifying
- Write in active voice where possible
- If nothing in the chunk is a requirement, return empty list

Return ONLY valid JSON, no markdown:
{{"requirements": ["sentence 1", "sentence 2"]}}

Examples:

Input: "This document outlines the payment module. The system shall process payments via Stripe. Users must receive a confirmation email after payment."
Output: {{"requirements": ["The system shall process payments via Stripe", "Users must receive a confirmation email after payment"]}}

Input: "The project was initiated in Q3 2025 by the product team. Stakeholders include finance and operations."
Output: {{"requirements": []}}

Input: "All mandatory fields must be validated before form submission. The Submit button is disabled until all fields are filled. Invalid email format triggers an inline error."
Output: {{"requirements": ["All mandatory fields must be validated before form submission", "The Submit button is disabled until all fields are filled", "Invalid email format triggers an inline error"]}}

Chunk:
{chunk}"""


GLEANING_PROMPT = """You are an expert requirements analyst.

A first pass extracted requirements from this document chunk but SEVERAL REQUIREMENTS WERE MISSED.

ALREADY EXTRACTED (incomplete list):
{extracted}

ORIGINAL CHUNK:
{chunk}

The above extraction is incomplete. Find the requirements that were missed.

Focus on:
- Trigger-based behaviours ("when X happens, system does Y")
- Conditional behaviours ("if condition, system does Z")
- Automatic behaviours ("system shall automatically...")
- Reactive behaviours tied to external module changes
- Actions where the condition is in a nearby sentence
- Any testable system behaviour not already listed above

Do NOT repeat anything already extracted.
Do NOT include non-requirements (context, definitions, entity relationships, role descriptions).
IMPORTANT: Only extract requirements EXPLICITLY written in the chunk. Do NOT infer, assume, or derive requirements that are not directly stated in the text.

Return ONLY valid JSON, no markdown:
{{"requirements": ["missed req 1", "missed req 2"]}}"""


async def glean_requirements(chunk: str, already_extracted: list) -> list:
    """Second pass — asserts the first pass missed requirements and finds them."""
    if not already_extracted:
        return []

    prompt = GLEANING_PROMPT.format(
        chunk=chunk,
        extracted="\n".join(f"- {r}" for r in already_extracted)
    )
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0, "maxOutputTokens": 4096}
    }
    url = f"{GEMINI_URL}?key={GEMINI_API_KEY}"
    try:
        async with httpx.AsyncClient(timeout=60) as client:
            resp = await client.post(url, json=payload)
        if not resp.is_success:
            return []
        data = resp.json()
        raw  = data["candidates"][0]["content"]["parts"][0]["text"].strip()
        raw  = re.sub(r'```json|```', '', raw).strip()
        try:
            return json.loads(raw).get("requirements", [])
        except json.JSONDecodeError:
            matches = re.findall(r'"([^"]{15,})"', raw)
            return [m for m in matches if m != "requirements"]
    except Exception as e:
        print(f"[warn] Gleaning failed: {e}")
        return []


# ── Text extraction ────────────────────────────────────────────────────────────

def extract_text(file_path: Path) -> str:
    suffix = file_path.suffix.lower()

    if suffix == ".pdf":
        try:
            import pdfplumber
            with pdfplumber.open(file_path) as pdf:
                return "\n".join(p.extract_text() or "" for p in pdf.pages)
        except ImportError:
            print("[error] pdfplumber not installed. Run: pip install pdfplumber")
            sys.exit(1)

    elif suffix == ".docx":
        try:
            import docx
            doc = docx.Document(file_path)
            return "\n".join(p.text for p in doc.paragraphs if p.text.strip())
        except ImportError:
            print("[error] python-docx not installed. Run: pip install python-docx")
            sys.exit(1)

    else:
        return file_path.read_text(encoding="utf-8", errors="replace")


# ── Chunker ────────────────────────────────────────────────────────────────────

def chunk_document(text: str, max_chars: int = 800) -> list[str]:
    # Split on any newline — single or double
    paragraphs = re.split(r'\n+', text)
    chunks, current = [], ""
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        if len(current) + len(para) > max_chars:
            if current:
                chunks.append(current.strip())
            current = para
        else:
            current = current + " " + para if current else para
    if current:
        chunks.append(current.strip())
    return chunks


# ── LLM call ───────────────────────────────────────────────────────────────────

async def extract_requirements_from_chunk(chunk: str) -> list[str]:
    if not GEMINI_API_KEY:
        print("[error] GEMINI_API_KEY not set in .env")
        sys.exit(1)

    prompt  = PROMPT.replace("{chunk}", chunk)
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0, "maxOutputTokens": 8192}
    }
    url = f"{GEMINI_URL}?key={GEMINI_API_KEY}"

    try:
        async with httpx.AsyncClient(timeout=60) as client:
            resp = await client.post(url, json=payload)
        if not resp.is_success:
            print(f"[warn] Gemini error on chunk: {resp.status_code}")
            return []
        data = resp.json()
        raw  = data["candidates"][0]["content"]["parts"][0]["text"].strip()
        raw  = re.sub(r'```json|```', '', raw).strip()

        # Try full parse first
        try:
            return json.loads(raw).get("requirements", [])
        except json.JSONDecodeError:
            # Truncated — recover complete strings
            matches = re.findall(r'"([^"]{10,})"', raw)
            reqs = [m for m in matches if m != "requirements" and len(m) > 15]
            if reqs:
                print(f"[warn] Partial JSON recovered — {len(reqs)} requirements")
            return reqs

    except Exception as e:
        print(f"[warn] Failed chunk: {e}")
        return []


# ── Main ───────────────────────────────────────────────────────────────────────

async def run(file_path: Path, out_path: Path | None):
    print(f"Reading: {file_path}")
    text   = extract_text(file_path)
    chunks = chunk_document(text)
    print(f"Chunks : {len(chunks)}")

    all_requirements = []

    for i, chunk in enumerate(chunks, 1):
        print(f"Chunk {i}/{len(chunks)} — pass 1...", end="\r")
        reqs = await extract_requirements_from_chunk(chunk)

        print(f"Chunk {i}/{len(chunks)} — gleaning... ", end="\r")
        gleaned = await glean_requirements(chunk, reqs)

        combined = reqs + gleaned
        print(f"Chunk {i}/{len(chunks)}: {len(reqs)} found + {len(gleaned)} gleaned = {len(combined)} total")
        all_requirements.extend(combined)

    print(f"\nFound  : {len(all_requirements)} requirements")

    # Output
    output = "\n".join(all_requirements)

    if out_path:
        out_path.write_text(output, encoding="utf-8")
        print(f"Saved  : {out_path}")
    else:
        print("\n" + "─" * 52)
        print(output)
        print("─" * 52)

    return all_requirements


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract requirements from any document")
    parser.add_argument("--file", required=True, help="Path to PDF, DOCX, TXT, or MD file")
    parser.add_argument("--out",  default=None,  help="Output .txt file (default: print to terminal)")
    args = parser.parse_args()

    file_path = Path(args.file)
    if not file_path.exists():
        print(f"[error] File not found: {file_path}")
        sys.exit(1)

    out_path = Path(args.out) if args.out else None
    asyncio.run(run(file_path, out_path))
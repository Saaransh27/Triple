from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import httpx
import os
import re
import json
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
import openpyxl
import io

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_URL = os.getenv(
    "GEMINI_URL",
    "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent"
)

TEST_CASES_FILE = Path("test_cases.txt")
HISTORY_FILE    = Path("history.json")

SYSTEM_PROMPT = """You are a precise requirements triple extractor.

Your job: convert a requirement sentence into knowledge graph triples.

Rules — follow these exactly:
1. Extract ONLY what is explicitly stated. Never infer or assume.
2. One triple per stated fact. No more.
3. Triple format: (Subject, PREDICATE, Object)
4. Subject is always the actor performing the action (usually "User" or a system component).
5. Predicate is always an UPPER_SNAKE_CASE verb phrase.
6. Object is the target of the action.
7. If a sentence has two facts, produce two triples. If one fact, one triple.
8. Never add triples for implied context.
9. Never split one fact into multiple triples.

Location rule — always extract location as a second triple:
- If a sentence says "X on the Y page" or "X in the Y modal" or "X on the Y screen", produce two triples:
  Triple 1: the action on X
  Triple 2: (X, IS_ON, Y) or (X, IS_IN, Y)
- Example: "Locate the Settings icon on the Test Studio page"
  Triple 1: (User, LOCATES, Settings)
  Triple 2: (Settings, IS_ON, Test Studio)
  Annotations: [Settings :: Icon] [Test Studio :: Page]

URL rules — critical, follow exactly:
- A URL is always treated as a single atomic string. Never split or truncate it at any character.
- URLs contain colons, slashes, dots, and port numbers — all are part of the URL, keep them intact.
- The colon in a URL (e.g. :8080, :443, :3128, :1080) is PART of the URL, NOT a triple separator.
- When extracting a triple containing a URL, the URL is always the complete object.
- When a sentence says "enter X into Y" — produce: (User, ENTERS, X) and (User, ENTERS_INTO, Y)
- When a sentence says "field contains X" or "verify X contains Y" — produce: (X, CONTAINS, Y)
- URL examples — these are correct complete triples:
  (User, ENTERS, http://proxy.example.com:8080)
  (Proxy URL, CONTAINS, https://secureproxy.example.com:443)
  (Proxy URL, CONTAINS, socks5://proxyhost:1080)
  (User, NAVIGATES_TO, https://automate.nogrunt.com/new/test-studio)

Empty object rule:
- If a requirement states an action with no explicit object, use True as the object.
- Example: "Verify the modal closes" → (Modal, CLOSES, True)
- Example: "Settings are saved" → (Settings, ARE_SAVED, True)

Entity naming rules:
- Strip UI element type words from entity names: Icon, Button, Modal, Page, Dropdown, Input, Field, Link, Tab, Toggle, Switch, Checkbox, Menu
- Capture the stripped type word as an annotation: [EntityName :: Type]
- Allowed type values only: Icon, Button, Modal, Page, Dropdown, Input, Toggle, Switch, Checkbox, Menu, Link, Tab
- Use the same type label consistently for the same entity:
    Settings    → always [Settings :: Icon]
    Save        → always [Save :: Button]
    Description → always [Description :: Input]
    Proxy URL   → always [Proxy URL :: Input]
    Run Once    → always [Run Once :: Toggle]
- Exception: if removing the type word changes the meaning, keep it ("Test Suite" stays as is)

Output format:
- First all triples, one per line
- Then all entity annotations, one per line
- No explanation. No markdown. No extra text.

Examples:

Input: "Locate the Settings icon on the Test Studio page"
Output:
(User, LOCATES, Settings)
(Settings, IS_ON, Test Studio)
[Settings :: Icon]
[Test Studio :: Page]

Input: "Navigate to https://automate.nogrunt.com/new/test-studio"
Output:
(User, NAVIGATES_TO, https://automate.nogrunt.com/new/test-studio)

Input: "Locate and click the Settings icon to open the Test Scenario Settings modal"
Output:
(User, CLICKS, Settings)
(Settings, OPENS, Test Scenario Settings)
[Settings :: Icon]
[Test Scenario Settings :: Modal]

Input: "Enter 'http://proxy.example.com:8080' into the Proxy URL input field"
Output:
(User, ENTERS, http://proxy.example.com:8080)
(User, ENTERS_INTO, Proxy URL)
[Proxy URL :: Input]

Input: "Verify the Proxy URL field contains 'https://secureproxy.example.com:443'"
Output:
(Proxy URL, CONTAINS, https://secureproxy.example.com:443)

Input: "Verify the modal closes and settings are saved"
Output:
(Modal, CLOSES, True)
(Settings, ARE_SAVED, True)

Input: "Click the Dev button to select the Dev environment"
Output:
(User, CLICKS, Dev)
(Dev, SELECTS, Dev Environment)
[Dev :: Button]

Input: "Disable the Run Once toggle switch"
Output:
(User, DISABLES, Run Once)
[Run Once :: Toggle]"""


# ── Models ──────────────────────────────────────────────────────────────────────

class RequirementRequest(BaseModel):
    requirement: str


class TripleResponse(BaseModel):
    requirement: str
    triples: list[str]
    annotations: list[str]


# ── History helpers ─────────────────────────────────────────────────────────────

def load_history() -> list:
    if not HISTORY_FILE.exists():
        return []
    try:
        return json.loads(HISTORY_FILE.read_text(encoding="utf-8"))
    except Exception:
        return []


def save_history(history: list):
    HISTORY_FILE.write_text(json.dumps(history, indent=2, ensure_ascii=False), encoding="utf-8")


def next_run_number(history: list) -> int:
    if not history:
        return 1
    return max(h.get("run_number", 0) for h in history) + 1


# ── Core extraction ─────────────────────────────────────────────────────────────

async def extract_triples(requirement: str) -> dict:
    if not GEMINI_API_KEY:
        raise HTTPException(status_code=500, detail="GEMINI_API_KEY not configured")

    payload = {
        "contents": [
            {
                "parts": [{"text": f"{SYSTEM_PROMPT}\n\nInput: {requirement}"}]
            }
        ],
        "generationConfig": {
            "temperature": 0,
            "maxOutputTokens": 512,
        }
    }

    url = f"{GEMINI_URL}?key={GEMINI_API_KEY}"

    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(url, json=payload)

    if not resp.is_success:
        raise HTTPException(status_code=500, detail=f"Gemini API error: {resp.text}")

    data = resp.json()

    try:
        content = data["candidates"][0]["content"]["parts"][0]["text"].strip()
    except (KeyError, IndexError):
        raise HTTPException(status_code=500, detail=f"Unexpected response: {data}")

    triples = []
    annotations = []

    for line in content.splitlines():
        line = line.strip()
        if line.startswith("(") and line.endswith(")"):
            triples.append(line)
        elif line.startswith("[") and line.endswith("]") and "::" in line:
            annotations.append(line)

    if not triples:
        raise HTTPException(status_code=500, detail=f"No triples found: {content}")

    return {"triples": triples, "annotations": annotations}


# ── Excel parsing ───────────────────────────────────────────────────────────────

def parse_excel(file_bytes: bytes) -> list[dict]:
    wb = openpyxl.load_workbook(filename=io.BytesIO(file_bytes), read_only=True, data_only=True)
    ws = wb.active

    rows = list(ws.iter_rows(values_only=True))
    if not rows:
        raise HTTPException(status_code=400, detail="Excel file is empty")

    header = [str(c).strip().lower() if c else "" for c in rows[0]]

    title_col = next((i for i, h in enumerate(header) if "test case title" in h or "title" in h), None)
    steps_col = next((i for i, h in enumerate(header) if "execution step" in h or "step" in h), None)

    if title_col is None:
        raise HTTPException(status_code=400, detail="Could not find 'Test Case Title' column in Excel")
    if steps_col is None:
        raise HTTPException(status_code=400, detail="Could not find 'Execution Steps' column in Excel")

    test_cases = []
    for row in rows[1:]:
        title     = str(row[title_col]).strip() if row[title_col] else ""
        steps_raw = str(row[steps_col]).strip() if row[steps_col] else ""

        if not title or not steps_raw or title == "None" or steps_raw == "None":
            continue

        steps = re.split(r'\n|(?<!\d)(?=[0-9]+[\.\)])\s*|•|·|◦|‣|▪|■', steps_raw)
        steps = [re.sub(r'^[\d]+[\.\)]\s*', '', s).strip() for s in steps]
        steps = [s for s in steps if len(s) > 3]

        if steps:
            test_cases.append({"title": title, "steps": steps})

    wb.close()
    return test_cases


# ── Routes ──────────────────────────────────────────────────────────────────────

@app.post("/extract", response_model=TripleResponse)
async def extract(req: RequirementRequest):
    if not req.requirement.strip():
        raise HTTPException(status_code=400, detail="Requirement cannot be empty")

    result = await extract_triples(req.requirement.strip())
    return TripleResponse(
        requirement=req.requirement,
        triples=result["triples"],
        annotations=result["annotations"]
    )


@app.post("/extract-bulk")
async def extract_bulk(body: dict):
    requirements = body.get("requirements", [])
    if not requirements:
        raise HTTPException(status_code=400, detail="No requirements provided")

    results = []
    for req in requirements:
        req = req.strip()
        if not req:
            continue
        try:
            result = await extract_triples(req)
            results.append({
                "requirement": req,
                "triples"    : result["triples"],
                "annotations": result["annotations"],
                "error"      : None
            })
        except Exception as e:
            results.append({
                "requirement": req,
                "triples"    : [],
                "annotations": [],
                "error"      : str(e)
            })

    return {"results": results}


@app.post("/upload-excel")
async def upload_excel(file: UploadFile = File(...)):
    if not file.filename.endswith((".xlsx", ".xls")):
        raise HTTPException(status_code=400, detail="Only .xlsx files are accepted")

    file_bytes = await file.read()
    test_cases = parse_excel(file_bytes)

    if not test_cases:
        raise HTTPException(status_code=400, detail="No valid test cases found in Excel")

    all_results = []

    for tc in test_cases:
        tc_results = []
        for step in tc["steps"]:
            try:
                result = await extract_triples(step)
                tc_results.append({
                    "requirement": step,
                    "triples"    : result["triples"],
                    "annotations": result["annotations"],
                    "error"      : None
                })
            except Exception as e:
                tc_results.append({
                    "requirement": step,
                    "triples"    : [],
                    "annotations": [],
                    "error"      : str(e)
                })

        all_results.append({
            "test_case": tc["title"],
            "results"  : tc_results
        })

    return {"test_cases": all_results}


@app.post("/store-test-case")
async def store_test_case(body: dict):
    tc_number = body.get("test_case_number", "").strip()
    results   = body.get("results", [])
    mode      = body.get("mode", "single")
    source    = body.get("source", "")

    if not tc_number:
        raise HTTPException(status_code=400, detail="test_case_number is required")
    if not results:
        raise HTTPException(status_code=400, detail="No results to save")

    # ── Write to test_cases.txt ──────────────────────────────────────────────
    width  = 52
    header = f"── Test Case {tc_number} " + "─" * max(0, width - len(tc_number) - 14)
    lines  = [header, ""]

    for item in results:
        lines.append(item["requirement"])
        for t in item.get("triples", []):
            lines.append(t)
        for a in item.get("annotations", []):
            lines.append(a)
        lines.append("")

    block = "\n".join(lines) + "\n"

    with TEST_CASES_FILE.open("a", encoding="utf-8") as f:
        f.write(block)

    # ── Write to history.json ────────────────────────────────────────────────
    history    = load_history()
    run_number = next_run_number(history)

    total_triples     = sum(len(r.get("triples", [])) for r in results)
    total_annotations = sum(len(r.get("annotations", [])) for r in results)

    run_entry = {
        "run_number"        : run_number,
        "test_case_number"  : tc_number,
        "timestamp"         : datetime.utcnow().isoformat() + "Z",
        "mode"              : mode,
        "source"            : source,
        "requirements_count": len(results),
        "triples_count"     : total_triples,
        "annotations_count" : total_annotations,
        "results"           : results,
    }

    history.append(run_entry)
    save_history(history)

    return {
        "file"      : str(TEST_CASES_FILE),
        "run_number": run_number,
        "written"   : len(lines),
    }


@app.get("/history")
async def get_history():
    history = load_history()
    summary = [
        {
            "run_number"        : h["run_number"],
            "test_case_number"  : h["test_case_number"],
            "timestamp"         : h["timestamp"],
            "mode"              : h.get("mode", ""),
            "source"            : h.get("source", ""),
            "requirements_count": h["requirements_count"],
            "triples_count"     : h["triples_count"],
        }
        for h in history
    ]
    return {"runs": list(reversed(summary))}


@app.get("/history/{run_number}")
async def get_run(run_number: int):
    history = load_history()
    run = next((h for h in history if h["run_number"] == run_number), None)
    if not run:
        raise HTTPException(status_code=404, detail=f"Run {run_number} not found")
    return run


@app.get("/download-test-cases")
async def download_test_cases():
    if not TEST_CASES_FILE.exists():
        raise HTTPException(status_code=404, detail="No test cases saved yet")
    return FileResponse(
        path=TEST_CASES_FILE,
        filename="test_cases.txt",
        media_type="text/plain"
    )


app.mount("/", StaticFiles(directory="static", html=True), name="static")
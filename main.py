from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import httpx, os, re, json, io, openpyxl
from pathlib import Path
from datetime import datetime

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

GEMINI_API_KEY  = os.getenv("GEMINI_API_KEY", "")
GEMINI_URL      = os.getenv("GEMINI_URL", "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent")
TEST_CASES_FILE = Path("test_cases.txt")
HISTORY_FILE    = Path("history.json")
ELEM_DICT_FILE  = Path("element_dict.json")

# ── Prompts ──────────────────────────────────────────────────────────────────────

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

Navigation rule — when a click leads to a page change:
- If a sentence says "click X to navigate to Y" or "click X link to go to Y page", produce two triples:
  Triple 1: (User, CLICKS, X)
  Triple 2: (User, NAVIGATES_TO, Y)
  Annotations: [X :: Link] [Y :: Page]
- Example: "Click on the Test Suite link to navigate to the Test Suite page"
  (User, CLICKS, Test Suite)
  (User, NAVIGATES_TO, Test Suite)
  [Test Suite :: Link]

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

Input: "Click on the Test Suite link to navigate to the Test Suite page"
Output:
(User, CLICKS, Test Suite)
(User, NAVIGATES_TO, Test Suite)
[Test Suite :: Link]

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

Input: "Select 'nogrunt' from the Product dropdown on the Test Suite page"
Output:
(User, SELECTS, nogrunt)
(nogrunt, IS_FROM, Product)
(Product, IS_ON, Test Suite)
[Product :: Dropdown]
[Test Suite :: Page]

Input: "Click the SAVE button in the Add Test Case modal"
Output:
(User, CLICKS, SAVE)
(SAVE, IS_IN, Add Test Case)
[SAVE :: Button]
[Add Test Case :: Modal]

Input: "Disable the Run Once toggle switch"
Output:
(User, DISABLES, Run Once)
[Run Once :: Toggle]"""


ENRICHMENT_PROMPT = """You are a UI element analyst reading a software test requirement.

Given the requirement and a list of UI element names, extract for each element:
1. location  — where on screen (e.g. "Test Scenario Settings Modal"). If unclear, write "Unknown".
2. description — one sentence on what this element does. Do not infer beyond what is stated.

Return ONLY valid JSON, no markdown:
{{"elements": [{{"name": "Settings", "location": "Test Studio page", "description": "Opens the Test Scenario Settings modal when clicked"}}]}}

Requirement: {requirement}
Elements: {entities}"""


# ── Models ───────────────────────────────────────────────────────────────────────

class RequirementRequest(BaseModel):
    requirement: str

class TripleResponse(BaseModel):
    requirement: str
    triples: list[str]
    annotations: list[str]
    current_page: str = "Unknown"


# ── File helpers ─────────────────────────────────────────────────────────────────

def load_history() -> list:
    try:
        return json.loads(HISTORY_FILE.read_text(encoding="utf-8")) if HISTORY_FILE.exists() else []
    except Exception:
        return []

def save_history(h: list):
    HISTORY_FILE.write_text(json.dumps(h, indent=2, ensure_ascii=False), encoding="utf-8")

def next_run_number(history: list) -> int:
    return max((h.get("run_number", 0) for h in history), default=0) + 1

def load_elem_dict() -> dict:
    try:
        return json.loads(ELEM_DICT_FILE.read_text(encoding="utf-8")) if ELEM_DICT_FILE.exists() else {}
    except Exception:
        return {}

def save_elem_dict(data: dict):
    ELEM_DICT_FILE.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

def elem_key(tc: str, name: str) -> str:
    return f"{tc.strip()} | {name.strip()}"

def write_enrichment_to_dict(tc_title: str, enrichment: dict, annot_types: dict):
    d = load_elem_dict()
    for name_lower, info in enrichment.items():
        name = info.get("name", name_lower)
        key  = elem_key(tc_title, name)
        typ, location, desc = annot_types.get(name_lower, ""), info.get("location", ""), info.get("desc", "")
        if key not in d:
            d[key] = {"name": name, "type": typ, "test_case": tc_title, "location": location, "desc": desc}
        else:
            if not d[key].get("type")     and typ:      d[key]["type"]     = typ
            if not d[key].get("location") and location: d[key]["location"] = location
            if not d[key].get("desc")     and desc:     d[key]["desc"]     = desc
    save_elem_dict(d)


# ── LLM helpers ──────────────────────────────────────────────────────────────────

async def _gemini(prompt: str, max_tokens: int = 512) -> str:
    if not GEMINI_API_KEY:
        raise HTTPException(status_code=500, detail="GEMINI_API_KEY not configured")
    payload = {"contents": [{"parts": [{"text": prompt}]}],
               "generationConfig": {"temperature": 0, "maxOutputTokens": max_tokens}}
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(f"{GEMINI_URL}?key={GEMINI_API_KEY}", json=payload)
    if not resp.is_success:
        raise HTTPException(status_code=500, detail=f"Gemini error: {resp.text}")
    try:
        return resp.json()["candidates"][0]["content"]["parts"][0]["text"].strip()
    except (KeyError, IndexError):
        raise HTTPException(status_code=500, detail=f"Unexpected response: {resp.text}")


async def extract_triples(requirement: str) -> dict:
    # Clean up trailing " page" or " page." after a URL — confuses the extractor
    cleaned = re.sub(r'(https?://\S+)\s+page\.?', r'\1', requirement, flags=re.IGNORECASE).strip()
    content = await _gemini(f"{SYSTEM_PROMPT}\n\nInput: {cleaned}")
    triples, annotations = [], []
    for line in content.splitlines():
        line = line.strip()
        if line.startswith("(") and line.endswith(")"):
            triples.append(line)
        elif line.startswith("[") and line.endswith("]") and "::" in line:
            annotations.append(line)
    if not triples:
        raise HTTPException(status_code=500, detail=f"No triples found: {content}")
    return {"triples": triples, "annotations": annotations}


async def enrich_entities(requirement: str, entities: list) -> dict:
    if not entities:
        return {}
    try:
        raw = await _gemini(
            ENRICHMENT_PROMPT.format(requirement=requirement, entities=", ".join(entities)),
            max_tokens=1024
        )
        raw = re.sub(r'```json\s*|```', '', raw).strip()
        if not raw.startswith('{'):
            return {}
        parsed = json.loads(raw)
        if not isinstance(parsed.get("elements"), list):
            return {}
        return {
            str(e.get("name", "")).strip().lower(): {
                "name"    : str(e.get("name", "")).strip(),
                "location": str(e.get("location", "")).strip(),
                "desc"    : str(e.get("description", e.get("desc", ""))).strip(),
            }
            for e in parsed["elements"] if isinstance(e, dict) and e.get("name")
        }
    except Exception:
        return {}


def entities_from_result(triples: list, annotations: list) -> list:
    skip  = {"user", "system", "true", "false"}
    found = set()
    for a in annotations:
        m = re.match(r'\[(.+?)\s*::', a.strip())
        if m:
            name = m.group(1).strip()
            if name.lower() not in skip:
                found.add(name)
    for t in triples:
        parts = [p.strip() for p in t.strip().strip("()").split(",")]
        if len(parts) >= 3:
            obj = ",".join(parts[2:]).strip()
            if obj.lower() not in skip and not re.match(r'^https?://', obj) and len(obj) > 1:
                found.add(obj)
    return sorted(found)


# ── Page context inference ───────────────────────────────────────────────────────

# Patterns that signal a page change in the requirement text
# Each tuple: (regex_pattern, group_index_for_page_name)
_PAGE_PATTERNS = [
    # navigate to X page / navigate to X
    (re.compile(r'navigate to (?:the )?(.+?)(?:\s+page)[.,\s]', re.IGNORECASE), (1,)),
    (re.compile(r'navigate to (?:the )?(.+?)(?:\s+page)$', re.IGNORECASE), (1,)),
    (re.compile(r'navigate to (?:the )?([A-Z][\w\s]+?)$', re.IGNORECASE), (1,)),
    # go to X page / go to X
    (re.compile(r'go to (?:the )?(.+?)(?:\s+page)[.,\s]', re.IGNORECASE), (1,)),
    (re.compile(r'go to (?:the )?(.+?)(?:\s+page)$', re.IGNORECASE), (1,)),
    (re.compile(r'go to (?:the )?([A-Z][\w\s]+?)$', re.IGNORECASE), (1,)),
    # open / reopen / to open X modal/page
    (re.compile(r'(?:to open|reopen|open) (?:the )?(.+?)(?:\s+(?:page|modal))[.,\s]', re.IGNORECASE), (1,)),
    (re.compile(r'(?:to open|reopen|open) (?:the )?(.+?)(?:\s+(?:page|modal))$', re.IGNORECASE), (1,)),
    # redirects to X page
    (re.compile(r'redirects? to (?:the )?(.+?)(?:\s+page)[.,\s]', re.IGNORECASE), (1,)),
    (re.compile(r'redirects? to (?:the )?(.+?)(?:\s+page)$', re.IGNORECASE), (1,)),
    # on the X page — requires capital start to avoid matching "the"
    (re.compile(r'on (?:the )?([A-Z][A-Za-z]+(?:\s[A-Za-z]+)*) page[.,\s]'), (1,)),
    (re.compile(r'on (?:the )?([A-Z][A-Za-z]+(?:\s[A-Za-z]+)*) page$'), (1,)),
    # in the X modal — requires capital start
    (re.compile(r'in (?:the )?([A-Z][A-Za-z]+(?:\s[A-Za-z]+)*) modal[.,\s]'), (1,)),
    (re.compile(r'in (?:the )?([A-Z][A-Za-z]+(?:\s[A-Za-z]+)*) modal$'), (1,)),
]

_URL_PATTERN = re.compile(r'https?://[^\s]+')

_GENERIC = {"modal", "page", "screen", "dialog", "popup", "window", "the modal", "the page"}


def _extract_page_from_text(text: str):
    """
    Read the requirement text and return the page name if a page change is detected.
    Returns None if no page change is found.
    """
    # URL navigation — derive page from last path segment
    url_match = re.search(r'(navigate to|go to)\s+(https?://[^\s]+)', text, re.IGNORECASE)
    if url_match:
        url  = url_match.group(2).rstrip("/")
        slug = url.split("/")[-1].replace("-", " ").replace("_", " ").title()
        return slug or url

    for pattern, groups in _PAGE_PATTERNS:
        m = pattern.search(text)
        if m:
            for g in groups:
                val = m.group(g)
                if not val:
                    continue
                val = val.strip()
                # Must be at least 2 chars, not a generic word, not just articles/pronouns
                skip = _GENERIC | {"the", "a", "an", "this", "that", "its", "my"}
                if val.lower() not in skip and len(val) >= 2:
                    return val

    return None


def infer_page_context(results: list) -> list:
    """
    Walk steps in sequence. Page change takes effect from the NEXT step.
    The step that navigates to a page shows the PREVIOUS page.
    """
    current_page = "Unknown"

    for item in results:
        # 1. Assign current page to THIS step first
        item["current_page"] = current_page

        if item.get("error"):
            continue

        # 2. Detect page change from this step's text
        detected = _extract_page_from_text(item.get("requirement", ""))

        # 3. Update current_page for the NEXT step
        if detected:
            current_page = detected

    return results


# ── Excel parser ─────────────────────────────────────────────────────────────────

def parse_excel(file_bytes: bytes) -> list[dict]:
    wb   = openpyxl.load_workbook(filename=io.BytesIO(file_bytes), read_only=True, data_only=True)
    ws   = wb.active
    rows = list(ws.iter_rows(values_only=True))
    if not rows:
        raise HTTPException(status_code=400, detail="Excel file is empty")

    header    = [str(c).strip().lower() if c else "" for c in rows[0]]
    title_col = next((i for i, h in enumerate(header) if "test case title" in h or "title" in h), None)
    steps_col = next((i for i, h in enumerate(header) if "execution step" in h or "step" in h), None)

    if title_col is None: raise HTTPException(status_code=400, detail="Could not find 'Test Case Title' column")
    if steps_col is None: raise HTTPException(status_code=400, detail="Could not find 'Execution Steps' column")

    # Action verbs that start a new test step
    step_verb_re = re.compile(
        r'^(Navigate|Click|Locate|Verify|Select|Enter|Open|Close|Ensure|Check|'
        r'Find|Go|Drag|Scroll|Type|Clear|Enable|Disable|Toggle|Expand|Collapse|'
        r'Upload|Download|Submit|Cancel|Confirm|Reject|Add|Remove|Delete|Create|'
        r'Save|Load|Refresh|Reload|Reopen|Press|Tap|Hover|Focus|Blur|Wait)\b', re.IGNORECASE
    )
    numbered_re = re.compile(r'^[0-9]+[\.)]\ [A-Za-z]')

    def merge_step_lines(steps_raw: str) -> list:
        # Split only on bullet chars and "digit.) Letter" patterns — not bare numbers
        raw_steps = re.split(r'(?<!\d)(?=[0-9]+[\.)]\ +[A-Za-z])|[•·◦‣▪■]', steps_raw)
        joined = []
        for chunk in raw_steps:
            merged = []
            for line in chunk.split('\n'):
                line = line.strip()
                if not line:
                    continue
                if not merged:
                    merged.append(line)
                elif step_verb_re.match(line) or numbered_re.match(line) or re.match(r"^In the ['\"\ A-Z]", line):
                    merged.append(line)
                else:
                    merged[-1] += ' ' + line   # fragment — join to previous
            joined.extend(merged)
        return [re.sub(r'^[\d]+[\.\ )]\s*', '', s).strip() for s in joined if len(s.strip()) > 10]

    test_cases = []
    for row in rows[1:]:
        title     = str(row[title_col]).strip() if row[title_col] else ""
        steps_raw = str(row[steps_col]).strip() if row[steps_col] else ""
        if not title or not steps_raw or title == "None" or steps_raw == "None":
            continue
        steps = merge_step_lines(steps_raw)
        if steps:
            test_cases.append({"title": title, "steps": steps})

    wb.close()
    return test_cases


# ── Routes ───────────────────────────────────────────────────────────────────────

@app.post("/extract", response_model=TripleResponse)
async def extract(req: RequirementRequest):
    if not req.requirement.strip():
        raise HTTPException(status_code=400, detail="Requirement cannot be empty")
    result   = await extract_triples(req.requirement.strip())
    enriched = infer_page_context([{"requirement": req.requirement, **result}])
    return TripleResponse(requirement=req.requirement, **result,
                          current_page=enriched[0].get("current_page", "Unknown"))


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
            results.append({"requirement": req, **result, "error": None})
        except Exception as e:
            results.append({"requirement": req, "triples": [], "annotations": [], "error": str(e)})
    return {"results": infer_page_context(results)}


@app.post("/upload-excel")
async def upload_excel(file: UploadFile = File(...)):
    if not file.filename.endswith((".xlsx", ".xls")):
        raise HTTPException(status_code=400, detail="Only .xlsx files are accepted")
    test_cases = parse_excel(await file.read())
    if not test_cases:
        raise HTTPException(status_code=400, detail="No valid test cases found in Excel")
    all_results = []
    for tc in test_cases:
        tc_results = []
        for step in tc["steps"]:
            # Skip fragments — too short or no verb/action word
            if len(step.strip()) < 10 or not re.search(r'[a-zA-Z]{3,}', step):
                tc_results.append({"requirement": step, "triples": [], "annotations": [],
                                   "error": "Skipped — fragment or incomplete step"})
                continue
            try:
                result = await extract_triples(step)
                tc_results.append({"requirement": step, **result, "error": None})
            except Exception as e:
                tc_results.append({"requirement": step, "triples": [], "annotations": [], "error": str(e)})
        all_results.append({"test_case": tc["title"], "results": infer_page_context(tc_results)})
    return {"test_cases": all_results}


@app.post("/store-test-case")
async def store_test_case(body: dict):
    run_label  = body.get("test_case_number", "").strip()
    mode, source = body.get("mode", "single"), body.get("source", "")
    test_cases, flat = body.get("test_cases"), body.get("results", [])

    if not run_label:
        raise HTTPException(status_code=400, detail="test_case_number is required")

    groups = test_cases or ([{"test_case": run_label, "results": flat}] if flat else None)
    if not groups:
        raise HTTPException(status_code=400, detail="No results to save")

    divider = "=" * 52
    lines   = [divider, f"  Run: {run_label}  |  {datetime.utcnow().strftime('%Y-%m-%d %H:%M')} UTC", divider, ""]
    total_reqs = total_triples = total_annots = 0

    for group in groups:
        tc_title       = group.get("test_case", "Unknown")
        results        = infer_page_context(group.get("results", []))
        tc_annot_types = {}

        lines.extend([f"-- {tc_title} " + "-" * max(0, 50 - len(tc_title) - 4), ""])

        for item in results:
            if item.get("error"):
                lines.extend([f"[ERROR] {item['requirement']}: {item['error']}", ""])
                continue
            lines.append(f"PAGE: {item.get('current_page', 'Unknown')}")
            lines.append(item["requirement"])
            for t in item.get("triples", []):
                lines.append(t); total_triples += 1
            for a in item.get("annotations", []):
                lines.append(a); total_annots += 1
                m = re.match(r'\[(.+?)\s*::\s*(.+?)\]', a.strip())
                if m:
                    tc_annot_types[m.group(1).strip().lower()] = m.group(2).strip()
            lines.append("")
            total_reqs += 1

        all_entities = entities_from_result(
            [t for item in results for t in item.get("triples", [])],
            [a for item in results for a in item.get("annotations", [])]
        )
        if all_entities:
            combined_req = " | ".join(
                item["requirement"] for item in results if item.get("requirement") and not item.get("error")
            )[:2000]
            tc_enrichment = await enrich_entities(combined_req, all_entities)
            if tc_enrichment:
                write_enrichment_to_dict(tc_title, tc_enrichment, tc_annot_types)

        lines.append("")

    with TEST_CASES_FILE.open("a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    history    = load_history()
    run_number = next_run_number(history)
    history.append({
        "run_number": run_number, "test_case_number": run_label,
        "timestamp" : datetime.utcnow().isoformat() + "Z",
        "mode": mode, "source": source,
        "test_cases_count": len(groups), "requirements_count": total_reqs,
        "triples_count": total_triples, "annotations_count": total_annots,
        "test_cases": groups,
    })
    save_history(history)
    return {"file": str(TEST_CASES_FILE), "run_number": run_number,
            "test_cases_count": len(groups), "triples_count": total_triples}


@app.get("/history")
async def get_history():
    return {"runs": list(reversed([{
        "run_number": h["run_number"], "test_case_number": h["test_case_number"],
        "timestamp" : h["timestamp"], "mode": h.get("mode", ""), "source": h.get("source", ""),
        "requirements_count": h["requirements_count"], "triples_count": h["triples_count"],
    } for h in load_history()]))}


@app.get("/history/{run_number}")
async def get_run(run_number: int):
    run = next((h for h in load_history() if h["run_number"] == run_number), None)
    if not run:
        raise HTTPException(status_code=404, detail=f"Run {run_number} not found")
    return run


@app.get("/download-test-cases")
async def download_test_cases():
    if not TEST_CASES_FILE.exists():
        raise HTTPException(status_code=404, detail="No test cases saved yet")
    return FileResponse(path=TEST_CASES_FILE, filename="test_cases.txt", media_type="text/plain")


@app.get("/api/elem-dict")
async def get_elem_dict():
    return load_elem_dict()


@app.get("/api/elem-dict/download")
async def download_elem_dict():
    if not ELEM_DICT_FILE.exists():
        raise HTTPException(status_code=404, detail="No element dictionary yet")
    return FileResponse(path=ELEM_DICT_FILE, filename="element_dict.json", media_type="application/json")


@app.post("/api/elem-dict/{key:path}")
async def update_elem_dict_entry(key: str, body: dict):
    d = load_elem_dict()
    if key not in d:
        raise HTTPException(status_code=404, detail="Key not found")
    for field in ("type", "location", "desc"):
        if field in body:
            d[key][field] = body[field].strip()
    save_elem_dict(d)
    return {"ok": True}


app.mount("/", StaticFiles(directory="static", html=True), name="static")
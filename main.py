"""KAAM backend. Run with: .venv\\Scripts\\python.exe -m uvicorn main:app"""

import io
import json
import os
import time
import wave
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from sarvamai import SarvamAI
from sarvamai.core.api_error import ApiError

from agent import Agent
from operations import Operations
from store import DataStore, now_ist


ROOT = Path(__file__).parent
STATIC_DIR = ROOT / "static"
TRACE_DIR = ROOT / "traces"
STATIC_DIR.mkdir(exist_ok=True)
TRACE_DIR.mkdir(exist_ok=True)

load_dotenv(ROOT / ".env")
api_key = os.getenv("SARVAM_API_KEY")
if not api_key:
    raise RuntimeError("Set SARVAM_API_KEY in .env before starting KAAM")

client = SarvamAI(api_subscription_key=api_key)
store = DataStore(ROOT / "data", ROOT / "seed")
operations = Operations(store)


def save_trace(run):
    timestamp = now_ist().strftime("%Y%m%dT%H%M%S%f")
    path = TRACE_DIR / f"{timestamp}_{run.id}.json"
    payload = {"run_id": run.id, "status": run.status, "events": run.events}
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


agent = Agent(client, operations, on_complete=save_trace)
app = FastAPI(title="KAAM Backend", version="0.2.0")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
from call_api import install as install_call_api
install_call_api(app, agent, operations, client)


class AgentRequest(BaseModel):
    instruction_text: str = Field(min_length=1)
    khata_json: dict


class ResumeRequest(BaseModel):
    messages: list[dict]
    conversation_outcome: dict


def sse(events):
    for event in events:
        yield "data: " + json.dumps(event, ensure_ascii=False) + "\n\n"


def stream(events):
    return StreamingResponse(
        sse(events),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/")
def index():
    path = STATIC_DIR / "index.html"
    if not path.is_file():
        raise HTTPException(status_code=404, detail="static/index.html is not available yet")
    return FileResponse(path)


@app.post("/api/reset")
def reset():
    with store.lock:
        agent.clear_runs()
        result = store.reset()
    return {**result, "run_log_cleared": True}


@app.post("/api/agent")
def start_agent(body: AgentRequest):
    run = agent.create_run(body.instruction_text, body.khata_json)

    def events():
        yield agent.record_event(
            run,
            "input",
            {},
            {"instruction_text": body.instruction_text, "khata_json": body.khata_json},
            "received",
        )
        yield from agent.run_steps(run)

    return stream(events())


@app.post("/api/agent/resume")
def resume_agent(body: ResumeRequest):
    try:
        run = agent.find_pending(body.messages)
        resumed_events = agent.resume(run, body.conversation_outcome)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc

    def events():
        yield from resumed_events
        yield from agent.run_steps(run)

    return stream(events())


@app.get("/api/runs/{run_id}")
def get_run_log(run_id: str):
    run = agent.runs.get(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Run not found")
    return {"run_id": run.id, "status": run.status, "events": run.events}


@app.get("/api/traces/latest")
def latest_trace():
    paths = sorted(TRACE_DIR.glob("*.json"))
    if not paths:
        raise HTTPException(status_code=404, detail="No completed trace yet")
    return json.loads(paths[-1].read_text(encoding="utf-8"))


def build_receipt(events: list[dict]) -> dict:
    """Build every receipt field from events, without consulting mutable data files."""
    if not events:
        raise ValueError("Run has no events")
    run_id = events[0]["run_id"]
    input_event = next((e for e in events if e["tool"] == "input"), None)
    stt_event = next((e for e in events if e["tool"] == "stt" and e["status"] == "ok"), None)
    extract_event = next(
        (e for e in events if e["tool"] == "extract" and e["status"] == "ok"), None
    )
    instruction = (
        stt_event["result"].get("transcript") if stt_event
        else input_event["result"].get("instruction_text") if input_event else None
    )
    if extract_event:
        khata_evidence = {
            "source": "document_ai_extract",
            "fields": extract_event["result"].get("result"),
            "confidence": extract_event["result"].get("field_confidence"),
            "job_id": extract_event["result"].get("job_id"),
        }
    else:
        khata_evidence = {
            "source": "khata_json_input",
            "fields": input_event["result"].get("khata_json") if input_event else None,
            "confidence": [
                e["args"].get("confidence") for e in events
                if e["tool"] == "record_ledger_correction" and e["status"] == "ok"
            ],
        }
    rules = []
    for event in events:
        if event["tool"] == "check_policy" and event["status"] == "ok":
            rules.append({
                "rule_id": event["result"]["rule_id"],
                "allowed_actions": event["result"]["allowed_actions"],
                "matched_rules": event["result"].get("matched_rules"),
            })
        elif event["tool"] == "assign_task" and event["status"] == "ok":
            rule_id = event["args"].get("rule_id")
            if rule_id and not any(rule["rule_id"] == rule_id for rule in rules):
                rules.append({"rule_id": rule_id, "allowed_actions": ["assign_task"]})
    changes = [
        {
            "invoice_id": e["args"]["invoice_id"],
            "amount": e["args"]["amount"],
            "source": e["args"]["source"],
            "confidence": e["args"]["confidence"],
            "before": e["result"]["before_balance"],
            "after": e["result"]["after_balance"],
        }
        for e in events
        if e["tool"] == "record_ledger_correction" and e["status"] == "ok"
    ]
    handoffs = [
        {
            "task_id": e["result"]["id"],
            "member": e["result"]["member_name"],
            "rule_id": e["args"]["rule_id"],
            "task": e["args"]["task"],
            "status": e["result"]["status"],
        }
        for e in events if e["tool"] == "assign_task" and e["status"] == "ok"
    ]
    resumed = next(
        (e for e in reversed(events) if e["tool"] == "start_customer_conversation" and e["status"] == "resumed"),
        None,
    )
    transcript_lines = [
        e["result"]["line"] for e in events if e["tool"] == "call_transcript"
    ]
    call_outcome = ({**resumed["result"], "transcript_lines": transcript_lines} if resumed else None)
    if events[-1]["status"] == "error":
        status = "error"
    elif any(e["tool"] == "notify_owner" and e["status"] == "completed" for e in events):
        status = "completed"
    elif any(e["status"] == "awaiting_conversation" for e in events):
        status = "awaiting_conversation"
    else:
        status = "running"
    receipt_number = int(run_id.replace("-", "")[:10], 16) % 100000
    return {
        "receipt_id": f"K-{receipt_number:05d}",
        "run_id": run_id,
        "instruction": instruction,
        "khata_evidence": khata_evidence,
        "rules_applied": rules,
        "call_outcome": call_outcome,
        "changes": changes,
        "handoffs": handoffs,
        "status": status,
    }


@app.get("/api/receipt")
def receipt():
    if not agent.runs:
        raise HTTPException(status_code=404, detail="No run since reset")
    run = list(agent.runs.values())[-1]
    return build_receipt(run.events)


def sarvam_error(exc: ApiError):
    raise HTTPException(
        status_code=exc.status_code if 400 <= exc.status_code <= 599 else 502,
        detail={"error": "sarvam_api_error", "status_code": exc.status_code},
    ) from exc


def transcribe_audio(filename: str, audio: bytes, content_type: str, mode: str, language_code: str | None):
    if not audio:
        raise HTTPException(status_code=422, detail="Audio file is empty")
    if len(audio) > 20_000_000:
        raise HTTPException(status_code=413, detail="Audio file is too large")
    if filename.lower().endswith(".wav"):
        try:
            with wave.open(io.BytesIO(audio), "rb") as wav:
                duration = wav.getnframes() / wav.getframerate()
        except (wave.Error, ZeroDivisionError, EOFError) as exc:
            raise HTTPException(status_code=422, detail="Invalid WAV file") from exc
        if duration > 30:
            raise HTTPException(status_code=422, detail="Audio must be at most 30 seconds")
    kwargs = {
        "file": (filename, audio, content_type),
        "model": "saaras:v3",
        "mode": mode,
    }
    if language_code:
        kwargs["language_code"] = language_code
    try:
        response = client.speech_to_text.transcribe(**kwargs)
    except ApiError as exc:
        sarvam_error(exc)
    return {
        "model": "saaras:v3",
        "mode": mode,
        "transcript": response.transcript,
        "language_code": response.language_code,
    }


@app.post("/api/stt")
def speech_to_text(
    file: UploadFile = File(...),
    mode: Literal["codemix", "transcribe"] = Form("codemix"),
    language_code: str | None = Form(None),
):
    return transcribe_audio(
        file.filename or "audio.wav",
        file.file.read(),
        file.content_type or "audio/wav",
        mode,
        language_code,
    )


EXTRACT_SCHEMA = {
    "type": "object",
    "properties": {
        "customer_name": {
            "type": "string",
            "description": "Business or customer name written on the khata page",
        },
        "entries": {
            "type": "array",
            "description": "Each invoice or cash received entry on the khata page",
            "items": {
                "type": "object",
                "description": "One dated khata transaction",
                "properties": {
                    "date": {"type": "string", "description": "Transaction date in YYYY-MM-DD format"},
                    "description": {"type": "string", "description": "Written transaction description"},
                    "amount": {"type": "number", "description": "Transaction amount in INR"},
                    "type": {
                        "type": "string",
                        "description": "Whether this is an invoice or cash received",
                        "enum": ["invoice", "cash_received"],
                    },
                },
            },
        },
    },
}


def confidence_tree(value):
    if isinstance(value, dict) and "confidence" in value:
        return value["confidence"]
    if isinstance(value, dict):
        return {key: confidence_tree(child) for key, child in value.items()}
    if isinstance(value, list):
        return [confidence_tree(child) for child in value]
    return None


def extract_image(filename: str, image: bytes, language: str):
    suffix = Path(filename).suffix.lower()
    mime = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg"}.get(suffix)
    if mime is None or not image:
        raise HTTPException(status_code=422, detail="Upload a non-empty PNG or JPEG image")
    try:
        job = client.doc_ai.extract(
            file=[(filename, image, mime)],
            schema=json.dumps(EXTRACT_SCHEMA),
            language=language,
            output_format="json",
        )
        deadline = time.monotonic() + 180
        while True:
            status = client.doc_ai.get_status(job_id=job.job_id)
            state = str(status.status).lower()
            if state in {"completed", "partially_completed", "failed", "rejected"}:
                break
            if time.monotonic() >= deadline:
                raise HTTPException(
                    status_code=504,
                    detail={"error": "extract_timeout", "job_id": job.job_id, "status": state},
                )
            time.sleep(6)
        if state not in {"completed", "partially_completed"}:
            raise HTTPException(
                status_code=502,
                detail={"error": "extract_failed", "job_id": job.job_id, "status": state},
            )
        results = client.doc_ai.get_results(job_id=job.job_id)
    except ApiError as exc:
        sarvam_error(exc)
    return {
        "job_id": job.job_id,
        "status": state,
        "result": results.result,
        "annotations": results.annotations,
        "field_confidence": confidence_tree(results.annotations),
    }


@app.post("/api/extract")
def extract_khata(file: UploadFile = File(...), language: str = Form("en-IN")):
    return extract_image(file.filename or "", file.file.read(), language)


def attach_field_confidence(extraction: dict) -> dict:
    khata = deepcopy(extraction["result"])
    entries = khata.get("entries", [])
    confidence_entries = extraction.get("field_confidence", {}).get("entries", [])
    for entry, confidence in zip(entries, confidence_entries):
        if isinstance(entry, dict) and isinstance(confidence, dict):
            values = [
                confidence.get(field) for field in ("date", "amount", "type")
                if isinstance(confidence.get(field), (int, float))
            ]
            if values:
                entry["confidence"] = min(values)
    return khata


@app.post("/api/run")
def one_shot_run(
    audio: UploadFile = File(...),
    image: UploadFile = File(...),
    language: str = Form("en-IN"),
):
    audio_name = audio.filename or "audio.wav"
    image_name = image.filename or "khata.png"
    audio_bytes = audio.file.read()
    image_bytes = image.file.read()
    audio_type = audio.content_type or "audio/wav"
    run = agent.create_run("", {})
    run.status = "processing_inputs"

    def events():
        jobs = {
            "stt": (
                transcribe_audio,
                (audio_name, audio_bytes, audio_type, "codemix", None),
                {"filename": audio_name, "mode": "codemix"},
            ),
            "extract": (
                extract_image,
                (image_name, image_bytes, language),
                {"filename": image_name, "language": language},
            ),
        }
        results = {}
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = {
                pool.submit(fn, *arguments): (name, event_args)
                for name, (fn, arguments, event_args) in jobs.items()
            }
            for future in as_completed(futures):
                if not agent.is_active(run):
                    return
                name, event_args = futures[future]
                try:
                    result = future.result()
                except HTTPException as exc:
                    run.status = "error"
                    yield agent.record_event(run, name, event_args, {"error": exc.detail}, "error")
                    return
                except Exception as exc:
                    run.status = "error"
                    yield agent.record_event(
                        run, name, event_args, {"error": type(exc).__name__}, "error"
                    )
                    return
                results[name] = result
                yield agent.record_event(run, name, event_args, result, "ok")
        instruction = results["stt"]["transcript"].strip()
        if not agent.is_active(run):
            return
        if not instruction:
            run.status = "error"
            yield agent.record_event(run, "stt", {}, {"error": "Empty transcript"}, "error")
            return
        khata = attach_field_confidence(results["extract"])
        agent.set_input(run, instruction, khata)
        yield from agent.run_steps(run)

    return stream(events())

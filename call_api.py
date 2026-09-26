"""Browser bridge for the Sarvam conversation session in voice_test.py."""

import asyncio
import base64
import json
import os
import uuid
from dataclasses import dataclass, field
from datetime import date

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, SecretStr
from sarvam_conv_ai_sdk import AsyncSamvaadAgent, InteractionConfig, InteractionType, Role
from sarvam_conv_ai_sdk.messages.types import UserIdentifierType

from config import SARVAM_MODEL, SARVAM_REASONING_EFFORT, SARVAM_TEMPERATURE

router = APIRouter()
sessions = {}


@dataclass
class Session:
    id: str
    run: object
    queue: asyncio.Queue = field(default_factory=asyncio.Queue)
    lines: list = field(default_factory=list)
    voice: object = None
    ended: bool = False

    def emit(self, tool, result, status="ok"):
        event = {"step": len(self.run.events) + 1, "tool": tool, "args": {},
                 "result": result, "status": status, "ts": __import__("datetime").datetime.now().astimezone().isoformat()}
        self.queue.put_nowait(event)


def install(app, agent, operations, client):
    def variables(run):
        if run.status != "awaiting_conversation" or not run.customer_id or not run.invoice_id:
            raise ValueError("Run must be paused after the ledger correction")
        if not any(e["tool"] == "record_ledger_correction" and e["status"] == "ok" for e in run.events):
            raise ValueError("Ledger correction is required before a call")
        customer = next(c for c in operations.store.snapshot("customers") if c["id"] == run.customer_id)
        ledger = operations.get_ledger(run.customer_id)
        invoice = next(i for i in ledger["invoices"] if i["id"] == run.invoice_id)
        payments = [p for p in ledger["payments"] if p["invoice_id"] == run.invoice_id]
        latest = max(payments, key=lambda p: p["date"])
        money = lambda n: f"₹{float(n):,.0f}"
        return {"company_name": "Noetos", "customer_name": customer["contact"],
                "gender": customer.get("gender", "male"), "invoice_number": invoice["id"],
                "invoice_amount": money(invoice["amount"]), "amount_paid": money(sum(p["amount"] for p in payments)),
                "payment_date": date.fromisoformat(latest["date"]).strftime("%d %B"),
                "balance_due": money(invoice["balance"]),
                "due_date": date.fromisoformat(invoice["due_date"]).strftime("%d %B")}

    def extract(lines):
        schema = {"type": "function", "function": {"name": "record_call_outcome",
            "description": "Extract only facts stated in the customer call.",
            "parameters": {"type": "object", "properties": {
                "promised_date": {"type": "string", "description": "YYYY-MM-DD or empty string"},
                "discount_requested": {"type": "boolean"}, "discount_amount": {"type": "number"},
                "dispute": {"type": "boolean"}, "disputed_amount": {"type": "number"},
                "dispute_reason": {"type": "string"}},
                "required": ["promised_date", "discount_requested", "discount_amount", "dispute", "disputed_amount", "dispute_reason"]}}}
        response = client.chat.completions(model=SARVAM_MODEL, temperature=SARVAM_TEMPERATURE,
            reasoning_effort=SARVAM_REASONING_EFFORT, tools=[schema], tool_choice="required",
            messages=[{"role": "user", "content": "Call record_call_outcome once. Today is " + date.today().isoformat() +
                ". Resolve relative dates. Use empty string/zero/false for absent facts. Transcript: " + json.dumps(lines, ensure_ascii=False)}])
        calls = response.choices[0].message.tool_calls or []
        if len(calls) != 1 or calls[0].function.name != "record_call_outcome":
            raise ValueError("Outcome extraction did not return one tool call")
        result = json.loads(calls[0].function.arguments)
        if not result.get("promised_date"):
            result.pop("promised_date", None)
        result["transcript_lines"] = lines
        return result

    async def finish(session):
        if session.ended:
            return
        session.ended = True
        try:
            outcome = await asyncio.to_thread(extract, session.lines)
            session.emit("call_outcome", outcome)
            for event in agent.resume(session.run, outcome):
                session.queue.put_nowait(event)
            def continue_agent():
                for event in agent.run_steps(session.run):
                    loop.call_soon_threadsafe(session.queue.put_nowait, event)
            loop = asyncio.get_running_loop()
            await asyncio.to_thread(continue_agent)
        except Exception as exc:
            session.emit("call", {"error": str(exc)}, "error")
        finally:
            if session.voice:
                await session.voice.stop()
            session.queue.put_nowait(None)

    class Start(BaseModel):
        run_id: str

    @router.post("/api/call/start")
    async def start(body: Start):
        run = agent.runs.get(body.run_id)
        if not run:
            raise HTTPException(404, "Run not found")
        try:
            values = variables(run)
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from exc
        needed = ("SARVAM_API_KEY", "SARVAM_ORG_ID", "SARVAM_WORKSPACE_ID", "SARVAM_APP_ID")
        if any(not os.getenv(key) for key in needed):
            raise HTTPException(503, "Voice credentials are incomplete")
        session = Session(str(uuid.uuid4()), run)
        async def transcript(msg):
            line = {"speaker": "user" if msg.role == Role.USER else "bot", "text": msg.content, "source": "live_call"}
            session.lines.append(line)
            session.emit("call_transcript", {"line": line})
        async def event(msg):
            if str(msg.type).endswith("interaction_end"):
                asyncio.create_task(finish(session))
            else:
                session.emit("call_event", {"type": str(msg.type)})
        async def audio(msg):
            session.emit("call_audio", {"audio_base64": msg.audio_base64, "sample_rate": msg.sample_rate})
        config = InteractionConfig(user_identifier_type=UserIdentifierType.CUSTOM,
            user_identifier="kaam_" + session.id, org_id=os.environ["SARVAM_ORG_ID"],
            workspace_id=os.environ["SARVAM_WORKSPACE_ID"], app_id=os.environ["SARVAM_APP_ID"],
            interaction_type=InteractionType.CALL, sample_rate=16000, agent_variables=values)
        session.voice = AsyncSamvaadAgent(api_key=SecretStr(os.environ["SARVAM_API_KEY"]),
            config=config, transcript_callback=transcript, event_callback=event, audio_callback=audio)
        sessions[session.id] = session
        try:
            await session.voice.start()
            if not await session.voice.wait_for_connect(timeout=10):
                raise RuntimeError("Voice session did not connect")
        except Exception as exc:
            sessions.pop(session.id, None)
            await session.voice.stop()
            raise HTTPException(502, f"Voice session failed: {type(exc).__name__}") from exc
        return {"call_id": session.id, "run_id": run.id, "agent_variables": values}

    @router.get("/api/call/stream")
    async def stream(call_id: str):
        session = sessions.get(call_id)
        if not session:
            raise HTTPException(404, "Call not found")
        async def events():
            while True:
                item = await session.queue.get()
                if item is None:
                    break
                yield "data: " + json.dumps(item, ensure_ascii=False) + "\n\n"
        return StreamingResponse(events(), media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    class Audio(BaseModel):
        call_id: str
        audio_base64: str

    @router.post("/api/call/audio")
    async def send_audio(body: Audio):
        session = sessions.get(body.call_id)
        if not session or session.ended:
            raise HTTPException(404, "Active call not found")
        await session.voice.send_audio(base64.b64decode(body.audio_base64, validate=True))
        return {"status": "sent"}

    class Simulate(BaseModel):
        run_id: str
        conversation_outcome: dict

    @router.post("/api/call/simulate")
    def simulate(body: Simulate):
        run = agent.runs.get(body.run_id)
        if not run:
            raise HTTPException(404, "Run not found")
        try:
            events = agent.resume(run, body.conversation_outcome)
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from exc
        def output():
            for event in events:
                yield "data: " + json.dumps(event, ensure_ascii=False) + "\n\n"
            for event in agent.run_steps(run):
                yield "data: " + json.dumps(event, ensure_ascii=False) + "\n\n"
        return StreamingResponse(output(), media_type="text/event-stream")

    app.include_router(router)

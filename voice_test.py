"""Standalone smoke test for the Sarvam Voice Agent (talk to it through your mic).

Setup (once):
    pip install "sarvam-conv-ai-sdk[all]" python-dotenv
    # Windows: the PyAudio wheel bundles PortAudio. macOS: brew install portaudio.

Needs in .env: SARVAM_VA_API_KEY, SARVAM_ORG_ID, SARVAM_WORKSPACE_ID, SARVAM_APP_ID

Run:
    python voice_test.py
"""

import asyncio
import os
import sys
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from pydantic import SecretStr
from sarvam_conv_ai_sdk import (
    AsyncDefaultAudioInterface,
    AsyncSamvaadAgent,
    InteractionConfig,
    InteractionType,
    Role,
    ServerEventBase,
    ServerTranscriptMsg,
)
from sarvam_conv_ai_sdk.messages.types import UserIdentifierType

SESSION_TIMEOUT_S = 120
REQUIRED_ENV = ("SARVAM_VA_API_KEY", "SARVAM_ORG_ID", "SARVAM_WORKSPACE_ID", "SARVAM_APP_ID")

AGENT_VARIABLES = {
    "company_name": "Noetos",
    "customer_name": "Senthil",
    "gender": "male",
    "invoice_number": "INV-1042",
    "invoice_amount": "₹82,500",
    "amount_paid": "₹20,000",
    "payment_date": "12 September",
    "balance_due": "₹62,500",
    "due_date": "9 September",
}

# Windows consoles default to cp1252; ₹ and Tamil/Hindi text need UTF-8.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def ts() -> str:
    return datetime.now().strftime("%H:%M:%S")


async def on_transcript(msg: ServerTranscriptMsg) -> None:
    role = "user" if msg.role == Role.USER else "bot"
    print(f"[{ts()}] {role}: {msg.content}", flush=True)


async def on_event(msg: ServerEventBase) -> None:
    print(f"[{ts()}] event: {msg.type}", flush=True)


async def main() -> int:
    load_dotenv(Path(__file__).with_name(".env"))
    missing = [name for name in REQUIRED_ENV if not os.getenv(name)]
    if missing:
        print(f"Missing in .env: {', '.join(missing)}")
        return 1

    config = InteractionConfig(
        user_identifier_type=UserIdentifierType.CUSTOM,
        user_identifier="kaam_voice_test",
        org_id=os.environ["SARVAM_ORG_ID"],
        workspace_id=os.environ["SARVAM_WORKSPACE_ID"],
        app_id=os.environ["SARVAM_APP_ID"],
        interaction_type=InteractionType.CALL,
        sample_rate=16000,
        agent_variables=AGENT_VARIABLES,
    )
    agent = AsyncSamvaadAgent(
        api_key=SecretStr(os.environ["SARVAM_VA_API_KEY"]),
        config=config,
        audio_interface=AsyncDefaultAudioInterface(input_sample_rate=16000),
        transcript_callback=on_transcript,
        event_callback=on_event,
    )

    try:
        print(f"[{ts()}] starting session...")
        await agent.start()
        if not await agent.wait_for_connect(timeout=10.0):
            print(f"[{ts()}] ERROR: did not connect within 10s")
            return 1
        print(f"[{ts()}] connected, interaction_id={agent.get_interaction_id()}")
        print(f"Speak into your mic. Auto-stops after {SESSION_TIMEOUT_S}s (Ctrl+C to quit).\n")
        await asyncio.wait_for(agent.wait_for_disconnect(), timeout=SESSION_TIMEOUT_S)
        print(f"[{ts()}] session ended by server")
    except asyncio.TimeoutError:
        print(f"[{ts()}] timeout after {SESSION_TIMEOUT_S}s")
    except Exception as exc:
        print(f"[{ts()}] ERROR: {type(exc).__name__}: {exc}")
        raise
    finally:
        await agent.stop()
        print(f"[{ts()}] agent stopped")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(asyncio.run(main()))
    except KeyboardInterrupt:
        print("\ninterrupted")

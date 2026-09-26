"""Three small Sarvam chat calls to check access and compare latency."""

import os
import time
from pathlib import Path

from dotenv import load_dotenv
from sarvamai import SarvamAI
from config import SARVAM_MODEL, SARVAM_REASONING_EFFORT, SARVAM_TEMPERATURE


load_dotenv(Path(__file__).with_name(".env"))
api_key = os.getenv("SARVAM_API_KEY")
if not api_key:
    raise SystemExit("Add SARVAM_API_KEY to .env before running this test.")

client = SarvamAI(api_subscription_key=api_key)

customer_tool = [
    {
        "type": "function",
        "function": {
            "name": "find_customer",
            "description": "Find a customer account by name (dummy tool; never executed).",
            "parameters": {
                "type": "object",
                "properties": {"name": {"type": "string"}},
                "required": ["name"],
            },
        },
    }
]
customer_message = [{"role": "user", "content": "Check Murugan Traders' account"}]


def run_call(label, **kwargs):
    started = time.perf_counter()
    response = client.chat.completions(**kwargs)
    elapsed = time.perf_counter() - started
    message = response.choices[0].message

    print(f"{label}: {elapsed:.3f} s")
    print(f"Reply: {message.content or '(no text reply)'}")
    if "tools" in kwargs:
        tool_calls = message.tool_calls or []
        print(f"Tool call returned: {bool(tool_calls)}")
        if not tool_calls:
            print("Arguments: (none)")
        for tool_call in tool_calls:
            print(f"Tool: {tool_call.function.name}")
            print(f"Arguments: {tool_call.function.arguments}")
    print()
    return elapsed


run_call(
    "a) Tamil greeting",
    model="sarvam-105b-conversations",
    messages=[{"role": "user", "content": "தமிழில் ஒரு வரியில் வணக்கம் சொல்லுங்கள்."}],
    temperature=SARVAM_TEMPERATURE,
)

with_reasoning = run_call(
    "b) Customer lookup, default reasoning",
    model=SARVAM_MODEL,
    messages=customer_message,
    tools=customer_tool,
    tool_choice="auto",
    temperature=SARVAM_TEMPERATURE,
)

without_reasoning = run_call(
    f"c) Customer lookup, reasoning {SARVAM_REASONING_EFFORT or 'disabled'}",
    model=SARVAM_MODEL,
    messages=customer_message,
    tools=customer_tool,
    tool_choice="auto",
    temperature=SARVAM_TEMPERATURE,
    reasoning_effort=SARVAM_REASONING_EFFORT,
)

print(f"Latency difference (c - b): {without_reasoning - with_reasoning:+.3f} s")

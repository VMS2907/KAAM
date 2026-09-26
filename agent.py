"""Sarvam tool loop and the in-memory, resumable execution log."""

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, time, timedelta

from config import (
    MAX_AGENT_STEPS,
    SARVAM_MODEL,
    SARVAM_REASONING_EFFORT,
    SARVAM_TEMPERATURE,
)
from operations import Operations, date_from_iso
from store import IST, now_ist, today_ist


SYSTEM_PROMPT = (
    "You are KAAM, an AI operations worker for Noetos Pvt Ltd. You complete "
    "objectives; you do not chat. Work out the outcome the owner wants. Gather "
    "facts with tools. Compare the handwritten khata entries against the ledger "
    "and record any correction, citing the khata as the source. Call check_policy "
    "before any customer contact or concession. Act only within the allowed "
    "actions; assign anything outside them to the right teammate with evidence "
    "and a due time. Never claim an action happened unless a tool confirmed it. "
    "When finished, call notify_owner with a 3-sentence English summary."
)


def function_tool(name, description, properties, required):
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        },
    }


TEXT = {"type": "string"}
NUMBER = {"type": "number"}
INTEGER = {"type": "integer"}
TOOLS = [
    function_tool("find_customer", "Find a customer by business name.", {"name": TEXT}, ["name"]),
    function_tool(
        "get_ledger", "Get invoices, payments and balance.", {"customer_id": TEXT}, ["customer_id"]
    ),
    function_tool(
        "record_ledger_correction",
        "Record a cash payment seen in the handwritten khata but missing from the ledger.",
        {"invoice_id": TEXT, "amount": NUMBER, "date": TEXT, "source": TEXT, "confidence": NUMBER},
        ["invoice_id", "amount", "date", "source", "confidence"],
    ),
    function_tool(
        "check_policy",
        "Apply the company playbook deterministically before contact or concessions.",
        {"days_overdue": INTEGER, "balance": NUMBER, "situation": TEXT},
        ["days_overdue", "balance", "situation"],
    ),
    function_tool(
        "start_customer_conversation",
        "Prepare a customer conversation and pause for its externally supplied outcome; does not place a call.",
        {"customer_id": TEXT, "language": TEXT, "objective": TEXT, "amount_due": NUMBER},
        ["customer_id", "language", "objective", "amount_due"],
    ),
    function_tool(
        "record_commitment",
        "Record the customer's promised payment date and amount.",
        {"invoice_id": TEXT, "amount": NUMBER, "promised_date": TEXT},
        ["invoice_id", "amount", "promised_date"],
    ),
    function_tool(
        "schedule_followup",
        "Schedule a conditional follow-up for an invoice.",
        {"invoice_id": TEXT, "datetime": TEXT, "condition": TEXT},
        ["invoice_id", "datetime", "condition"],
    ),
    function_tool(
        "assign_task",
        "Assign policy-bound work to the teammate who handles it.",
        {
            "member_id": TEXT,
            "task": TEXT,
            "reason": TEXT,
            "rule_id": TEXT,
            "evidence": TEXT,
            "due_at": TEXT,
            "depends_on": TEXT,
        },
        ["member_id", "task", "reason", "rule_id", "evidence", "due_at"],
    ),
    function_tool(
        "notify_owner", "Record a three-sentence English owner summary.",
        {"summary_en": TEXT}, ["summary_en"]
    ),
]


class RunCancelled(Exception):
    """Raised when a reset invalidates a run that was still executing."""


@dataclass
class Run:
    id: str
    instruction: str
    khata: dict
    messages: list[dict]
    todo: list[str] = field(default_factory=lambda: ["find_customer", "get_ledger"])
    events: list[dict] = field(default_factory=list)
    step: int = 0
    agent_steps: int = 0
    generation: int = 0
    customer_id: str | None = None
    invoice_id: str | None = None
    policy: dict | None = None
    correction_entries: list[dict] = field(default_factory=list)
    assignments: list[dict] = field(default_factory=list)
    outcome: dict | None = None
    pending_tool_call_id: str | None = None
    status: str = "running"


class Agent:
    def __init__(self, client, operations: Operations, on_complete=None):
        self.client = client
        self.operations = operations
        self.on_complete = on_complete
        self.runs: dict[str, Run] = {}
        self.generation = 0

    def _finance_id(self) -> str:
        return next(
            member["id"] for member in self.operations.store.snapshot("team")
            if member["role"] == "Finance Manager"
        )

    def _dispute_member_id(self, outcome: dict) -> str:
        team = self.operations.store.snapshot("team")
        reason = str(outcome.get("dispute_reason") or outcome.get("reason") or "").casefold()
        if any(word in reason for word in ("delivery", "damage", "dispatch")):
            return next(
                member["id"] for member in team
                if any("delivery" in handle or "damage" in handle for handle in member["handles"])
            )
        return next(
            member["id"] for member in team
            if "disputes" in member["handles"]
        )

    def create_run(self, instruction: str, khata: dict) -> Run:
        run_id = str(uuid.uuid4())
        run = Run(
            id=run_id,
            instruction=instruction,
            khata=khata,
            messages=self._build_messages(instruction, khata),
            generation=self.generation,
        )
        self.runs[run_id] = run
        return run

    @staticmethod
    def _build_messages(instruction: str, khata: dict) -> list[dict]:
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "system",
                "content": "Call exactly the named tool selected by the controller. Give one tool call per turn. Use the facts in the tool results; do not invent completed actions.",
            },
            {
                "role": "user",
                "content": json.dumps(
                    {"instruction_text": instruction, "khata_json": khata, "today": today_ist().isoformat()},
                    ensure_ascii=False,
                ),
            },
        ]
    def set_input(self, run: Run, instruction: str, khata: dict):
        if not self.is_active(run):
            raise RunCancelled()
        run.instruction = instruction
        run.khata = khata
        run.messages = self._build_messages(instruction, khata)
        run.status = "running"

    def clear_runs(self):
        self.generation += 1
        for run in self.runs.values():
            run.status = "cancelled"
        self.runs.clear()

    def is_active(self, run: Run) -> bool:
        return run.generation == self.generation and self.runs.get(run.id) is run

    def record_event(self, run: Run, tool: str, args: dict, result: dict, status: str) -> dict:
        run.step += 1
        return self._event(run, tool, args, result, status)

    def find_pending(self, messages: list[dict]) -> Run:
        for run in self.runs.values():
            if run.status == "awaiting_conversation" and run.messages == messages:
                return run
        raise ValueError("No matching paused run; resume with the exact messages from the pause event")

    def resume(self, run: Run, outcome: dict) -> list[dict]:
        if run.status != "awaiting_conversation" or not run.pending_tool_call_id:
            raise ValueError("Run is not awaiting a conversation")
        if not isinstance(outcome, dict):
            raise ValueError("conversation_outcome must be an object")
        if outcome.get("promised_date"):
            date_from_iso(outcome["promised_date"])
        if outcome.get("dispute"):
            disputed = float(outcome.get("disputed_amount") or 0)
            if not 0 < disputed < self._balance(run):
                raise ValueError("Dispute requires disputed_amount between zero and the outstanding balance")
        run.outcome = outcome
        run.messages.append(
            {
                "role": "tool",
                "tool_call_id": run.pending_tool_call_id,
                "content": json.dumps(outcome, ensure_ascii=False),
            }
        )
        run.pending_tool_call_id = None
        run.status = "running"
        if outcome.get("dispute"):
            run.todo.append("check_policy")
        if outcome.get("promised_date"):
            run.todo.extend(["record_commitment", "schedule_followup"])
        if outcome.get("discount_requested"):
            run.assignments.append({"kind": "discount", "rule_id": "R4", "member_id": self._finance_id()})
            run.todo.append("assign_task")
        if outcome.get("dispute"):
            member_id = self._dispute_member_id(outcome)
            run.assignments.append({"kind": "dispute", "rule_id": "R5", "member_id": member_id})
            run.todo.append("assign_task")
        run.todo.append("notify_owner")
        events = [self.record_event(run, "start_customer_conversation", {}, outcome, "resumed")]
        for line in outcome.get("transcript_lines", []):
            events.append(self.record_event(run, "call_transcript", {}, {"line": line}, "ok"))
        return events

    def _event(self, run: Run, tool: str, args: dict, result: dict, status: str) -> dict:
        event = {
            "run_id": run.id,
            "step": run.step,
            "tool": tool,
            "args": args,
            "result": result,
            "status": status,
            "ts": now_ist().isoformat(),
        }
        run.events.append(event)
        return event

    def _customer_name(self, run: Run, model_args: dict) -> str:
        if run.khata.get("customer_name"):
            return str(run.khata["customer_name"])
        for customer in self.operations.store.snapshot("customers"):
            if customer["name"].casefold() in run.instruction.casefold():
                return customer["name"]
        return str(model_args.get("name", ""))

    def _invoice(self, run: Run) -> dict:
        if not run.invoice_id:
            raise ValueError("No invoice selected")
        invoice = next(
            (i for i in self.operations.store.snapshot("invoices") if i["id"] == run.invoice_id),
            None,
        )
        if invoice is None:
            raise ValueError("Invoice disappeared from store")
        return invoice

    def _balance(self, run: Run) -> float:
        return self.operations.get_ledger(run.customer_id)["balance"]

    def _next_business_due(self) -> str:
        due = today_ist() + timedelta(days=1)
        while due.weekday() >= 5:
            due += timedelta(days=1)
        return datetime.combine(due, time(17, 0), IST).isoformat()

    def _summary(self, run: Run) -> str:
        customer = next(
            c for c in self.operations.store.snapshot("customers") if c["id"] == run.customer_id
        )
        correction = next((e for e in run.events if e["tool"] == "record_ledger_correction"), None)
        if correction:
            first = (
                f"I recorded the khata cash payment of INR {correction['args']['amount']:,.0f} "
                f"for {customer['name']}, reducing {run.invoice_id} from INR "
                f"{correction['result']['before_balance']:,.0f} to INR "
                f"{correction['result']['after_balance']:,.0f}."
            )
        else:
            first = f"I checked {customer['name']}'s ledger for {run.invoice_id}, with INR {self._balance(run):,.0f} outstanding."
        if run.outcome and run.outcome.get("promised_date"):
            promised_amount = self._balance(run) - float(run.outcome.get("disputed_amount") or 0)
            second = (
                f"The supplied customer conversation outcome records a promise to pay "
                f"INR {promised_amount:,.0f} on {run.outcome['promised_date']}, and a conditional follow-up was scheduled."
            )
        elif run.status == "awaiting_conversation":
            second = "The customer conversation is awaiting an externally supplied outcome."
        else:
            second = "The current playbook was checked before deciding the next action."
        discount_task = next(
            (e for e in run.events if e["tool"] == "assign_task" and e["args"]["rule_id"] == "R4"),
            None,
        )
        if discount_task:
            discount_text = (
                f"A requested INR {run.outcome.get('discount_amount', 0):,.0f} discount "
                f"was assigned to {discount_task['result']['member_name']} under R4 for review"
            )
        else:
            discount_text = "No unapproved concession was made"
        dispute_task = next(
            (e for e in run.events if e["tool"] == "assign_task" and e["args"]["rule_id"] == "R5"),
            None,
        )
        if dispute_task:
            third = (
                f"The INR {float(run.outcome['disputed_amount']):,.0f} disputed portion was assigned "
                f"to {dispute_task['result']['member_name']} under R5; {discount_text[0].lower() + discount_text[1:]}."
            )
        elif discount_task:
            third = discount_text + "; no discount was approved."
        else:
            third = discount_text + "."
        return " ".join((first, second, third))

    def _args(self, run: Run, name: str, model_args: dict) -> dict:
        if name == "find_customer":
            return {"name": self._customer_name(run, model_args)}
        if name == "get_ledger":
            return {"customer_id": run.customer_id}
        if name == "record_ledger_correction":
            entry = run.correction_entries[0]
            return {
                "invoice_id": run.invoice_id,
                "amount": entry["amount"],
                "date": entry["date"],
                "source": "khata: handwritten cash_received entry",
                "confidence": float(entry.get("confidence", 0.8)),
            }
        if name == "check_policy":
            invoice = self._invoice(run)
            days = max(0, (today_ist() - date_from_iso(invoice["due_date"])).days)
            return {
                "days_overdue": days,
                "balance": self._balance(run),
                "situation": "dispute" if run.outcome and run.outcome.get("dispute") else "overdue collection",
            }
        if name == "start_customer_conversation":
            customer = next(
                c for c in self.operations.store.snapshot("customers") if c["id"] == run.customer_id
            )
            return {
                "customer_id": run.customer_id,
                "language": customer["language"],
                "objective": "Discuss the overdue invoice in Tamil and confirm a payment date",
                "amount_due": self._balance(run),
            }
        if name == "record_commitment":
            disputed = float(run.outcome.get("disputed_amount") or 0) if run.outcome.get("dispute") else 0
            return {
                "invoice_id": run.invoice_id,
                "amount": round(self._balance(run) - disputed, 2),
                "promised_date": run.outcome["promised_date"],
            }
        if name == "schedule_followup":
            if run.outcome and run.outcome.get("promised_date"):
                due = datetime.combine(
                    date_from_iso(run.outcome["promised_date"]), time(18, 0), IST
                )
                condition = "If the promised payment has not been received"
            else:
                due = datetime.fromisoformat(self._next_business_due()).replace(hour=9)
                condition = "Customer contact only during permitted hours"
            return {
                "invoice_id": run.invoice_id,
                "datetime": due.isoformat(),
                "condition": condition,
            }
        if name == "assign_task":
            assignment = run.assignments[0]
            if assignment["kind"] == "discount":
                amount = float(run.outcome.get("discount_amount") or 0)
                return {
                    "member_id": assignment["member_id"],
                    "task": f"Review INR {amount:,.0f} discount request for {run.invoice_id}; do not apply without approval",
                    "reason": "Customer requested a discount; KAAM cannot approve it",
                    "rule_id": "R4",
                    "evidence": json.dumps(
                        {"invoice_id": run.invoice_id, "discount_requested": True, "discount_amount": amount, "conversation_outcome": run.outcome},
                        ensure_ascii=False,
                    ),
                    "due_at": self._next_business_due(),
                }
            if assignment["kind"] == "dispute":
                disputed = float(run.outcome.get("disputed_amount") or 0)
                return {
                    "member_id": assignment["member_id"],
                    "task": f"Resolve disputed INR {disputed:,.0f} on {run.invoice_id}; continue collecting undisputed INR {max(0, self._balance(run) - disputed):,.0f}",
                    "reason": "Customer disputed part of the balance",
                    "rule_id": "R5",
                    "evidence": json.dumps(run.outcome, ensure_ascii=False),
                    "due_at": self._next_business_due(),
                }
            return {
                "member_id": assignment["member_id"],
                "task": f"Review overdue {run.invoice_id}",
                "reason": "R3 escalation threshold reached",
                "rule_id": "R3",
                "evidence": json.dumps(run.policy, ensure_ascii=False),
                "due_at": self._next_business_due(),
            }
        if name == "notify_owner":
            return {"summary_en": self._summary(run)}
        raise ValueError(f"Unknown planned tool {name}")

    def _advance(self, run: Run, name: str, result: dict):
        run.todo.pop(0)
        if name == "find_customer":
            run.customer_id = result["id"]
        elif name == "get_ledger":
            if not result["invoices"]:
                raise ValueError("Customer has no invoices")
            run.invoice_id = result["invoices"][0]["id"]
            payments = result["payments"]
            run.correction_entries = [
                e for e in run.khata.get("entries", [])
                if e.get("type") == "cash_received"
                and not any(
                    float(p["amount"]) == float(e["amount"]) and p["date"] == e["date"]
                    for p in payments
                )
            ]
            run.todo.extend(["record_ledger_correction"] * len(run.correction_entries))
            run.todo.append("check_policy")
        elif name == "record_ledger_correction":
            run.correction_entries.pop(0)
        elif name == "check_policy":
            run.policy = result
            if run.outcome is not None:
                return
            if result["rule_id"] in {"R2", "R5"} and "start_customer_conversation" in result["allowed_actions"]:
                run.todo.append("start_customer_conversation")
            elif result["rule_id"] == "R3":
                run.assignments.append({"kind": "overdue", "rule_id": "R3", "member_id": self._finance_id()})
                run.todo.extend(["assign_task", "notify_owner"])
            elif not result["contact_allowed"]:
                run.todo.extend(["schedule_followup", "notify_owner"])
            else:
                run.todo.append("notify_owner")
        elif name == "assign_task":
            run.assignments.pop(0)

    def run_steps(self, run: Run):
        while run.todo:
            if not self.is_active(run):
                run.status = "cancelled"
                yield self.record_event(run, run.todo[0], {}, {"reason": "reset"}, "cancelled")
                return
            if run.agent_steps >= MAX_AGENT_STEPS:
                run.status = "error"
                yield self.record_event(run, run.todo[0], {}, {"error": "12-step limit reached"}, "error")
                return
            name = run.todo[0]
            try:
                selected_tool = next(tool for tool in TOOLS if tool["function"]["name"] == name)
                planned_args = self._args(run, name, {})
                call_messages = run.messages + [
                    {
                        "role": "user",
                        "content": (
                            f"Controller's next action: call the available {name} function "
                            f"once with these arguments: {json.dumps(planned_args, ensure_ascii=False)}. "
                            "Return a structured function tool call, not text."
                        ),
                    }
                ]
                response = self.client.chat.completions(
                    model=SARVAM_MODEL,
                    messages=call_messages,
                    tools=[selected_tool],
                    tool_choice="auto",
                    reasoning_effort=SARVAM_REASONING_EFFORT,
                    temperature=SARVAM_TEMPERATURE,
                    max_tokens=1024,
                )
                message = response.choices[0].message
                calls = message.tool_calls or []
                if len(calls) != 1 or calls[0].function.name != name:
                    raise ValueError(
                        f"Sarvam did not return {name}; finish_reason="
                        f"{response.choices[0].finish_reason}, calls="
                        f"{[call.function.name for call in calls]}, content="
                        f"{(message.content or '')[:300]}"
                    )
                call = calls[0]
                model_args = json.loads(call.function.arguments)
                args = self._args(run, name, model_args)
                if name == "start_customer_conversation":
                    if not run.policy or name not in run.policy["allowed_actions"]:
                        raise ValueError("Policy did not allow customer contact")
                with self.operations.store.lock:
                    if not self.is_active(run):
                        raise RunCancelled()
                    result = self.operations.dispatch(name, args)
                run.step += 1
                run.agent_steps += 1
                assistant_message = {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": call.id,
                            "type": "function",
                            "function": {"name": name, "arguments": json.dumps(args, ensure_ascii=False)},
                        }
                    ],
                }
                if message.content:
                    assistant_message["content"] = message.content
                run.messages.append(assistant_message)
                self._advance(run, name, result)
                if name == "start_customer_conversation":
                    run.pending_tool_call_id = call.id
                    run.status = "awaiting_conversation"
                    event = self._event(run, name, args, result, "awaiting_conversation")
                    event["messages"] = json.loads(json.dumps(run.messages, ensure_ascii=False))
                    yield event
                    return
                run.messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.id,
                        "content": json.dumps(result, ensure_ascii=False),
                    }
                )
                status = "completed" if name == "notify_owner" else "ok"
                if status == "completed":
                    run.status = "completed"
                event = self._event(run, name, args, result, status)
                if status == "completed" and self.on_complete:
                    try:
                        self.on_complete(run)
                    except OSError:
                        run.status = "error"
                        yield self.record_event(run, "trace", {}, {"error": "Could not save trace"}, "error")
                        return
                yield event
            except RunCancelled:
                run.status = "cancelled"
                yield self.record_event(run, name, {}, {"reason": "reset"}, "cancelled")
                return
            except Exception as exc:
                run.status = "error"
                detail = str(exc) if isinstance(exc, ValueError) else type(exc).__name__
                yield self.record_event(run, name, {}, {"error": detail}, "error")
                return

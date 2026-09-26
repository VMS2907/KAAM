"""Deterministic business tools used by the agent."""

from datetime import date, datetime

from store import DataStore, now_ist


class Operations:
    def __init__(self, store: DataStore):
        self.store = store

    def _customer(self, customer_id: str):
        return next(
            (c for c in self.store.data["customers"] if c["id"] == customer_id), None
        )

    def _invoice(self, invoice_id: str):
        return next((i for i in self.store.data["invoices"] if i["id"] == invoice_id), None)

    def _invoice_balance(self, invoice_id: str) -> float:
        invoice = self._invoice(invoice_id)
        if invoice is None:
            raise ValueError(f"Unknown invoice {invoice_id}")
        paid = sum(
            float(p["amount"])
            for p in self.store.data["payments"]
            if p["invoice_id"] == invoice_id
        )
        return round(float(invoice["amount"]) - paid, 2)

    def find_customer(self, name: str) -> dict:
        with self.store.lock:
            name = name.strip().casefold()
            matches = [
                c for c in self.store.data["customers"] if name in c["name"].casefold()
            ]
            if len(matches) != 1:
                raise ValueError(f"Expected one customer match; found {len(matches)}")
            return dict(matches[0])

    def get_ledger(self, customer_id: str) -> dict:
        with self.store.lock:
            if self._customer(customer_id) is None:
                raise ValueError(f"Unknown customer {customer_id}")
            invoices = [
                {**i, "balance": self._invoice_balance(i["id"])}
                for i in self.store.data["invoices"]
                if i["customer_id"] == customer_id
            ]
            invoice_ids = {i["id"] for i in invoices}
            payments = [
                dict(p)
                for p in self.store.data["payments"]
                if p["invoice_id"] in invoice_ids
            ]
            return {
                "customer_id": customer_id,
                "invoices": invoices,
                "payments": payments,
                "balance": round(sum(i["balance"] for i in invoices), 2),
            }

    def record_ledger_correction(
        self, invoice_id: str, amount: float, date: str, source: str, confidence: float
    ) -> dict:
        with self.store.lock:
            invoice = self._invoice(invoice_id)
            if invoice is None:
                raise ValueError(f"Unknown invoice {invoice_id}")
            amount = round(float(amount), 2)
            if amount <= 0:
                raise ValueError("Correction amount must be positive")
            date_value = date_from_iso(date)
            if date_value > now_ist().date():
                raise ValueError("Payment date cannot be in the future")
            if "khata" not in source.casefold():
                raise ValueError("A khata source citation is required")
            if not 0 <= float(confidence) <= 1:
                raise ValueError("Confidence must be between 0 and 1")
            before = self._invoice_balance(invoice_id)
            duplicate = next(
                (
                    p
                    for p in self.store.data["payments"]
                    if p["invoice_id"] == invoice_id
                    and float(p["amount"]) == amount
                    and p["date"] == date
                    and p["source"] == source
                ),
                None,
            )
            if duplicate:
                return {
                    "payment": dict(duplicate),
                    "before_balance": before,
                    "after_balance": before,
                    "already_recorded": True,
                }
            if amount > before:
                raise ValueError("Correction exceeds outstanding invoice balance")
            payment = {
                "id": f"PAY-{len(self.store.data['payments']) + 1:04d}",
                "invoice_id": invoice_id,
                "customer_id": invoice["customer_id"],
                "amount": amount,
                "date": date,
                "source": source,
                "confidence": float(confidence),
            }
            self.store.data["payments"].append(payment)
            self.store.save("payments")
            after = self._invoice_balance(invoice_id)
            if after == 0:
                invoice["status"] = "paid"
                self.store.save("invoices")
            return {
                "payment": dict(payment),
                "before_balance": before,
                "after_balance": after,
                "already_recorded": False,
            }

    def check_policy(self, days_overdue: int, balance: float, situation: str) -> dict:
        days_overdue = int(days_overdue)
        balance = round(float(balance), 2)
        situation_lower = situation.casefold()
        if "discount" in situation_lower:
            rule_id, actions = "R4", ["assign_task", "notify_owner"]
        elif "dispute" in situation_lower:
            rule_id, actions = "R5", [
                "collect_undisputed",
                "start_customer_conversation",
                "assign_task",
                "notify_owner",
            ]
        elif days_overdue > 30 or balance > 100000:
            rule_id, actions = "R3", ["assign_task", "notify_owner"]
        elif days_overdue >= 8:
            rule_id, actions = "R2", [
                "start_customer_conversation",
                "record_commitment",
                "schedule_followup",
                "notify_owner",
            ]
        else:
            rule_id, actions = "R1", ["voice_note_reminder", "notify_owner"]
        contact_allowed = now_ist().hour < 19
        if not contact_allowed:
            actions = [
                action for action in actions
                if action not in {"start_customer_conversation", "voice_note_reminder", "collect_undisputed"}
            ]
            if "schedule_followup" not in actions:
                actions.insert(0, "schedule_followup")
        return {
            "rule_id": rule_id,
            "allowed_actions": actions,
            "description": self.store.data["playbook"][rule_id],
            "matched_rules": [rule_id] + ([] if contact_allowed else ["R6"]),
            "contact_allowed": contact_allowed,
            "days_overdue": days_overdue,
            "balance": balance,
        }

    def start_customer_conversation(
        self, customer_id: str, language: str, objective: str, amount_due: float
    ) -> dict:
        with self.store.lock:
            customer = self._customer(customer_id)
            if customer is None:
                raise ValueError(f"Unknown customer {customer_id}")
            if now_ist().hour >= 19:
                raise ValueError("R6 prohibits customer contact after 7 PM India time")
            if language != customer["language"]:
                raise ValueError("Customer language does not match the customer record")
            actual_due = self.get_ledger(customer_id)["balance"]
            if round(float(amount_due), 2) != actual_due:
                raise ValueError("Conversation amount must equal the reconciled ledger")
            return {
                "status": "awaiting_conversation",
                "customer_id": customer_id,
                "contact": customer["contact"],
                "language": language,
                "objective": objective,
                "amount_due": actual_due,
            }

    def record_commitment(
        self, invoice_id: str, amount: float, promised_date: str
    ) -> dict:
        with self.store.lock:
            amount = round(float(amount), 2)
            if amount <= 0 or amount > self._invoice_balance(invoice_id):
                raise ValueError("Commitment amount must fit the current balance")
            if date_from_iso(promised_date) < now_ist().date():
                raise ValueError("Promised date cannot be in the past")
            existing = next(
                (
                    c
                    for c in self.store.data["commitments"]
                    if c["invoice_id"] == invoice_id
                    and c["amount"] == amount
                    and c["promised_date"] == promised_date
                ),
                None,
            )
            if existing:
                return {**existing, "already_recorded": True}
            commitment = {
                "id": f"COM-{len(self.store.data['commitments']) + 1:04d}",
                "invoice_id": invoice_id,
                "amount": amount,
                "promised_date": promised_date,
                "recorded_at": now_ist().isoformat(),
            }
            self.store.data["commitments"].append(commitment)
            self.store.save("commitments")
            return {**commitment, "already_recorded": False}

    def schedule_followup(self, invoice_id: str, datetime: str, condition: str) -> dict:
        with self.store.lock:
            if self._invoice(invoice_id) is None:
                raise ValueError(f"Unknown invoice {invoice_id}")
            due = parse_datetime(datetime)
            if due <= now_ist():
                raise ValueError("Follow-up must be in the future")
            followup = {
                "id": f"FOL-{len(self.store.data['followups']) + 1:04d}",
                "invoice_id": invoice_id,
                "datetime": due.isoformat(),
                "condition": condition,
                "status": "scheduled",
            }
            self.store.data["followups"].append(followup)
            self.store.save("followups")
            return dict(followup)

    def assign_task(
        self,
        member_id: str,
        task: str,
        reason: str,
        rule_id: str,
        evidence: str,
        due_at: str,
        depends_on: str | None = None,
    ) -> dict:
        with self.store.lock:
            member = next(
                (m for m in self.store.data["team"] if m["id"] == member_id), None
            )
            if member is None:
                raise ValueError(f"Unknown team member {member_id}")
            if rule_id not in self.store.data["playbook"]:
                raise ValueError(f"Unknown policy rule {rule_id}")
            due = parse_datetime(due_at)
            if due <= now_ist():
                raise ValueError("Task due time must be in the future")
            if rule_id in {"R3", "R4"} and member["role"] != "Finance Manager":
                raise ValueError(f"{rule_id} requires the Finance Manager")
            task_record = {
                "id": f"TASK-{len(self.store.data['tasks']) + 1:04d}",
                "member_id": member_id,
                "member_name": member["name"],
                "task": task,
                "reason": reason,
                "rule_id": rule_id,
                "evidence": evidence,
                "due_at": due.isoformat(),
                "depends_on": depends_on,
                "status": "assigned",
            }
            self.store.data["tasks"].append(task_record)
            self.store.save("tasks")
            return dict(task_record)

    def notify_owner(self, summary_en: str) -> dict:
        with self.store.lock:
            notification = {
                "id": f"NOT-{len(self.store.data['notifications']) + 1:04d}",
                "owner": self.store.data["company"]["owner"],
                "summary_en": summary_en,
                "created_at": now_ist().isoformat(),
                "status": "recorded",
            }
            self.store.data["notifications"].append(notification)
            self.store.save("notifications")
            return dict(notification)

    def dispatch(self, name: str, args: dict) -> dict:
        method = getattr(self, name, None)
        if name.startswith("_") or name not in {
            "find_customer",
            "get_ledger",
            "record_ledger_correction",
            "check_policy",
            "start_customer_conversation",
            "record_commitment",
            "schedule_followup",
            "assign_task",
            "notify_owner",
        }:
            raise ValueError(f"Unknown tool {name}")
        return method(**args)


def date_from_iso(value: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise ValueError("Date must be YYYY-MM-DD") from exc


def parse_datetime(value: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise ValueError("Datetime must be ISO 8601") from exc
    if parsed.tzinfo is None:
        raise ValueError("Datetime must include a timezone offset")
    return parsed

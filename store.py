"""Small JSON-backed store restored from clean seed files on startup."""

import json
import os
import re
import threading
from copy import deepcopy
from datetime import date, datetime, timedelta, timezone
from pathlib import Path


IST = timezone(timedelta(hours=5, minutes=30))


def today_ist() -> date:
    return datetime.now(IST).date()


def now_ist() -> datetime:
    return datetime.now(IST)


class DataStore:
    def __init__(self, directory: Path, seed_directory: Path):
        self.directory = directory
        self.seed_directory = seed_directory
        self.directory.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        self.data = {}
        self.reset()

    @staticmethod
    def _materialize(value):
        if isinstance(value, str):
            match = re.fullmatch(r"@TODAY-(\d+)D", value)
            if match:
                return (today_ist() - timedelta(days=int(match.group(1)))).isoformat()
            return value
        if isinstance(value, list):
            return [DataStore._materialize(item) for item in value]
        if isinstance(value, dict):
            return {key: DataStore._materialize(item) for key, item in value.items()}
        return value

    def reset(self) -> dict:
        with self.lock:
            paths = sorted(self.seed_directory.glob("*.json"))
            if not paths:
                raise RuntimeError("The seed folder has no JSON files")
            clean = {
                path.stem: self._materialize(json.loads(path.read_text(encoding="utf-8")))
                for path in paths
            }
            for path in self.directory.glob("*.json"):
                if path.stem not in clean:
                    path.unlink()
            for name, value in clean.items():
                self._write(self.directory / f"{name}.json", value)
            self.data = clean
            return {"status": "reset", "date": today_ist().isoformat(), "files": sorted(clean)}

    @staticmethod
    def _write(path: Path, value):
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        os.replace(temporary, path)

    def save(self, name: str):
        with self.lock:
            self._write(self.directory / f"{name}.json", self.data[name])

    def snapshot(self, name: str):
        with self.lock:
            return deepcopy(self.data[name])

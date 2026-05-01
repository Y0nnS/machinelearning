"""Runtime bridge for non-Python AI engines."""
from __future__ import annotations

import json
import math
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional


ROOT_DIR = Path(__file__).resolve().parents[2]
GO_TEXT_ENGINE = ROOT_DIR / "polyglot" / "go_text_ai"
TOKEN_PATTERN = re.compile(r"[a-zA-Z0-9_]+")

INTENTS: Dict[str, Dict[str, float]] = {
    "code": {
        "api": 1.3,
        "backend": 1.4,
        "frontend": 1.2,
        "bug": 1.4,
        "error": 1.2,
        "fix": 1.1,
        "function": 1.1,
        "model": 0.8,
        "python": 1.5,
        "rust": 1.5,
        "go": 1.4,
        "cpp": 1.5,
        "typescript": 1.3,
        "database": 1.1,
    },
    "learning": {
        "learn": 1.6,
        "belajar": 1.6,
        "explain": 1.4,
        "explanation": 1.4,
        "why": 1.1,
        "how": 1.1,
        "tutorial": 1.5,
        "konsep": 1.3,
        "materi": 1.2,
    },
    "idea": {
        "idea": 1.4,
        "design": 1.2,
        "plan": 1.2,
        "brainstorm": 1.5,
        "buat": 0.8,
        "build": 0.9,
        "fitur": 1.2,
        "produk": 1.1,
    },
    "data": {
        "data": 1.5,
        "dataset": 1.6,
        "csv": 1.4,
        "feature": 1.1,
        "train": 1.4,
        "accuracy": 1.3,
        "predict": 1.2,
        "classification": 1.2,
    },
}

SENTIMENT: Dict[str, Dict[str, float]] = {
    "positive": {
        "good": 1.0,
        "great": 1.3,
        "clean": 1.1,
        "simple": 1.0,
        "bagus": 1.2,
        "suka": 1.2,
        "mantap": 1.3,
        "cepat": 0.8,
    },
    "negative": {
        "bad": 1.0,
        "error": 0.8,
        "broken": 1.2,
        "susah": 1.0,
        "bingung": 1.1,
        "gagal": 1.2,
        "lambat": 1.0,
        "jelek": 1.2,
    },
}


@dataclass(frozen=True)
class EngineStatus:
    id: str
    name: str
    runtime: str
    status: str
    detail: str

    def to_dict(self) -> Dict[str, str]:
        return {
            "id": self.id,
            "name": self.name,
            "runtime": self.runtime,
            "status": self.status,
            "detail": self.detail,
        }


class PolyglotEngine:
    """Runs small native engines as isolated subprocesses."""

    def __init__(self, timeout_seconds: float = 8.0):
        self.timeout_seconds = timeout_seconds

    def go_status(self) -> EngineStatus:
        if not GO_TEXT_ENGINE.exists():
            return EngineStatus(
                id="go_text_ai",
                name="Go Text AI",
                runtime="Go",
                status="missing",
                detail="Engine source was not found.",
            )

        if shutil.which("go") is None:
            return EngineStatus(
                id="go_text_ai",
                name="Go Text AI",
                runtime="Go / Python fallback",
                status="fallback",
                detail="Install Go to run the native engine. Python fallback is active.",
            )

        return EngineStatus(
            id="go_text_ai",
            name="Go Text AI",
            runtime="Go",
            status="ready",
            detail="Local lexical classifier and response planner.",
        )

    def analyze_text(self, text: str) -> Dict[str, Any]:
        status = self.go_status()
        if status.status != "ready":
            return self._fallback_analyze(text, status.detail)

        payload = {"task": "analyze", "text": text}
        completed = subprocess.run(
            ["go", "run", "."],
            cwd=GO_TEXT_ENGINE,
            input=json.dumps(payload),
            capture_output=True,
            text=True,
            timeout=self.timeout_seconds,
            check=False,
        )

        if completed.returncode != 0:
            stderr = completed.stderr.strip() or "Go engine failed."
            raise RuntimeError(stderr)

        try:
            return json.loads(completed.stdout)
        except json.JSONDecodeError as exc:
            raise RuntimeError("Go engine returned invalid JSON.") from exc

    def _fallback_analyze(self, text: str, detail: str) -> Dict[str, Any]:
        tokens = [
            token
            for token in TOKEN_PATTERN.findall(text.lower())
            if len(token) >= 2
        ]
        intent_signals = self._score_groups(tokens, INTENTS)
        sentiment_signals = self._score_groups(tokens, SENTIMENT)

        label = "general"
        confidence = 0.45
        if intent_signals and intent_signals[0]["score"] > 0:
            label = intent_signals[0]["label"]
            confidence = min(
                0.96,
                max(0.05, 0.48 + math.tanh(intent_signals[0]["score"] / 4) * 0.42),
            )

        signals = [
            signal
            for signal in sorted(
                intent_signals + sentiment_signals,
                key=lambda item: item["score"],
                reverse=True,
            )
            if signal["score"] > 0
        ][:6]

        return {
            "engine": "go_text_ai",
            "runtime": "python_fallback",
            "native_available": False,
            "detail": detail,
            "label": label,
            "confidence": round(confidence, 3),
            "tokens": self._top_tokens(tokens),
            "signals": signals,
            "reply": self._build_reply(label, sentiment_signals),
        }

    @staticmethod
    def _score_groups(
        tokens: List[str],
        groups: Dict[str, Dict[str, float]],
    ) -> List[Dict[str, Any]]:
        counts: Dict[str, int] = {}
        for token in tokens:
            counts[token] = counts.get(token, 0) + 1

        signals: List[Dict[str, Any]] = []
        for label, weights in groups.items():
            score = 0.0
            reasons: List[str] = []
            for token, count in counts.items():
                if token in weights:
                    score += weights[token] * count
                    reasons.append(token)
            signals.append(
                {
                    "label": label,
                    "score": round(score, 3),
                    "reason": ", ".join(sorted(reasons)),
                }
            )

        return sorted(signals, key=lambda item: item["score"], reverse=True)

    @staticmethod
    def _top_tokens(tokens: List[str], limit: int = 14) -> List[str]:
        counts: Dict[str, int] = {}
        for token in tokens:
            counts[token] = counts.get(token, 0) + 1
        return sorted(counts, key=lambda token: (-counts[token], token))[:limit]

    @staticmethod
    def _build_reply(label: str, sentiment_signals: List[Dict[str, Any]]) -> str:
        tone = "neutral"
        if sentiment_signals and sentiment_signals[0]["score"] > 0:
            tone = sentiment_signals[0]["label"]

        if label == "code":
            if tone == "negative":
                return "Break the issue into input, process, and output, then test the smallest failing path."
            return "Start with a small endpoint or function, measure the output, then iterate."
        if label == "learning":
            return "Turn the topic into one concept, one example, and one small experiment."
        if label == "idea":
            return "Pick the smallest version that proves the core behavior."
        if label == "data":
            return "Check the dataset shape, target label, baseline score, then model changes."
        return "Ask a narrower question or add sample input for a stronger result."


polyglot_engine = PolyglotEngine()

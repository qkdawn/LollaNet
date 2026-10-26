from __future__ import annotations

import argparse
import csv
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
from threading import Lock
from typing import Any
from urllib.parse import urlparse


SCORE_FIELDS = ("formulaic_0_4", "specificity_0_4", "usefulness_0_4", "validity_0_4")
CSV_FIELDS = ("item_id", *SCORE_FIELDS, "exclude_0_1", "notes")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def validate_score_payload(payload: dict[str, Any]) -> dict[str, str]:
    result = {"item_id": str(payload.get("item_id", "")).strip()}
    if not result["item_id"]:
        raise ValueError("item_id is required")
    for field in SCORE_FIELDS:
        raw = payload.get(field)
        try:
            value = float(raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{field} must be numeric") from exc
        if value < 0 or value > 4 or value * 2 != round(value * 2):
            raise ValueError(f"{field} must be 0-4 in half-point increments")
        result[field] = f"{value:g}"
    exclude = payload.get("exclude_0_1", 0)
    if exclude is True:
        exclude = 1
    if exclude is False:
        exclude = 0
    try:
        exclude_value = int(exclude)
    except (TypeError, ValueError) as exc:
        raise ValueError("exclude_0_1 must be 0 or 1") from exc
    if exclude_value not in {0, 1}:
        raise ValueError("exclude_0_1 must be 0 or 1")
    result["exclude_0_1"] = str(exclude_value)
    result["notes"] = str(payload.get("notes", "")).strip()
    return result


class ScoreStore:
    def __init__(self, packet_path: Path, scores_path: Path, rubric_path: Path):
        self.packet_path = packet_path.resolve()
        self.scores_path = scores_path.resolve()
        self.rubric_path = rubric_path.resolve()
        self.lock = Lock()
        self.items = _read_jsonl(self.packet_path)
        item_ids = [str(row.get("item_id", "")) for row in self.items]
        if not item_ids or len(item_ids) != len(set(item_ids)):
            raise ValueError("annotation packet has empty or duplicate item IDs")
        if any(set(row) != {"item_id", "prompt", "response"} for row in self.items):
            raise ValueError("annotation packet has unexpected or provenance-leaking fields")
        self.item_ids = item_ids
        self.rubric = self.rubric_path.read_text(encoding="utf-8")
        self.scores = self._load_scores()

    def _load_scores(self) -> dict[str, dict[str, str]]:
        with self.scores_path.open(encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
        if not rows or tuple(rows[0]) != CSV_FIELDS:
            raise ValueError("score sheet columns differ from the frozen contract")
        observed = [str(row["item_id"]) for row in rows]
        if observed != self.item_ids:
            raise ValueError("score sheet order or coverage differs from the blind packet")
        return {str(row["item_id"]): {field: str(row.get(field, "")) for field in CSV_FIELDS} for row in rows}

    @staticmethod
    def _complete(row: dict[str, str]) -> bool:
        return all(str(row.get(field, "")).strip() != "" for field in (*SCORE_FIELDS, "exclude_0_1"))

    def state(self) -> dict[str, Any]:
        with self.lock:
            items = []
            for packet_row in self.items:
                item_id = str(packet_row["item_id"])
                score = self.scores[item_id]
                items.append({
                    **packet_row,
                    "scores": {field: score[field] for field in CSV_FIELDS if field != "item_id"},
                    "complete": self._complete(score),
                })
            return {
                "items": items,
                "completed_count": sum(bool(row["complete"]) for row in items),
                "total_count": len(items),
                "rubric": self.rubric,
            }

    def save(self, payload: dict[str, Any]) -> dict[str, Any]:
        value = validate_score_payload(payload)
        item_id = value["item_id"]
        with self.lock:
            if item_id not in self.scores:
                raise ValueError("item_id is outside this blind packet")
            self.scores[item_id] = value
            temporary = self.scores_path.with_suffix(self.scores_path.suffix + ".tmp")
            with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
                writer.writeheader()
                writer.writerows([self.scores[record_id] for record_id in self.item_ids])
            temporary.replace(self.scores_path)
            completed = sum(self._complete(row) for row in self.scores.values())
        return {"item_id": item_id, "saved": True, "completed_count": completed, "total_count": len(self.item_ids)}


def make_handler(store: ScoreStore, html_path: Path) -> type[BaseHTTPRequestHandler]:
    html = html_path.read_bytes()

    class Handler(BaseHTTPRequestHandler):
        server_version = "TemplateAttractorAnnotation/1.0"

        def _json(self, value: Any, status: HTTPStatus = HTTPStatus.OK) -> None:
            body = json.dumps(value, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:  # noqa: N802
            path = urlparse(self.path).path
            if path == "/":
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(html)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(html)
                return
            if path == "/api/state":
                self._json(store.state())
                return
            self._json({"error": "not found"}, HTTPStatus.NOT_FOUND)

        def do_POST(self) -> None:  # noqa: N802
            if urlparse(self.path).path != "/api/save":
                self._json({"error": "not found"}, HTTPStatus.NOT_FOUND)
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if length < 2 or length > 64 * 1024:
                    raise ValueError("request body size is invalid")
                payload = json.loads(self.rfile.read(length).decode("utf-8"))
                self._json(store.save(payload))
            except (ValueError, json.JSONDecodeError) as exc:
                self._json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)

        def log_message(self, format: str, *args: Any) -> None:
            return

    return Handler


def serve(packet: Path, scores: Path, rubric: Path, host: str, port: int) -> None:
    store = ScoreStore(packet, scores, rubric)
    html_path = Path(__file__).with_name("annotation_app.html")
    server = ThreadingHTTPServer((host, port), make_handler(store, html_path))
    print(f"annotation_app http://{host}:{port} completed={store.state()['completed_count']}/{len(store.item_ids)}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve one isolated blind annotation packet locally.")
    parser.add_argument("--packet", type=Path, required=True)
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--rubric", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    if args.host not in {"127.0.0.1", "localhost"}:
        parser.error("the blind annotation app must remain bound to localhost")
    serve(args.packet, args.scores, args.rubric, args.host, args.port)


if __name__ == "__main__":
    main()


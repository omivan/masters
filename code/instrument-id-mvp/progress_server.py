"""Tiny HTTP server that parses combo_exp logs and serves progress as JSON."""
import json, re, sys
from pathlib import Path
from http.server import BaseHTTPRequestHandler, HTTPServer

LOG   = Path(__file__).parent / "run_full.log"
ERR   = Path(__file__).parent / "run_full_err.log"
PORT  = 8765

CONFIGS = [
    "HarmonicCNN",
    "HarmonicCNN+Mixup",
    "HarmonicCNN+pitch2+SpecAug",
    "HarmonicCNN+CosineLR",
    "DeltaHarmonicCNN+CosineLR",
]

def parse_log():
    text = LOG.read_text(encoding="utf-8", errors="replace") if LOG.exists() else ""
    err  = ERR.read_text(encoding="utf-8", errors="replace") if ERR.exists() else ""

    # parse total seeds/epochs from header line
    m = re.search(r"\((\d+) seeds.*?(\d+) epochs\)", text)
    total_seeds  = int(m.group(1)) if m else 3
    total_epochs = int(m.group(2)) if m else 30

    results = {}
    for cfg in CONFIGS:
        seeds_done = [float(x) for x in re.findall(
            rf"\[{re.escape(cfg)} seed \d+\] F1-macro: ([0-9.]+)", text)]
        results[cfg] = seeds_done

    # current config running
    starts = re.findall(r"=== (.+?) \(", text)
    current_cfg = starts[-1].strip() if starts else None

    # current epoch from tqdm err (last "ep X/Y" line in stdout)
    ep_match = re.findall(r"\[.+? seed (\d+)\] ep\s+(\d+)/(\d+)", text)
    current_epoch = 0
    current_seed  = 0
    if ep_match:
        _, ep, total = ep_match[-1]
        current_epoch = int(ep)

    # also try tqdm stderr for epoch bar
    tqdm_ep = re.findall(r"ep \d+.*?(\d+)/(\d+) \[", err)
    if tqdm_ep:
        a, b = tqdm_ep[-1]
        current_epoch = max(current_epoch, int(a))

    # find current seed from log
    seed_starts = re.findall(r"\[.+? seed (\d+)\] F1", text)
    seeds_in_current = 0
    if current_cfg and current_cfg in results:
        seeds_in_current = len(results[current_cfg])

    # overall done
    total_runs = len(CONFIGS) * total_seeds
    done_runs  = sum(len(v) for v in results.values())

    finished = "Final comparison" in text or "Best:" in text

    # best result so far
    best_cfg, best_f1 = None, 0.0
    for cfg, scores in results.items():
        if scores:
            avg = sum(scores) / len(scores)
            if avg > best_f1:
                best_f1, best_cfg = avg, cfg

    return {
        "configs": CONFIGS,
        "results": results,
        "current_cfg": current_cfg,
        "current_epoch": current_epoch,
        "total_epochs": total_epochs,
        "total_seeds": total_seeds,
        "done_runs": done_runs,
        "total_runs": total_runs,
        "finished": finished,
        "best_cfg": best_cfg,
        "best_f1": round(best_f1, 4),
        "seeds_in_current": seeds_in_current,
    }

class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        data = json.dumps(parse_log()).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(data)
    def log_message(self, *_): pass  # silence request logs

print(f"Progress server → http://localhost:{PORT}", flush=True)
HTTPServer(("", PORT), Handler).serve_forever()

# -*- coding: utf-8 -*-
"""
ComfyUI per-run timing monitor.

Tracks every completed generation by tailing ComfyUI's log, pairs it with GPU
telemetry, and separates "clean" runs from runs that paid a model-staging cost.

Why this exists (two non-obvious traps):

1. `got prompt` is NOT a valid start anchor. It fires when the HTTP request
   arrives, which can be minutes before execution starts when jobs are queued.
   Use `Prompt executed in X seconds` instead - each generation emits exactly one.

2. Almost every run prints `prepared for dynamic VRAM loading` for the main
   model and both VAEs, because they are re-staged before decode. Those are
   routine. The only reliable signal that the *resolution* changed is
   `MiniMaxH3TEModel` being re-staged. Use that to flag switch runs.

Usage:
    python monitor.py --log D:/ComfyUI/user/comfyui.log --out ./run.csv

    # stop automatically after N runs, or after 15 min of silence
    python monitor.py --max-runs 6 --idle-stop 900
"""
import argparse
import asyncio
import csv
import datetime
import os
import re
import sys
import time

TQDM_RE = re.compile(r"(\d+)/(\d+) \[(\d+:[\d:]+)<[^,]*, ([\d.]+)s/it\]")
EXEC_RE = re.compile(r"in ([\d.]+) seconds")
STAGE_RE = re.compile(r"Model (\S+) prepared for dynamic VRAM loading")

# A run during which any of these was re-staged counts as a resolution switch.
SWITCH_KEYS = ("TEModel",)


def ts():
    return datetime.datetime.now().strftime("%H:%M:%S")


def hms(t):
    parts = [int(x) for x in t.split(":")]
    if len(parts) == 2:
        return parts[0] * 60 + parts[1]
    return parts[0] * 3600 + parts[1] * 60 + parts[2]


def start_offset(path):
    """Resume after the last completed run so an in-flight run keeps its tqdm."""
    try:
        size = os.path.getsize(path)
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            f.seek(max(0, size - 600_000))
            tail = f.read()
        i = tail.rfind("Prompt executed")
        if i < 0:
            return size
        j = tail.find("\n", i)
        return max(0, size - 600_000) + (j + 1 if j > 0 else len(tail))
    except OSError:
        return 0


async def read_gpu():
    try:
        p = await asyncio.create_subprocess_exec(
            "nvidia-smi",
            "--query-gpu=memory.used,utilization.gpu,temperature.gpu,power.draw",
            "--format=csv,noheader,nounits",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        out, _ = await p.communicate()
        v = [x.strip() for x in out.decode("utf-8", "ignore").strip().split(",")]
        return v if len(v) >= 4 else None
    except Exception:
        return None


class Monitor:
    def __init__(self, args):
        self.args = args
        self.runs = []
        self.pend = {"sample": None, "steps": None, "sic": None, "staged": []}
        self.last_event = time.time()
        self.peak = {"mem": 0, "temp": 0.0, "power": 0.0, "util": 0}
        self.rows = []

    # -- GPU sampling -------------------------------------------------------
    async def sample_gpu(self, stop):
        interval = self.args.gpu_interval
        while not stop.is_set():
            v = await read_gpu()
            if v:
                try:
                    mem = int(float(v[0]))
                    util = int(float(v[1]))
                    self.peak["mem"] = max(self.peak["mem"], mem)
                    self.peak["util"] = max(self.peak["util"], util)
                    self.peak["temp"] = max(self.peak["temp"], float(v[2]))
                    self.peak["power"] = max(self.peak["power"], float(v[3]))
                    self.rows.append([ts(), mem, util, v[2], v[3]])
                except ValueError:
                    pass
            await asyncio.sleep(interval)

    # -- log tail -----------------------------------------------------------
    async def tail_log(self, stop, finished):
        path = self.args.log
        pos = start_offset(path)
        while not stop.is_set():
            try:
                if os.path.exists(path):
                    size = os.path.getsize(path)
                    if size > pos:
                        with open(path, "r", encoding="utf-8", errors="ignore") as f:
                            f.seek(pos)
                            chunk = f.read()
                            pos = f.tell()
                        self.parse(chunk, finished)
                    elif size < pos:          # log truncated / rotated
                        pos = 0
            except OSError:
                pass
            await asyncio.sleep(1.0)

    def parse(self, chunk, finished):
        for raw in chunk.replace("\r", "\n").splitlines():
            s = raw.strip()
            if not s:
                continue
            m = re.match(r"\[(\d{4}-\d\d-\d\d [\d:.]+)\]\s*(.*)", s)
            body = m.group(2) if m else s
            stamp = m.group(1)[11:19] if m else ts()

            g = TQDM_RE.search(body)
            if g:
                self.pend["steps"] = "%s/%s" % (g.group(1), g.group(2))
                self.pend["sic"] = float(g.group(4))
                self.pend["sample"] = hms(g.group(3))
                self.last_event = time.time()
                continue

            if "prepared for dynamic VRAM loading" in body:
                nm = STAGE_RE.search(body)
                self.pend["staged"].append(nm.group(1) if nm else "?")
                self.last_event = time.time()
                continue

            if "Prompt executed" in body:
                mm = EXEC_RE.search(body)
                total = float(mm.group(1)) if mm else None
                switch = [x for x in self.pend["staged"]
                          if any(k in x for k in SWITCH_KEYS)]
                run = {
                    "n": len(self.runs) + 1,
                    "stamp": stamp,
                    "total": total,
                    "sample": self.pend["sample"],
                    "steps": self.pend["steps"],
                    "sic": self.pend["sic"],
                    "switch": bool(switch),
                    "staged": list(self.pend["staged"]),
                }
                self.runs.append(run)
                self.report(run)
                self.pend = {"sample": None, "steps": None, "sic": None, "staged": []}
                self.last_event = time.time()
                if self.args.max_runs and len(self.runs) >= self.args.max_runs:
                    finished.set()

    def report(self, r):
        tag = "  [SWITCH - exclude]" if r["switch"] else ""
        print("[%s] run %d  total %s s | sampling %s s | %s | %s s/step%s"
              % (r["stamp"], r["n"],
                 "%.2f" % r["total"] if r["total"] else "-",
                 r["sample"] or "-", r["steps"] or "-", r["sic"] or "-", tag))
        if r["switch"]:
            print("        re-staged: %s" % ", ".join(r["staged"]))

    # -- summary ------------------------------------------------------------
    def summary(self):
        clean = [r for r in self.runs if not r["switch"] and r["total"]]
        print("\n" + "=" * 68)
        print("run   total(s)  sampling(s)  s/step   note")
        for r in self.runs:
            note = "switch" if r["switch"] else ""
            if r["total"] and r["total"] < 1:
                note = "cache hit"
            print("%-4d %9s %12s %8s   %s"
                  % (r["n"],
                     "%.2f" % r["total"] if r["total"] else "-",
                     r["sample"] or "-", r["sic"] or "-", note))
        if clean:
            v = [r["total"] for r in clean]
            print("\nclean runs: n=%d  mean %.2f s  range %.2f - %.2f s"
                  % (len(v), sum(v) / len(v), min(v), max(v)))
            s = [r["sample"] for r in clean if r["sample"]]
            if s:
                print("mean sampling: %.1f s" % (sum(s) / len(s)))
        print("peak VRAM %d MiB | peak temp %.1f C | peak power %.1f W"
              % (self.peak["mem"], self.peak["temp"], self.peak["power"]))
        print("=" * 68)

    def write_csv(self, path):
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["time", "gpu_mem_MiB", "gpu_util_pct", "temp_C", "power_W"])
            w.writerows(self.rows)
        print("telemetry -> %s (%d samples)" % (path, len(self.rows)))


async def main(args):
    mon = Monitor(args)
    stop, finished = asyncio.Event(), asyncio.Event()

    tasks = [asyncio.create_task(mon.sample_gpu(stop)),
             asyncio.create_task(mon.tail_log(stop, finished))]

    async def guard():
        while not stop.is_set():
            await asyncio.sleep(2)
            if time.time() - mon.last_event > args.idle_stop:
                print("\n[%s] idle %d s, stopping" % (ts(), args.idle_stop))
                finished.set()
                return
            if len(mon.runs) >= (args.max_runs or 10 ** 9):
                return

    tasks.append(asyncio.create_task(guard()))
    await finished.wait()
    await asyncio.sleep(5)
    stop.set()
    for t in tasks:
        t.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)

    mon.summary()
    if args.out:
        mon.write_csv(args.out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", default=os.environ.get(
        "COMFY_LOG", "D:/ComfyUI/user/comfyui.log"), help="path to comfyui.log")
    ap.add_argument("--out", default="run.csv", help="telemetry CSV output")
    ap.add_argument("--gpu-interval", type=float, default=1.5)
    ap.add_argument("--idle-stop", type=int, default=900)
    ap.add_argument("--max-runs", type=int, default=0)
    a = ap.parse_args()
    if not os.path.exists(a.log):
        sys.exit("log not found: %s  (pass --log)" % a.log)
    try:
        asyncio.run(main(a))
    except KeyboardInterrupt:
        pass

import argparse, hashlib, json, subprocess, time
from pathlib import Path
import pandas as pd

from ccd_config import (PROMPT_CODES, CODES, MANIFEST, WINDOW_DIR, RUNS_DIR, MODEL_REGISTRY, WINDOW_SEC, SNAP_TOL_SEC,
                        NUM_FRAMES, IMAGE_SIZE, MAX_NEW_TOKENS, FEW_SHOT_K, load_few_shot)
from prompt import build_prompt, prompt_hash
from runners import get_runner, parse_counts, parse_trace, counts_from_trace
import metrics

RETRY_SUFFIX = "\n\nYour previous answer was not a valid JSON object. Return only the JSON object."
TRACE_RETRY_SUFFIX = "\n\nYour previous answer contained no valid turn lines. Output one JSON object per turn as specified."
TRACE_MAX_NEW_TOKENS = 4096


def code_hash():
    h = hashlib.sha256()
    for f in ("ccd_config.py", "prompt.py", "runners.py", "run_eval.py", "few_shot_examples.json"):
        h.update((Path(__file__).parent / f).read_bytes())
    return h.hexdigest()[:12]


def git_hash():
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=Path(__file__).parent,
                              capture_output=True, text=True).stdout.strip() or None
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(MODEL_REGISTRY))
    ap.add_argument("--few-shot", action="store_true")
    ap.add_argument("--use-audio", action="store_true", help="also give the model the audio of each window (audio-capable backends only; ignored otherwise)")
    ap.add_argument("--device", default="cuda:0"); ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--cids", nargs="*", help="restrict the run to these couple IDs")
    ap.add_argument("--trace", action="store_true", help="trace mode: one line per speaker turn with its text and a confidence for every code")
    a = ap.parse_args()

    spec = MODEL_REGISTRY[a.model]
    setting = "few" if a.few_shot else "zero"
    run_name = f"{a.model}__{setting}" + ("__audio" if a.use_audio else "") + ("__trace" if a.trace else "")
    max_new = TRACE_MAX_NEW_TOKENS if a.trace else MAX_NEW_TOKENS
    run_dir = RUNS_DIR / run_name; run_dir.mkdir(parents=True, exist_ok=True)
    raw_path = run_dir / "raw_outputs.jsonl"

    inputs_note = ("16 frames sampled uniformly across the segment, the audio of the segment, and the speech transcript of the segment"
                   if a.use_audio else None)
    phash = prompt_hash(a.few_shot, trace=a.trace, inputs_note=inputs_note)

    cfg_path = run_dir / "config.json"
    if cfg_path.exists() and raw_path.exists() and raw_path.stat().st_size > 0:
        prev = json.loads(cfg_path.read_text(encoding="utf-8"))
        cur = dict(prompt_hash=phash, model=a.model, use_audio=a.use_audio,
                   window_sec=WINDOW_SEC, num_frames=NUM_FRAMES, max_new_tokens=max_new)
        old = dict(prompt_hash=prev.get("prompt_hash"), model=prev.get("model"), use_audio=prev.get("use_audio"),
                   window_sec=prev.get("protocol", {}).get("window_sec"), num_frames=prev.get("protocol", {}).get("num_frames"),
                   max_new_tokens=prev.get("protocol", {}).get("max_new_tokens"))
        if cur != old:
            diff = {k: (old[k], cur[k]) for k in cur if cur[k] != old[k]}
            raise SystemExit(f"run '{run_name}' already has outputs from a different configuration {diff}. "
                             f"Delete {run_dir} to start over.")

    config = dict(model=a.model, backend=spec["backend"], hf_path=spec["hf_path"], setting=setting, use_audio=a.use_audio,
                  prompt_hash=phash, trace=a.trace, code_hash=code_hash(), git=git_hash(),
                  protocol=dict(window_sec=WINDOW_SEC, snap_tol_sec=SNAP_TOL_SEC, num_frames=NUM_FRAMES, image_size=IMAGE_SIZE,
                                max_new_tokens=max_new, decoding="greedy", few_shot_k=(FEW_SHOT_K or len(load_few_shot())) if a.few_shot else 0,
                                codes=CODES, inputs=inputs_note or "16 frames + transcript"),
                  started=time.strftime("%Y-%m-%d %H:%M:%S"), device=a.device, dtype=a.dtype)
    (run_dir / "config.json").write_text(json.dumps(config, indent=1))

    done, done_counts = set(), {}
    if raw_path.exists():
        for line in raw_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line); done.add((r["CID"], r["video_id"], r["window"])); done_counts[(r["CID"], r["video_id"], r["window"])] = r["counts"]
    print(f"run {run_name}: {len(done)} windows already done")

    m = pd.read_csv(MANIFEST, dtype={"CID": str})
    if a.cids: m = m[m.CID.isin(a.cids)]

    runner = get_runner(spec["backend"], spec["hf_path"], a.device, a.dtype)
    runner.max_new_tokens = max_new
    if a.use_audio and not runner.supports_audio:
        print(f"WARNING: {a.model} does not accept audio; --use-audio ignored")

    n_calls = n_fail = 0
    with open(raw_path, "a", encoding="utf-8") as raw:
        for vi, r in enumerate(m.itertuples(), 1):
            vdir = WINDOW_DIR / r.CID / f"video_{r.video_id}"
            wj = vdir / "windows.json"
            if not wj.exists():
                print(f"  {r.CID}/video_{r.video_id}: windows missing, skipped"); continue
            windows = json.loads(wj.read_text(encoding="utf-8"))
            t0 = time.time(); vid_counts = []
            for w in windows:
                key = (r.CID, int(r.video_id), w["window"])
                if key in done: vid_counts.append(done_counts.get(key)); continue
                wdir = vdir / f"w{w['window']:02d}"
                frames = sorted((wdir / "frames").glob("f*.jpg"))
                transcript = (wdir / "transcript.txt").read_text(encoding="utf-8")
                prompt = build_prompt(transcript, w["window"], w["n_windows"], w["start"], w["end"], a.few_shot, inputs_note, trace=a.trace)
                text = runner.generate(prompt, frames, wdir / "clip.mp4", use_audio=a.use_audio)
                retried = False
                if a.trace:
                    rows0, _ = parse_trace(text)
                    if not rows0:
                        retried = True
                        text = runner.generate(prompt + TRACE_RETRY_SUFFIX, frames, wdir / "clip.mp4", use_audio=a.use_audio)
                        rows0, _ = parse_trace(text)
                    counts = counts_from_trace(rows0) if rows0 else None
                else:
                    counts = parse_counts(text)
                    if counts is None:
                        retried = True
                        text = runner.generate(prompt + RETRY_SUFFIX, frames, wdir / "clip.mp4", use_audio=a.use_audio)
                        counts = parse_counts(text)
                n_calls += 1; n_fail += counts is None; vid_counts.append(counts)
                rec = {"CID": r.CID, "video_id": int(r.video_id), "window": w["window"], "start": w["start"],
                       "end": w["end"], "n_frames": len(frames), "n_words": w["n_words"], "retried": retried,
                       "raw": text, "counts": counts}
                if a.trace:
                    rows, bad = parse_trace(text); rec["trace"] = rows; rec["trace_bad_lines"] = bad
                raw.write(json.dumps(rec, ensure_ascii=False) + "\n"); raw.flush()
            vid_total = {c: sum(d[c] for d in vid_counts if isinstance(d, dict)) for c in CODES}
            gold = {c: int(getattr(r, c)) for c in CODES}
            print(f"[{vi}/{len(m)}] {r.CID}/video_{r.video_id}: {len(windows)} windows, {time.time()-t0:.0f}s "
                  f"(cumulative parse failures {n_fail}/{n_calls})", flush=True)
            print(f"    pred {json.dumps(vid_total)}", flush=True)
            print(f"    gold {json.dumps(gold)}", flush=True)

    rows = [json.loads(l) for l in raw_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    wdf = pd.DataFrame(rows)
    for c in CODES:
        wdf[f"pred_{c}"] = wdf["counts"].apply(lambda d: d[c] if isinstance(d, dict) else None)
    wdf["parse_failed"] = wdf["counts"].isna()
    wdf.drop(columns=["counts"]).to_csv(run_dir / "window_predictions.csv", index=False)

    agg = wdf.groupby(["CID", "video_id"]).agg(n_windows=("window", "size"), n_failed_windows=("parse_failed", "sum"),
                                              **{f"pred_{c}": (f"pred_{c}", "sum") for c in CODES}).reset_index()
    agg["CID"] = agg["CID"].astype(str)
    pred = m.merge(agg, on=["CID", "video_id"], how="inner")
    pred = pred.rename(columns={c: f"gold_{c}" for c in CODES})
    pred.to_csv(run_dir / "predictions.csv", index=False)
    print(f"\npredictions.csv: {len(pred)} videos, windows with parse failure: {int(agg.n_failed_windows.sum())}/{int(agg.n_windows.sum())}")

    metrics.evaluate_run(run_dir)
    config["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    (run_dir / "config.json").write_text(json.dumps(config, indent=1))


if __name__ == "__main__":
    main()

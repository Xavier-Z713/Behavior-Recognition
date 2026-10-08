import argparse, json
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats
from ccd_config import CODES, RUNS_DIR


def _metrics(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    out = dict(n=int(len(y)), MAE=float(np.abs(y - p).mean()), RMSE=float(np.sqrt(((y - p) ** 2).mean())))
    if np.std(p) > 0 and np.std(y) > 0:
        r, pr = stats.pearsonr(y, p); rho, prho = stats.spearmanr(y, p)
        out.update(pearson_r=float(r), p_r=float(pr), spearman_rho=float(rho), p_rho=float(prho))
    else:
        out.update(pearson_r=np.nan, p_r=np.nan, spearman_rho=np.nan, p_rho=np.nan)
    out["constant_prediction"] = bool(np.std(p) == 0)
    return out


def evaluate_run(run_dir: Path):
    run_dir = Path(run_dir)
    d = pd.read_csv(run_dir / "predictions.csv", dtype={"CID": str})
    G = d[[f"gold_{c}" for c in CODES]].to_numpy(float); P = d[[f"pred_{c}" for c in CODES]].to_numpy(float)

    rows = [dict(code=c, **_metrics(G[:, i], P[:, i])) for i, c in enumerate(CODES)]
    for i, c in enumerate(CODES):
        y = G[:, i]
        rows.append(dict(code=c, model="baseline_zeros", **_metrics(y, np.zeros_like(y))))
        rows.append(dict(code=c, model="baseline_mean", **_metrics(y, np.full_like(y, y.mean()))))
    per = pd.DataFrame(rows); per["model"] = per["model"].fillna("model")
    per = per[["model", "code", "n", "MAE", "RMSE", "spearman_rho", "p_rho", "pearson_r", "p_r", "constant_prediction"]]
    per.to_csv(run_dir / "per_code_metrics.csv", index=False)

    pooled = []
    for name, PP in (("model", P), ("baseline_zeros", np.zeros_like(G)), ("baseline_mean", np.tile(G.mean(0), (len(G), 1)))):
        mm = _metrics(G.ravel(), PP.ravel()); mm.pop("p_r"); mm.pop("p_rho")
        pooled.append(dict(model=name, n_pairs=int(G.size), **mm))
    pd.DataFrame(pooled).to_csv(run_dir / "pooled_metrics.csv", index=False)

    model_rows = per[per.model == "model"]
    summary = dict(
        n_videos=int(len(d)), n_windows=int(d.n_windows.sum()), n_failed_windows=int(d.n_failed_windows.sum()),
        codes_with_constant_prediction=[c for c in CODES if model_rows.set_index("code").loc[c, "constant_prediction"]],
        codes_beating_zeros_on_RMSE=[c for c in CODES if model_rows.set_index("code").loc[c, "RMSE"]
                                     < per[(per.model == "baseline_zeros") & (per.code == c)].RMSE.iloc[0]],
        codes_with_rho_p_below_05=[c for c in CODES if model_rows.set_index("code").loc[c, "p_rho"] < 0.05],
        median_per_code_rho=float(np.nanmedian(model_rows.spearman_rho)),
        pooled=pooled[0],
    )
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=1))
    print(f"\n=== {run_dir.name} ===")
    print(model_rows[["code", "MAE", "RMSE", "spearman_rho", "p_rho", "constant_prediction"]].round(3).to_string(index=False))
    print("codes beating all-zeros RMSE:", summary["codes_beating_zeros_on_RMSE"], "| rho p<.05:", summary["codes_with_rho_p_below_05"])
    print("pooled (secondary):", {k: round(v, 4) for k, v in pooled[0].items() if isinstance(v, float)})
    return summary


def collect():
    rows = []
    for run in sorted(RUNS_DIR.iterdir()):
        f = run / "per_code_metrics.csv"
        if not f.exists(): continue
        cfg = json.loads((run / "config.json").read_text()); per = pd.read_csv(f)
        for _, r in per[per.model == "model"].iterrows():
            rows.append(dict(run=run.name, model=cfg["model"], setting=cfg["setting"], audio=cfg.get("use_audio", False),
                             code=r.code, MAE=r.MAE, RMSE=r.RMSE, rho=r.spearman_rho, p_rho=r.p_rho, constant=r.constant_prediction))
    out = pd.DataFrame(rows); out.to_csv(RUNS_DIR / "summary_all_runs.csv", index=False)
    if len(out):
        piv = out.pivot_table(index="code", columns=["model", "setting"], values="rho").round(2)
        print("Spearman rho per code (rows) x model/setting (cols):\n", piv.to_string())
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("run_dir", nargs="?"); ap.add_argument("--collect", action="store_true")
    a = ap.parse_args()
    if a.collect: collect()
    elif a.run_dir: evaluate_run(Path(a.run_dir))
    else: ap.print_help()

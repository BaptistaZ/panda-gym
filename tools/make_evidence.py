#!/usr/bin/env python3
# tools/make_evidence.py (OVERLAY-ONLY)

import argparse
import csv
import hashlib
import json
import math
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Matplotlib headless
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# -------------------------
# Utils / Hash / Git / Env
# -------------------------
def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def try_cmd(cmd: List[str], cwd: Optional[Path] = None) -> Optional[str]:
    try:
        out = subprocess.check_output(cmd, cwd=str(cwd) if cwd else None, stderr=subprocess.DEVNULL)
        s = out.decode().strip()
        return s if s else None
    except Exception:
        return None


def try_git_info(repo_root: Path) -> Dict[str, Optional[str]]:
    return {
        "commit": try_cmd(["git", "rev-parse", "HEAD"], cwd=repo_root),
        "status_porcelain": try_cmd(["git", "status", "--porcelain"], cwd=repo_root),
    }


def try_pip_freeze() -> Optional[str]:
    return try_cmd([sys.executable, "-m", "pip", "freeze"])


# -------------------------
# Import eval module
# -------------------------
def load_eval_module(repo_root: Path):
    import importlib.util
    eval_path = repo_root / "scripts" / "teacher" / "reach" / "eval_reach_ppo.py"
    if not eval_path.exists():
        raise FileNotFoundError(f"Não encontrei {eval_path}")

    spec = importlib.util.spec_from_file_location("eval_reach_ppo", str(eval_path))
    if spec is None or spec.loader is None:
        raise RuntimeError("Falha a carregar o módulo scripts/teacher/reach/eval_reach_ppo.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["eval_reach_ppo"] = mod
    spec.loader.exec_module(mod)

    if not hasattr(mod, "eval_model"):
        raise RuntimeError("eval_reach_ppo.py não expõe eval_model(...)")
    return mod


# -------------------------
# Runs / TB scalars
# -------------------------
def find_latest_run_dir(run_root: Path) -> Optional[Path]:
    if not run_root.exists():
        return None
    candidates = [p for p in run_root.iterdir() if p.is_dir()]
    if not candidates:
        return None
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def list_event_files(run_dir: Path) -> List[Path]:
    if not run_dir.exists():
        return []
    return sorted(
        [p for p in run_dir.glob("events.out.tfevents.*") if p.is_file()],
        key=lambda p: p.stat().st_mtime,
    )


def extract_tb_scalars(run_dir: Path) -> Dict[str, List[Tuple[int, float]]]:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    ea = EventAccumulator(str(run_dir), size_guidance={"scalars": 0})
    ea.Reload()

    out: Dict[str, List[Tuple[int, float]]] = {}
    for tag in ea.Tags().get("scalars", []):
        events = ea.Scalars(tag)
        out[tag] = [(int(e.step), float(e.value)) for e in events]
    return out


def save_scalars_csv_long(scalars: Dict[str, List[Tuple[int, float]]], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["step", "tag", "value"])
        for tag, pts in scalars.items():
            for step, val in pts:
                w.writerow([step, tag, val])


def save_scalar_tags_txt(scalars: Dict[str, List[Tuple[int, float]]], out_txt: Path) -> None:
    out_txt.parent.mkdir(parents=True, exist_ok=True)
    tags = sorted(list(scalars.keys()))
    out_txt.write_text("\n".join(tags) + ("\n" if tags else ""), encoding="utf-8")


def get_series(scalars: Dict[str, List[Tuple[int, float]]], tag: str) -> Tuple[List[int], List[float]]:
    pts = scalars.get(tag, [])
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    return xs, ys


# -------------------------
# Milestones / Metrics
# -------------------------
def first_stable_step(xs: List[int], ys: List[float], threshold: float, mode: str = "ge") -> Optional[int]:
    if not xs or not ys or len(xs) != len(ys):
        return None

    def cond(v: float) -> bool:
        return (v >= threshold) if mode == "ge" else (v <= threshold)

    for i in range(len(ys)):
        if all(cond(v) for v in ys[i:]):
            return xs[i]
    return None


def last_value(xs: List[int], ys: List[float]) -> Optional[Tuple[int, float]]:
    if not xs or not ys:
        return None
    return xs[-1], ys[-1]


# -------------------------
# Plot helpers (OVERLAY ONLY)
# -------------------------
def plot_overlay(series: Dict[str, Tuple[List[int], List[float]]], out_png: Path, title: str) -> bool:
    """
    series: label -> (xs, ys)
    """
    ok_any = False
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.figure()
    for label, (xs, ys) in series.items():
        if xs:
            plt.plot(xs, ys, label=label)
            ok_any = True
    if not ok_any:
        plt.close()
        return False
    plt.title(title)
    plt.xlabel("timesteps")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=180)
    plt.close()
    return True


# -------------------------
# Evaluation
# -------------------------
@dataclass
class EvalRow:
    eval_type: str    # deterministic | stochastic
    mode: str         # ee | joints
    env_id: str
    checkpoint: str
    model_path: str
    seed: int
    episodes: int
    avg_return: float
    success_rate: float
    avg_len: float


def discover_checkpoints(model_dir: Path) -> List[str]:
    zips = sorted([p.name for p in model_dir.glob("*.zip") if p.is_file()])

    def key(name: str):
        m = re.search(r"ckpt(\d+)", name)
        if m:
            return (0, int(m.group(1)))
        if name.startswith("mid"):
            return (1, 0)
        if name.startswith("final"):
            return (2, 0)
        return (3, name)

    return sorted(zips, key=key)


def eval_models(
    eval_mod,
    eval_type: str,
    mode: str,
    env_id: str,
    model_dir: Path,
    checkpoints: List[str],
    seeds: List[int],
    episodes: int,
    deterministic: bool,
) -> List[EvalRow]:
    rows: List[EvalRow] = []
    for ck in checkpoints:
        mp = model_dir / ck
        if not mp.exists():
            continue
        for s in seeds:
            res = eval_mod.eval_model(
                env_id=env_id,
                model_path=str(mp),
                episodes=episodes,
                seed=s,
                deterministic=deterministic,
            )
            rows.append(
                EvalRow(
                    eval_type=eval_type,
                    mode=mode,
                    env_id=env_id,
                    checkpoint=ck,
                    model_path=str(mp),
                    seed=s,
                    episodes=episodes,
                    avg_return=float(res["avg_return"]),
                    success_rate=float(res["success_rate"]),
                    avg_len=float(res["avg_len"]),
                )
            )
    return rows


def write_eval_csv(rows: List[EvalRow], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "eval_type", "mode", "env_id", "checkpoint", "model_path", "seed", "episodes",
            "avg_return", "success_rate", "avg_len"
        ])
        for r in rows:
            w.writerow([
                r.eval_type, r.mode, r.env_id, r.checkpoint, r.model_path, r.seed, r.episodes,
                r.avg_return, r.success_rate, r.avg_len
            ])


def agg_eval(rows: List[EvalRow]) -> Dict[Tuple[str, str, str], Dict[str, float]]:
    buckets: Dict[Tuple[str, str, str], List[EvalRow]] = {}
    for r in rows:
        buckets.setdefault((r.eval_type, r.mode, r.checkpoint), []).append(r)

    def mean(vals):
        return sum(vals) / len(vals) if vals else float("nan")

    def std(vals):
        if len(vals) < 2:
            return 0.0
        m = mean(vals)
        return math.sqrt(sum((v - m) ** 2 for v in vals) / (len(vals) - 1))

    def vmin(vals):
        return min(vals) if vals else float("nan")

    def vmax(vals):
        return max(vals) if vals else float("nan")

    out: Dict[Tuple[str, str, str], Dict[str, float]] = {}
    for key, rs in buckets.items():
        rets = [r.avg_return for r in rs]
        succ = [r.success_rate for r in rs]
        lens = [r.avg_len for r in rs]
        out[key] = {
            "n": float(len(rs)),
            "return_mean": mean(rets),
            "return_std": std(rets),
            "return_min": vmin(rets),
            "return_max": vmax(rets),
            "success_mean": mean(succ),
            "success_std": std(succ),
            "success_min": vmin(succ),
            "success_max": vmax(succ),
            "len_mean": mean(lens),
            "len_std": std(lens),
            "len_min": vmin(lens),
            "len_max": vmax(lens),
        }
    return out


def write_eval_summary_md(agg: Dict[Tuple[str, str, str], Dict[str, float]], out_md: Path) -> None:
    out_md.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted(agg.keys(), key=lambda k: (k[0], k[1], k[2]))
    lines = []
    lines.append("# Sumário de avaliação (robusto)\n")
    lines.append("| eval | modo | checkpoint | n | return (m±sd) | return [min,max] | success (m±sd) | success [min,max] | len (m±sd) | len [min,max] |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for (eval_type, mode, ck) in keys:
        s = agg[(eval_type, mode, ck)]
        lines.append(
            f"| {eval_type} | {mode} | {ck} | {int(s['n'])} | "
            f"{s['return_mean']:.3f}±{s['return_std']:.3f} | [{s['return_min']:.3f},{s['return_max']:.3f}] | "
            f"{s['success_mean']:.3f}±{s['success_std']:.3f} | [{s['success_min']:.3f},{s['success_max']:.3f}] | "
            f"{s['len_mean']:.2f}±{s['len_std']:.2f} | [{s['len_min']:.2f},{s['len_max']:.2f}] |"
        )
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


# -------------------------
# Report (OVERLAY ONLY)
# -------------------------
def write_report_md(
    out_md: Path,
    task: str,
    run_info: Dict[str, Dict],
    milestones: Dict[str, Dict],
    eval_agg: Dict[Tuple[str, str, str], Dict[str, float]],
    produced: Dict[str, List[str]],
    manifest_name: str = "manifest.json",
) -> None:
    out_md.parent.mkdir(parents=True, exist_ok=True)

    lines: List[str] = []
    lines.append(f"# Relatório de evidências: {task}\n")

    lines.append("## Artefactos gerados\n")
    lines.append(f"- Manifest: `{manifest_name}`")
    for k, items in produced.items():
        if items:
            lines.append(f"- {k}:")
            for it in items:
                lines.append(f"  - `{it}`")
    lines.append("")

    lines.append("## Fluxo completo (o que foi medido)\n")
    lines.append("1) Extração de *scalars* do TensorBoard (ficheiros `events.out.tfevents.*`).")
    lines.append("2) Identificação de marcos do treino (thresholds e estabilização).")
    lines.append("3) Avaliação de checkpoints (determinística e/ou estocástica) para várias seeds.")
    lines.append("4) Consolidação em tabelas e gráficos (apenas overlays EE vs Joints).\n")

    lines.append("## Logs de treino encontrados\n")
    for mode in ["ee", "joints"]:
        info = run_info.get(mode, {})
        lines.append(f"### {mode}")
        lines.append(f"- run_dir: `{info.get('run_dir','(missing)')}`")
        lines.append(f"- status: `{info.get('status','unknown')}`")
        if info.get("status") == "ok":
            lines.append(f"- n_tags: `{info.get('n_tags','?')}`")
        if info.get("tags_txt"):
            lines.append(f"- tags_txt: `{info.get('tags_txt')}`")
        lines.append("")

    lines.append("## Marcos do treino (a partir de `rollout/*`)\n")
    for mode in ["ee", "joints"]:
        m = milestones.get(mode, {})
        lines.append(f"### {mode}")
        for key in [
            "final_step",
            "final_success",
            "final_ep_len",
            "step_success_95_stable",
            "step_success_99_stable",
            "step_success_100_stable",
            "step_len_le_3_stable",
        ]:
            if key in m:
                lines.append(f"- {key}: `{m[key]}`")
        lines.append("")

    lines.append("## Avaliação por checkpoint (média±desvio; min/max)\n")
    keys = sorted(eval_agg.keys(), key=lambda k: (k[0], k[1], k[2]))
    lines.append("| eval | modo | checkpoint | n | return (m±sd) | success (m±sd) | len (m±sd) |")
    lines.append("|---|---|---:|---:|---:|---:|---:|")
    for (eval_type, mode, ck) in keys:
        s = eval_agg[(eval_type, mode, ck)]
        lines.append(
            f"| {eval_type} | {mode} | {ck} | {int(s['n'])} | "
            f"{s['return_mean']:.3f}±{s['return_std']:.3f} | "
            f"{s['success_mean']:.3f}±{s['success_std']:.3f} | "
            f"{s['len_mean']:.2f}±{s['len_std']:.2f} |"
        )
    lines.append("")

    lines.append("## Gráficos principais\n")
    lines.append("### Comparação EE vs Joints (overlay)\n")
    for img in produced.get("plots_overlay", []):
        lines.append(f"![]({img})")
    lines.append("")

    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


# -------------------------
# Main
# -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", default=".", help="raiz do repo")
    ap.add_argument("--task", default="PandaReach", help="nome lógico para output")

    ap.add_argument("--ee-env", default="PandaReach-v3")
    ap.add_argument("--joints-env", default="PandaReachJoints-v3")

    ap.add_argument("--ee-model-dir", default="models/PandaReach/ee")
    ap.add_argument("--joints-model-dir", default="models/PandaReach/joints")

    ap.add_argument("--ee-run-root", default="runs/PandaReach/ee")
    ap.add_argument("--joints-run-root", default="runs/PandaReach/joints")

    ap.add_argument("--checkpoints", default="ckpt10.zip,mid.zip,final.zip", help="lista separada por vírgulas")
    ap.add_argument("--discover-checkpoints", action="store_true", help="descobre *.zip no model_dir e ignora --checkpoints")

    ap.add_argument("--seeds", default="0", help="ex: 0 ou 0,1,2,3,4")
    ap.add_argument("--episodes", type=int, default=20)

    ap.add_argument("--eval-both", action="store_true", help="faz deterministic + stochastic")
    ap.add_argument("--stochastic-only", action="store_true", help="faz apenas stochastic")

    ap.add_argument("--out", default="", help="pasta de output (default: evidence/<task>_<timestamp>)")
    ap.add_argument("--hash", action="store_true", help="sha256 de modelos e event files")
    ap.add_argument("--pip-freeze", action="store_true", help="inclui pip freeze no manifest")

    ap.add_argument("--dump-tags", action="store_true", help="guarda a lista de tags TB encontradas num txt por modo")

    args = ap.parse_args()

    repo_root = Path(args.repo_root).resolve()
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out) if args.out else (repo_root / "evidence" / f"{args.task}_{ts}")
    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    eval_mod = load_eval_module(repo_root)
    seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]

    ee_model_dir = (repo_root / args.ee_model_dir).resolve()
    joints_model_dir = (repo_root / args.joints_model_dir).resolve()

    if args.discover_checkpoints:
        ee_ck = discover_checkpoints(ee_model_dir)
        joints_ck = discover_checkpoints(joints_model_dir)
        common = sorted(list(set(ee_ck).intersection(set(joints_ck))))
        checkpoints = common if common else sorted(list(set(ee_ck + joints_ck)))
    else:
        checkpoints = [c.strip() for c in args.checkpoints.split(",") if c.strip()]

    run_info: Dict[str, Dict] = {}
    scalars_by_mode: Dict[str, Dict[str, List[Tuple[int, float]]]] = {}

    for mode, run_root_str in [("ee", args.ee_run_root), ("joints", args.joints_run_root)]:
        run_root = (repo_root / run_root_str).resolve()
        run_dir = find_latest_run_dir(run_root)
        if run_dir is None:
            run_info[mode] = {"status": "missing_run_dir", "run_root": str(run_root)}
            continue

        run_info[mode] = {"status": "ok", "run_dir": str(run_dir)}

        try:
            scalars = extract_tb_scalars(run_dir)
            scalars_by_mode[mode] = scalars
            run_info[mode]["n_tags"] = len(scalars.keys())

            save_scalars_csv_long(scalars, out_dir / "train_scalars" / f"{mode}_scalars.csv")

            if args.dump_tags:
                tags_txt = out_dir / "train_scalars" / f"{mode}_tags.txt"
                save_scalar_tags_txt(scalars, tags_txt)
                run_info[mode]["tags_txt"] = str(tags_txt.relative_to(out_dir))

        except ImportError:
            run_info[mode]["status"] = "tensorboard_not_installed"
        except Exception as e:
            run_info[mode]["status"] = "tb_parse_error"
            run_info[mode]["error"] = str(e)

    produced = {"plots_overlay": [], "tables": []}

    # Overlays EE vs Joints (comparação direta) — ÚNICO tipo de gráfico gerado
    overlay_tags = [
        ("rollout/success_rate", "overlay_success_rate"),
        ("rollout/ep_len_mean", "overlay_ep_len_mean"),
        ("rollout/ep_rew_mean", "overlay_ep_rew_mean"),
        ("train/value_loss", "overlay_value_loss"),
        ("train/policy_gradient_loss", "overlay_policy_gradient_loss"),
        ("train/entropy_loss", "overlay_entropy_loss"),
        ("train/std", "overlay_std"),
        ("train/approx_kl", "overlay_approx_kl"),
        ("train/clip_fraction", "overlay_clip_fraction"),
        ("train/explained_variance", "overlay_explained_variance"),
    ]

    for tag, fname in overlay_tags:
        series = {}
        for mode in ["ee", "joints"]:
            scalars = scalars_by_mode.get(mode, {})
            xs, ys = get_series(scalars, tag)
            series[mode] = (xs, ys)
        out_png = out_dir / "plots" / f"{fname}.png"
        ok = plot_overlay(series, out_png, title=f"EE vs Joints: {tag}")
        if ok:
            produced["plots_overlay"].append(str(out_png.relative_to(out_dir)))

    # Milestones (treino)
    milestones: Dict[str, Dict] = {}
    for mode in ["ee", "joints"]:
        scalars = scalars_by_mode.get(mode, {})
        xs_sr, ys_sr = get_series(scalars, "rollout/success_rate")
        xs_len, ys_len = get_series(scalars, "rollout/ep_len_mean")

        final_sr = last_value(xs_sr, ys_sr)
        final_len = last_value(xs_len, ys_len)

        m = {}
        if final_sr:
            m["final_step"] = final_sr[0]
            m["final_success"] = round(final_sr[1], 6)
            m["step_success_95_stable"] = first_stable_step(xs_sr, ys_sr, 0.95, "ge")
            m["step_success_99_stable"] = first_stable_step(xs_sr, ys_sr, 0.99, "ge")
            m["step_success_100_stable"] = first_stable_step(xs_sr, ys_sr, 1.0, "ge")
        if final_len:
            m["final_ep_len"] = round(final_len[1], 6)
            m["step_len_le_3_stable"] = first_stable_step(xs_len, ys_len, 3.0, "le")
        milestones[mode] = m

    # Evaluation (deterministic / stochastic)
    rows: List[EvalRow] = []

    do_stoch = args.stochastic_only or args.eval_both
    do_det = (not args.stochastic_only)

    if do_det:
        rows += eval_models(
            eval_mod, "deterministic",
            "ee", args.ee_env, ee_model_dir, checkpoints, seeds, args.episodes,
            deterministic=True,
        )
        rows += eval_models(
            eval_mod, "deterministic",
            "joints", args.joints_env, joints_model_dir, checkpoints, seeds, args.episodes,
            deterministic=True,
        )

    if do_stoch:
        rows += eval_models(
            eval_mod, "stochastic",
            "ee", args.ee_env, ee_model_dir, checkpoints, seeds, args.episodes,
            deterministic=False,
        )
        rows += eval_models(
            eval_mod, "stochastic",
            "joints", args.joints_env, joints_model_dir, checkpoints, seeds, args.episodes,
            deterministic=False,
        )

    write_eval_csv(rows, out_dir / "eval_results.csv")
    produced["tables"].append("eval_results.csv")

    eval_agg = agg_eval(rows)
    write_eval_summary_md(eval_agg, out_dir / "eval_summary.md")
    produced["tables"].append("eval_summary.md")

    manifest = {
        "timestamp": ts,
        "task": args.task,
        "repo_root": str(repo_root),
        "git": try_git_info(repo_root),
        "python": sys.version,
        "argv": sys.argv,
        "args": vars(args),
        "run_info": run_info,
        "milestones": milestones,
        "produced": produced,
    }
    if args.pip_freeze:
        manifest["pip_freeze"] = try_pip_freeze()

    if args.hash:
        manifest["hashes"] = {"models": {}, "events": {}}
        for mode, mdir in [("ee", ee_model_dir), ("joints", joints_model_dir)]:
            for ck in checkpoints:
                p = mdir / ck
                if p.exists():
                    manifest["hashes"]["models"][f"{mode}:{ck}"] = {
                        "path": str(p),
                        "sha256": sha256_file(p),
                        "bytes": p.stat().st_size,
                    }
        for mode in ["ee", "joints"]:
            run_dir_str = run_info.get(mode, {}).get("run_dir")
            if run_dir_str:
                evs = list_event_files(Path(run_dir_str))
                if evs:
                    p = evs[-1]
                    manifest["hashes"]["events"][f"{mode}:latest_event"] = {
                        "path": str(p),
                        "sha256": sha256_file(p),
                        "bytes": p.stat().st_size,
                    }

    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    write_report_md(
        out_dir / "report.md",
        task=args.task,
        run_info=run_info,
        milestones=milestones,
        eval_agg=eval_agg,
        produced=produced,
        manifest_name="manifest.json",
    )

    print(f"[OK] evidence written to: {out_dir}")
    print(f"[OK] report:  {out_dir / 'report.md'}")
    print(f"[OK] eval:    {out_dir / 'eval_results.csv'}")
    print(f"[OK] summary: {out_dir / 'eval_summary.md'}")
    print(f"[OK] plots:   {out_dir / 'plots'}")

    if any(run_info.get(m, {}).get("status") == "tensorboard_not_installed" for m in ["ee", "joints"]):
        print("[WARN] tensorboard não está instalado -> sem extração de scalars/gráficos (avaliação continua OK).")


if __name__ == "__main__":
    main()

"""
audit_polygon_containment.py
----------------------------
Reproducible post-hoc geometric containment and simplicity audit script.

Verifies that building footprints satisfying OutlineAssessment vertex-level
containment and angle checks also satisfy full Shapely Polygon.within(envelope)
containment (no edges bridging concave setback indentations) and topological
simplicity (Polygon.is_simple / non-self-intersection) across all 1,000 test
parcels (seeds 0-999) and, optionally, classical solvers and the 3 Synthetic
Distribution-Shift (OOD) regimes (seeds 2000-2059).

Usage:
    ./.venv/bin/python scripts/audit_polygon_containment.py --n-parcels 1000 --audit-ood
    ./.venv/bin/python scripts/audit_polygon_containment.py --n-parcels 10 --recompute-classical
"""

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

import gymnasium as gym
import torch
from shapely.geometry import Polygon
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

import urban_planner  # noqa: F401  (registers UrbanPlanner-v0)
from urban_planner.domain.geometries import Plot
from scripts.baselines.run_cma_es import run_cma_es
from scripts.baselines.run_constrained_opt import run_cobyla, run_slsqp
from scripts.baselines.run_de import run_de
from scripts.baselines.run_hybrid_ppo import run_hybrid_ppo


def _load_generate_ood_plot() -> Callable[[str], Plot]:
    """Loads shared OOD parcel generator from scripts/run-synthetic-ood-eval.py."""
    ood_path = PROJECT_ROOT / "scripts" / "run-synthetic-ood-eval.py"
    spec = importlib.util.spec_from_file_location("run_synthetic_ood_eval", str(ood_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {ood_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.generate_ood_plot


def _init_method_stats() -> Dict[str, int]:
    return {
        "n_total": 0,
        "n_solved": 0,
        "solved_within": 0,
        "solved_simple": 0,
        "terminal_vertex_outside": 0,
        "terminal_non_within": 0,
        "terminal_non_simple": 0,
    }


def _evaluate_polygon_checks(
    coords: List[Tuple[int, int]],
    env_unwrapped: Any,
) -> Tuple[bool, bool, bool]:
    """Returns (vertices_inside, polygon_within, is_simple) for integer coords."""
    assessment = env_unwrapped.assessment
    assessment.update(coords)
    envelope = env_unwrapped.plot.permitted_outline
    poly = Polygon(coords)
    vertices_inside = all(d == 0 for d in assessment.outside_shape_distances)
    polygon_within = bool(poly.within(envelope))
    is_simple = bool(getattr(assessment, "is_simple", True) and poly.is_simple)
    return vertices_inside, polygon_within, is_simple


def _record_result(
    stats: Dict[str, int],
    solved: bool,
    coords: List[Tuple[int, int]],
    env_unwrapped: Any,
) -> None:
    vertices_inside, polygon_within, is_simple = _evaluate_polygon_checks(coords, env_unwrapped)
    stats["n_total"] += 1
    if not vertices_inside:
        stats["terminal_vertex_outside"] += 1
    if not polygon_within:
        stats["terminal_non_within"] += 1
    if not is_simple:
        stats["terminal_non_simple"] += 1
    if solved:
        stats["n_solved"] += 1
        if polygon_within:
            stats["solved_within"] += 1
        if is_simple:
            stats["solved_simple"] += 1


def _load_cached_v2_classical(max_index: int) -> Dict[str, List[Dict[str, Any]]]:
    """Loads recorded classical final_coords from logs/comparative_evaluation_v2_results.json if present."""
    v2_path = PROJECT_ROOT / "logs" / "comparative_evaluation_v2_results.json"
    if not v2_path.exists():
        return {}
    with open(v2_path, "r") as f:
        raw = json.load(f).get("raw_results", {})
    cached = {}
    for name in ("CMA-ES", "Diff-Evolution", "COBYLA"):
        entries = raw.get(name, [])
        if entries and "final_coords" in entries[0] and len(entries) > max_index:
            cached[name] = entries
    return cached


def _audit_suite(
    model: PPO,
    vec_env: VecNormalize,
    raw_env: gym.Env,
    n_parcels: int,
    seed_offset: int,
    recompute_classical: bool = False,
    cached_v2: Dict[str, List[Dict[str, Any]]] | None = None,
    progress_label: str = "Main",
) -> Dict[str, Dict[str, int]]:
    """Audits containment and simplicity across seeds [seed_offset, seed_offset + n_parcels)."""
    methods: Dict[str, Dict[str, int]] = {
        "CL-PPO": _init_method_stats(),
        "CL-PPO + Refinement": _init_method_stats(),
    }
    cached_v2 = cached_v2 or {}
    for name in cached_v2:
        if not recompute_classical:
            methods[name] = _init_method_stats()

    classical_runners = {}
    if recompute_classical:
        classical_runners = {
            "CMA-ES": run_cma_es,
            "SLSQP (5x)": run_slsqp,
            "COBYLA (10x)": run_cobyla,
            "Diff-Evolution": run_de,
        }
        for name in classical_runners:
            methods[name] = _init_method_stats()

    unwrapped = raw_env.unwrapped
    for idx in range(n_parcels):
        seed = seed_offset + idx
        hyb = run_hybrid_ppo(model, vec_env, raw_env, seed=seed)
        hyb_coords = hyb["final_coords"]
        _record_result(methods["CL-PPO + Refinement"], bool(hyb["solved"]), hyb_coords, unwrapped)

        if hyb["solved"] and hyb.get("polish_evals", 0) == 0:
            _record_result(methods["CL-PPO"], True, hyb_coords, unwrapped)

        if not recompute_classical:
            for name, entries in cached_v2.items():
                rec = entries[seed]
                c_coords = [(int(round(pt[0])), int(round(pt[1]))) for pt in rec["final_coords"]]
                _record_result(methods[name], bool(rec["solved"]), c_coords, unwrapped)
        else:
            for name, runner in classical_runners.items():
                res = runner(raw_env, seed=seed)
                _record_result(methods[name], bool(res["solved"]), res["final_coords"], unwrapped)

        if (idx + 1) % 100 == 0 or idx + 1 == n_parcels:
            print(f"  [{progress_label}] Audited {idx + 1}/{n_parcels} parcels...")

    return methods


def _print_stats_table(title: str, stats_by_method: Dict[str, Dict[str, int]]) -> None:
    print(f"\n=== {title} ===")
    for name, s in stats_by_method.items():
        n_sol = s["n_solved"]
        n_tot = s["n_total"]
        w_pct = (s["solved_within"] / n_sol * 100.0) if n_sol > 0 else 0.0
        s_pct = (s["solved_simple"] / n_sol * 100.0) if n_sol > 0 else 0.0
        line = (
            f"  {name:20s}: Solved within={s['solved_within']}/{n_sol} ({w_pct:5.1f}%), "
            f"simple={s['solved_simple']}/{n_sol} ({s_pct:5.1f}%)"
        )
        if name == "CL-PPO + Refinement" and n_tot > 0:
            line += (
                f" | Terminal violations (N={n_tot}): "
                f"vertex={s['terminal_vertex_outside']}/{n_tot} ({s['terminal_vertex_outside'] / n_tot * 100:.2f}%), "
                f"non-within={s['terminal_non_within']}/{n_tot} ({s['terminal_non_within'] / n_tot * 100:.2f}%), "
                f"non-simple={s['terminal_non_simple']}/{n_tot} ({s['terminal_non_simple'] / n_tot * 100:.2f}%)"
            )
        print(line)


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit Polygon.within & topological simplicity.")
    parser.add_argument("--n-parcels", type=int, default=1000, help="Number of main test parcels (seeds 0..N-1).")
    parser.add_argument("--seed-offset", type=int, default=0)
    parser.add_argument("--recompute-classical", action="store_true", help="Re-run CMA-ES, SLSQP, COBYLA, and DE from scratch instead of using cached v2 final_coords.")
    parser.add_argument("--audit-ood", action="store_true", help="Also audit CL-PPO / Hybrid across 3 OOD regimes.")
    parser.add_argument("--ood-parcels", type=int, default=60, help="Parcels per OOD regime.")
    parser.add_argument("--ood-seed-offset", type=int, default=2000)
    parser.add_argument("--save-json", type=Path, default=None, help="Optional JSON output path.")
    args = parser.parse_args()

    torch.set_num_threads(1)
    raw_env = gym.make("UrbanPlanner-v0", stage=4)
    dummy = DummyVecEnv([lambda: gym.make("UrbanPlanner-v0", stage=4)])
    stats_path = PROJECT_ROOT / "models" / "stage_4_tunning_1757784606" / "best_model_vec_normalize.pkl"
    model_path = PROJECT_ROOT / "models" / "stage_4_tunning_1757784606" / "best_model.zip"
    vec_env = VecNormalize.load(str(stats_path), dummy)
    vec_env.training = False
    vec_env.norm_reward = False
    model = PPO.load(model_path, env=vec_env, device="cpu")

    cached_v2 = (
        {}
        if args.recompute_classical
        else _load_cached_v2_classical(args.seed_offset + args.n_parcels - 1)
    )
    results: Dict[str, Any] = {}
    main_stats = _audit_suite(
        model, vec_env, raw_env,
        n_parcels=args.n_parcels,
        seed_offset=args.seed_offset,
        recompute_classical=args.recompute_classical,
        cached_v2=cached_v2,
        progress_label="Main Test Suite",
    )
    _print_stats_table(f"In-Distribution Test Audit (N={args.n_parcels})", main_stats)
    results["main_test_suite"] = main_stats

    if args.audit_ood:
        generate_ood_plot = _load_generate_ood_plot()
        orig_from_random = Plot.from_random_generation
        ood_summary: Dict[str, Any] = {}
        total_ood_ppo_solved = 0
        total_ood_ppo_within = 0
        try:
            for regime in ("ood_tight_setbacks", "ood_irregular_nonconvex", "ood_high_coverage"):
                Plot.from_random_generation = classmethod(lambda cls, rk=regime: generate_ood_plot(rk))
                reg_stats = _audit_suite(
                    model, vec_env, raw_env,
                    n_parcels=args.ood_parcels,
                    seed_offset=args.ood_seed_offset,
                    recompute_classical=False,
                    cached_v2=None,
                    progress_label=regime,
                )
                _print_stats_table(f"OOD Audit: {regime} (N={args.ood_parcels})", reg_stats)
                ood_summary[regime] = reg_stats
                total_ood_ppo_solved += reg_stats["CL-PPO"]["n_solved"]
                total_ood_ppo_within += reg_stats["CL-PPO"]["solved_within"]
        finally:
            Plot.from_random_generation = orig_from_random
        print(
            f"\nCombined OOD CL-PPO Polygon.within: "
            f"{total_ood_ppo_within}/{total_ood_ppo_solved}"
        )
        results["ood_suites"] = ood_summary

    if args.save_json is not None:
        args.save_json.parent.mkdir(parents=True, exist_ok=True)
        with open(args.save_json, "w") as f:
            json.dump(results, f, indent=4)
        print(f"Saved audit results to {args.save_json}")


if __name__ == "__main__":
    main()

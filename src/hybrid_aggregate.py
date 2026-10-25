"""
AGREGACAO: HIBRIDOS x CLASSICO, ANIMAL A ANIMAL
===============================================

Junta os bracos deste repositorio com os do classico (por padrao a cabeca
linear e a floresta da ResNet50) e roda as MESMAS funcoes do aggregate.py
classico: metricas por fold, tabela pareada por animal, teste de sinal com
piso do p, resumo e comparacao entre sementes.

Acrescenta o que a pergunta "a hibrida generaliza melhor para um animal
novo?" exige e o classico nao calculava por braco:

  diagnostics_by_run.csv   por (braco, fold, semente): queda validacao->teste
                           do F1 macro, colapso (>= 80% das predicoes numa
                           classe), adjacencia dos erros, revocacao da classe
                           mais intensa
  decision.csv             o criterio fixado antes de rodar (ver abaixo)

CRITERIO DE DECISAO
-------------------
Uma cabeca hibrida H e "melhor" que uma referencia R num cenario quando,
com a media das sementes por animal de teste:

  (1) H supera R em TODOS os animais;
  (2) a diferenca media entre animais passa de 2x o desvio entre sementes
      (raiz da media dos desvios combinados por animal);
  (3) H nao colapsa mais, nem zera a revocacao da classe mais intensa mais
      vezes, que R (taxa por treino).

Sem replicas por semente o criterio (2) fica indeterminado. Nenhum p-valor
entra na decisao: com 4 animais o piso do teste de sinal e 0,125.

Uso:
    python src/hybrid_aggregate.py --results_dir resultados_q/binary \\
        --classical_results ../Classical/main_results/binary --task binary
"""

from __future__ import annotations

import classical_bridge  # noqa: F401

from common import silence_known_warnings
silence_known_warnings()

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from aggregate import (aggregate_metrics, collect_arms, collect_selection,
                       p_floor, paired_by_subject, paired_comparison,
                       seed_paired, seed_summary)
from manifest import TASKS, class_names_for

LINEAR_SUFFIX = "_linear_head"
DEFAULT_CLASSICAL_ARMS = ("resnet50_rf_linear_head", "resnet50_rf")
REFERENCE_ARMS = ("c1_bottleneck", "c1_ordinal", "c0_linear", "resnet50_rf_linear_head")
COLLAPSE_THRESHOLD = 0.8


# --------------------------------------------------------------------------
# Coleta de mais de uma raiz
# --------------------------------------------------------------------------
def _filter(df: pd.DataFrame, keep: set[str] | None) -> pd.DataFrame:
    return df if keep is None or df.empty else df[df.arm.isin(keep)]


def collect_roots(roots: list[tuple[Path, set[str] | None]], folds: list[str],
                  seeds: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    """collect_arms do classico sobre cada raiz, filtrando os bracos de cada uma."""
    pf, ps = [], []
    for root, keep in roots:
        presentes = [f for f in folds if (root / f).is_dir()]
        if not presentes:
            continue
        a, b = collect_arms(root, presentes, seeds=seeds)
        pf.append(_filter(a, keep)); ps.append(_filter(b, keep))
    return (pd.concat(pf, ignore_index=True) if pf else pd.DataFrame(),
            pd.concat(ps, ignore_index=True) if ps else pd.DataFrame())


def check_same_subjects(por_sujeito: pd.DataFrame) -> None:
    """Cada fold tem de testar o mesmo animal em todos os bracos."""
    g = por_sujeito.groupby("fold")["subject_id"].nunique()
    ruins = g[g > 1]
    if not ruins.empty:
        raise RuntimeError(
            f"folds com animais de teste diferentes entre bracos: {list(ruins.index)}. "
            "As raizes nao usam a mesma particao; a comparacao pareada seria invalida.")


# --------------------------------------------------------------------------
# Diagnosticos por treino
# --------------------------------------------------------------------------
def _runs(root: Path, folds: list[str], keep: set[str] | None):
    """(fold, braco, semente, pasta_base, prefixo, pasta_familia) de cada treino avaliado."""
    for fold in folds:
        if not (root / fold).is_dir():
            continue
        for fam_dir in sorted(p for p in (root / fold).iterdir() if p.is_dir()):
            bases = [(fam_dir, None)] + [
                (d, int(d.name.split("_", 1)[1]))
                for d in sorted((fam_dir / "seeds").glob("seed_*")) if d.is_dir()]
            for base, seed in bases:
                for arm, prefix in ((fam_dir.name, "test"),
                                    (fam_dir.name + LINEAR_SUFFIX, "test_linear_head")):
                    if keep is not None and arm not in keep:
                        continue
                    if (base / f"predictions_{prefix}.csv").exists():
                        yield fold, arm, seed, base, prefix, fam_dir


def _val_score(fam_dir: Path, seed, linear: bool, metric: str) -> float:
    """F1 de validacao do treino: best_summary (vencedor) ou seeds_summary (replica)."""
    col = ("linear_val_" if linear else "val_") + metric
    if seed is None:
        f = fam_dir / "best_summary.csv"
        if f.exists():
            df = pd.read_csv(f)
            if col in df.columns:
                return float(df[col].iloc[0])
        return float("nan")
    f = fam_dir / "seeds" / "seeds_summary.csv"
    if linear or not f.exists():
        return float("nan")                       # o classico nao grava a validacao da cabeca linear por semente
    df = pd.read_csv(f)
    linha = df[df.seed == seed]
    return float(linha[col].iloc[0]) if col in df.columns and len(linha) else float("nan")


def diagnostics(roots, folds, class_names, metric: str = "f1_macro") -> pd.DataFrame:
    intensa = class_names[-1]
    linhas = []
    for root, keep in roots:
        for fold, arm, seed, base, prefix, fam_dir in _runs(root, folds, keep):
            pred = pd.read_csv(base / f"predictions_{prefix}.csv")
            m = pd.read_csv(base / f"metrics_{prefix}.csv").iloc[0]
            erros = pred[pred.y_pred != pred.y_true]
            frac = float(pred.y_pred.value_counts(normalize=True).max())
            val = _val_score(fam_dir, seed, prefix.endswith("linear_head"), metric)
            teste = float(m.get(metric, np.nan))
            rec = m.get(f"recall_{intensa}", np.nan)
            linhas.append({
                "arm": arm, "fold": fold, "seed": seed,
                "subject_id": int(pred.subject_id.iloc[0]) if pred.subject_id.nunique() == 1 else np.nan,
                f"val_{metric}": val, f"test_{metric}": teste,
                "val_test_gap": val - teste,
                "majority_pred_fraction": frac,
                "collapsed": frac >= COLLAPSE_THRESHOLD,
                "adjacent_error_rate": float(((erros.y_pred - erros.y_true).abs() == 1).mean())
                if len(erros) else np.nan,
                f"recall_{intensa}": float(rec),
                "zero_recall_most_intense": bool(float(rec) == 0.0) if pd.notna(rec) else False,
            })
    return pd.DataFrame(linhas)


def diagnostics_summary(diag: pd.DataFrame) -> pd.DataFrame:
    if diag.empty:
        return diag
    num = [c for c in diag.columns if c not in ("arm", "fold", "seed", "subject_id")]
    g = diag.groupby("arm")
    out = g[num].mean(numeric_only=True)
    out["n_runs"] = g.size()
    out["n_collapsed"] = g["collapsed"].sum()
    out["n_zero_recall_most_intense"] = g["zero_recall_most_intense"].sum()
    return out.reset_index()


# --------------------------------------------------------------------------
# Criterio de decisao
# --------------------------------------------------------------------------
def default_pairs(arms: list[str]) -> list[tuple[str, str]]:
    """Cada hibrido (nome comecando com 'h') contra cada referencia presente.

    Hibridos ordinais se comparam com c1_ordinal; os demais com c1_bottleneck.
    Ambos se comparam com a cabeca linear (classica e/ou refeita aqui).
    """
    pares = []
    for h in sorted(a for a in arms if a.startswith("h")):
        ordinal = "ordinal" in h
        for r in REFERENCE_ARMS:
            if r not in arms:
                continue
            if (r == "c1_ordinal" and not ordinal) or (r == "c1_bottleneck" and ordinal):
                continue
            pares.append((h, r))
    return pares


def decision_table(rep: pd.DataFrame, diag: pd.DataFrame, pairs, metric: str) -> pd.DataFrame:
    """Aplica os criterios (1)-(3) a cada par (hibrido, referencia).

    `rep` tem uma linha por (fold, animal, braco[, semente]). Sem coluna seed
    (so o vencedor do grid), cada animal tem uma observacao por braco.
    """
    linhas = []
    for h, r in pairs:
        sub = rep[rep.arm.isin([h, r])]
        if sub.empty or metric not in sub.columns:
            continue
        stats = sub.groupby(["subject_id", "arm"])[metric].agg(["mean", "std", "count"]).unstack("arm")
        if h not in stats["mean"].columns or r not in stats["mean"].columns:
            continue
        ok = stats["mean"][[h, r]].dropna().index
        mh, mr = stats["mean"].loc[ok, h], stats["mean"].loc[ok, r]
        d = mh - mr
        sd = stats["std"].loc[ok, [h, r]]
        ruido = float(np.sqrt(((sd[h] ** 2 + sd[r] ** 2) / 2).mean())) if sd.notna().all().all() else np.nan

        def taxa(arm, col):
            g = diag[diag.arm == arm] if len(diag) else diag
            return float(g[col].mean()) if len(g) else np.nan

        c1 = bool(len(d) > 0 and (d > 0).all())
        c2 = None if np.isnan(ruido) else bool(d.mean() > 2 * ruido)
        col_h, col_r = taxa(h, "collapsed"), taxa(r, "collapsed")
        zero_h, zero_r = taxa(h, "zero_recall_most_intense"), taxa(r, "zero_recall_most_intense")
        c3 = (None if np.isnan([col_h, col_r, zero_h, zero_r]).any()
              else bool(col_h <= col_r and zero_h <= zero_r))
        if c1 and c2 and c3:
            veredito = "melhor"
        elif c1 and (c2 is None or c3 is None):
            veredito = "indeterminado"
        else:
            veredito = "nao superior"

        linhas.append({
            "metric": metric, "hybrid": h, "reference": r, "n_subjects": len(d),
            "n_seeds_hybrid": int(stats["count"].loc[ok, h].min()) if len(ok) else 0,
            "n_seeds_reference": int(stats["count"].loc[ok, r].min()) if len(ok) else 0,
            "mean_hybrid": float(mh.mean()), "mean_reference": float(mr.mean()),
            "mean_diff": float(d.mean()), "min_diff": float(d.min()), "max_diff": float(d.max()),
            "wins_hybrid": int((d > 0).sum()), "wins_reference": int((d < 0).sum()),
            "seed_noise": ruido,
            "collapse_rate_hybrid": col_h, "collapse_rate_reference": col_r,
            "zero_recall_rate_hybrid": zero_h, "zero_recall_rate_reference": zero_r,
            "val_test_gap_hybrid": taxa(h, "val_test_gap"),
            "val_test_gap_reference": taxa(r, "val_test_gap"),
            "c1_all_subjects": c1, "c2_above_2x_noise": c2, "c3_no_extra_failure": c3,
            "verdict": veredito,
        })
    return pd.DataFrame(linhas)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description="Agregacao hibridos x classico.")
    ap.add_argument("--results_dir", required=True, help="<resultados quanticos>/<tarefa>")
    ap.add_argument("--classical_results", default=None,
                    help="<resultados classicos>/<tarefa>, para incluir os bracos classicos.")
    ap.add_argument("--classical_arms", nargs="*", default=list(DEFAULT_CLASSICAL_ARMS))
    ap.add_argument("--out_dir", default=None, help="Padrao: <results_dir>/_tabelas")
    ap.add_argument("--task", required=True, choices=list(TASKS))
    ap.add_argument("--folds", nargs="*", default=None)
    ap.add_argument("--metrics", nargs="*", default=["f1_macro", "balanced_accuracy"])
    ap.add_argument("--pairs", nargs="*", default=None,
                    help="Pares hibrido:referencia. Padrao: ver default_pairs.")
    args = ap.parse_args()

    res = Path(args.results_dir)
    out = Path(args.out_dir) if args.out_dir else res / "_tabelas"
    out.mkdir(parents=True, exist_ok=True)
    folds = args.folds or sorted(d.name for d in res.glob("fold_*") if d.is_dir())
    roots: list[tuple[Path, set[str] | None]] = [(res, None)]
    if args.classical_results:
        roots.append((Path(args.classical_results), set(args.classical_arms)))
    principal = args.metrics[0]

    collect_selection(res, folds).to_csv(out / "best_by_fold.csv", index=False)
    por_fold, por_sujeito = collect_roots(roots, folds)
    if por_fold.empty:
        print(f"[ERRO] nenhum metrics_test*.csv sob {[str(r) for r, _ in roots]}")
        return 1
    check_same_subjects(por_sujeito)

    aggregate_metrics(por_fold).to_csv(out / "aggregate_metrics.csv", index=False)
    por_sujeito.to_csv(out / "metrics_by_subject_all_arms.csv", index=False)
    tabela = paired_by_subject(por_sujeito, args.metrics)
    tabela.to_csv(out / "paired_by_subject.csv", index=False)
    paired_comparison(por_sujeito, args.metrics).to_csv(out / "paired_comparison.csv", index=False)

    diag = diagnostics(roots, folds, class_names_for(args.task), principal)
    diag.to_csv(out / "diagnostics_by_run.csv", index=False)
    diagnostics_summary(diag).to_csv(out / "diagnostics_summary.csv", index=False)

    _, rep = collect_roots(roots, folds, seeds=True)
    if rep.empty:
        print("(sem replicas por semente: o criterio (2) fica indeterminado)")
        base_decisao = por_sujeito
    else:
        check_same_subjects(rep)
        rep.to_csv(out / "seed_replicates_by_subject.csv", index=False)
        seed_summary(rep, args.metrics).to_csv(out / "seed_summary_by_subject.csv", index=False)
        seed_paired(rep, args.metrics).to_csv(out / "seed_paired_by_subject.csv", index=False)
        base_decisao = rep

    arms = sorted(por_sujeito.arm.unique())
    pares = ([tuple(p.split(":", 1)) for p in args.pairs] if args.pairs
             else default_pairs(arms))
    dec = decision_table(base_decisao, diag, pares, principal)
    dec.to_csv(out / "decision.csv", index=False)

    n_animais = por_sujeito.subject_id.nunique()
    print("=" * 78)
    print(f"AGREGACAO HIBRIDA — tarefa={args.task}  {len(folds)} folds, {n_animais} animais, "
          f"bracos: {arms}")
    print("=" * 78)
    t = tabela[tabela.metric == principal].drop(columns="metric")
    cols = ["subject_id", "fold"] + arms
    print(t[[c for c in cols if c in t.columns]].to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    if not dec.empty:
        print()
        print(dec[["hybrid", "reference", "mean_diff", "wins_hybrid", "wins_reference",
                   "seed_noise", "c1_all_subjects", "c2_above_2x_noise",
                   "c3_no_extra_failure", "verdict"]]
              .to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"\npiso do p com {n_animais} animais: {p_floor(n_animais):.4f} "
          "(a decisao nao usa p-valor)")
    print(f"[ok] tabelas em {out}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

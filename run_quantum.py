#!/usr/bin/env python3
"""
ORQUESTRADOR HIBRIDO (Regime B)
===============================

Mesma estrutura do run_all.py classico, de quem reaproveita o registro de
parametros por etapa, o log e a execucao de cada (fold, familia) em
subprocesso proprio — pelas mesmas razoes: fragmentacao de memoria de GPU,
vazamentos e falhas isoladas por par.

Etapas por tarefa:

  1. particao   copia as dobras do classico (--classical_results), o que
                garante que o teste de cada fold e o MESMO animal nos dois
                repositorios. Sem resultados classicos, gera as dobras com o
                manifest.py e o splits.py do classico, em fixed_paired.
  2. grid       hybrid_grid.py por (fold, familia).
  3. sementes   hybrid_seeds.py, retreino do vencedor com --seeds.
  4. agregacao  hybrid_aggregate.py, incluindo os bracos classicos.

Familias (configs/grid_<familia>.json):

  c1_bottleneck    C1   gargalo classico casado em parametros com o H1
  h1_vqc           H1   dressed quantum circuit
  h2_reupload      H2   circuito com re-uploading
  h3_vqc_ordinal   H3   circuito com saida ordinal (pulado no binario)
  c1_ordinal       C1-ord  controle classico da saida ordinal (idem)
  c0_linear        C0   cabeca linear refeita neste codigo (opcional; a
                        referencia principal e a cabeca linear do classico)

Uso:
    python run_quantum.py --task binary --images_dir ../../_dataset/images \\
        --classical_results ../../Classical/main_results --seeds 1 2 3 4 5
    python run_quantum.py --task multiclass --families c1_bottleneck h1_vqc h3_vqc_ordinal c1_ordinal
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

import classical_bridge  # noqa: E402  (poe o classico no sys.path)

import argparse  # noqa: E402
import json  # noqa: E402
import shutil  # noqa: E402
import time  # noqa: E402
from datetime import timedelta  # noqa: E402

from run_all import file_sha, grava_stamp, precisa_rodar, run_step, setup_logging  # noqa: E402

CONFIGS = ROOT / "configs"
CLASSICAL_SRC = classical_bridge.CLASSICAL_ROOT / "src"
TASK_CHOICES = ["multiclass", "three_bernini", "three_mlss", "binary"]
ALL_FAMILIES = ("c1_bottleneck", "h1_vqc", "h2_reupload", "h3_vqc_ordinal",
                "c1_ordinal", "c0_linear")
DEFAULT_FAMILIES = ("c1_bottleneck", "h1_vqc")


def prepare_splits(args, res: Path, log) -> bool:
    """Etapa 1. Devolve False se nao ha como obter as dobras."""
    splits = res / "splits"
    stamp = splits / "params.json"
    origem = (Path(args.classical_results) / args.task / "splits"
              if args.classical_results else None)

    if origem is not None and sorted(origem.glob("fold_*.csv")):
        dobras = sorted(origem.glob("fold_*.csv"))
        params = {"source": str(origem.resolve()),
                  "folds": {f.name: file_sha(f) for f in dobras}}
        if precisa_rodar(stamp, splits / "fold_00.csv", params, args.force, log, "particao"):
            splits.mkdir(parents=True, exist_ok=True)
            for f in dobras:
                shutil.copy2(f, splits / f.name)
            for extra in ("splits_summary.csv",):
                if (origem / extra).exists():
                    shutil.copy2(origem / extra, splits / extra)
            grava_stamp(stamp, params)
            log.info(f"1/4 dobras copiadas do classico: {origem}")
        else:
            log.info(f"1/4 reusando dobras copiadas de {origem}")
        return True

    if origem is not None:
        log.warning(f"  {origem} nao tem fold_*.csv; gerando as dobras aqui.")
    if not args.fixed_test_subjects:
        log.error("sem dobras do classico e sem --fixed_test_subjects.")
        return False
    manifest = res / f"manifest_{args.task}.csv"
    py = sys.executable
    p_man = {"metadata_sha": file_sha(args.metadata) if Path(args.metadata).exists() else None,
             "images_dir": args.images_dir, "task": args.task}
    if precisa_rodar(manifest.with_suffix(".params.json"), manifest, p_man, args.force, log,
                     "manifesto"):
        ok, _ = run_step([py, str(CLASSICAL_SRC / "manifest.py"), "--metadata", args.metadata,
                          "--images_dir", args.images_dir, "--task", args.task,
                          "--out", str(manifest), "--check_files"], log, "1/4 manifesto")
        if not ok:
            return False
        grava_stamp(manifest.with_suffix(".params.json"), p_man)
    params = {"manifest_sha": file_sha(manifest), "mode": "fixed_paired",
              "fixed_test_subjects": args.fixed_test_subjects, "seed": args.seed,
              "n_val_subjects": args.n_val_subjects}
    if precisa_rodar(stamp, splits / "fold_00.csv", params, args.force, log, "particao"):
        ok, _ = run_step([py, str(CLASSICAL_SRC / "splits.py"), "--manifest", str(manifest),
                          "--out_dir", str(splits), "--mode", "fixed_paired",
                          "--n_val_subjects", str(args.n_val_subjects),
                          "--seed", str(args.seed), "--fixed_test_subjects"]
                         + [str(s) for s in args.fixed_test_subjects], log, "1/4 particao")
        if not ok:
            return False
        grava_stamp(stamp, params)
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description="Orquestrador das familias hibridas.")
    ap.add_argument("--task", default="binary", choices=TASK_CHOICES)
    ap.add_argument("--images_dir", default="imagens")
    ap.add_argument("--metadata", default="metadata.csv")
    ap.add_argument("--results_dir", default="resultados")
    ap.add_argument("--classical_results", default=None,
                    help="Raiz dos resultados classicos (contendo <tarefa>/splits e "
                         "<tarefa>/fold_*). Fonte das dobras e dos bracos classicos.")
    ap.add_argument("--fixed_test_subjects", type=int, nargs="*", default=[1, 9, 10, 11])
    ap.add_argument("--n_val_subjects", type=int, default=2,
                    help="Mesmo padrao do run_all.py classico, para gerar dobras identicas.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--seeds", type=int, nargs="*", default=None)
    ap.add_argument("--keep_seed_models", action="store_true")
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--families", nargs="*", default=list(DEFAULT_FAMILIES),
                    choices=list(ALL_FAMILIES))
    ap.add_argument("--only-folds", type=int, nargs="*", default=None)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    res = Path(args.results_dir) / args.task
    log = setup_logging(res / "run_quantum.log")
    log.info("=" * 70)
    log.info(f"tarefa={args.task}  familias={args.families}  seed={args.seed}  "
             f"replicas={args.seeds or '-'}  device={args.device}")
    log.info(f"classico: {classical_bridge.CLASSICAL_ROOT}  "
             f"(commit {classical_bridge.git_commit(classical_bridge.CLASSICAL_ROOT)})")
    log.info(f"saida: {res}/")
    log.info("=" * 70)
    t0 = time.time()
    falhas: list[str] = []
    py = sys.executable

    if not prepare_splits(args, res, log):
        log.error("sem particao; nada mais pode rodar.")
        return 2
    todos = sorted((res / "splits").glob("fold_*.csv"))
    folds = [f for f in todos
             if args.only_folds is None or int(f.stem.split("_")[1]) in args.only_folds]
    log.info(f"folds a processar: {[f.stem for f in folds]}")

    familias = []
    for fam in args.families:
        cfg = json.loads((CONFIGS / f"grid_{fam}.json").read_text(encoding="utf-8"))
        if args.task in cfg.get("skip_tasks", []):
            log.info(f"  {fam}: pulada em {args.task} ({cfg.get('_nota_skip', 'skip_tasks')})")
        else:
            familias.append(fam)

    # --- 2. grid ------------------------------------------------------
    log.info("2/4 grid")
    for fold_csv in folds:
        for fam in familias:
            out = res / fold_csv.stem / fam
            grid_json = CONFIGS / f"grid_{fam}.json"
            p = {"fold_sha": file_sha(fold_csv), "grid_sha": file_sha(grid_json),
                 "seed": args.seed, "task": args.task}
            if not precisa_rodar(out / "params.json", out / "best_summary.csv", p,
                                 args.force, log, f"{fold_csv.stem}/{fam}"):
                log.info(f"  {fold_csv.stem}/{fam} ja concluido, pulando")
                continue
            for velho in ("best_summary.csv", "best_selection.json", "params.json"):
                (out / velho).unlink(missing_ok=True)
            ok, _ = run_step([py, str(SRC / "hybrid_grid.py"), "--fold", str(fold_csv),
                              "--grid", str(grid_json), "--images_dir", args.images_dir,
                              "--out_dir", str(out), "--device", args.device,
                              "--task", args.task, "--seed", str(args.seed)],
                             log, f"grid {fold_csv.stem}/{fam}")
            if ok:
                grava_stamp(out / "params.json", p)
            else:
                falhas.append(f"grid {fold_csv.stem}/{fam}")

    # --- 3. sementes --------------------------------------------------
    if not args.seeds:
        log.info("3/4 replicas por semente desligadas (use --seeds)")
    else:
        log.info(f"3/4 replicas por semente {args.seeds}")
        for fold_csv in folds:
            for fam in familias:
                out = res / fold_csv.stem / fam
                if not (out / "best_selection.json").exists():
                    log.warning(f"  {fold_csv.stem}/{fam} sem vencedor, pulando replicas")
                    continue
                cmd = [py, str(SRC / "hybrid_seeds.py"), "--family_dir", str(out),
                       "--fold", str(fold_csv), "--images_dir", args.images_dir,
                       "--device", args.device, "--task", args.task,
                       "--seeds"] + [str(s) for s in args.seeds]
                if args.force:
                    cmd.append("--force")
                if args.keep_seed_models:
                    cmd.append("--keep_models")
                ok, _ = run_step(cmd, log, f"replicas {fold_csv.stem}/{fam}")
                if not ok:
                    falhas.append(f"replicas {fold_csv.stem}/{fam}")

    # --- 4. agregacao -------------------------------------------------
    log.info("4/4 agregacao")
    cmd = [py, str(SRC / "hybrid_aggregate.py"), "--results_dir", str(res),
           "--task", args.task, "--folds"] + [f.stem for f in todos]
    if args.classical_results:
        cmd += ["--classical_results", str(Path(args.classical_results) / args.task)]
    ok, _ = run_step(cmd, log, "agregacao")
    if not ok:
        falhas.append("agregacao")

    total = timedelta(seconds=int(time.time() - t0))
    log.info("=" * 70)
    if falhas:
        log.warning(f"CONCLUIDO COM {len(falhas)} FALHA(S) em {total}")
        for f in falhas:
            log.warning(f"  - {f}")
        log.warning("Rode de novo sem --force: as etapas concluidas sao puladas.")
    else:
        log.info(f"CONCLUIDO SEM FALHAS em {total}")
    log.info(f"tabelas em {res}/_tabelas/   decisao em {res}/_tabelas/decision.csv")
    log.info("=" * 70)
    return 1 if falhas else 0


if __name__ == "__main__":
    raise SystemExit(main())

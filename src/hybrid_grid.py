"""
GRID E AVALIACAO DE UMA FAMILIA HIBRIDA EM UM FOLD
==================================================

Mesmo invariante do grid_search.py classico: cada candidato so ve treino e
validacao; o vencedor e escolhido pela validacao; o teste e avaliado uma
unica vez, depois, com tudo congelado. Todos os candidatos de um fold
recebem a mesma semente.

Saida no mesmo formato do classico (best_summary.csv, best_selection.json,
metrics_test*.csv, predictions_test.csv, ...), o que deixa o aggregate.py
classico ler as familias hibridas sem adaptacao. grid_summary.csv guarda a
validacao de todos os candidatos; o model.pt dos que perderam e apagado
(--keep_all_models mantem).

Uso:
    python src/hybrid_grid.py --fold splits/fold_00.csv \\
        --grid configs/grid_h1_vqc.json --images_dir imagens \\
        --out_dir resultados/binary/fold_00/h1_vqc --task binary
"""

from __future__ import annotations

import classical_bridge  # noqa: F401

from common import resolve_device, silence_known_warnings
silence_known_warnings()

import argparse
import gc
import hashlib
import itertools
import json
import sys
import traceback
from pathlib import Path

import pandas as pd
import torch

from manifest import TASKS, class_names_for

from hybrid_train import evaluate_on_test, run_candidate


def expand_grid(grid: dict) -> list[dict]:
    """Produto cartesiano do bloco `grid`. Valor escalar vira lista de um."""
    chaves = list(grid)
    valores = [v if isinstance(v, list) else [v] for v in grid.values()]
    return [dict(zip(chaves, c)) for c in itertools.product(*valores)]


def load_family_config(path) -> dict:
    """JSON de familia: blocos 'base' e 'grid', e 'selection' opcional."""
    cfg = json.loads(Path(path).read_text(encoding="utf-8"))
    if "base" not in cfg:
        raise ValueError(f"{path}: falta o bloco 'base'")
    return cfg


def main() -> int:
    ap = argparse.ArgumentParser(description="Grid de uma familia hibrida em um fold.")
    ap.add_argument("--fold", required=True)
    ap.add_argument("--grid", required=True)
    ap.add_argument("--images_dir", default="imagens")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--task", default="multiclass", choices=list(TASKS))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit_candidates", type=int, default=None,
                    help="Usa so os N primeiros candidatos do grid (piloto).")
    ap.add_argument("--force", action="store_true",
                    help="Retreina todos os candidatos, mesmo os ja concluidos.")
    ap.add_argument("--keep_all_models", action="store_true",
                    help="Mantem model.pt dos candidatos que perderam (por padrao sao apagados).")
    args = ap.parse_args()

    device = resolve_device(args.device)
    class_names = class_names_for(args.task)
    n_classes = len(class_names)

    cfg = load_family_config(args.grid)
    base, grid = cfg["base"], cfg.get("grid", {})
    metrica = cfg.get("selection", {}).get("metric", "val_f1_macro")
    candidatos = expand_grid(grid) or [{}]
    if args.limit_candidates:
        # So para piloto: corta o grid nos N primeiros candidatos.
        candidatos = candidatos[:args.limit_candidates]

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[{out_dir.name}] dispositivo={device}  candidatos={len(candidatos)}  "
          f"selecao por {metrica}")

    fold_sha = hashlib.sha1(Path(args.fold).read_bytes()).hexdigest()[:16]
    resumos = []
    for i, combo in enumerate(candidatos):
        c = {**base, **combo, "config_id": f"net_{i:03d}", "seed": args.seed}
        net_dir = out_dir / f"net_{i:03d}"
        print(f"  net_{i:03d}  {combo}")

        # Retomada: um grid de 36 candidatos leva mais de um dia por dobra.
        # Se o processo cair no meio, os candidatos ja concluidos (mesma
        # config, mesmo fold, model.pt presente) nao sao treinados de novo.
        marca = net_dir / "candidate_done.json"
        params = {"config": c, "fold_sha": fold_sha}
        if (not args.force and marca.exists() and (net_dir / "model.pt").exists()):
            feito = json.loads(marca.read_text())
            if feito.get("params") == json.loads(json.dumps(params, default=str)):
                print("    ja concluido, reaproveitando")
                resumos.append({"net_id": f"net_{i:03d}", **combo, **feito["result"],
                                "status": "ok"})
                pd.DataFrame(resumos).to_csv(out_dir / "grid_summary.csv", index=False)
                continue
        try:
            r = run_candidate(args.fold, args.images_dir, c, net_dir, device,
                              n_classes, class_names)
            marca.write_text(json.dumps({"params": params, "result": r}, indent=2, default=str))
            resumos.append({"net_id": f"net_{i:03d}", **combo, **r, "status": "ok"})
        except Exception as e:
            print(f"    ERRO: {e!r}")
            traceback.print_exc(file=sys.stdout)       # vai para o log do orquestrador
            resumos.append({"net_id": f"net_{i:03d}", **combo, "status": "erro",
                            "error": repr(e), "net_dir": str(net_dir),
                            "cand_dir": str(net_dir)})
        finally:
            # 36 treinos no mesmo processo: devolve o cache da GPU entre eles.
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        pd.DataFrame(resumos).to_csv(out_dir / "grid_summary.csv", index=False)

    df = pd.DataFrame(resumos)
    ok = df[df.status == "ok"]
    if ok.empty:
        print("[ERRO] nenhum candidato concluiu.", file=sys.stderr)
        return 1
    if metrica not in ok.columns:
        print(f"[ERRO] metrica '{metrica}' ausente.", file=sys.stderr)
        return 1

    melhor = ok.loc[ok[metrica].astype(float).idxmax()]
    print(f"  vencedor: {melhor.net_id}  {metrica}={melhor[metrica]:.4f}")
    net_dir = Path(melhor["net_dir"])
    config = json.loads((net_dir / "config_used.json").read_text())
    for k in ("_env", "_params"):
        config.pop(k, None)
    m_teste = evaluate_on_test(net_dir, args.fold, args.images_dir, config, device,
                               n_classes, class_names, out_dir)

    pd.DataFrame([{**melhor.to_dict(), **{f"test_{k}": v for k, v in m_teste.items()}}]) \
        .to_csv(out_dir / "best_summary.csv", index=False)
    (out_dir / "best_selection.json").write_text(json.dumps(
        {"best_config_id": melhor.net_id, "net_dir": str(net_dir),
         "cand_dir": str(net_dir), "selected_by": metrica}, indent=2))
    print(f"  teste: f1_macro={m_teste.get('f1_macro', float('nan')):.4f}")

    # Pesos dos candidatos que perderam: ~90 MB cada, e nada mais os le
    # (replicas e SHAP usam so o vencedor). Logs, config e predicoes de
    # validacao ficam, para a analise do grid.
    if not args.keep_all_models:
        for outro in ok.net_dir:
            if Path(outro) != net_dir:
                (Path(outro) / "model.pt").unlink(missing_ok=True)
                (Path(outro) / "rf_head.joblib").unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

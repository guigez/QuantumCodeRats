"""
GRID E AVALIACAO DE UMA FAMILIA HIBRIDA EM UM FOLD
==================================================

Mesmo invariante do grid_search.py classico: cada candidato so ve treino e
validacao; o vencedor e escolhido pela validacao; o teste e avaliado uma
unica vez, depois, com tudo congelado. Todos os candidatos de um fold
recebem a mesma semente.

Saida no mesmo formato do classico (best_summary.csv, best_selection.json,
metrics_test*.csv, predictions_test.csv, ...), o que deixa o aggregate.py
classico ler as familias hibridas sem adaptacao.

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
import itertools
import json
import sys
from pathlib import Path

import pandas as pd

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
    args = ap.parse_args()

    device = resolve_device(args.device)
    class_names = class_names_for(args.task)
    n_classes = len(class_names)

    cfg = load_family_config(args.grid)
    base, grid = cfg["base"], cfg.get("grid", {})
    metrica = cfg.get("selection", {}).get("metric", "val_f1_macro")
    candidatos = expand_grid(grid) or [{}]

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[{out_dir.name}] dispositivo={device}  candidatos={len(candidatos)}  "
          f"selecao por {metrica}")

    resumos = []
    for i, combo in enumerate(candidatos):
        c = {**base, **combo, "config_id": f"net_{i:03d}", "seed": args.seed}
        net_dir = out_dir / f"net_{i:03d}"
        print(f"  net_{i:03d}  {combo}")
        try:
            r = run_candidate(args.fold, args.images_dir, c, net_dir, device,
                              n_classes, class_names)
            resumos.append({"net_id": f"net_{i:03d}", **combo, **r, "status": "ok"})
        except Exception as e:
            print(f"    ERRO: {e!r}")
            resumos.append({"net_id": f"net_{i:03d}", **combo, "status": "erro",
                            "error": repr(e), "net_dir": str(net_dir),
                            "cand_dir": str(net_dir)})
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

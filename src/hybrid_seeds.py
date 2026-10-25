"""
REPLICAS POR SEMENTE DE UMA FAMILIA HIBRIDA
===========================================

Mesmo papel do seed_replicates.py classico: retreina a configuracao vencedora
de um (fold, familia) com outras sementes, no mesmo split, e avalia o teste
uma vez por replica. Separa diferenca sistematica entre bracos de sorte de
inicializacao — e no circuito a inicializacao pesa: os angulos comecam
uniformes em [0, 2pi).

Saida em <family_dir>/seeds/seed_<s>/, no formato que o aggregate.py
classico le. model.pt e apagado ao fim de cada replica, a menos de
--keep_models.

Uso:
    python src/hybrid_seeds.py --family_dir resultados/binary/fold_00/h1_vqc \\
        --fold resultados/binary/splits/fold_00.csv --images_dir imagens \\
        --task binary --seeds 1 2 3 4 5
"""

from __future__ import annotations

import classical_bridge  # noqa: F401

from common import resolve_device, silence_known_warnings
silence_known_warnings()

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

import pandas as pd

from manifest import TASKS, class_names_for

from hybrid_train import evaluate_on_test, run_candidate


def load_winner(family_dir: Path) -> tuple[dict, Path]:
    """(selecao, net_dir) do vencedor. Cai para family_dir/<net_id> se o caminho gravado sumiu."""
    sel = json.loads((family_dir / "best_selection.json").read_text())
    net_dir = Path(sel["net_dir"])
    if not net_dir.exists():
        net_dir = family_dir / net_dir.name
    return sel, net_dir


def main() -> int:
    ap = argparse.ArgumentParser(description="Replicas por semente do vencedor hibrido.")
    ap.add_argument("--family_dir", required=True)
    ap.add_argument("--fold", required=True)
    ap.add_argument("--images_dir", default="imagens")
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--task", default="multiclass", choices=list(TASKS))
    ap.add_argument("--seeds", type=int, nargs="+", required=True)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--keep_models", action="store_true")
    args = ap.parse_args()

    device = resolve_device(args.device)
    class_names = class_names_for(args.task)
    n_classes = len(class_names)
    family_dir = Path(args.family_dir)

    sel, net_dir = load_winner(family_dir)
    config = json.loads((net_dir / "config_used.json").read_text())
    for k in ("_env", "_params"):
        config.pop(k, None)

    base_params = {
        "winner": sel.get("best_config_id"),
        "config": {k: v for k, v in config.items() if k not in ("seed", "config_id")},
        "fold_sha": hashlib.sha1(Path(args.fold).read_bytes()).hexdigest()[:16],
    }
    print(f"[{family_dir}] vencedor={sel.get('best_config_id')}  sementes={args.seeds}  "
          f"dispositivo={device}")

    seeds_dir = family_dir / "seeds"
    resumos, falhas = [], []
    for s in args.seeds:
        out = seeds_dir / f"seed_{s}"
        stamp = out / "params.json"
        params = {**base_params, "seed": s}
        if (not args.force and (out / "metrics_test.csv").exists() and stamp.exists()
                and json.loads(stamp.read_text()) == params):
            print(f"  seed {s}: ja concluida, pulando")
            continue
        if out.exists():
            shutil.rmtree(out)

        c = {**config, "seed": s, "config_id": f"seed_{s}"}
        print(f"  seed {s}: treinando...")
        try:
            r = run_candidate(args.fold, args.images_dir, c, out, device, n_classes, class_names)
            m = evaluate_on_test(out, args.fold, args.images_dir, c, device,
                                 n_classes, class_names, out)
        except Exception as e:
            print(f"    ERRO: {e!r}")
            falhas.append(s)
            continue

        if not args.keep_models:
            (out / "model.pt").unlink(missing_ok=True)
        stamp.write_text(json.dumps(params, indent=2))
        resumos.append({"seed": s, "best_epoch": r.get("best_epoch"),
                        "val_f1_macro": r.get("val_f1_macro"),
                        "test_f1_macro": m.get("f1_macro"),
                        "test_balanced_accuracy": m.get("balanced_accuracy")})
        print(f"    best_epoch={r.get('best_epoch')}  val_f1={r.get('val_f1_macro', float('nan')):.4f}"
              f"  teste_f1={m.get('f1_macro', float('nan')):.4f}")

    if resumos:
        novo = pd.DataFrame(resumos)
        resumo_csv = seeds_dir / "seeds_summary.csv"
        if resumo_csv.exists():
            antigo = pd.read_csv(resumo_csv)
            novo = pd.concat([antigo[~antigo.seed.isin(novo.seed)], novo]).sort_values("seed")
        novo.to_csv(resumo_csv, index=False)

    if falhas:
        print(f"[ERRO] sementes com falha: {falhas}", file=sys.stderr)
        return 1
    print("[ok] replicas concluidas")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

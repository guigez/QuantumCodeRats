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
(--keep_all_models mantem). Candidatos que constroem a mesma rede que um
anterior (ver effective_signature) nao sao treinados: entram no resumo com
status "duplicado" e o resultado do original, e ficam fora da selecao.

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

from hybrid_models import matched_hidden, quantum_param_count
from hybrid_train import evaluate_on_test, run_candidate


def expand_grid(grid: dict) -> list[dict]:
    """Produto cartesiano do bloco `grid`. Valor escalar vira lista de um."""
    chaves = list(grid)
    valores = [v if isinstance(v, list) else [v] for v in grid.values()]
    return [dict(zip(chaves, c)) for c in itertools.product(*valores)]


def effective_signature(config: dict) -> str:
    """Chave da rede que a config de fato constroi e treina.

    Combinacoes diferentes do grid podem gerar a mesma rede. No C1, por
    exemplo, c1_match=basic e n_layers=1 resolvem quase sempre para h=1: o
    MLP nao fica menor que isso. Com a mesma semente e treino deterministico,
    treina-las de novo reproduz o mesmo resultado bit a bit. Aqui os
    hiperparametros que a cabeca escolhida ignora saem da chave, e os que ela
    deriva (largura do MLP, taxa do nucleo) entram ja resolvidos.
    """
    c = {k: v for k, v in config.items() if k not in ("config_id",) and not k.startswith("_")}
    head = c.get("head", "vqc")
    lr_head = c.get("head_learning_rate", 1e-3)
    lr_core = c.pop("core_learning_rate", c.pop("quantum_learning_rate", lr_head))
    usa = {"linear": set(), "bottleneck": {"n_qubits"},
           "vqc": {"n_qubits", "n_layers", "ansatz"},
           "reupload": {"n_qubits", "n_layers"}}[head]
    for k in ("n_qubits", "n_layers", "ansatz", "c1_hidden", "c1_match"):
        if k not in usa:
            c.pop(k, None)
    if head == "bottleneck":
        h = config.get("c1_hidden", "match")
        if h == "match":
            n = int(config.get("n_qubits", 8))
            h = matched_hidden(n, quantum_param_count(config.get("c1_match", "strongly"),
                                                      n, int(config.get("n_layers", 4))))
        c["_c1_hidden"] = int(h)
    if head != "linear":
        c["_lr_core"] = float(lr_core)
    return json.dumps(c, sort_keys=True, default=str)


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
    feitos: dict[str, dict] = {}          # assinatura -> linha do primeiro candidato com ela
    for i, combo in enumerate(candidatos):
        c = {**base, **combo, "config_id": f"net_{i:03d}", "seed": args.seed}
        net_dir = out_dir / f"net_{i:03d}"
        print(f"  net_{i:03d}  {combo}")

        assinatura = effective_signature(c)
        if assinatura in feitos:
            orig = feitos[assinatura]
            print(f"    mesma rede que {orig['net_id']}, nao treina de novo")
            resumos.append({**{k: v for k, v in orig.items() if k not in combo},
                            "net_id": f"net_{i:03d}", **combo,
                            "status": "duplicado", "duplicate_of": orig["net_id"]})
            pd.DataFrame(resumos).to_csv(out_dir / "grid_summary.csv", index=False)
            continue

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
                feitos[assinatura] = resumos[-1]
                pd.DataFrame(resumos).to_csv(out_dir / "grid_summary.csv", index=False)
                continue
        try:
            r = run_candidate(args.fold, args.images_dir, c, net_dir, device,
                              n_classes, class_names)
            marca.write_text(json.dumps({"params": params, "result": r}, indent=2, default=str))
            resumos.append({"net_id": f"net_{i:03d}", **combo, **r, "status": "ok"})
            feitos[assinatura] = resumos[-1]
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
    # Varre todas as pastas net_*: inclui duplicatas treinadas antes da
    # deduplicacao existir, que nao aparecem como "ok" no resumo.
    if not args.keep_all_models:
        for outro in out_dir.glob("net_*"):
            if outro.resolve() != net_dir.resolve():
                (outro / "model.pt").unlink(missing_ok=True)
                (outro / "rf_head.joblib").unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

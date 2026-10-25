"""
ATRIBUICAO SHAP DAS FAMILIAS HIBRIDAS
=====================================

Mesma analise do explain_shap.py classico, aplicada ao vencedor de cada
(fold, familia): GradientExplainer por pixel sobre um acerto e um erro por
classe do animal de teste, e o enriquecimento de atribuicao

    E = (atribuicao por pixel dentro da mascara do animal)
        / (atribuicao por pixel fora dela)

Mascara, enriquecimento, escolha de exemplos, amostra de referencia e painel
sao as funcoes do classico, importadas — o numero e o mesmo do artigo.

IMAGENS COMPLETAS (COM ESTEIRA) POR PADRAO
------------------------------------------
O dataset ainda tem a esteira, entao a mascara padrao e a do classico com
--with_background: limiar de Otsu sobre o canal R-B, por quadro. Ela marca a
regiao QUENTE do quadro, nao o corpo do animal; faixas mornas da baia podem
entrar. E sai, portanto, como limite superior da atencao ao animal, como
registrado no artigo. --background_removed so deve ser usado com imagens de
fundo removido; se a mascara de fundo removido cobrir mais de 90% do quadro,
o script avisa e volta para o limiar, como o classico faz.

O QUE E EXPLICADO
-----------------
  softmax   os k logits, como no classico.
  ordinal   log P(y = c) para cada classe. A saida crua do modelo sao k-1
            logits cumulativos; indexa-los pela classe real explicaria a
            quantidade errada (e estouraria o indice na ultima classe).
  c0_rf     a cabeca linear (braco c0_rf_linear_head), que compartilha o
            backbone com a floresta. A floresta nao e diferenciavel; dela sai
            so a importancia das 2048 features (explain_forest do classico).
            Os exemplos vem das predicoes da propria cabeca linear.

O circuito e diferenciavel (backprop no default.qubit), entao o gradiente da
entrada atravessa a cabeca quantica sem tratamento especial.

Uso:
    python src/hybrid_shap.py --family_dir resultados/binary/fold_00/h1_vqc \\
        --fold resultados/binary/splits/fold_00.csv --images_dir imagens --task binary
"""

from __future__ import annotations

import classical_bridge  # noqa: F401

from common import resolve_device, silence_known_warnings
silence_known_warnings()

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from explain_shap import (animal_mask, attribution_enrichment, background_ratio,
                          explain_forest, painel, pick_examples, sample_background,
                          to_display)
from manifest import TASKS, class_names_for

from hybrid_models import create_hybrid_model
from hybrid_train import build_loaders, uses_forest

COVERAGE_LIMIT = 0.9          # mascara de "fundo removido" cobrindo o quadro = imagem com esteira


class ExplainTarget(nn.Module):
    """Uma saida por classe para o explicador (ver docstring do modulo)."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.model(x)
        if self.model.output_kind == "ordinal":
            return torch.log(self.model.probabilities(out).clamp_min(1e-12))
        return out


def explain(target: nn.Module, background: torch.Tensor, X: torch.Tensor,
            nsamples: int = 200) -> np.ndarray:
    """GradientExplainer; devolve (N, C, H, W, k), o formato do classico."""
    import shap
    valores = shap.GradientExplainer(target, background).shap_values(X, nsamples=nsamples)
    if isinstance(valores, list):
        valores = np.stack([np.asarray(v) for v in valores], axis=-1)
    return np.asarray(valores)


def load_winner(family_dir: Path) -> tuple[Path, dict]:
    sel = json.loads((family_dir / "best_selection.json").read_text())
    net_dir = Path(sel["net_dir"])
    if not net_dir.exists():
        net_dir = family_dir / net_dir.name
    config = json.loads((net_dir / "config_used.json").read_text())
    for k in ("_env", "_params"):
        config.pop(k, None)
    return net_dir, config


def run_shap(family_dir, fold_csv, images_dir, task, device, out_dir=None,
             background_removed: bool = False, background_size: int = 32,
             nsamples: int = 200, seed: int = 42) -> pd.DataFrame:
    """Explica o vencedor de uma (fold, familia) e grava painel e tabela de E."""
    family_dir = Path(family_dir)
    out_dir = Path(out_dir) if out_dir else family_dir / "shap"
    out_dir.mkdir(parents=True, exist_ok=True)
    class_names = class_names_for(task)

    net_dir, config = load_winner(family_dir)
    floresta = uses_forest(config)
    model = create_hybrid_model({**config, "pretrained": False}, len(class_names)).to(device)
    model.load_state_dict(torch.load(net_dir / "model.pt", map_location=device)["model_state_dict"])
    model.eval()
    target = ExplainTarget(model).eval()

    loaders = build_loaders(fold_csv, images_dir, config)
    preds_csv = family_dir / ("predictions_test_linear_head.csv" if floresta
                              else "predictions_test.csv")
    arm = family_dir.name + ("_linear_head" if floresta else "")
    exemplos = pick_examples(preds_csv, class_names, seed)
    teste = loaders["test"].dataset
    por_nome = {n: i for i, n in enumerate(teste.df.filename)}
    exemplos = [e for e in exemplos if e["filename"] in por_nome]
    if not exemplos:
        raise RuntimeError(f"nenhum exemplo de {preds_csv} encontrado no teste do fold")
    X = torch.stack([teste[por_nome[e["filename"]]][0] for e in exemplos]).to(device)
    bg = sample_background(loaders["train_eval"].dataset, background_size, seed, device)

    print(f"[{family_dir}] braco={arm}  explicando {len(exemplos)} exemplos "
          f"com referencia de {len(bg)} imagens ({nsamples} amostras)")
    valores = explain(target, bg, X, nsamples)

    if background_removed:
        cobertura = float(np.mean([animal_mask(to_display(x, True), True).mean() for x in X]))
        if cobertura > COVERAGE_LIMIT:
            print(f"[aviso] a mascara de fundo removido cobre {cobertura:.0%} do quadro: "
                  "as imagens ainda tem a esteira. Usando a mascara por limiar.")
            background_removed = False

    painel(exemplos, X, valores, class_names, out_dir / "shap_panel.png",
           True, background_removed)
    linhas = []
    for i, e in enumerate(exemplos):
        heat = np.abs(valores[i, ..., e["y_true"]]).sum(0)
        mask = animal_mask(to_display(X[i], True), background_removed)
        linhas.append({"arm": arm, **e, "background_ratio": background_ratio(heat, mask),
                       **attribution_enrichment(heat, mask),
                       "mask": "fundo_removido" if background_removed else "limiar_R-B"})
    df = pd.DataFrame(linhas)
    df.to_csv(out_dir / "shap_background_ratio.csv", index=False)
    if floresta:
        explain_forest(net_dir, out_dir)

    enr = df.enrichment.replace([np.inf, -np.inf], np.nan)
    print(f"  enriquecimento medio E={enr.mean():.2f}  "
          f"(acertos {enr[df.tipo == 'acerto'].mean():.2f}, erros {enr[df.tipo == 'erro'].mean():.2f})  "
          f"area media do animal {df.animal_area_frac.mean():.1%}  mascara={linhas[0]['mask']}")
    return df


def main() -> int:
    ap = argparse.ArgumentParser(description="SHAP do vencedor de uma familia hibrida.")
    ap.add_argument("--family_dir", required=True)
    ap.add_argument("--fold", required=True)
    ap.add_argument("--images_dir", default="imagens")
    ap.add_argument("--out_dir", default=None, help="Padrao: <family_dir>/shap")
    ap.add_argument("--task", default="multiclass", choices=list(TASKS))
    ap.add_argument("--background_size", type=int, default=32)
    ap.add_argument("--nsamples", type=int, default=200)
    ap.add_argument("--background_removed", action="store_true",
                    help="Imagens SEM a esteira. Por padrao assume imagens completas.")
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    run_shap(args.family_dir, args.fold, args.images_dir, args.task,
             resolve_device(args.device), args.out_dir, args.background_removed,
             args.background_size, args.nsamples, args.seed)
    print("[ok]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

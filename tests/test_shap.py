"""SHAP do vencedor sobre imagens sinteticas: formato, mascara e saida ordinal."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
import torch

pytest.importorskip("shap")

from hybrid_aggregate import shap_summary
from hybrid_models import HybridResNet
from hybrid_shap import ExplainTarget, run_shap
from hybrid_train import evaluate_on_test, run_candidate

CFG = {"input_size": [32, 32], "pretrained": False, "train_layer4": True,
       "n_qubits": 2, "n_layers": 1, "epochs": 1, "patience": 0, "lr_patience": 1,
       "batch_size": 4, "num_workers": 0, "augment": False, "seed": 7,
       "rf_n_estimators": 10}


def _vencedor(synthetic_fold, family_dir, **cfg):
    """Treina, avalia e grava best_selection.json como o hybrid_grid faria."""
    fold_csv, img_dir, nomes = synthetic_fold
    c = {**CFG, **cfg}
    net = family_dir / "net_000"
    run_candidate(fold_csv, img_dir, c, net, torch.device("cpu"), len(nomes), nomes)
    evaluate_on_test(net, fold_csv, img_dir, c, torch.device("cpu"), len(nomes), nomes, family_dir)
    (family_dir / "best_selection.json").write_text(json.dumps(
        {"best_config_id": "net_000", "net_dir": str(net), "cand_dir": str(net)}))
    return fold_csv, img_dir


@pytest.mark.parametrize("familia,cfg,braco", [
    ("h1_vqc", {"head": "vqc"}, "h1_vqc"),
    ("c0_rf", {"head": "linear", "rf_head": True}, "c0_rf_linear_head"),
])
def test_shap_imagens_com_esteira(synthetic_fold, tmp_path, familia, cfg, braco):
    fam = tmp_path / "binary" / "fold_00" / familia
    fold_csv, img_dir = _vencedor(synthetic_fold, fam, **cfg)
    df = run_shap(fam, fold_csv, img_dir, "binary", torch.device("cpu"),
                  background_size=4, nsamples=8)

    assert (fam / "shap" / "shap_panel.png").exists()
    salvo = pd.read_csv(fam / "shap" / "shap_background_ratio.csv")
    assert len(salvo) == len(df) > 0
    assert set(salvo.arm) == {braco}
    assert set(salvo["mask"]) == {"limiar_R-B"}              # padrao: imagens completas
    assert salvo.enrichment.replace([np.inf, -np.inf], np.nan).notna().any()
    if familia == "c0_rf":
        assert (fam / "shap" / "rf_feature_importance.csv").exists()

    _, resumo = shap_summary(tmp_path / "binary", ["fold_00"])
    assert resumo.arm.tolist() == [braco]


def test_alvo_ordinal_tem_uma_saida_por_classe():
    m = HybridResNet(4, head="vqc", output="ordinal", pretrained=False,
                     n_qubits=2, n_layers=1).eval()
    out = ExplainTarget(m)(torch.rand(2, 3, 32, 32))
    assert out.shape == (2, 4)
    assert torch.allclose(out.exp().sum(1), torch.ones(2), atol=1e-4)

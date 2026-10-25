"""Treino ponta a ponta de poucas epocas sobre imagens sinteticas."""

from __future__ import annotations

import json

import pandas as pd
import pytest
import torch

from hybrid_grid import expand_grid
from hybrid_train import evaluate_on_test, run_candidate

CFG = {"input_size": [32, 32], "pretrained": False, "train_layer4": True,
       "n_qubits": 2, "n_layers": 1, "epochs": 2, "patience": 0, "lr_patience": 1,
       "batch_size": 4, "num_workers": 0, "augment": False, "seed": 7,
       "use_class_weights": True, "class_weight_scheme": "balanced"}


@pytest.mark.parametrize("head,output", [("vqc", "softmax"), ("bottleneck", "softmax"),
                                         ("reupload", "softmax"), ("vqc", "ordinal")])
def test_candidato_e_teste(synthetic_fold, tmp_path, head, output):
    fold_csv, img_dir, nomes = synthetic_fold
    cfg = {**CFG, "head": head, "output": output}
    out = tmp_path / f"{head}_{output}"
    r = run_candidate(fold_csv, img_dir, cfg, out, torch.device("cpu"), len(nomes), nomes)

    assert r["epochs_run"] == 2 and "val_f1_macro" in r
    log = pd.read_csv(out / "train_log.csv")
    assert len(log) == 2
    if head != "bottleneck":
        assert log["quantum_grad_norm"].notna().all()
    usado = json.loads((out / "config_used.json").read_text())
    assert "pennylane" in usado["_env"] and "_params" in usado

    m = evaluate_on_test(out, fold_csv, img_dir, cfg, torch.device("cpu"),
                         len(nomes), nomes, out)
    assert 0.0 <= m["f1_macro"] <= 1.0
    pred = pd.read_csv(out / "predictions_test.csv")
    assert set(pred.subject_id) == {1}
    assert (out / "metrics_test_by_subject.csv").exists()


def test_c0_rf_gera_dois_bracos(synthetic_fold, tmp_path):
    fold_csv, img_dir, nomes = synthetic_fold
    cfg = {**CFG, "head": "linear", "output": "softmax", "rf_head": True,
           "rf_n_estimators": 10, "rf_min_samples_leaf": 1}
    out = tmp_path / "c0_rf"
    r = run_candidate(fold_csv, img_dir, cfg, out, torch.device("cpu"), len(nomes), nomes)
    assert "linear_val_f1_macro" in r and "val_f1_macro" in r
    assert (out / "rf_head.joblib").exists()
    assert (out / "predictions_val_linear_head.csv").exists()

    evaluate_on_test(out, fold_csv, img_dir, cfg, torch.device("cpu"), len(nomes), nomes, out)
    for prefixo in ("test", "test_linear_head"):
        assert (out / f"metrics_{prefixo}.csv").exists()
        assert (out / f"metrics_{prefixo}_by_subject.csv").exists()


def test_rf_exige_cabeca_linear(synthetic_fold, tmp_path):
    fold_csv, img_dir, nomes = synthetic_fold
    cfg = {**CFG, "head": "vqc", "rf_head": True}
    with pytest.raises(ValueError):
        run_candidate(fold_csv, img_dir, cfg, tmp_path / "x", torch.device("cpu"), 2, nomes)


def test_mesma_semente_mesmo_resultado(synthetic_fold, tmp_path):
    fold_csv, img_dir, nomes = synthetic_fold
    cfg = {**CFG, "head": "vqc", "output": "softmax", "epochs": 1}
    a = run_candidate(fold_csv, img_dir, cfg, tmp_path / "a", torch.device("cpu"), 2, nomes)
    b = run_candidate(fold_csv, img_dir, cfg, tmp_path / "b", torch.device("cpu"), 2, nomes)
    la = pd.read_csv(tmp_path / "a" / "train_log.csv")
    lb = pd.read_csv(tmp_path / "b" / "train_log.csv")
    assert la.train_loss.iloc[0] == pytest.approx(lb.train_loss.iloc[0], rel=1e-5)
    assert a["val_f1_macro"] == pytest.approx(b["val_f1_macro"])


def test_grid_vazio_e_um_candidato():
    assert expand_grid({}) == [{}]
    assert len(expand_grid({"n_qubits": [4, 8], "augment": True})) == 2

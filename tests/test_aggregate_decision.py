"""Agregacao entre raizes e criterio de decisao, sobre resultados sinteticos."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from metrics import predictions_dataframe, save_all

from hybrid_aggregate import (check_same_subjects, collect_roots, decision_table,
                              default_pairs, diagnostics)

NOMES = ["warm-up+moderate", "heavy+severe"]
ANIMAIS = {"fold_00": 1, "fold_01": 9, "fold_02": 10, "fold_03": 11}


def _escreve(base, sid, acc, rng, prefix="test", n=60):
    y = np.arange(n) % 2
    certo = rng.random(n) < acc
    yp = np.where(certo, y, 1 - y)
    probs = np.eye(2)[yp]
    save_all(predictions_dataframe([f"f{i}" for i in range(n)], [sid] * n, y, yp, probs),
             NOMES, base, prefix=prefix)


def _familia(root, arm, accs, rng, seeds=(1, 2, 3), linear=False):
    """accs: fold -> acuracia media; cada semente perturba um pouco."""
    for fold, sid in ANIMAIS.items():
        fam = root / fold / arm
        _escreve(fam, sid, accs[fold], rng)
        cols = {"val_f1_macro": [accs[fold]]}
        if linear:
            _escreve(fam, sid, accs[fold], rng, prefix="test_linear_head")
            cols["linear_val_f1_macro"] = [accs[fold]]
        pd.DataFrame(cols).to_csv(fam / "best_summary.csv", index=False)
        linhas = []
        for s in seeds:
            a = float(np.clip(accs[fold] + rng.normal(0, 0.01), 0, 1))
            _escreve(fam / "seeds" / f"seed_{s}", sid, a, rng)
            if linear:
                _escreve(fam / "seeds" / f"seed_{s}", sid, a, rng, prefix="test_linear_head")
            linhas.append({"seed": s, "val_f1_macro": a})
        pd.DataFrame(linhas).to_csv(fam / "seeds" / "seeds_summary.csv", index=False)


@pytest.fixture
def raizes(tmp_path):
    rng = np.random.default_rng(3)
    q, c = tmp_path / "quantum", tmp_path / "classical"
    _familia(q, "h1_vqc", {f: 0.95 for f in ANIMAIS}, rng)
    _familia(q, "c1_bottleneck", {f: 0.70 for f in ANIMAIS}, rng)
    _familia(c, "resnet50_rf", {f: 0.60 for f in ANIMAIS}, rng, linear=True)
    return [(q, None), (c, {"resnet50_rf_linear_head"})]


def test_coleta_filtra_bracos_classicos(raizes):
    _, ps = collect_roots(raizes, list(ANIMAIS))
    assert set(ps.arm) == {"h1_vqc", "c1_bottleneck", "resnet50_rf_linear_head"}
    check_same_subjects(ps)


def test_pares_padrao():
    pares = default_pairs(["c1_bottleneck", "c1_ordinal", "h1_vqc", "h3_vqc_ordinal",
                           "resnet50_rf_linear_head"])
    assert ("h1_vqc", "c1_bottleneck") in pares
    assert ("h1_vqc", "c1_ordinal") not in pares
    assert ("h3_vqc_ordinal", "c1_ordinal") in pares
    assert ("h3_vqc_ordinal", "c1_bottleneck") not in pares
    assert ("h1_vqc", "resnet50_rf_linear_head") in pares


def test_hibrido_claramente_melhor(raizes):
    _, rep = collect_roots(raizes, list(ANIMAIS), seeds=True)
    diag = diagnostics(raizes, list(ANIMAIS), NOMES)
    dec = decision_table(rep, diag, [("h1_vqc", "c1_bottleneck"),
                                     ("h1_vqc", "resnet50_rf_linear_head")], "f1_macro")
    assert set(dec.verdict) == {"melhor"}
    assert (dec.wins_hybrid == 4).all() and (dec.n_seeds_hybrid == 3).all()


def test_derrota_em_um_animal_nao_e_superior(tmp_path):
    rng = np.random.default_rng(5)
    q = tmp_path / "q"
    accs = {f: 0.95 for f in ANIMAIS}
    accs["fold_02"] = 0.50
    _familia(q, "h1_vqc", accs, rng)
    _familia(q, "c1_bottleneck", {f: 0.70 for f in ANIMAIS}, rng)
    roots = [(q, None)]
    _, rep = collect_roots(roots, list(ANIMAIS), seeds=True)
    dec = decision_table(rep, diagnostics(roots, list(ANIMAIS), NOMES),
                         [("h1_vqc", "c1_bottleneck")], "f1_macro")
    assert dec.verdict.iloc[0] == "nao superior"
    assert dec.wins_hybrid.iloc[0] == 3


def test_sem_sementes_criterio_2_indeterminado(raizes):
    _, ps = collect_roots(raizes, list(ANIMAIS))
    dec = decision_table(ps, diagnostics(raizes, list(ANIMAIS), NOMES),
                         [("h1_vqc", "c1_bottleneck")], "f1_macro")
    assert dec.c2_above_2x_noise.iloc[0] is None
    assert dec.verdict.iloc[0] == "indeterminado"


def test_animais_diferentes_abortam():
    df = pd.DataFrame({"fold": ["fold_00", "fold_00"], "subject_id": [1, 9],
                       "arm": ["a", "b"]})
    with pytest.raises(RuntimeError):
        check_same_subjects(df)

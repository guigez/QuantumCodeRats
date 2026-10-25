"""Fixtures compartilhadas. src/ deste repositorio entra no sys.path, e o
classical_bridge poe o src/ do classico logo em seguida."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

import classical_bridge  # noqa: E402,F401

from manifest import class_names_for  # noqa: E402


@pytest.fixture
def synthetic_fold(tmp_path):
    """Imagens RGB pequenas e um CSV de fold binario: treino {2,3}, val {9}, teste {1}."""
    from PIL import Image

    rng = np.random.default_rng(0)
    img_dir = tmp_path / "imgs"
    img_dir.mkdir()
    linhas = []
    for sid, split in ((2, "train"), (3, "train"), (9, "val"), (1, "test")):
        for label in (0, 1):
            for i in range(6):
                nome = f"s{sid}_c{label}_{i}.png"
                base = 60 + 120 * label                      # classe 1 mais "quente"
                arr = np.clip(rng.normal(base, 25, (40, 40, 3)), 0, 255).astype("uint8")
                Image.fromarray(arr).save(img_dir / nome)
                linhas.append({"filename": nome, "filepath": nome, "subject_id": sid,
                               "class_name": "warm-up" if label == 0 else "heavy",
                               "label": label, "split": split, "fold": 0})
    fold_csv = tmp_path / "fold_00.csv"
    pd.DataFrame(linhas).to_csv(fold_csv, index=False)
    return fold_csv, img_dir, class_names_for("binary")

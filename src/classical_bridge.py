"""
PONTE PARA O REPOSITORIO CLASSICO
=================================

Este repositorio nao copia nada do classicalCodeRats. Particao, acesso as
imagens, metricas, agregacao e o proprio backbone ResNet50 sao importados de
la. E isso que torna a comparacao verificavel: dobras, transformacoes,
pesos de classe, metricas por animal e testes pareados sao o MESMO codigo nos
dois bracos, e nao uma reimplementacao que pode divergir em silencio.

Os modulos classicos se importam pelo nome curto (`from common import ...`),
entao o `src/` de la entra no sys.path. Nenhum modulo deste repositorio usa um
nome que colida com os de la — por isso todos comecam com `hybrid_`.

Onde procurar o repositorio classico, nesta ordem:

  1. variavel de ambiente CLASSICAL_CODE;
  2. Bernini/Classical/classicalCodeRats ou ClassicalCodeRats;
  3. Bernini/Classical/Code                 (estrutura local antiga);
  4. external/classicalCodeRats             (submodulo git, se existir).

O orquestrador resolve o caminho uma vez e o repassa aos subprocessos pela
mesma variavel, para que todos usem a mesma copia.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]          # raiz deste repositorio
ENV_VAR = "CLASSICAL_CODE"

# Modulos classicos usados aqui. Se algum estiver corrompido ou ausente, falha
# na hora com mensagem clara, em vez de um SyntaxError no meio de um treino.
REQUIRED = ("common", "dataset", "manifest", "splits", "metrics", "aggregate",
            "models", "train", "explain_shap")


def _candidates() -> list[Path]:
    quantum_dir = ROOT.parent                       # Bernini/Quantum
    bernini = quantum_dir.parent                    # Bernini
    out = []
    if os.environ.get(ENV_VAR):
        out.append(Path(os.environ[ENV_VAR]))
    # As duas grafias: o servidor e Linux, onde a caixa do nome importa.
    out += [bernini / "Classical" / "classicalCodeRats",
            bernini / "Classical" / "ClassicalCodeRats",
            bernini / "Classical" / "Code",
            ROOT / "external" / "classicalCodeRats"]
    return out


def find_classical() -> Path:
    """Raiz do repositorio classico (a pasta que contem `src/` e `run_all.py`)."""
    tentados = []
    for c in _candidates():
        c = c.resolve()
        tentados.append(str(c))
        if (c / "src" / "aggregate.py").exists() and (c / "run_all.py").exists():
            return c
    raise FileNotFoundError(
        "repositorio classico nao encontrado. Defina CLASSICAL_CODE com o caminho "
        "do classicalCodeRats. Tentados:\n  " + "\n  ".join(tentados))


def check_classical(root: Path) -> None:
    """Compila os modulos usados; aborta com instrucao de reparo se falhar."""
    ruins = []
    for nome in REQUIRED:
        f = root / "src" / f"{nome}.py"
        try:
            compile(f.read_text(encoding="utf-8"), str(f), "exec")
        except (SyntaxError, ValueError, UnicodeDecodeError, FileNotFoundError) as e:
            ruins.append(f"{f}: {e.__class__.__name__}")
    if ruins:
        raise ImportError(
            "modulos do repositorio classico ilegiveis:\n  " + "\n  ".join(ruins) +
            f"\nSe o repositorio esta limpo no git, restaure com:\n"
            f"  git -C {root} checkout -- src/")


def install() -> Path:
    """Poe src/ e a raiz do classico no sys.path. Idempotente."""
    root = find_classical()
    check_classical(root)
    for p in (root / "src", root):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))
    os.environ[ENV_VAR] = str(root)
    return root


def git_commit(repo_dir: Path) -> str | None:
    """Hash do commit, com sufixo '-dirty' se houver alteracao local."""
    try:
        sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo_dir,
                             capture_output=True, text=True, check=True).stdout.strip()
        sujo = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"],
                              cwd=repo_dir, capture_output=True, text=True,
                              check=True).stdout.strip()
        return sha + ("-dirty" if sujo else "")
    except (OSError, subprocess.CalledProcessError):
        return None


CLASSICAL_ROOT = install()

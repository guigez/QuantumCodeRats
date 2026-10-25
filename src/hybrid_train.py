"""
TREINO DE UM CANDIDATO HIBRIDO (Regime B, ponta a ponta)
========================================================

Espelha `train.py` do classico — mesmo dataset, mesmas transformacoes, mesmos
pesos de classe, mesma metrica de selecao, mesmo early stopping e
ReduceLROnPlateau na validacao — com tres diferencas, todas por necessidade:

  1. Perda pluggable: entropia cruzada ou ordinal (H3 / C1-ord).
  2. Tres grupos de parametros com taxa propria:
       backbone (layer4)   learning_rate        (1e-4, a do classico)
       cabeca classica     head_learning_rate
       circuito            quantum_learning_rate (padrao = head_learning_rate)
     O circuito NAO recebe weight decay: os parametros sao angulos
     periodicos, e puxa-los para zero e um vies sem sentido fisico.
     C1 e H1 usam a mesma head_learning_rate, entao a comparacao entre eles
     nao e confundida pela taxa.
  3. Diagnostico de barren plateau: a norma media do gradiente dos
     parametros do circuito vai para train_log.csv a cada epoca. Gradiente
     que encolhe para ~0 com o treino ainda longe da convergencia e o
     sintoma.

O TESTE nunca e tocado aqui. Quem avalia o teste e hybrid_grid.py, uma vez,
depois de escolher o vencedor.
"""

from __future__ import annotations

import classical_bridge  # noqa: F401

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from common import environment_info, seed_everything
from dataset import ThermalFrameDataset, build_transforms, class_weights_from_split
from metrics import compute_metrics, predictions_dataframe, save_all

from hybrid_models import create_hybrid_model

DEFAULT_INPUT_SIZE = (224, 224)


def build_loaders(fold_csv, images_dir, config) -> dict:
    """Loaders de treino (com augmentation), validacao e teste.

    Normalizacao ImageNet sempre ligada: o backbone e a ResNet50 pre-treinada,
    como na familia resnet50_rf do classico.
    """
    size = tuple(config.get("input_size", DEFAULT_INPUT_SIZE))
    tf_treino, tf_eval = build_transforms(
        size, imagenet_norm=True,
        augment=bool(config.get("augment", True)),
        hflip=bool(config.get("hflip", False)),
    )
    bs = int(config.get("batch_size", 32))
    nw = int(config.get("num_workers", 4))
    pin = torch.cuda.is_available()
    g = torch.Generator(); g.manual_seed(int(config.get("seed", 42)))

    loaders = {"train": DataLoader(ThermalFrameDataset(fold_csv, images_dir, "train", tf_treino),
                                   batch_size=bs, shuffle=True, generator=g,
                                   num_workers=nw, pin_memory=pin, drop_last=True)}
    for split in ("val", "test"):
        loaders[split] = DataLoader(ThermalFrameDataset(fold_csv, images_dir, split, tf_eval),
                                    batch_size=bs, shuffle=False,
                                    num_workers=nw, pin_memory=pin)
    return loaders


@torch.no_grad()
def infer(model, loader, device):
    """Probabilidades, rotulos, sujeitos e nomes de arquivo."""
    model.eval()
    P, Y, S, F = [], [], [], []
    for x, y, s, f in loader:
        P.append(model.probabilities(model(x.to(device))).float().cpu().numpy())
        Y.append(y.numpy()); S.append(s.numpy()); F.extend(list(f))
    return np.concatenate(P), np.concatenate(Y), np.concatenate(S), F


def build_optimizer(model, config):
    lr = float(config.get("learning_rate", 1e-4))
    lr_head = float(config.get("head_learning_rate", 1e-3))
    lr_q = float(config.get("quantum_learning_rate", lr_head))
    wd = float(config.get("weight_decay", 1e-4))
    grupos = model.parameter_groups()
    param_groups = [{"params": grupos["backbone"], "lr": lr, "weight_decay": wd, "name": "backbone"},
                    {"params": grupos["head"], "lr": lr_head, "weight_decay": wd, "name": "head"}]
    if grupos["quantum"]:
        param_groups.append({"params": grupos["quantum"], "lr": lr_q,
                             "weight_decay": 0.0, "name": "quantum"})
    return torch.optim.Adam([g for g in param_groups if g["params"]])


def _grad_norm(params) -> float:
    gs = [p.grad.detach().norm().item() for p in params if p.grad is not None]
    return float(np.mean(gs)) if gs else float("nan")


def train_model(model, loaders, config, device, n_classes, class_names, out_dir):
    """Laco de treino com early stopping no escore de validacao."""
    epochs = int(config.get("epochs", 200))
    patience = int(config.get("patience", 15))
    metrica = str(config.get("selection_metric", "f1_macro"))

    pesos = (class_weights_from_split(loaders["train"].dataset, n_classes,
                                      config.get("class_weight_scheme", "balanced")).to(device)
             if config.get("use_class_weights", True) else None)
    otimizador = build_optimizer(model, config)
    reduz = torch.optim.lr_scheduler.ReduceLROnPlateau(
        otimizador, mode="max", factor=0.1, patience=int(config.get("lr_patience", 6)))
    q_params = model.parameter_groups()["quantum"]

    melhor, melhor_estado, melhor_epoca, espera = -np.inf, None, -1, 0
    log = []
    t0 = time.time()
    for ep in range(1, epochs + 1):
        model.train()
        perda_acc, n, gq = 0.0, 0, []
        for x, y, _, _ in loaders["train"]:
            x, y = x.to(device), y.to(device)
            otimizador.zero_grad(set_to_none=True)
            perda = model.loss(model(x), y, pesos)
            perda.backward()
            if q_params:
                gq.append(_grad_norm(q_params))
            otimizador.step()
            perda_acc += float(perda.item()) * x.size(0); n += x.size(0)

        probs, y_true, _, _ = infer(model, loaders["val"], device)
        m = compute_metrics(y_true, probs.argmax(1), class_names)
        score = float(m.get(metrica, m["f1_macro"]))
        reduz.step(score)
        log.append({"epoch": ep, "train_loss": perda_acc / max(n, 1),
                    **{f"lr_{g['name']}": g["lr"] for g in otimizador.param_groups},
                    "quantum_grad_norm": float(np.nanmean(gq)) if gq else float("nan"),
                    **{f"val_{k}": v for k, v in m.items()}})
        print(f"    ep {ep:3d}  loss={log[-1]['train_loss']:.4f}  val_{metrica}={score:.4f}"
              + (f"  |grad_q|={log[-1]['quantum_grad_norm']:.2e}" if gq else ""))

        if score > melhor:
            melhor, melhor_epoca, espera = score, ep, 0
            melhor_estado = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            espera += 1
            if patience > 0 and espera >= patience:
                break

    if melhor_estado is not None:
        model.load_state_dict(melhor_estado)
    pd.DataFrame(log).to_csv(Path(out_dir) / "train_log.csv", index=False)
    return {"best_epoch": melhor_epoca, "best_val_score": melhor,
            "epochs_run": len(log), "seconds": time.time() - t0}


def run_environment() -> dict:
    """Versoes do classico (torch, sklearn, commit do classico) + PennyLane e este repo."""
    import pennylane as qml
    env = environment_info()
    env["classical_git_commit"] = env.pop("git_commit", None)
    env["classical_root"] = str(classical_bridge.CLASSICAL_ROOT)
    env["pennylane"] = qml.__version__
    env["quantum_git_commit"] = classical_bridge.git_commit(classical_bridge.ROOT)
    return env


def run_candidate(fold_csv, images_dir, config, out_dir, device, n_classes, class_names):
    """Treina UMA configuracao e devolve o resumo com as metricas de validacao."""
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(int(config.get("seed", 42)))

    loaders = build_loaders(fold_csv, images_dir, config)
    model = create_hybrid_model(config, n_classes).to(device)
    desc = model.describe()
    print(f"    parametros: {desc}")
    resumo = train_model(model, loaders, config, device, n_classes, class_names, out_dir)

    torch.save({"model_state_dict": model.state_dict(), "config": config}, out_dir / "model.pt")
    with open(out_dir / "config_used.json", "w") as fh:
        json.dump({**config, "_params": desc, "_env": run_environment()}, fh,
                  indent=2, default=str)

    probs, y, s, f = infer(model, loaders["val"], device)
    predictions_dataframe(f, s, y, probs.argmax(1), probs) \
        .to_csv(out_dir / "predictions_val.csv", index=False)
    m = compute_metrics(y, probs.argmax(1), class_names)
    return {"net_dir": str(out_dir), "cand_dir": str(out_dir), **resumo, **desc,
            **{f"val_{k}": v for k, v in m.items()}}


def evaluate_on_test(net_dir, fold_csv, images_dir, config, device, n_classes,
                     class_names, out_dir) -> dict:
    """Avaliacao unica no teste, com o modelo ja congelado."""
    loaders = build_loaders(fold_csv, images_dir, config)
    model = create_hybrid_model({**config, "pretrained": False}, n_classes).to(device)
    estado = torch.load(Path(net_dir) / "model.pt", map_location=device)
    model.load_state_dict(estado["model_state_dict"])
    probs, y, s, f = infer(model, loaders["test"], device)
    return save_all(predictions_dataframe(f, s, y, probs.argmax(1), probs),
                    class_names, out_dir, prefix="test")

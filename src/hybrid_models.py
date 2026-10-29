"""
MODELOS HIBRIDOS (PennyLane + PyTorch)
======================================

Todas as familias compartilham o backbone do classico — a MESMA classe
`ResNet50Layer4` do classicalCodeRats, pre-treinada no ImageNet, com tudo
congelado exceto layer4. O que muda entre familias e so a cabeca sobre o
vetor de 2048 dimensoes do avgpool:

  head="linear"      2048 -> k                       (C0; com rf_head, tambem C-RF)
  head="bottleneck"  2048 -> n -> MLP pequeno -> k              (C1, controle)
  head="vqc"         2048 -> n -> circuito variacional -> k     (H1)
  head="reupload"    2048 -> n -> circuito com re-uploading -> k (H2)

e, ortogonal a isso, a saida:

  output="softmax"   k logits, entropia cruzada
  output="ordinal"   um escore e k-1 limiares ordenados (H3 / C1-ord)

Tudo e treinado ponta a ponta (Regime B): o gradiente do circuito volta para
a layer4.

POR QUE O CONTROLE C1 EXISTE
----------------------------
A cabeca hibrida difere da linear em DUAS coisas ao mesmo tempo: um gargalo
de n dimensoes e o circuito. Sem C1, um ganho nao separa o efeito do gargalo
do efeito do circuito. C1 tem o mesmo gargalo, a mesma compressao, o mesmo
squash (pi/2)*tanh e uma saida limitada a [-1, 1] como <Z>, com numero de
parametros casado com o do circuito. A unica diferenca que sobra e o
circuito ser quantico.

CODIFICACAO
-----------
x = (pi/2) * tanh(W z + b) leva cada componente para [-pi/2, pi/2], e
AngleEmbedding com RY gira cada qubit por esse angulo. A faixa evita a
ambiguidade de RY ser periodica: angulos alem de pi voltariam a estados ja
usados por outras entradas.

ONDE O CIRCUITO RODA
--------------------
`quantum_run_on="cpu"` (padrao) mantem os parametros do circuito na CPU e
copia as n ativacoes de cada lote para la e de volta. Com 8 qubits o vetor de
estado tem 256 amplitudes; a copia custa microssegundos, e evita depender de
o simulador aceitar tensores CUDA na versao instalada do PennyLane.
`quantum_run_on="input"` roda onde o dado estiver.
"""

from __future__ import annotations

import math

import classical_bridge  # noqa: F401  (poe o src/ classico no sys.path)

import pennylane as qml
import torch
import torch.nn as nn
import torch.nn.functional as F

from models import ResNet50Layer4

FEATURE_DIM = ResNet50Layer4.FEATURE_DIM
HEADS = ("linear", "bottleneck", "vqc", "reupload")
OUTPUTS = ("softmax", "ordinal")
ANSATZE = ("strongly", "basic")


# ==========================================================================
# Contagem de parametros do circuito, para casar o controle classico
# ==========================================================================
def quantum_param_count(kind: str, n_qubits: int, n_layers: int) -> int:
    """Parametros treinaveis do circuito, sem contar compressao e saida."""
    if kind == "strongly":
        return 3 * n_qubits * n_layers           # Rot(phi, theta, omega) por qubit
    if kind == "basic":
        return n_qubits * n_layers               # um RY por qubit
    if kind == "reupload":
        return 4 * n_qubits * n_layers           # Rot + um peso de escala por entrada
    raise ValueError(f"circuito '{kind}' desconhecido")


def matched_hidden(n: int, target: int) -> int:
    """Largura h do MLP n -> h -> n cujo total (2nh + h + n) fica mais perto de target."""
    return max(1, round((target - n) / (2 * n + 1)))


def count_parameters(module: nn.Module, trainable_only: bool = True) -> int:
    return sum(p.numel() for p in module.parameters()
               if p.requires_grad or not trainable_only)


# ==========================================================================
# Nucleos: o que fica entre a compressao e a saida
# ==========================================================================
class ClassicalCore(nn.Module):
    """C1: n -> h -> n com tanh, saida em [-1, 1] como um vetor de <Z>."""

    def __init__(self, n: int, hidden: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n, hidden), nn.Tanh(),
                                 nn.Linear(hidden, n), nn.Tanh())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _uniform_angles(t: torch.Tensor) -> torch.Tensor:
    return nn.init.uniform_(t, 0.0, 2 * math.pi)


class QuantumCore(nn.Module):
    """Circuito variacional como camada torch (`qml.qnn.TorchLayer`).

    kind="strongly"  AngleEmbedding(RY) + StronglyEntanglingLayers x L
    kind="basic"     AngleEmbedding(RY) + BasicEntanglerLayers(RY) x L
    kind="reupload"  [AngleEmbedding(RY, s_l * x) + StronglyEntangling x 1] x L

    Mede <Z_i> em cada qubit: observaveis locais, que atenuam barren plateau
    em relacao a uma medicao global.
    """

    def __init__(self, n_qubits: int, n_layers: int, kind: str = "strongly",
                 device_name: str = "default.qubit", diff_method: str = "backprop",
                 run_on: str = "cpu"):
        super().__init__()
        if run_on not in ("cpu", "input"):
            raise ValueError("quantum_run_on deve ser 'cpu' ou 'input'")
        self.kind, self.n_qubits, self.n_layers, self.run_on = kind, n_qubits, n_layers, run_on
        wires = list(range(n_qubits))
        dev = qml.device(device_name, wires=n_qubits)

        if kind == "strongly":
            def circuit(inputs, weights):
                qml.AngleEmbedding(inputs, wires=wires, rotation="Y")
                qml.StronglyEntanglingLayers(weights, wires=wires)
                return [qml.expval(qml.PauliZ(w)) for w in wires]
            shapes = {"weights": (n_layers, n_qubits, 3)}
            init = {"weights": _uniform_angles}
        elif kind == "basic":
            def circuit(inputs, weights):
                qml.AngleEmbedding(inputs, wires=wires, rotation="Y")
                qml.BasicEntanglerLayers(weights, wires=wires, rotation=qml.RY)
                return [qml.expval(qml.PauliZ(w)) for w in wires]
            shapes = {"weights": (n_layers, n_qubits)}
            init = {"weights": _uniform_angles}
        elif kind == "reupload":
            def circuit(inputs, weights, scales):
                for layer in range(n_layers):
                    qml.AngleEmbedding(inputs * scales[layer], wires=wires, rotation="Y")
                    qml.StronglyEntanglingLayers(weights[layer:layer + 1], wires=wires)
                return [qml.expval(qml.PauliZ(w)) for w in wires]
            shapes = {"weights": (n_layers, n_qubits, 3), "scales": (n_layers, n_qubits)}
            # Escalas comecam em 1: a primeira passada ve a mesma codificacao do H1.
            init = {"weights": _uniform_angles, "scales": nn.init.ones_}
        else:
            raise ValueError(f"circuito '{kind}' desconhecido")

        qnode = qml.QNode(circuit, dev, interface="torch", diff_method=diff_method)
        self.layer = qml.qnn.TorchLayer(qnode, shapes, init_method=init)

    def _apply(self, fn, *args, **kwargs):
        # model.to(cuda) nao arrasta o circuito quando ele roda na CPU. O
        # load_state_dict continua funcionando: copy_ atravessa dispositivos.
        if self.run_on == "cpu":
            return self
        return super()._apply(fn, *args, **kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.run_on == "cpu" and x.device.type != "cpu":
            return self.layer(x.cpu()).to(device=x.device, dtype=x.dtype)
        return self.layer(x).to(dtype=x.dtype)


# ==========================================================================
# Saidas
# ==========================================================================
class OrdinalOutput(nn.Module):
    """Um escore s e k-1 limiares b_1 < ... < b_{k-1}; logit_j = s - b_j.

    A ordem dos limiares e garantida por construcao: b_1 livre e cada
    incremento positivo via softplus. Comeca em b = (-1, 0, 1, ...).
    Responde a F4 da analise: ~90% dos erros caem na classe vizinha, e uma
    saida softmax trata Baixo->Severo como erro do mesmo tamanho que
    Intenso->Severo.
    """

    def __init__(self, in_dim: int, n_classes: int):
        super().__init__()
        if n_classes < 2:
            raise ValueError("saida ordinal exige ao menos 2 classes")
        self.score = nn.Linear(in_dim, 1)
        self.first = nn.Parameter(torch.tensor([-1.0]))
        self.gaps = nn.Parameter(torch.full((n_classes - 2,), math.log(math.e - 1)))

    def thresholds(self) -> torch.Tensor:
        if self.gaps.numel() == 0:
            return self.first
        return torch.cat([self.first, self.first + torch.cumsum(F.softplus(self.gaps), 0)])

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.score(h) - self.thresholds()


def ordinal_probabilities(logits: torch.Tensor) -> torch.Tensor:
    """P(y = c) a partir de P(y > j) = sigmoid(logit_j)."""
    c = torch.sigmoid(logits)
    um = torch.ones_like(c[:, :1])
    zero = torch.zeros_like(c[:, :1])
    p = (torch.cat([um, c], 1) - torch.cat([c, zero], 1)).clamp_min(0)
    return p / p.sum(1, keepdim=True).clamp_min(1e-12)


def ordinal_loss(logits: torch.Tensor, y: torch.Tensor,
                 class_weights: torch.Tensor | None = None) -> torch.Tensor:
    """BCE sobre os alvos cumulativos (y > j), somada nos limiares (CORAL).

    Pesos de classe entram por amostra, normalizados pela soma dos pesos —
    o mesmo que a CrossEntropyLoss ponderada faz, para manter as duas saidas
    na mesma escala de perda.
    """
    j = torch.arange(logits.shape[1], device=y.device)
    alvo = (y[:, None] > j[None, :]).to(logits.dtype)
    por_amostra = F.binary_cross_entropy_with_logits(logits, alvo, reduction="none").sum(1)
    if class_weights is None:
        return por_amostra.mean()
    w = class_weights[y]
    return (por_amostra * w).sum() / w.sum()


# ==========================================================================
# Modelo completo
# ==========================================================================
class HybridResNet(nn.Module):
    """ResNet50 (layer4 treinavel) + uma das cabecas acima."""

    def __init__(self, n_classes: int, head: str = "vqc", output: str = "softmax",
                 n_qubits: int = 8, n_layers: int = 4, ansatz: str = "strongly",
                 c1_hidden: int | str = "match", c1_match: str = "strongly",
                 pretrained: bool = True, train_layer4: bool = True,
                 quantum_device: str = "default.qubit", diff_method: str = "backprop",
                 quantum_run_on: str = "cpu"):
        super().__init__()
        if head not in HEADS:
            raise ValueError(f"head '{head}' desconhecida. Opcoes: {HEADS}")
        if output not in OUTPUTS:
            raise ValueError(f"output '{output}' desconhecido. Opcoes: {OUTPUTS}")
        self.head, self.output_kind, self.n_classes = head, output, n_classes

        enc = ResNet50Layer4(n_classes=n_classes, pretrained=pretrained,
                             train_layer4=train_layer4)
        enc.backbone.fc = nn.Identity()          # a saida e da cabeca
        self.encoder = enc

        if head == "linear":
            self.compress, self.core, largura = None, None, FEATURE_DIM
        else:
            self.compress = nn.Linear(FEATURE_DIM, n_qubits)
            largura = n_qubits
            if head == "bottleneck":
                if c1_hidden == "match":
                    c1_hidden = matched_hidden(
                        n_qubits, quantum_param_count(c1_match, n_qubits, n_layers))
                self.core = ClassicalCore(n_qubits, int(c1_hidden))
            else:
                kind = "reupload" if head == "reupload" else ansatz
                if kind not in ANSATZE + ("reupload",):
                    raise ValueError(f"ansatz '{ansatz}' desconhecido. Opcoes: {ANSATZE}")
                self.core = QuantumCore(n_qubits, n_layers, kind, quantum_device,
                                        diff_method, quantum_run_on)

        self.out = (OrdinalOutput(largura, n_classes) if output == "ordinal"
                    else nn.Linear(largura, n_classes))

    def bottleneck(self, z: torch.Tensor) -> torch.Tensor:
        """Vetor de n angulos em [-pi/2, pi/2] que entra no nucleo."""
        return (math.pi / 2) * torch.tanh(self.compress(z))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder.features(x)
        h = z if self.compress is None else self.core(self.bottleneck(z))
        return self.out(h)

    def probabilities(self, out: torch.Tensor) -> torch.Tensor:
        if self.output_kind == "ordinal":
            return ordinal_probabilities(out)
        return torch.softmax(out, 1)

    def loss(self, out: torch.Tensor, y: torch.Tensor,
             class_weights: torch.Tensor | None = None) -> torch.Tensor:
        if self.output_kind == "ordinal":
            return ordinal_loss(out, y, class_weights)
        return F.cross_entropy(out, y, weight=class_weights)

    def parameter_groups(self) -> dict[str, list[nn.Parameter]]:
        """Grupos com taxa propria: backbone (layer4), cabeca (compressao e
        saida), nucleo quantico e nucleo classico (MLP do C1).

        Os dois nucleos ocupam o mesmo lugar e recebem a mesma taxa
        (core_learning_rate), o que deixa o grid da taxa espelhavel no C1.
        """
        nucleo = list(self.core.parameters()) if self.core is not None else []
        quantum = nucleo if isinstance(self.core, QuantumCore) else []
        classical_core = [] if quantum else nucleo
        ids = {id(p) for p in nucleo}
        backbone = [p for p in self.encoder.parameters() if p.requires_grad]
        head = [p for n, p in self.named_parameters()
                if not n.startswith("encoder.") and id(p) not in ids and p.requires_grad]
        return {"backbone": backbone, "head": head, "quantum": quantum,
                "classical_core": classical_core}

    def describe(self) -> dict:
        """Contagens de parametros, gravadas em config_used.json."""
        g = self.parameter_groups()
        return {
            "params_backbone_trainable": sum(p.numel() for p in g["backbone"]),
            "params_compress": count_parameters(self.compress) if self.compress is not None else 0,
            "params_core": count_parameters(self.core) if self.core is not None else 0,
            "params_core_is_quantum": isinstance(self.core, QuantumCore),
            "params_output": count_parameters(self.out),
            "params_head_total": sum(p.numel() for p in
                                     g["head"] + g["quantum"] + g["classical_core"]),
        }


def create_hybrid_model(config: dict, n_classes: int) -> HybridResNet:
    """Constroi o modelo a partir do dicionario de configuracao."""
    return HybridResNet(
        n_classes=n_classes,
        head=config.get("head", "vqc"),
        output=config.get("output", "softmax"),
        n_qubits=int(config.get("n_qubits", 8)),
        n_layers=int(config.get("n_layers", 4)),
        ansatz=config.get("ansatz", "strongly"),
        c1_hidden=config.get("c1_hidden", "match"),
        c1_match=config.get("c1_match", "strongly"),
        pretrained=bool(config.get("pretrained", True)),
        train_layer4=bool(config.get("train_layer4", True)),
        quantum_device=config.get("quantum_device", "default.qubit"),
        diff_method=config.get("diff_method", "backprop"),
        quantum_run_on=config.get("quantum_run_on", "cpu"),
    )

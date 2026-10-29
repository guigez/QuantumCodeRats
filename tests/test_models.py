"""Cabecas hibridas: formatos, probabilidades, contagem de parametros e gradiente."""

from __future__ import annotations

import pytest
import torch

from hybrid_models import (HEADS, OUTPUTS, HybridResNet, OrdinalOutput, QuantumCore,
                           matched_hidden, ordinal_loss, ordinal_probabilities,
                           quantum_param_count)

X = torch.rand(2, 3, 64, 64)


def make(head, output="softmax", k=4, n_qubits=4, n_layers=2, **kw):
    return HybridResNet(k, head=head, output=output, pretrained=False,
                        n_qubits=n_qubits, n_layers=n_layers, **kw)


@pytest.mark.parametrize("output", OUTPUTS)
@pytest.mark.parametrize("head", HEADS)
def test_forward_e_probabilidades(head, output):
    m = make(head, output).eval()
    with torch.no_grad():
        out = m(X)
    assert out.shape == (2, 4 if output == "softmax" else 3)
    p = m.probabilities(out)
    assert torch.allclose(p.sum(1), torch.ones(2), atol=1e-5)
    assert (p >= 0).all()


def test_contagem_casada_c1_h1():
    h1 = HybridResNet(4, head="vqc", pretrained=False, n_qubits=8, n_layers=4)
    c1 = HybridResNet(4, head="bottleneck", pretrained=False, n_qubits=8, n_layers=4)
    h2 = HybridResNet(4, head="reupload", pretrained=False, n_qubits=8, n_layers=4)
    assert h1.describe()["params_core"] == 96 == quantum_param_count("strongly", 8, 4)
    assert c1.describe()["params_core"] == 93            # h = 5: 2*8*5 + 5 + 8
    assert h2.describe()["params_core"] == 128
    assert matched_hidden(8, 96) == 5
    assert h1.describe()["params_compress"] == c1.describe()["params_compress"] == 2048 * 8 + 8


def test_contagem_ansatz_basic_e_casamento():
    h1 = HybridResNet(4, head="vqc", ansatz="basic", pretrained=False, n_qubits=4, n_layers=2)
    c1 = HybridResNet(4, head="bottleneck", c1_match="basic", pretrained=False,
                      n_qubits=4, n_layers=2)
    assert h1.describe()["params_core"] == 8 == quantum_param_count("basic", 4, 2)
    assert c1.describe()["params_core"] == 13                 # h minimo 1: 2*4 + 1 + 4


@pytest.mark.parametrize("head,chave,grupo", [("vqc", "quantum_learning_rate", "quantum"),
                                             ("bottleneck", "core_learning_rate", "classical_core")])
def test_taxa_do_nucleo_espelhada(head, chave, grupo):
    from hybrid_train import build_optimizer
    m = make(head)
    taxas = {g["name"]: (g["lr"], g["weight_decay"]) for g in
             build_optimizer(m, {chave: 0.01, "head_learning_rate": 0.001}).param_groups}
    assert taxas[grupo][0] == 0.01 and taxas["head"][0] == 0.001
    assert taxas[grupo][1] == (0.0 if grupo == "quantum" else 1e-4)


def test_grupos_de_parametros_disjuntos_e_completos():
    m = make("vqc")
    g = m.parameter_groups()
    ids = [id(p) for grupo in g.values() for p in grupo]
    assert len(ids) == len(set(ids))
    treinaveis = {id(p) for p in m.parameters() if p.requires_grad}
    assert set(ids) == treinaveis
    assert g["quantum"] and g["backbone"] and g["head"]


def test_backbone_congelado_exceto_layer4():
    m = make("vqc")
    for nome, p in m.encoder.named_parameters():
        assert p.requires_grad == nome.startswith("backbone.layer4"), nome


@pytest.mark.parametrize("head", ["vqc", "reupload"])
def test_gradiente_chega_ao_circuito_e_a_layer4(head):
    m = make(head).train()
    perda = m.loss(m(X), torch.tensor([0, 3]))
    perda.backward()
    for p in m.parameter_groups()["quantum"]:
        assert p.grad is not None and torch.isfinite(p.grad).all()
        assert p.grad.abs().sum() > 0
    assert m.compress.weight.grad.abs().sum() > 0
    assert any(p.grad is not None and p.grad.abs().sum() > 0
               for p in m.encoder.backbone.layer4.parameters())


def test_limiares_ordinais_ordenados_e_probabilidades():
    out = OrdinalOutput(8, 4)
    b = out.thresholds()
    assert torch.all(b[1:] > b[:-1])
    logits = torch.randn(5, 3)
    p = ordinal_probabilities(logits - torch.tensor([0.0, 1.0, 2.0]))
    assert torch.allclose(p.sum(1), torch.ones(5), atol=1e-5)
    # duas classes: [1 - sigma, sigma]
    l2 = torch.tensor([[0.3]])
    assert torch.allclose(ordinal_probabilities(l2),
                          torch.tensor([[1 - torch.sigmoid(l2).item(), torch.sigmoid(l2).item()]]))


def test_perda_ordinal_prefere_o_acerto():
    certo = torch.tensor([[8.0, 8.0, -8.0]])           # P(y>0), P(y>1) altos; P(y>2) baixo -> classe 2
    errado = torch.tensor([[-8.0, -8.0, -8.0]])        # -> classe 0
    y = torch.tensor([2])
    assert ordinal_loss(certo, y) < ordinal_loss(errado, y)
    w = torch.tensor([1.0, 1.0, 5.0, 1.0])
    assert torch.isclose(ordinal_loss(certo, y, w), ordinal_loss(certo, y))


def test_circuito_fica_na_cpu_quando_pedido():
    q = QuantumCore(3, 1, run_on="cpu")
    q.to(torch.float32)
    assert all(p.device.type == "cpu" for p in q.parameters())
    if torch.cuda.is_available():
        m = make("vqc").cuda()
        assert all(p.device.type == "cpu" for p in m.core.parameters())
        assert m.compress.weight.device.type == "cuda"
        out = m(X.cuda())
        assert out.device.type == "cuda"
        m.loss(out, torch.tensor([0, 1], device="cuda")).backward()
        assert all(p.grad is not None for p in m.core.parameters())

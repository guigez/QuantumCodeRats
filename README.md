# Rede híbrida quântico-clássica para termografia de ratos em esteira

Extensão híbrida (PennyLane + PyTorch) do pipeline clássico
[classicalCodeRats](https://github.com/guigez/classicalCodeRats). A pergunta
é se uma cabeça com circuito variacional **generaliza melhor para um animal
novo** do que a cabeça clássica sobre o mesmo backbone. A avaliação usa as
mesmas dobras por sujeito, as mesmas métricas e a mesma comparação pareada
por animal do clássico.

## Desenho

Todas as famílias usam o backbone da família `resnet50_rf` do clássico: a
mesma classe `ResNet50Layer4`, pré-treinada no ImageNet, com a `layer4`
destravada. As famílias diferem só na cabeça sobre o vetor de 2048 dimensões.
Todas treinam ponta a ponta (Regime B), então o gradiente do circuito volta
para a `layer4`.

| família | braço | cabeça |
|---|---|---|
| `c1_bottleneck` | C1 | 2048→8, (π/2)·tanh, MLP 8→5→8 com tanh, 8→k. 93 parâmetros no núcleo, casados com os 96 do H1 |
| `h1_vqc` | H1 | 2048→8, (π/2)·tanh, AngleEmbedding(RY) + StronglyEntanglingLayers ×4, ⟨Z⟩ por qubit, 8→k |
| `h2_reupload` | H2 | mesma compressão; dados reinseridos em cada uma das 4 camadas, com escala treinável |
| `h3_vqc_ordinal` | H3 | H1 com saída ordinal: um escore e k−1 limiares ordenados (pulado no binário) |
| `c1_ordinal` | C1-ord | C1 com saída ordinal (pulado no binário) |
| `c0_linear` | C0 | cabeça linear refeita neste código (opcional) |

A referência principal é a cabeça linear do clássico
(`resnet50_rf_linear_head`), lida dos resultados clássicos. O C1 existe
porque a cabeça híbrida difere da linear em duas coisas ao mesmo tempo: o
gargalo de 8 dimensões e o circuito. O C1 tem o mesmo gargalo, o mesmo squash
e a mesma faixa de saída. Ganhar do C1 é o que atribui o efeito ao circuito.

Diferenças de treino em relação ao `resnet50_rf` clássico, todas registradas
em `config_used.json`:

- a cabeça (compressão, núcleo e saída) treina a `head_learning_rate = 1e-3`;
  a `layer4` fica em `1e-4`, como no clássico. C1 e H1 usam a mesma taxa;
- os ângulos do circuito não recebem weight decay;
- `train_log.csv` grava a norma média do gradiente do circuito por época, para
  diagnosticar barren plateau.

Cada família tem uma única configuração, fixada antes de rodar, sem grid. A
variação de treino é medida pelas réplicas por semente.

## Estrutura

```
run_quantum.py          orquestrador: dobras → grid → sementes → agregação
configs/                uma configuração por família
src/
  classical_bridge.py   localiza o repositório clássico e o põe no sys.path
  hybrid_models.py      cabeças C0, C1, H1, H2, saída ordinal
  hybrid_train.py       treino ponta a ponta de um candidato
  hybrid_grid.py        candidato(s) no fold, seleção na validação, teste uma vez
  hybrid_seeds.py       réplicas por semente do vencedor
  hybrid_aggregate.py   tabelas pareadas com os braços clássicos e decision.csv
scripts/run_server.sh   execução no servidor com nohup
tests/                  testes (pytest)
```

Do clássico são importados `dataset`, `splits`, `manifest`, `metrics`,
`aggregate`, `common`, `models.ResNet50Layer4` e os utilitários do
`run_all.py`. Nada é copiado.

## Repositório clássico

O código procura o clássico nesta ordem:

1. variável `CLASSICAL_CODE`;
2. `Bernini/Classical/classicalCodeRats` (servidor);
3. `Bernini/Classical/Code` (local);
4. `external/classicalCodeRats`, se você preferir fixá-lo como submódulo:
   ```bash
   git submodule add https://github.com/guigez/classicalCodeRats external/classicalCodeRats
   ```

Os commits dos dois repositórios são gravados em cada `config_used.json`.

## Instalação

```bash
pip install -r requirements.txt
```

## Execução

No servidor, com `Bernini/Quantum/QuantumCodeRats/` (este repositório), o
clássico em `Bernini/Classical/classicalCodeRats/` e os resultados clássicos
em `Bernini/Classical/main_results/`:

```bash
nohup bash scripts/run_server.sh > /dev/null 2>&1 &
tail -f ../main_results/run_server.log
```

Por padrão, roda `binary` e `multiclass` com C1 e H1 e as sementes 1 a 5. Para
incluir a saída ordinal no multiclasse:

```bash
TASKS="multiclass" FAMILIES="h3_vqc_ordinal c1_ordinal" nohup bash scripts/run_server.sh > /dev/null 2>&1 &
```

As dobras são copiadas de `<resultados clássicos>/<tarefa>/splits/`, o que
garante os mesmos animais de teste nos dois repositórios. A agregação aborta
se algum fold testar animais diferentes entre braços.

## Leitura dos resultados

Em `<resultados>/<tarefa>/_tabelas/`:

| arquivo | conteúdo |
|---|---|
| `decision.csv` | critério fixado antes de rodar, por par híbrido × referência |
| `paired_by_subject.csv` | métrica de cada braço por animal e diferenças pareadas |
| `paired_comparison.csv` | direção, tamanho do efeito, teste de sinal e piso do p |
| `seed_paired_by_subject.csv` | diferença entre braços comparada ao ruído entre sementes |
| `diagnostics_by_run.csv` | queda validação→teste, colapso, adjacência, revocação da classe mais intensa |
| `aggregate_metrics.csv` | média, desvio e extremos entre folds |

Uma cabeça híbrida é considerada **melhor** que a referência num cenário
quando, com a média das sementes por animal: (1) vence nos 4 animais; (2) a
diferença média passa de 2× o desvio entre sementes; (3) não colapsa mais, nem
zera a revocação da classe mais intensa mais vezes, que a referência.

## Testes

```bash
python -m pytest tests
```

Os testes constroem os modelos sem baixar pesos (`pretrained=False`), treinam
uma época sobre imagens sintéticas e checam a agregação e o critério de
decisão sobre resultados sintéticos.

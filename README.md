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

| família | braços | cabeça |
|---|---|---|
| `c0_rf` | C0 (`c0_rf_linear_head`) e C-RF (`c0_rf`) | reprodução do `resnet50_rf` clássico: cabeça linear 2048→k e, sobre a rede congelada, StandardScaler + RandomForest (100 árvores, balanceada, mínimo de 5 por folha). Um treino gera os dois braços |
| `c1_bottleneck` | C1 | 2048→8, (π/2)·tanh, MLP 8→5→8 com tanh, 8→k. 93 parâmetros no núcleo, casados com os 96 do H1 |
| `h1_vqc` | H1 | 2048→8, (π/2)·tanh, AngleEmbedding(RY) + StronglyEntanglingLayers ×4, ⟨Z⟩ por qubit, 8→k |
| `h2_reupload` | H2 | mesma compressão; dados reinseridos em cada uma das 4 camadas, com escala treinável |
| `h3_vqc_ordinal` | H3 | H1 com saída ordinal: um escore e k−1 limiares ordenados (pulado no binário) |
| `c1_ordinal` | C1-ord | C1 com saída ordinal (pulado no binário) |

C0 e C-RF são as referências clássicas, treinadas neste mesmo código. O C1
existe porque a cabeça híbrida difere da linear em duas coisas ao mesmo tempo:
o gargalo de 8 dimensões e o circuito. O C1 tem o mesmo gargalo, o mesmo squash
e a mesma faixa de saída. Ganhar do C1 é o que atribui o efeito ao circuito.

Diferenças de treino entre as famílias, todas registradas em
`config_used.json`:

- `c0_rf` treina tudo a `1e-4`, inclusive a cabeça linear, como o clássico.
  Nas demais, a cabeça (compressão, núcleo e saída) treina a
  `head_learning_rate = 1e-3` e a `layer4` a `1e-4`. C1 e H1 usam a mesma taxa,
  por isso o C1 é o controle casado do H1;
- os ângulos do circuito não recebem weight decay;
- `train_log.csv` grava a norma média do gradiente do circuito por época, para
  diagnosticar barren plateau.

### Grid

| famílias | grid | candidatos por dobra |
|---|---|---|
| `h1_vqc`, `h3_vqc_ordinal` | `n_qubits` {4, 6, 8} × `n_layers` {1, 2, 4} × `quantum_learning_rate` {1e-3, 1e-2} × `ansatz` {strongly, basic} | 36 |
| `h2_reupload` | `n_qubits` {4, 6, 8} × `n_layers` {1, 2, 4} × `quantum_learning_rate` {1e-3, 1e-2} | 18 |
| `c1_bottleneck`, `c1_ordinal` | o grid do H1, espelhado: `n_qubits` × `n_layers` × `core_learning_rate` {1e-3, 1e-2} × `c1_match` {strongly, basic} | 36 |
| `c0_rf` | nenhum (configuração vencedora do clássico, fixa) | 1 |

O vencedor de cada dobra é escolhido pelo F1 macro da validação, e o teste é
avaliado uma vez, só nele. O H2 não tem `ansatz` no grid: ele usa sempre uma
camada StronglyEntangling por reinserção, e o parâmetro seria ignorado.

O grid é espelhado no C1 para que as duas cabeças tenham a mesma busca:

- `n_qubits` vira a largura do gargalo do C1;
- `core_learning_rate` é a taxa do MLP, no papel da `quantum_learning_rate`
  (o circuito fica sem weight decay; o MLP mantém o da cabeça);
- `n_layers` e `c1_match` definem o alvo do casamento de parâmetros: 3nL
  (strongly) ou nL (basic).

Com alvos pequenos o MLP não consegue ser tão pequeno quanto o circuito: com
h = 1 ele já tem 2n + 1 + n parâmetros (13 com n = 4, 25 com n = 8), contra
nL do circuito basic (4 a 32).

| n × L | strongly (H1/H3) | basic (H1/H3) | H2 | MLP casado com strongly | MLP casado com basic |
|---|---|---|---|---|---|
| 4 × 1 | 12 | 4 | 16 | 13 (h = 1) | 13 (h = 1) |
| 4 × 2 | 24 | 8 | 32 | 22 (h = 2) | 13 (h = 1) |
| 4 × 4 | 48 | 16 | 64 | 49 (h = 5) | 13 (h = 1) |
| 6 × 1 | 18 | 6 | 24 | 19 (h = 1) | 19 (h = 1) |
| 6 × 2 | 36 | 12 | 48 | 32 (h = 2) | 19 (h = 1) |
| 6 × 4 | 72 | 24 | 96 | 71 (h = 5) | 19 (h = 1) |
| 8 × 1 | 24 | 8 | 32 | 25 (h = 1) | 25 (h = 1) |
| 8 × 2 | 48 | 16 | 64 | 42 (h = 2) | 25 (h = 1) |
| 8 × 4 | 96 | 32 | 128 | 93 (h = 5) | 25 (h = 1) |

As réplicas por semente e o SHAP usam a configuração vencedora de cada dobra.
`grid_summary.csv` guarda a validação de todos os candidatos, e o `model.pt`
dos que perderam é apagado. Os demais hiperparâmetros são fixos.

Um grid de 36 candidatos leva mais de um dia por dobra. Se a execução cair no
meio, rodar de novo retoma de onde parou: cada candidato concluído grava
`net_XXX/candidate_done.json` e não é treinado outra vez enquanto a
configuração e a dobra forem as mesmas. `EXTRA_ARGS="--force"` retreina tudo.

## Estrutura

```
run_quantum.py          orquestrador: dobras → grid → sementes → SHAP → agregação
configs/                uma configuração por família
src/
  classical_bridge.py   localiza o repositório clássico e o põe no sys.path
  hybrid_models.py      cabeças C0, C1, H1, H2, saída ordinal
  hybrid_train.py       treino ponta a ponta de um candidato (e a floresta do c0_rf)
  hybrid_grid.py        candidato(s) no fold, seleção na validação, teste uma vez
  hybrid_seeds.py       réplicas por semente do vencedor
  hybrid_shap.py        atribuição SHAP e enriquecimento E no vencedor
  hybrid_aggregate.py   tabelas pareadas por animal e decision.csv
scripts/run_server.sh   execução no servidor com nohup
tests/                  testes (pytest)
```

Do clássico é importado só código: `dataset`, `splits`, `manifest`,
`metrics`, `aggregate`, `common`, `models` (`ResNet50Layer4`,
`extract_features`, `make_rf_pipeline`), as funções de máscara, enriquecimento
e painel do `explain_shap` e os utilitários do `run_all.py`.
Nada é copiado, e nenhum resultado clássico é lido por padrão.

## Repositório clássico

O código procura o clássico nesta ordem:

1. variável `CLASSICAL_CODE`;
2. `Bernini/Classical/classicalCodeRats` ou `ClassicalCodeRats` (servidor);
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

### Antes da execução completa

A execução completa leva semanas. Antes dela, com o ambiente ativado e a
partir da raiz deste repositório:

1. Rode os testes (minutos, CPU, imagens sintéticas):
   ```bash
   python -m pytest tests -q
   ```
2. Rode um piloto com dados reais numa pasta separada: H1, binário, só o
   fold_00, só o primeiro candidato do grid (`--limit_candidates 1`) e sem
   réplicas. Ele percorre as cinco etapas (dobras, treino, SHAP, agregação) na
   GPU em cerca de uma hora, sem sujar `main_results`:
   ```bash
   RESULTS="$(pwd)/../piloto" TASKS="binary" FAMILIES="h1_vqc" SEEDS="" EXTRA_ARGS="--only-folds 0 --limit_candidates 1" nohup bash scripts/run_server.sh > /dev/null 2>&1 &
   ```
   Confira em `../piloto/run_server.log` que terminou sem falhas, em
   `../piloto/binary/fold_00/h1_vqc/net_00*/train_log.csv` quanto tempo leva
   cada época e se `quantum_grad_norm` não é zero, e em
   `../piloto/binary/fold_00/h1_vqc/shap/` que o painel saiu.

### Execução completa

No servidor, com `Bernini/Quantum/QuantumCodeRats/` (este repositório), o
clássico em `Bernini/Classical/classicalCodeRats/` e os dados em
`Bernini/_dataset/`:

```bash
nohup bash scripts/run_server.sh > /dev/null 2>&1 &
tail -f ../main_results/run_server.log
```

Por padrão é uma **execução limpa**: os quatro cenários (`binary`,
`three_bernini`, `three_mlss`, `multiclass`), as seis famílias e as sementes 1
a 5. As dobras são geradas aqui a partir do metadata, com o `manifest.py` e o
`splits.py` do clássico em `fixed_paired` (teste nos animais 1, 9, 10 e 11), e
nada é lido de outra pasta de resultados. Para refazer tudo numa pasta que já
tem resultados, use `EXTRA_ARGS="--force"` ou aponte `RESULTS` para uma pasta
nova.

Custo: cada treino da ResNet50 levou cerca de 45 min no servidor. Por dobra, o
binário treina 90 candidatos (C1 36, H1 36, H2 18) e cada um dos outros três
cenários 162 (mais H3 36 e C1-ord 36). Nas 4 dobras são 2.304 candidatos, mais
360 réplicas por semente (18 combinações de família × cenário × 4 dobras × 5) e
96 treinos do `c0_rf`: 2.760 treinos, cerca de **86 dias** de GPU, mais o custo
do circuito. Para um subconjunto:

```bash
TASKS="binary multiclass" FAMILIES="c0_rf c1_bottleneck h1_vqc" nohup bash scripts/run_server.sh > /dev/null 2>&1 &
```

### SHAP

A etapa 4 explica o vencedor de cada dobra e família (o treino com
`--seed 42`; as réplicas apagam o `model.pt`). Usa `GradientExplainer` sobre
um acerto e um erro por classe do animal de teste e calcula o enriquecimento
E, densidade de atribuição dentro da máscara do animal dividida pela de fora,
com as mesmas funções do `explain_shap.py` clássico.

- **Imagens completas, com esteira (padrão, `WITH_BACKGROUND=1`):** a máscara é
  o limiar de Otsu sobre R − B, por quadro. Ela marca a região quente do
  quadro, e faixas mornas da baia podem entrar, então E é um limite superior
  da atenção ao animal, como no artigo.
- **Imagens sem fundo (`WITH_BACKGROUND=0`):** a máscara é a região não vazia.
  Se ela cobrir mais de 90% do quadro, o script avisa e volta para o limiar.
- **Saída ordinal:** explica log P(y = c), uma saída por classe.
- **`c0_rf`:** a atribuição por pixel é da cabeça linear (braço
  `c0_rf_linear_head`); da floresta sai só a importância das 2048 features.

Resultados em `fold_*/<família>/shap/` (painel e `shap_background_ratio.csv`)
e, por braço, em `_tabelas/shap_summary.csv`. `--no-shap` desliga a etapa.

Opcionalmente, `CLASSICAL_RESULTS=<pasta>` copia as dobras de uma execução
clássica e inclui os braços dela na agregação. A agregação aborta se algum
fold testar animais diferentes entre braços.

## Leitura dos resultados

Em `<resultados>/<tarefa>/_tabelas/`:

| arquivo | conteúdo |
|---|---|
| `decision.csv` | critério fixado antes de rodar, por par híbrido × referência |
| `paired_by_subject.csv` | métrica de cada braço por animal e diferenças pareadas |
| `paired_comparison.csv` | direção, tamanho do efeito, teste de sinal e piso do p |
| `seed_paired_by_subject.csv` | diferença entre braços comparada ao ruído entre sementes |
| `diagnostics_by_run.csv` | queda validação→teste, colapso, adjacência, revocação da classe mais intensa |
| `shap_summary.csv` | enriquecimento E por braço, com acertos e erros separados |
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
uma época sobre imagens sintéticas, rodam o SHAP nelas e checam a agregação e o critério de
decisão sobre resultados sintéticos.

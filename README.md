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

Cada família tem uma única configuração, fixada antes de rodar, sem grid. A
variação de treino é medida pelas réplicas por semente.

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

Custo: cada treino da ResNet50 levou cerca de 45 min no servidor, e cada família
faz 4 dobras × (vencedor + 5 sementes) = 24 treinos, cerca de 18 h por cenário.
São 4 famílias no binário (H3 e C1-ord são puladas) e 6 nos outros três, ou
22 × 18 h ≈ 16 dias no total. Para um subconjunto:

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

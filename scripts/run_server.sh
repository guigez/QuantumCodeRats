#!/usr/bin/env bash
# Roda todas as familias (classicas e hibridas) em todos os cenarios, em
# sequencia, sem depender da sessao SSH: lance com nohup e pode desconectar.
#
# EXECUCAO LIMPA: por padrao nenhum resultado de outra pasta e lido. As
# dobras sao geradas aqui a partir do metadata (fixed_paired, teste 1 9 10 11)
# e C0/C-RF sao treinados aqui pela familia c0_rf. Do repositorio classico so
# vem CODIGO (dataset, splits, metricas, agregacao, backbone), importado.
#
# Estrutura esperada (caminhos derivados da posicao deste script):
#   Bernini/_dataset/images/                imagens
#   Bernini/_dataset/metadata-1.csv         metadata (filename, mice_id, class)
#   Bernini/Classical/classicalCodeRats/    repositorio classico (codigo importado;
#                                           ClassicalCodeRats tambem serve)
#   Bernini/Quantum/QuantumCodeRats/        este repositorio
#   Bernini/Quantum/main_results/           saida (criada se nao existir)
#
# Uso, com o ambiente Python ja ativado:
#   nohup bash scripts/run_server.sh > /dev/null 2>&1 &
#
# Acompanhar:   tail -f ../main_results/run_server.log
# Interromper:  kill -- -$(cat ../main_results/run_server.pid)
#
# Variaveis opcionais (ex.: TASKS="binary" SEEDS="1 2 3" nohup bash ...):
#   PYTHON             interpretador                    (padrao: python do PATH)
#   TASKS              cenarios, na ordem               (padrao: os quatro)
#   FAMILIES           familias                         (padrao: as seis)
#   SEEDS              sementes das replicas            (padrao: 1 2 3 4 5; "" desliga)
#   FIXED_TEST         animais de teste                 (padrao: 1 9 10 11)
#   DEVICE             auto | cuda | cpu                (padrao: auto)
#   WITH_BACKGROUND    1 = imagens completas, com a esteira (padrao: 1). Afeta so
#                      a mascara do SHAP: 1 usa o limiar R-B do classico; 0 assume
#                      fundo removido. Use 0 so com imagens sem fundo.
#   DATASET            pasta dos dados                  (padrao: Bernini/_dataset)
#   CLASSICAL_CODE     repositorio classico             (padrao: Bernini/Classical/[cC]lassicalCodeRats)
#   RESULTS            pasta de saida                   (padrao: Bernini/Quantum/main_results)
#   CLASSICAL_RESULTS  so se definida: copia as dobras de la e inclui os bracos
#                      classicos na agregacao (deixa de ser execucao limpa)
#   EXTRA_ARGS         argumentos a mais para run_quantum.py (ex.: "--force")

set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
QUANTUM="$(dirname "$REPO")"
BERNINI="$(dirname "$QUANTUM")"

DATASET="${DATASET:-$BERNINI/_dataset}"
if [ -z "${CLASSICAL_CODE:-}" ]; then
    CLASSICAL_CODE="$BERNINI/Classical/classicalCodeRats"
    [ -d "$CLASSICAL_CODE" ] || CLASSICAL_CODE="$BERNINI/Classical/ClassicalCodeRats"
fi
export CLASSICAL_CODE
CLASSICAL_RESULTS="${CLASSICAL_RESULTS:-}"
RESULTS="${RESULTS:-$QUANTUM/main_results}"
PYTHON="${PYTHON:-python}"
TASKS="${TASKS:-binary three_bernini three_mlss multiclass}"
FAMILIES="${FAMILIES:-c0_rf c1_bottleneck h1_vqc h2_reupload h3_vqc_ordinal c1_ordinal}"
SEEDS="${SEEDS-1 2 3 4 5}"
FIXED_TEST="${FIXED_TEST:-1 9 10 11}"
DEVICE="${DEVICE:-auto}"
WITH_BACKGROUND="${WITH_BACKGROUND:-1}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

mkdir -p "$RESULTS"
LOG="$RESULTS/run_server.log"
exec >>"$LOG" 2>&1
echo $$ > "$RESULTS/run_server.pid"

echo "======================================================================"
echo "inicio    $(date '+%F %T')   commit $(git -C "$REPO" rev-parse --short HEAD 2>/dev/null || echo '?')"
echo "classico  $CLASSICAL_CODE  (commit $(git -C "$CLASSICAL_CODE" rev-parse --short HEAD 2>/dev/null || echo '?'))"
echo "dados     $DATASET"
echo "saida     $RESULTS"
echo "modo      ${CLASSICAL_RESULTS:+usando resultados classicos de $CLASSICAL_RESULTS}${CLASSICAL_RESULTS:-execucao limpa (dobras geradas aqui, nada de fora e lido)}"
echo "python    $(command -v "$PYTHON")"
echo "tarefas   $TASKS"
echo "familias  $FAMILIES"
echo "sementes  ${SEEDS:-nenhuma}   teste: $FIXED_TEST   device: $DEVICE   com esteira: $WITH_BACKGROUND"
echo "======================================================================"

if [ ! -d "$DATASET/images" ]; then echo "ERRO: $DATASET/images nao existe"; exit 1; fi
if [ ! -f "$DATASET/metadata-1.csv" ]; then echo "ERRO: $DATASET/metadata-1.csv nao existe"; exit 1; fi
if [ ! -d "$CLASSICAL_CODE/src" ]; then echo "ERRO: $CLASSICAL_CODE/src nao existe"; exit 1; fi
if [ -n "$CLASSICAL_RESULTS" ] && [ ! -d "$CLASSICAL_RESULTS" ]; then
    echo "ERRO: CLASSICAL_RESULTS=$CLASSICAL_RESULTS nao existe"; exit 1
fi
if ! "$PYTHON" -c "import torch, torchvision, sklearn, pennylane, shap, matplotlib" ; then
    echo "ERRO: dependencias ausentes em $(command -v "$PYTHON"); rode pip install -r requirements.txt"
    exit 1
fi

cd "$REPO" || exit 1
seeds_arg=()
if [ -n "$SEEDS" ]; then seeds_arg=(--seeds $SEEDS); fi
bg_arg=()
if [ "$WITH_BACKGROUND" != "1" ]; then bg_arg=(--background_removed); fi
classical_arg=()
if [ -n "$CLASSICAL_RESULTS" ]; then classical_arg=(--classical_results "$CLASSICAL_RESULTS"); fi

# ${arr[@]+"${arr[@]}"}: com set -u, bash anterior ao 4.4 trata array vazio como
# variavel nao definida e aborta. classical_arg e vazio por padrao.
falhas=()
for task in $TASKS; do
    echo
    echo "------ $task  ($(date '+%F %T')) ------"
    # shellcheck disable=SC2086
    if ! "$PYTHON" run_quantum.py --task "$task" --families $FAMILIES \
            --fixed_test_subjects $FIXED_TEST \
            --images_dir "$DATASET/images" --metadata "$DATASET/metadata-1.csv" \
            --results_dir "$RESULTS" --device "$DEVICE" \
            ${classical_arg[@]+"${classical_arg[@]}"} ${seeds_arg[@]+"${seeds_arg[@]}"} \
            ${bg_arg[@]+"${bg_arg[@]}"} $EXTRA_ARGS; then
        falhas+=("$task")
    fi
done

echo
echo "======================================================================"
echo "fim     $(date '+%F %T')"
if [ ${#falhas[@]} -gt 0 ]; then
    echo "tarefas com falha: ${falhas[*]}  (rode de novo: as etapas concluidas sao puladas)"
else
    echo "todas as tarefas concluidas sem falhas"
fi
echo "tabelas: $RESULTS/<tarefa>/_tabelas/   decisao: $RESULTS/<tarefa>/_tabelas/decision.csv"
echo "======================================================================"
rm -f "$RESULTS/run_server.pid"
[ ${#falhas[@]} -eq 0 ]

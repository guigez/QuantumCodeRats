#!/usr/bin/env bash
# Roda as familias hibridas no servidor, em sequencia, sem depender da sessao
# SSH: lance com nohup e pode desconectar.
#
# Estrutura esperada (caminhos derivados da posicao deste script):
#   Bernini/_dataset/images/                imagens
#   Bernini/_dataset/metadata-1.csv         metadata (so usado se for preciso gerar as dobras)
#   Bernini/Classical/classicalCodeRats/    repositorio classico (importado)
#   Bernini/Classical/main_results/         resultados classicos (dobras + bracos de referencia)
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
#   TASKS              cenarios, na ordem               (padrao: binary multiclass)
#   FAMILIES           familias hibridas                (padrao: c1_bottleneck h1_vqc)
#   SEEDS              sementes das replicas            (padrao: 1 2 3 4 5; "" desliga)
#   DEVICE             auto | cuda | cpu                (padrao: auto)
#   DATASET            pasta dos dados                  (padrao: Bernini/_dataset)
#   CLASSICAL_CODE     repositorio classico             (padrao: Bernini/Classical/classicalCodeRats)
#   CLASSICAL_RESULTS  resultados classicos             (padrao: Bernini/Classical/main_results)
#   RESULTS            pasta de saida                   (padrao: Bernini/Quantum/main_results)
#   EXTRA_ARGS         argumentos a mais para run_quantum.py (ex.: "--force")

set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
QUANTUM="$(dirname "$REPO")"
BERNINI="$(dirname "$QUANTUM")"

DATASET="${DATASET:-$BERNINI/_dataset}"
export CLASSICAL_CODE="${CLASSICAL_CODE:-$BERNINI/Classical/classicalCodeRats}"
CLASSICAL_RESULTS="${CLASSICAL_RESULTS:-$BERNINI/Classical/main_results}"
RESULTS="${RESULTS:-$QUANTUM/main_results}"
PYTHON="${PYTHON:-python}"
TASKS="${TASKS:-binary multiclass}"
FAMILIES="${FAMILIES:-c1_bottleneck h1_vqc}"
SEEDS="${SEEDS-1 2 3 4 5}"
DEVICE="${DEVICE:-auto}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

mkdir -p "$RESULTS"
LOG="$RESULTS/run_server.log"
exec >>"$LOG" 2>&1
echo $$ > "$RESULTS/run_server.pid"

echo "======================================================================"
echo "inicio    $(date '+%F %T')   commit $(git -C "$REPO" rev-parse --short HEAD 2>/dev/null || echo '?')"
echo "classico  $CLASSICAL_CODE  (commit $(git -C "$CLASSICAL_CODE" rev-parse --short HEAD 2>/dev/null || echo '?'))"
echo "dados     $DATASET"
echo "ref.      $CLASSICAL_RESULTS"
echo "saida     $RESULTS"
echo "python    $(command -v "$PYTHON")"
echo "tarefas   $TASKS   familias: $FAMILIES   sementes: ${SEEDS:-nenhuma}   device: $DEVICE"
echo "======================================================================"

if [ ! -d "$DATASET/images" ]; then echo "ERRO: $DATASET/images nao existe"; exit 1; fi
if [ ! -d "$CLASSICAL_CODE/src" ]; then echo "ERRO: $CLASSICAL_CODE/src nao existe"; exit 1; fi
if [ ! -d "$CLASSICAL_RESULTS" ]; then
    echo "AVISO: $CLASSICAL_RESULTS nao existe: as dobras serao geradas aqui e nao havera bracos classicos na agregacao"
fi
if ! "$PYTHON" -c "import torch, torchvision, sklearn, pennylane" ; then
    echo "ERRO: dependencias ausentes em $(command -v "$PYTHON"); rode pip install -r requirements.txt"
    exit 1
fi

cd "$REPO" || exit 1
seeds_arg=()
if [ -n "$SEEDS" ]; then seeds_arg=(--seeds $SEEDS); fi
classical_arg=()
if [ -d "$CLASSICAL_RESULTS" ]; then classical_arg=(--classical_results "$CLASSICAL_RESULTS"); fi

falhas=()
for task in $TASKS; do
    echo
    echo "------ $task  ($(date '+%F %T')) ------"
    # shellcheck disable=SC2086
    if ! "$PYTHON" run_quantum.py --task "$task" --families $FAMILIES \
            --images_dir "$DATASET/images" --metadata "$DATASET/metadata-1.csv" \
            --results_dir "$RESULTS" --device "$DEVICE" \
            "${classical_arg[@]}" "${seeds_arg[@]}" $EXTRA_ARGS; then
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

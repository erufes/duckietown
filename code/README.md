# README da ERUS

## Getting Started

1. Crie um ambiente virtual do conda `conda create -n duckietown python`
2. Clone o repositório **e os submódulos**: `git clone https://github.com/erufes/duckietown --recurse-submodules`
3. Instale as dependências: `./setup.sh` (se der erro, rode os comandos do .sh manualmente)

## Treinamento

O script principal é o `article/train.py`. Ele é o responsável por de fato treinar o modelo. Pode executá-lo a partir desta pasta rodando
`python -m article.train`.

Dentro dele, há alguns parâmetros ajustáveis:

- `MODEL_PREFIX` é o nome do arquivo que será salvo para as runs no tensorboard e nos modelos resultantes
- `THREAD_COUNT` é a quantidade de threads paralelas de execução. Caso queira rodar single threaded THREAD_COUNT = 1

A rede usada no treinamento está definida em `/gym-duckietown/custom_net.py`

Pra rodar o script remotamente, precisa ter uma tela (mesmo que seja virtual). Pra isso, tem os seguintes comandos:

```Xvfb :1 -screen 0 1024x768x24 -ac +extension GLX +render -noreset &> xvfb.log &```

```export DISPLAY=:0```

### Ambiente de treino

O ambiente é na verdade um ambiente vetorizado de tamanho 5, com cada ambiente interno sendo um dos últimos 5 frames observados (FrameStack). Em `/gym-duckietown/__init__.py` os ambientes são registrados. Um ambiente customizado `Duckietown-udem1-v0_pietroluongo_train` foi implementado para facilitar o treino.

### Observação

Note que a observação retornada pela função `step()` do simulador está no formato `(h, w, 3)`, e o stable baselines precisa do formato `(3, h, w)`, enquanto o pygame usa `(w, h, 3)`. Dessa forma, frequentemente algumas transposições de observação são necessárias.

## Debugging

Alguns scripts auxiliares foram desenvolvidos para debug:

- `python -m utils.test_actions` para testar as ações discretizadas
- `python -m utils.test_observations` para testar observações diretas do sistema
- `python -m utils.test_observation_stack` para testar o frame stack com observações

## Análise de resultados

Para testar um modelo treinado, basta usar o script `python -m article.enjoy` alterando o `MODEL_NAME` para o nome do modelo que deseja visualizar.

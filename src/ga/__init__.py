"""Algoritmo genetico para otimizacao de hiperparametros (Fase 2).

Ordem de leitura dos modulos:
    search_space  -> Etapa 1: como um conjunto de hiperparametros vira
                     um vetor de floats em [0,1] (e como volta)
    operators     -> Etapa 2: selecao, cruzamento, mutacao
    fitness       -> Etapa 2: o que "bom" significa (CV no treino)
    engine        -> o laco geracional que junta os tres
    run_experiments / compare -> Etapas 3 e 4
"""

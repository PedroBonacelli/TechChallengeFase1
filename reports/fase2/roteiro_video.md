# Etapa 10 — Roteiro do vídeo de demonstração

**Limite: 15 minutos.** Roteiro alvo: ~13 min, deixando folga para
respiro e para o tempo real de execução dos comandos.

**Publicação:** YouTube ou Vimeo, público ou não listado. Colar o link
aqui e no `README_FASE2.md` depois de subir.

## Antes de gravar — checklist

- [ ] Rodar o pipeline inteiro uma vez, para que todos os artefatos de
      `reports/fase2/` estejam gerados e os comandos da demo não demorem
- [ ] `.env` preenchido (senão a demo da LLM cai no template)
- [ ] Terminal com fonte grande (14pt+) e tema claro — fonte pequena é o
      erro mais comum nesse tipo de vídeo
- [ ] Fechar notificações do sistema
- [ ] Abrir previamente, em abas: `comparison.md`, as figuras 30/32/34/35/36,
      `llm_explanations.md`, `llm_quality.md`
- [ ] Testar o áudio com 30 segundos de gravação antes de valer

> **Atenção ao tempo:** `run_experiments` leva ~28 minutos. **Não rodar
> ao vivo.** Mostrar o log já gerado e, se quiser demonstrar execução
> real, usar `--quick` (leva ~2 min) ou deixar rodando em segundo plano
> desde antes da gravação.

---

## Bloco 1 — Abertura e contexto (0:00 – 1:30)

**Mostrar:** slide ou o `README_FASE2.md`.

Falar:

- Fase 1 entregou três modelos de diagnóstico de câncer de mama sobre o
  Breast Cancer Wisconsin: Regressão Logística, Árvore de Decisão e
  Random Forest, com hiperparâmetros escolhidos à mão.
- A Fase 2 faz duas coisas: otimiza esses hiperparâmetros com um
  algoritmo genético implementado do zero, e acrescenta uma camada de
  LLM que traduz o resultado numérico em texto que um médico use.
- **Antecipar o resultado aqui, não no fim:** o GA melhorou o fitness de
  validação cruzada nos três modelos, mas esse ganho *não* transferiu
  para o teste. "Vou mostrar o número, mostrar por que aconteceu, e
  mostrar o que fiz a respeito."

> Dizer isso no primeiro minuto muda a leitura do vídeo inteiro: quem
> assiste para de esperar o anúncio de um ganho e passa a acompanhar uma
> investigação.

## Bloco 2 — Codificação do GA (1:30 – 3:30)

**Mostrar:** `src/ga/search_space.py` e rodar
`python -m src.ga.search_space` (é instantâneo).

Falar:

- Cromossomo é um vetor de floats em `[0,1]`; o decodificador traduz
  cada posição pro domínio do hiperparâmetro.
- Por que normalizada: cruzamento genérico, limites nunca violados, e a
  escala logarítmica do `C` sai de graça — amostrar linear em
  `[0.001, 100]` colocaria 99% da massa acima de 1.
- Mostrar na saída: o espaço da Random Forest tem **11,9 milhões de
  combinações discretas**. Busca exaustiva está fora de questão; é o que
  justifica o GA.
- Operador de reparo: `lbfgs` não aceita `penalty='l1'`. Em vez de
  descartar o indivíduo (o que jogaria material genético fora), conserta
  o fenótipo.

## Bloco 3 — Operadores e fitness (3:30 – 5:30)

**Mostrar:** `src/ga/operators.py` e `src/ga/fitness.py`.

Falar:

- Seleção por torneio e por roleta; cruzamento uniforme, 1 ponto e
  BLX-alfa; mutação sensível ao tipo do gene.
- **O detalhe que vale explicar:** a mutação de gene categórico é
  re-sorteio, não perturbação gaussiana. Com três categorias, cada faixa
  tem 0,33 de largura e um ruído de desvio 0,15 quase nunca atravessa a
  fronteira — a mutação viraria um no-op silencioso justamente nos genes
  mais discretos.
- Fitness: `0,60 × recall_mal + 0,25 × F1_mal + 0,15 × acurácia`.
  O peso do recall vem do custo clínico assimétrico da Fase 1. O F1
  existe para impedir a solução degenerada de chamar tudo de maligno.
- **Enfatizar:** o fitness é medido por validação cruzada **no treino**.
  Se fosse no teste, o GA otimizaria contra ele e todo número do
  comparativo seria vazamento.

## Bloco 4 — Os três experimentos (5:30 – 7:30)

**Mostrar:** `src/ga/run_experiments.py` (a lista `EXPERIMENTOS`) e a
figura `30_ga_convergencia.png`.

Falar:

- Três configurações: `equilibrado` (referência), `exploratorio`
  (população grande, mutação 3×, roleta, BLX) e `intensivo` (população
  pequena, 40 gerações, torneio de 5, elitismo forte).
- **Orçamento igual nos três: 600 avaliações.** Se um tivesse o dobro,
  um resultado melhor não diria nada sobre a estratégia.
- Mostrar a curva de convergência. Apontar que o `intensivo` converge
  nas primeiras gerações e fica plano o resto do tempo.
- Abrir `32_ga_diversidade.png`: a diversidade do `intensivo` despenca
  para perto de zero. É convergência prematura, visível no gráfico — a
  população virou cópia do elite e as 35 gerações restantes não
  exploraram nada.
- Mostrar o log: no `intensivo` da Random Forest, **95 avaliações reais
  e 505 do cache**. O cache de fitness economizou 84% do trabalho.

## Bloco 5 — O resultado e a investigação (7:30 – 10:00)

> **Este é o bloco mais importante do vídeo.** Não apressar.

**Mostrar:** `reports/fase2/comparison.md`, depois
`34_matrizes_confusao.png`, depois `35_cv_vs_teste.png`.

Falar:

1. **O fitness de CV melhorou nos três** (+1,3 / +4,3 / +1,7 p.p.). O GA
   fez o que foi pedido.
2. **No teste, não transferiu.** Mostrar a tabela: o recall do maligno
   ficou idêntico nos três, e a precisão caiu. Nas matrizes de confusão:
   a Random Forest saiu de 0 para 4 falsos positivos; a Árvore, de 4
   para 12.
3. **Por quê.** Abrir `35_cv_vs_teste.png` — nove pontos, fitness de CV
   no eixo x, F1 de teste no eixo y. Se o fitness previsse o teste, os
   pontos subiriam. Eles descem. A correlação de Spearman é **−1,00 na
   Árvore e −0,63 na Floresta**: quanto melhor o fitness, pior o teste.
4. **O que isso é:** sobreajuste do critério de seleção. 600 avaliações
   num espaço de 6 a 8 genes, sobre 455 amostras divididas em **5 folds
   fixos** — o GA encontra configurações que exploram o ruído daquela
   partição específica.
5. **O caso que fecha o argumento** (vale 30 segundos): na Random Forest,
   `equilibrado` e `exploratorio` deram fitness **idêntico até a 16ª casa
   decimal** (`0.9533482273924908` nos dois), com hiperparâmetros bem
   diferentes — 150 árvores contra 110, `min_samples_split` 3 contra 24.
   Uma tinha 1 falso negativo no teste, a outra 3, e o desempate foi por
   ordem de execução. O estimador não tem resolução para ordenar
   candidatos próximos.
6. **Mencionar o segundo fator, sem usá-lo como desculpa:** o teste tem
   42 casos malignos, então o recall só se move em passos de 2,4 pontos
   percentuais. Ele não tem resolução para mostrar ganho pequeno.

## Bloco 5b — A correção e o que ela provou (10:00 – 11:00)

**Mostrar:** `36_experimento_robusto.png`.

- **A correção:** trocar a CV de 5 folds por 5 folds × 3 repetições —
  três particionamentos, média das três. O ruído de um fold específico
  não sobrevive à média.
- **O resultado é a evidência mais forte do trabalho.** Os três painéis
  da figura: no painel 1 (régua 5-fold) o vencedor laranja ganha; no
  painel 2 (régua 5×3) o vencedor azul ganha; no painel 3 (teste)
  nenhum dos dois ganha. **Cada configuração vence apenas no estimador
  contra o qual foi otimizada — 6 de 6 comparações.**
- **Quantificar:** o ganho de 4,25 pontos da Árvore vira **0,61** quando
  medido na régua 5×3. Na Random Forest, de 1,72 vira **0,18**. Ou seja,
  86% e 90% do "ganho" eram ruído.
- **Ser honesto sobre o limite da correção:** ela melhorou a Árvore
  (falsos positivos 12 → 10), piorou marginalmente a Logística, e na
  Floresta trocou um falso positivo a menos por **um falso negativo a
  mais** — na direção errada, clinicamente. Medir melhor não substitui
  ter o que medir.

## Bloco 6 — LLM ao vivo (11:00 – 12:30)

**Rodar ao vivo:** `python -m src.llm.explain --n-casos 2`
(leva ~20 s com a API).

Falar enquanto roda:

- A LLM **nunca vê dado bruto nem chama modelo**. Recebe um payload JSON
  fechado, com todos os números já calculados em Python. Mostrar um
  payload em `llm_payloads.json`.
- Isso existe por três razões: auditabilidade, a LLM não precisa fazer
  aritmética, e o payload fica salvo para reconferir a explicação meses
  depois.
- Os casos **não são aleatórios** — o sistema escolhe falso negativo,
  discordância entre modelos e baixa confiança. Caso com 99,9% de
  certeza e consenso não precisa de explicação.
- Mostrar a saída: as quatro seções, e em particular um caso em que os
  três modelos divergem, com o texto abrindo por isso.
- Mostrar `prompts.py`: três versões documentadas, com o que mudou e
  **por quê**. A v1 inventava métricas; a v2 escolhia um "vencedor" nos
  casos de discordância e apagava a informação mais útil.

## Bloco 7 — Avaliação da qualidade (12:30 – 13:30)

**Mostrar:** `reports/fase2/llm_quality.md`.

Falar:

- Duas camadas. A **Camada 1 é código**: extrai todo número do texto e
  confere contra o payload. Alucinação numérica em contexto clínico é o
  modo de falha mais perigoso e o mais fácil de passar batido numa
  leitura casual.
- A **Camada 2 é a LLM como juiz**, instruída a ser cética, dando nota
  em cinco critérios.
- **Quando discordam, a Camada 1 ganha** — e a tabela registra a
  divergência, porque ela mede a confiabilidade do juiz.
- Mostrar a comparação entre prompts v1/v2/v3: é evidência medida, não
  impressão.
- **Mencionar:** a auditoria pegou dois defeitos no meu próprio gerador
  por template — jargão não traduzido e falta da ressalva de decisão
  clínica. Os dois viraram teste de regressão.

## Bloco 8 — Testes, monitoramento e nuvem (13:30 – 14:30)

**Rodar ao vivo:** `python -m pytest -q` — mas atenção, a suíte leva
~2,5 min. **Alternativa melhor:** rodar
`python -m pytest tests/test_operators.py tests/test_llm.py -q`
(≈10 s) e mostrar o resultado da suíte completa já capturado.

Falar:

- 201 testes. Os de operador testam **propriedade**, não saída fixa: "o
  torneio devolve o melhor quando o torneio é a população inteira", "a
  mutação nunca sai de [0,1]".
- O teste mais valioso treina de verdade 25 fenótipos sorteados em cada
  espaço — é o que garante que o decodificador e o reparo cobrem toda
  combinação inválida.
- Monitoramento: log estruturado JSONL, com monitor de previsões que
  acompanha taxa de maligno e confiança média — detecção de drift.
- Nuvem (opcional): mostrar o diagrama do `infra/README.md`. Batch para
  a otimização, Lambda para a inferência, S3 como fronteira. **Dizer com
  clareza que não foi aplicado** — a entrega é o código de
  infraestrutura.
- O detalhe que vale mostrar: os metric filters do CloudWatch leem o
  mesmo JSONL do `monitoring.py` e o transformam em métrica, sem agente
  nenhum. É a razão de o log ser JSON e não texto.

## Bloco 9 — Fechamento (14:30 – 15:00)

- O GA está implementado, testado e funciona: melhora a função objetivo
  que recebe.
- O aprendizado da fase não foi "o GA melhorou os modelos" — foi
  **medir que não melhorou e descobrir por quê**. Otimizar um estimador
  de desempenho não é a mesma coisa que melhorar o desempenho.
- A camada de LLM é reaproveitável no Módulo 3 (dados textuais).
- Aviso final: ferramenta de apoio, não substitui avaliação de
  profissional de saúde.

---

## Perguntas prováveis da banca

| Pergunta | Resposta curta |
|---|---|
| Por que não usou DEAP/PyGAD? | O enunciado pede a implementação dos operadores. Escrevendo à mão, cada um fica testável isoladamente — são 201 testes sobre eles. |
| Por que o GA não melhorou? | Melhorou o que foi medido (fitness de CV). Não transferiu porque o critério sobreajustou aos folds — Spearman −1,00 / −0,63 — e porque o teste tem 42 malignos, resolução de 2,4 p.p. |
| Como sabe que não houve vazamento? | O fitness nunca tocou o teste: `StratifiedKFold` sobre `X_train`. A escolha do vencedor também é por fitness de CV. A análise CV-vs-teste é diagnóstico posterior, não critério. |
| A LLM pode inventar número? | Pode — por isso existe a auditoria determinística que confere cada número contra o payload. É código, não confiança. |
| Por que o baseline foi retreinado? | Os `.joblib` da Fase 1 são de outra versão do scikit-learn e nem carregam. Retreinar também é mais justo: original e otimizado passam pela mesma biblioteca. Confere com a Fase 1 em 0,0000 de diferença. |
| Qual o custo da nuvem? | ~US$ 5/mês de infraestrutura; a API da LLM domina (~US$ 25-30 para mil exames). Daí o teto de concorrência e a métrica de custo por chamada. |

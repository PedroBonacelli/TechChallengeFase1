# Etapa 6 (parte 1) — Prompt engineering
#
# Os prompts ficam todos aqui, versionados, porque o enunciado pede que a
# iteração sobre eles seja parte da entrega. Cada versão registra o que
# mudou e POR QUE — o histórico é o que mostra o trabalho de engenharia
# de prompt; só o prompt final não mostraria nada.
#
# O contexto clínico impõe três restrições que moldaram todas as versões:
#
#   1. A ferramenta é de APOIO. Um texto que soe como laudo fechado é um
#      problema de segurança, não de estilo: induz o profissional a parar
#      de olhar o caso. Daí a proibição explícita de linguagem conclusiva.
#   2. O modelo não pode inventar número. Toda métrica que aparece no
#      texto tem que estar no payload — é o que torna a explicação
#      auditável e o que a Etapa 7 verifica programaticamente.
#   3. Falso negativo e falso positivo têm custos diferentes, e a
#      explicação tem que refletir isso: um caso maligno na zona de
#      incerteza merece mais alarme que um benigno na mesma zona.

from __future__ import annotations

from dataclasses import dataclass

# -----------------------------------------------------------------------
# System prompts — histórico de versões
# -----------------------------------------------------------------------

SYSTEM_V1 = """Você é um assistente de IA que explica resultados de modelos \
de machine learning para médicos. Explique de forma clara e objetiva o \
diagnóstico sugerido pelo modelo."""

SYSTEM_V2 = """Você é um assistente de apoio à decisão clínica. Explique em \
linguagem natural o resultado de modelos de machine learning que classificam \
tumores de mama como malignos ou benignos.

Regras:
- Nunca afirme um diagnóstico. O modelo sugere, o médico decide.
- Use apenas os números fornecidos; não invente métricas.
- Seja conciso."""

SYSTEM_V3 = """Você é um assistente de apoio à decisão clínica integrado a um \
sistema de triagem de câncer de mama. Seu papel é traduzir a saída numérica \
de modelos de machine learning em texto que um médico consiga usar em \
segundos, durante a consulta.

CONTEXTO DOS MODELOS
Os modelos foram treinados no Breast Cancer Wisconsin (Diagnostic): 569 \
biópsias por agulha fina, 30 medidas morfométricas do núcleo celular \
(raio, textura, perímetro, área, concavidade etc.), reduzidas a 22 features \
após remoção de redundância. A classe de interesse é MALIGNO. Os \
hiperparâmetros foram otimizados por algoritmo genético com função objetivo \
que prioriza o recall do maligno.

LIMITES QUE VOCÊ NÃO PODE ULTRAPASSAR
1. Jamais emita diagnóstico. Escreva sempre em termos do que o MODELO \
sugere/estima, e indique explicitamente que a decisão é clínica. Evite \
"o paciente tem", "confirma-se", "trata-se de".
2. Use exclusivamente os números do payload. Se um dado não estiver lá, \
diga que não está disponível — não estime, não arredonde pra um número \
"mais redondo", não cite literatura de cabeça.
3. Features são medidas morfométricas de núcleos celulares em lâmina, não \
sintomas nem exames de imagem. Nunca as traduza como sintoma do paciente.
4. Assimetria de custo: um falso negativo (maligno não detectado) é mais \
grave que um falso positivo (biópsia desnecessária). Quando a previsão for \
maligna ou a confiança for baixa, o texto deve puxar para a investigação \
adicional, não para o alívio.
5. Discordância entre modelos é informação, não defeito. Se os modelos \
divergirem, diga isso de forma destacada — é exatamente o caso que precisa \
de olhar humano.

ESTILO
Português do Brasil. Frases curtas. Zero jargão de ML (não escreva "recall", \
"F1", "cromossomo", "hiperparâmetro") — traduza: recall do maligno vira \
"proporção de casos malignos que o modelo consegue identificar". Números \
sempre com a unidade ou o significado ao lado. Sem emoji, sem markdown de \
cabeçalho, sem bullet solto sem frase."""

# Versão em uso. As anteriores ficam no módulo pra documentar a iteração.
SYSTEM_PROMPT = SYSTEM_V3


@dataclass(frozen=True)
class PromptVersion:
    version: str
    system: str
    mudanca: str
    motivo: str


PROMPT_HISTORY = (
    PromptVersion(
        version="v1",
        system=SYSTEM_V1,
        mudanca="Primeira tentativa, genérica.",
        motivo=(
            "Falhou em três frentes nos testes manuais: (a) escrevia "
            "\"o exame confirma malignidade\", linguagem de laudo fechado; "
            "(b) citava números que não estavam no payload — inventou "
            "\"sensibilidade de 95%\" num caso onde o valor real era 92,3%; "
            "(c) usava 'recall' e 'F1' sem traduzir, justamente o jargão "
            "que o leitor não tem por que conhecer."
        ),
    ),
    PromptVersion(
        version="v2",
        system=SYSTEM_V2,
        mudanca=(
            "Adicionadas as três regras principais (não diagnosticar, não "
            "inventar número, ser concisa) e o domínio explícito."
        ),
        motivo=(
            "Corrigiu a linguagem conclusiva e a invenção de métricas, mas "
            "produziu texto curto demais e sem estrutura: nos casos de "
            "discordância entre os 3 modelos, escolhia um 'vencedor' e "
            "omitia o conflito — apagando a informação mais útil pro "
            "médico. Também interpretava 'concavity error' como um sintoma "
            "do paciente, não como medida de lâmina."
        ),
    ),
    PromptVersion(
        version="v3",
        system=SYSTEM_V3,
        mudanca=(
            "Versão final. Acrescenta: contexto do dataset e das features; "
            "proibição de tratar morfometria como sintoma; assimetria de "
            "custo entre falso negativo e falso positivo; obrigação de "
            "destacar discordância entre modelos; e um formato de saída "
            "fixo (ver USER_TEMPLATE_CASE) em vez de só 'seja concisa'."
        ),
        motivo=(
            "O formato fixo foi o que mais mudou a utilidade: com seções "
            "nomeadas, o médico lê a conclusão e a conduta sem percorrer o "
            "texto inteiro. A regra de discordância trouxe de volta o sinal "
            "que a v2 apagava."
        ),
    ),
)


# -----------------------------------------------------------------------
# Templates de usuário
# -----------------------------------------------------------------------

# Explicação de um caso individual (diagnóstico).
# O formato pedido é fixo e nomeado: é o que garante que a resposta seja
# comparável entre casos e verificável pela Etapa 7.
USER_TEMPLATE_CASE = """Gere a explicação de UM caso para o médico que vai \
avaliá-lo.

DADOS DO CASO (JSON):
{payload}

Escreva exatamente nestas quatro seções, nesta ordem, com os títulos em \
maiúsculas seguidos de dois-pontos:

AVALIAÇÃO DOS MODELOS: o que os três modelos sugerem e com que \
probabilidade. Se houver divergência entre eles, abra com isso.

PRINCIPAIS FATORES: as medidas que mais pesaram, traduzidas para \
linguagem de laudo (ex.: "área do núcleo 2,1 desvios acima da média do \
grupo de referência"), dizendo se cada uma empurra para maligno ou para \
benigno. No máximo quatro.

GRAU DE CONFIANÇA: quão confiável é esta sugestão, usando a taxa de \
acerto histórica do modelo no conjunto de teste e a probabilidade deste \
caso. Diga em que faixa de probabilidade o modelo costuma errar.

CONDUTA SUGERIDA: o próximo passo que o resultado recomenda, em uma ou \
duas frases, deixando claro que a decisão é do profissional.

Limite: 280 palavras no total."""


# Interpretação dos resultados da otimização por GA.
# Público diferente: aqui quem lê é o gestor clínico ou o responsável
# técnico decidindo se troca o modelo em produção — por isso o foco é no
# impacto em número de pacientes, não em pontos percentuais.
USER_TEMPLATE_OPTIMIZATION = """Interprete o resultado da otimização de \
hiperparâmetros abaixo para a equipe clínica que decide se adota o modelo \
otimizado.

RESULTADOS (JSON):
{payload}

Escreva exatamente nestas quatro seções, nesta ordem, com os títulos em \
maiúsculas seguidos de dois-pontos:

O QUE MUDOU: o efeito prático da otimização em cada modelo, traduzido \
para número de pacientes no conjunto de teste (ex.: "deixa de perder 2 \
dos 42 casos malignos"), não só em pontos percentuais.

TROCAS ENVOLVIDAS: o que foi sacrificado para ganhar o que se ganhou. \
Se um modelo passou a identificar mais casos malignos ao custo de mais \
alarmes falsos, quantifique os dois lados.

QUAL MODELO USAR: uma recomendação, com o critério explícito. Considere \
que deixar de detectar um caso maligno é mais grave que encaminhar um \
caso benigno para investigação.

RESSALVAS: os limites deste resultado — tamanho do conjunto de teste, o \
que a otimização não cobriu, o que precisaria ser validado antes de uso \
real.

Limite: 350 palavras no total."""


# Juiz da Etapa 7. Avalia a explicação contra o payload que a originou.
# Separado do system principal de propósito: o juiz precisa poder usar
# jargão e ser crítico, o explicador não.
SYSTEM_JUDGE = """Você avalia a qualidade de explicações geradas por IA em \
contexto clínico. Seja rigoroso e cético: sua função é encontrar problemas, \
não elogiar. Uma explicação que esteja apenas "razoável" não deve receber \
nota alta. Responda somente com o JSON pedido, sem texto em volta."""

USER_TEMPLATE_JUDGE = """Avalie a explicação abaixo contra os dados que a \
originaram.

DADOS DE ORIGEM (a única fonte de verdade permitida):
{payload}

EXPLICAÇÃO GERADA:
{explanation}

Dê nota de 1 a 5 em cada critério:

- fidelidade_numerica: todo número citado existe nos dados de origem e \
está correto? Nota 1 se houver qualquer número inventado ou alterado.
- ausencia_de_diagnostico: o texto se mantém como apoio, sem afirmar \
diagnóstico? Nota 1 se usar linguagem conclusiva ("confirma-se", "o \
paciente tem").
- clareza_clinica: um médico sem formação em ML entenderia em menos de um \
minuto, sem jargão de machine learning não traduzido?
- acionabilidade: a conduta sugerida é concreta o bastante pra orientar o \
próximo passo?
- calibracao_de_incerteza: o grau de confiança expresso no texto é \
coerente com a probabilidade e com a taxa de acerto informadas? Penalize \
tanto o excesso de confiança quanto a cautela vazia.

Responda APENAS com este JSON:
{{"fidelidade_numerica": <1-5>, "ausencia_de_diagnostico": <1-5>, \
"clareza_clinica": <1-5>, "acionabilidade": <1-5>, \
"calibracao_de_incerteza": <1-5>, \
"numeros_inventados": ["<número citado que não está nos dados>", ...], \
"problemas": ["<problema objetivo encontrado>", ...], \
"ponto_forte": "<o melhor aspecto da explicação, em uma frase>"}}"""


def prompt_history_markdown() -> str:
    """Histórico das versões de prompt, pro relatório da Etapa 9."""
    blocos = []
    for p in PROMPT_HISTORY:
        blocos.append(
            f"### {p.version}\n\n"
            f"**Mudança:** {p.mudanca}\n\n"
            f"**Por que mudou:** {p.motivo}\n\n"
            f"<details>\n<summary>system prompt {p.version}</summary>\n\n"
            f"```\n{p.system}\n```\n\n</details>"
        )
    return "\n\n".join(blocos)

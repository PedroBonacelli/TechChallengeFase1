# Gera o documento de entrega da Fase 2 em .docx.
#
#   python reports/fase2/build_entrega_docx.py
#
# Por que .docx e não markdown + pandoc, como foi feito na Fase 1: o
# documento precisa marcar pendências EM VERMELHO, e não há pandoc,
# LaTeX nem Word neste ambiente para converter markdown com cor. O
# python-docx escreve o arquivo final direto, com a cor nativa, e o
# Word exporta o PDF a partir dele.
#
# Este script é a FONTE do documento — o .docx é saída regenerável.
# Depois de gerado, dá para editar livremente no Word; só lembre que
# reexecutar o script sobrescreve as edições.
#
# DEPOIS DE ABRIR NO WORD: o sumário é um campo, não texto. Ele aparece
# vazio até ser atualizado — Ctrl+A e depois F9 (ou clicar no sumário e
# escolher "Atualizar índice"). Em seguida, Arquivo > Exportar > PDF.
#
# Convenção de cor: VERMELHO marca o que ainda não foi entregue. Inclui
# tanto o que falta fazer (vídeo, LLM real, commit) quanto o que foi
# conscientemente deixado fora de escopo (nuvem não provisionada), para
# que o avaliador veja tudo que não está pronto sem caçar no texto.

from __future__ import annotations

from pathlib import Path

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Inches, Pt, RGBColor

BASE_DIR = Path(__file__).resolve().parent.parent.parent
FIG_DIR = BASE_DIR / "reports" / "fase2" / "figures"
SAIDA = BASE_DIR / "reports" / "fase2" / "entrega_final_fase2.docx"

# Vermelho escuro em vez de #FF0000 puro: imprime melhor e não vibra na
# tela, mantendo o contraste de "isto está pendente".
VERMELHO = RGBColor(0xC0, 0x00, 0x00)
CINZA = RGBColor(0x59, 0x59, 0x59)
LARGURA_FIG = Inches(6.3)  # A4 (21 cm) menos 2 x 2,5 cm de margem


# -----------------------------------------------------------------------
# Helpers de formatação
# -----------------------------------------------------------------------


def configurar_pagina(doc: Document) -> None:
    """A4 com margens de 2,5 cm — espelha o `geometry: margin=2.5cm`
    do frontmatter pandoc usado no documento da Fase 1."""
    for secao in doc.sections:
        secao.page_width = Cm(21.0)
        secao.page_height = Cm(29.7)
        for lado in ("top", "bottom", "left", "right"):
            setattr(secao, f"{lado}_margin", Cm(2.5))

    # Calibri 11 no corpo: é o padrão do Word e o que o PDF da Fase 1 usa
    normal = doc.styles["Normal"]
    normal.font.name = "Calibri"
    normal.font.size = Pt(11)
    normal.paragraph_format.space_after = Pt(8)
    normal.paragraph_format.line_spacing = 1.15


def _aplicar(run, estilo: dict) -> None:
    if estilo.get("bold"):
        run.bold = True
    if estilo.get("italic"):
        run.italic = True
    if estilo.get("red"):
        run.font.color.rgb = VERMELHO
    if estilo.get("gray"):
        run.font.color.rgb = CINZA
    if estilo.get("code"):
        run.font.name = "Consolas"
        run.font.size = Pt(9.5)
    if estilo.get("size"):
        run.font.size = Pt(estilo["size"])


def p(doc: Document, *trechos, align=None, space_after=None):
    """Parágrafo com runs mistos.

    Cada trecho é uma string (texto simples) ou uma tupla
    (texto, {"bold": True, "red": True, ...}). É isso que permite
    vermelho no meio da frase, em vez de só em parágrafos inteiros.
    """
    par = doc.add_paragraph()
    if align is not None:
        par.alignment = align
    if space_after is not None:
        par.paragraph_format.space_after = Pt(space_after)
    for trecho in trechos:
        texto, estilo = trecho if isinstance(trecho, tuple) else (trecho, {})
        _aplicar(par.add_run(texto), estilo)
    return par


def bullet(doc: Document, *trechos, nivel: int = 0):
    par = doc.add_paragraph(style="List Bullet" if nivel == 0 else "List Bullet 2")
    par.paragraph_format.space_after = Pt(4)
    for trecho in trechos:
        texto, estilo = trecho if isinstance(trecho, tuple) else (trecho, {})
        _aplicar(par.add_run(texto), estilo)
    return par


def numerado(doc: Document, *trechos):
    par = doc.add_paragraph(style="List Number")
    par.paragraph_format.space_after = Pt(4)
    for trecho in trechos:
        texto, estilo = trecho if isinstance(trecho, tuple) else (trecho, {})
        _aplicar(par.add_run(texto), estilo)
    return par


def h1(doc: Document, texto: str):
    """Heading 1 nativo — já vem azul no tema padrão do Office, que é
    exatamente o visual dos títulos no PDF da Fase 1."""
    return doc.add_heading(texto, level=1)


def h2(doc: Document, texto: str):
    return doc.add_heading(texto, level=2)


def quebra(doc: Document) -> None:
    """Quebra de página — equivalente ao `\\newpage` da Fase 1."""
    doc.add_paragraph().add_run().add_break(WD_BREAK.PAGE)


def sumario(doc: Document) -> None:
    """Insere o campo TOC do Word.

    O python-docx não tem API para sumário, então o campo é montado em
    XML bruto. Consequência que o usuário precisa saber: o Word NÃO
    preenche o campo sozinho ao abrir — é preciso Ctrl+A e F9. Está
    avisado no cabeçalho deste script e no próprio documento.
    """
    par = doc.add_paragraph()
    run = par.add_run()

    inicio = OxmlElement("w:fldChar")
    inicio.set(qn("w:fldCharType"), "begin")

    instrucao = OxmlElement("w:instrText")
    instrucao.set(qn("xml:space"), "preserve")
    # \o "1-2" = níveis 1 e 2; \h = hyperlinks; \z = oculta no modo web;
    # \u = usa os níveis de estrutura de tópicos
    instrucao.text = r'TOC \o "1-2" \h \z \u'

    separador = OxmlElement("w:fldChar")
    separador.set(qn("w:fldCharType"), "separate")

    aviso = OxmlElement("w:t")
    aviso.text = "Clique aqui e pressione F9 para gerar o sumário."

    fim = OxmlElement("w:fldChar")
    fim.set(qn("w:fldCharType"), "end")

    for elemento in (inicio, instrucao, separador, aviso, fim):
        run._r.append(elemento)


def tabela(doc: Document, cabecalho: list[str], linhas: list[list], larguras=None):
    """Tabela com grade e cabeçalho em negrito.

    Cada célula pode ser uma string ou uma tupla (texto, estilo) — o que
    permite destacar uma célula em negrito ou em vermelho.
    """
    tab = doc.add_table(rows=1, cols=len(cabecalho))
    tab.style = "Table Grid"
    tab.alignment = WD_TABLE_ALIGNMENT.CENTER

    for celula, texto in zip(tab.rows[0].cells, cabecalho):
        celula.paragraphs[0].text = ""
        run = celula.paragraphs[0].add_run(texto)
        run.bold = True
        run.font.size = Pt(9.5)

    for linha in linhas:
        celulas = tab.add_row().cells
        for celula, conteudo in zip(celulas, linha):
            texto, estilo = (
                conteudo if isinstance(conteudo, tuple) else (conteudo, {})
            )
            celula.paragraphs[0].text = ""
            run = celula.paragraphs[0].add_run(str(texto))
            run.font.size = Pt(9.5)
            _aplicar(run, estilo)

    if larguras:
        for linha in tab.rows:
            for celula, largura in zip(linha.cells, larguras):
                celula.width = Cm(largura)

    doc.add_paragraph().paragraph_format.space_after = Pt(2)
    return tab


def figura(doc: Document, nome: str, legenda: str, largura=LARGURA_FIG) -> None:
    caminho = FIG_DIR / nome
    if not caminho.exists():
        p(doc, (f"[figura ausente: {nome}]", {"red": True, "italic": True}))
        return

    par = doc.add_paragraph()
    par.alignment = WD_ALIGN_PARAGRAPH.CENTER
    par.paragraph_format.space_after = Pt(2)
    par.add_run().add_picture(str(caminho), width=largura)

    cap = doc.add_paragraph()
    cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
    cap.paragraph_format.space_after = Pt(12)
    run = cap.add_run(legenda)
    run.italic = True
    run.font.size = Pt(9)
    run.font.color.rgb = CINZA


def capa(doc: Document) -> None:
    titulo = doc.add_paragraph()
    titulo.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = titulo.add_run("Tech Challenge — Fase 2")
    run.bold = True
    run.font.size = Pt(24)
    run.font.color.rgb = RGBColor(0x1F, 0x4E, 0x79)

    sub = doc.add_paragraph()
    sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = sub.add_run(
        "Otimização de Modelos de Diagnóstico por Algoritmo Genético\n"
        "e Integração com LLM"
    )
    run.bold = True
    run.font.size = Pt(14)
    run.font.color.rgb = RGBColor(0x2E, 0x74, 0xB5)

    p(
        doc,
        "Diego Matias (rm375934), Pedro Bonacelli (rm374881),\n"
        "Mateus Augusto (rm375935), Yan Hudson (rm375933)",
        align=WD_ALIGN_PARAGRAPH.CENTER,
    )

    p(
        doc,
        ("[data da entrega — a definir]", {"red": True, "bold": True}),
        align=WD_ALIGN_PARAGRAPH.CENTER,
    )

    p(
        doc,
        (
            "Tudo em vermelho neste documento é pendência — o que ainda "
            "falta entregar e o que ficou conscientemente fora de "
            "escopo —, e nada além disso. A lista consolidada está na "
            "seção 11. Negrito em tabelas de métricas marca destaque ou "
            "piora, não pendência.",
            {"italic": True, "gray": True, "size": 9.5},
        ),
        align=WD_ALIGN_PARAGRAPH.CENTER,
    )

    doc.add_paragraph()
    h1(doc, "Sumário")
    sumario(doc)
    quebra(doc)


# -----------------------------------------------------------------------
# Seções do documento
# -----------------------------------------------------------------------

REPO = "https://github.com/PedroBonacelli/TechChallengeFase1"


def secao_1_entregaveis(doc: Document) -> None:
    h1(doc, "1. Entregáveis")

    p(doc, ("Repositório: ", {"bold": True}), REPO,
      "  — branch ", ("tehc-challenge-fase-2", {"code": True}), ".")

    p(doc,
      "O repositório contém o código-fonte do algoritmo genético ",
      ("(src/ga/)", {"code": True}),
      ", da camada de LLM ", ("(src/llm/)", {"code": True}),
      ", o monitoramento ", ("(src/monitoring.py)", {"code": True}),
      ", a suíte de testes ", ("(tests/)", {"code": True}),
      ", a infraestrutura como código ", ("(infra/)", {"code": True}),
      ", todos os resultados e figuras ", ("(reports/fase2/)", {"code": True}),
      " e o relatório técnico completo ",
      ("(reports/fase2/RELATORIO_TECNICO_FASE2.md)", {"code": True}),
      ", com instruções de execução em ",
      ("README_FASE2.md", {"code": True}), ".")

    p(doc, ("Vídeo de demonstração: ", {"bold": True}),
      ("pendente — não gravado nem publicado. O roteiro completo, com "
       "10 blocos e marcação de tempo, está pronto em "
       "reports/fase2/roteiro_video.md.", {"red": True}))

    p(doc, ("Apresentação: ", {"bold": True}),
      ("pendente — os slides da Fase 2 ainda não foram montados.",
       {"red": True}))

    p(doc, ("Publicação do código: ", {"bold": True}),
      ("pendente — os 17 arquivos novos/modificados da Fase 2 ainda não "
       "foram commitados nem enviados ao GitHub. Até que isso aconteça, "
       "o link acima mostra apenas a Fase 1.", {"red": True}))

    quebra(doc)


def secao_2_problema(doc: Document) -> None:
    h1(doc, "2. O Problema e o Ponto de Partida")

    p(doc,
      "A Fase 1 entregou a base do sistema de apoio ao diagnóstico de "
      "câncer de mama: três classificadores treinados sobre o Breast "
      "Cancer Wisconsin (Diagnostic), com hiperparâmetros escolhidos à "
      "mão, e uma CNN extra sobre o CBIS-DDSM.")

    p(doc, "A Fase 2 tem dois objetivos sobre essa base:")

    numerado(doc, ("Otimizar os hiperparâmetros ", {"bold": True}),
             "dos três modelos estruturados com um algoritmo genético "
             "implementado do zero.")
    numerado(doc, ("Traduzir os resultados numéricos ", {"bold": True}),
             "em linguagem natural, por meio de uma LLM, para que um "
             "profissional de saúde consiga usá-los.")

    p(doc,
      "Os dados são exatamente os mesmos da Fase 1, sem reprocessamento: "
      "455 amostras de treino (170 malignas, 37,4%), 114 de teste "
      "(42 malignas, 36,8%), 22 features após a poda de "
      "multicolinearidade, mesmo split com ",
      ("random_state=42", {"code": True}),
      ". Isso é o que torna o comparativo honesto — se o "
      "pré-processamento mudasse, não daria para saber se um ganho veio "
      "do algoritmo genético ou de um split diferente.")

    h2(doc, "O ponto de partida a ser batido")

    tabela(doc,
           ["Modelo (Fase 1)", "Acurácia", "Recall mal.", "F1 mal.",
            "ROC AUC", "Falsos neg."],
           [["Regressão Logística", "0,965", ("0,976", {"bold": True}),
             "0,954", "0,995", ("1", {"bold": True})],
            ["Árvore de Decisão", "0,939", "0,929", "0,918", "0,929", "3"],
            ["Random Forest", ("0,974", {"bold": True}), "0,929",
             ("0,963", {"bold": True}), ("0,996", {"bold": True}), "3"]],
           larguras=[4.4, 2.1, 2.3, 2.0, 2.0, 2.2])

    p(doc,
      ("Observação que condiciona toda a Fase 2: ", {"bold": True}),
      "estes modelos já operam perto do teto. Com ROC AUC de 0,995 e "
      "0,996 e apenas 1 a 3 falsos negativos em 42 casos malignos, o "
      "espaço disponível para melhora real é estreito — e o conjunto de "
      "teste, com 42 malignos, só mede variações de recall em degraus "
      "de 2,4 pontos percentuais.")

    quebra(doc)


def secao_3_ga(doc: Document) -> None:
    h1(doc, "3. Implementação do Algoritmo Genético")

    p(doc,
      "O GA foi escrito à mão, sem DEAP, PyGAD ou biblioteca "
      "equivalente. O enunciado pede a implementação dos operadores, e "
      "escrevendo cada um isoladamente eles ficam testáveis: 201 testes "
      "automatizados cobrem seleção, cruzamento, mutação, fitness, "
      "motor e integração ponta a ponta.")

    h2(doc, "3.1 Codificação dos genes")

    p(doc,
      ("Codificação real normalizada (random-key). ", {"bold": True}),
      "Cada cromossomo é um vetor de floats em [0, 1], um por "
      "hiperparâmetro; um decodificador traduz cada posição para o "
      "domínio real conforme o tipo do gene. O espaço de busca é misto "
      "— há categórico, inteiro e contínuo — mas o genótipo é homogêneo. "
      "Três razões:")

    bullet(doc, ("Cruzamento genérico. ", {"bold": True}),
           "Trocar posições entre dois vetores em [0, 1] é sempre "
           "válido; não é preciso um operador por tipo de gene.")
    bullet(doc, ("Escala logarítmica de graça. ", {"bold": True}),
           "O C da Regressão Logística varia de 1e−3 a 1e2. Amostrar "
           "uniformemente no expoente dá peso igual a cada ordem de "
           "magnitude; amostrar linear em [0,001, 100] colocaria 99% da "
           "massa acima de 1.")
    bullet(doc, ("Limites nunca violados. ", {"bold": True}),
           "Basta manter o vetor em [0, 1], o que o recorte da mutação "
           "garante.")

    p(doc,
      ("A mutação, mesmo assim, é sensível ao tipo. ", {"bold": True}),
      "Gene contínuo ou inteiro recebe perturbação gaussiana (busca "
      "local); gene categórico é re-sorteado uniformemente. O motivo é "
      "concreto: com três categorias cada faixa tem 0,33 de largura, e "
      "um ruído de desvio 0,15 quase nunca atravessa a fronteira — a "
      "mutação viraria um no-op silencioso justamente nos genes mais "
      "discretos do espaço.")

    tabela(doc,
           ["Modelo", "Genes", "Tamanho do espaço de busca"],
           [["Regressão Logística", "6",
             "450 combinações discretas × 2 eixos contínuos"],
            ["Árvore de Decisão", "7",
             "198.360 discretas × 1 eixo contínuo"],
            ["Random Forest", "8",
             ("11.901.600 combinações", {"bold": True})]],
           larguras=[4.5, 1.8, 8.7])

    p(doc,
      "O espaço da Random Forest sozinho descarta busca exaustiva — é "
      "o que justifica a busca heurística.")

    p(doc,
      ("Operador de reparo. ", {"bold": True}),
      "Algumas combinações são inválidas no scikit-learn (o solver "
      "lbfgs não aceita penalty l1; elasticnet só existe no saga). Em "
      "vez de descartar o indivíduo — o que jogaria material genético "
      "fora e distorceria a seleção — um operador de reparo conserta o "
      "fenótipo após a decodificação. Um teste sorteia 25 cromossomos "
      "por espaço e treina cada um de verdade, garantindo que nenhuma "
      "combinação inválida escapa.")

    h2(doc, "3.2 Operadores genéticos")

    tabela(doc,
           ["Operador", "Variantes implementadas"],
           [["Seleção", "torneio (tamanho configurável), roleta"],
            ["Cruzamento", "uniforme, 1 ponto, BLX-α"],
            ["Mutação",
             "gaussiana + re-sorteio categórico, taxa e σ configuráveis"],
            ["Elitismo", "N melhores copiados direto para a próxima geração"]],
           larguras=[3.5, 11.5])

    p(doc,
      ("Por que o torneio é o padrão. ", {"bold": True}),
      "As métricas dos modelos ficam comprimidas entre 0,90 e 0,99. Na "
      "roleta, a probabilidade é proporcional ao fitness, então um "
      "indivíduo ruim (0,90) tem quase a mesma chance de um ótimo "
      "(0,98) — a pressão seletiva quase desaparece. O torneio olha "
      "apenas a ordem, então continua separando bem mesmo com fitness "
      "comprimido.")

    p(doc,
      ("Elitismo garante monotonicidade. ", {"bold": True}),
      "Com elitismo ≥ 1, o melhor fitness nunca cai de uma geração para "
      "a outra. Há teste de propriedade para isso, e um teste de "
      "contraste sem elitismo que documenta a diferença.")

    h2(doc, "3.3 Função fitness")

    p(doc,
      ("fitness = 0,60 × recall_maligno + 0,25 × F1_maligno "
       "+ 0,15 × acurácia", {"bold": True, "code": True}),
      align=WD_ALIGN_PARAGRAPH.CENTER)

    p(doc, "medida por ",
      ("validação cruzada estratificada de 5 folds sobre o conjunto de "
       "TREINO", {"bold": True}), ".")

    p(doc,
      ("Os pesos. ", {"bold": True}),
      "O recall do maligno domina porque o custo dos dois erros é "
      "assimétrico, como já estabelecido na Fase 1: um falso negativo é "
      "um câncer que segue sem tratamento; um falso positivo é uma "
      "biópsia desnecessária. O F1 entra para impedir a solução "
      "degenerada — um modelo que chamasse tudo de maligno teria recall "
      "1,0 e fitness máximo se o recall fosse o único termo.")

    p(doc,
      ("Onde se mede importa mais que o que se mede. ", {"bold": True}),
      "O fitness nunca toca o conjunto de teste. Se tocasse, o GA "
      "otimizaria diretamente contra ele, o teste teria virado conjunto "
      "de treino de fato, e todo número da seção 6 seria vazamento em "
      "vez de desempenho.")

    p(doc,
      ("Cache de fitness. ", {"bold": True}),
      "Elitismo e convergência fazem o mesmo indivíduo reaparecer em "
      "muitas gerações, e o decodificador mapeia genótipos distintos no "
      "mesmo fenótipo. O cache por fenótipo economizou entre 5% e 84% "
      "das avaliações conforme a configuração.")

    quebra(doc)


def secao_4_experimentos(doc: Document) -> None:
    h1(doc, "4. Os Três Experimentos")

    p(doc,
      ("Orçamento idêntico nos três: 600 avaliações de fitness ",
       {"bold": True}),
      "(população × gerações). Isso é deliberado — se um experimento "
      "tivesse o dobro do orçamento, um resultado melhor não diria nada "
      "sobre a estratégia, apenas que ele treinou mais modelos. Com "
      "orçamento igual, a comparação mede o que se quer medir: dado o "
      "mesmo custo, exploração rende mais ou menos que intensificação?")

    tabela(doc,
           ["Parâmetro", "equilibrado", "exploratorio", "intensivo"],
           [["População", "30", "50", "15"],
            ["Gerações", "20", "12", "40"],
            ["Taxa de cruzamento", "0,90", "0,70", "0,95"],
            ["Operador de cruzamento", "uniforme", "BLX-α (0,5)",
             "BLX-α (0,3)"],
            ["Taxa de mutação", "0,10", ("0,30", {"bold": True}),
             ("0,05", {"bold": True})],
            ["σ da mutação", "0,15", "0,25", "0,08"],
            ["Seleção", "torneio (3)", ("roleta", {"bold": True}),
             ("torneio (5)", {"bold": True})],
            ["Elitismo", "2", "1", "4"],
            ["Semente", "42", "43", "44"]],
           larguras=[5.0, 3.3, 3.3, 3.4])

    bullet(doc, ("equilibrado", {"bold": True}),
           " — referência; parâmetros de meio de tabela.")
    bullet(doc, ("exploratorio", {"bold": True}),
           " — população grande, mutação 3× maior, roleta (pressão "
           "fraca) e BLX. Aposta em cobrir o espaço; o risco é não "
           "sobrar geração para refinar.")
    bullet(doc, ("intensivo", {"bold": True}),
           " — população pequena com o dobro de gerações, torneio de 5, "
           "mutação baixa, elitismo forte. Converge rápido; o risco é "
           "travar num ótimo local.")

    p(doc, "Tempo total das 9 execuções (3 modelos × 3 configurações): ",
      ("27,6 minutos", {"bold": True}), ".")

    figura(doc, "30_ga_convergencia.png",
           "Convergência do GA — melhor fitness por geração. O "
           "'intensivo' (verde) converge nas primeiras gerações e fica "
           "plano o resto do tempo.")

    figura(doc, "32_ga_diversidade.png",
           "Diversidade da população por geração. O 'exploratorio' "
           "(laranja) nunca converge; o 'intensivo' (verde) colapsa em "
           "quatro gerações. Nenhum dos extremos funcionou.")

    p(doc,
      ("Convergência prematura é visível no gráfico. ", {"bold": True}),
      "No 'intensivo' da Random Forest a diversidade cai de 0,270 na "
      "geração 1 para 0,029 na geração 5 — 89% em quatro gerações — e "
      "chega ao mínimo de 0,006 na geração 21. A população virou cópia "
      "do elite cedo, e as 35 gerações seguintes giraram em torno do "
      "mesmo ponto. Isso aparece no log como ",
      ("95 avaliações reais contra 505 do cache", {"bold": True}),
      ": 84% do orçamento foi gasto reavaliando indivíduos idênticos.")

    quebra(doc)


def secao_5_resultados_ga(doc: Document) -> None:
    h1(doc, "5. Resultados da Otimização")

    h2(doc, "5.1 Fitness de validação cruzada nas 9 execuções")

    tabela(doc,
           ["Modelo", "Experimento", "Fitness CV", "Recall mal.",
            "Tempo", "Aval. reais", "Cache"],
           [["Reg. Logística", "equilibrado", "0,9717", "0,9706",
             "40,0 s", "211", "389"],
            ["Reg. Logística", "exploratorio", "0,9707", "0,9706",
             "201,2 s", "542", "58"],
            ["Reg. Logística", ("intensivo", {"bold": True}),
             ("0,9732", {"bold": True}), ("0,9765", {"bold": True}),
             "19,1 s", "119", "481"],
            ["Árvore de Decisão", ("equilibrado", {"bold": True}),
             ("0,9331", {"bold": True}), ("0,9529", {"bold": True}),
             "12,9 s", "295", "305"],
            ["Árvore de Decisão", "exploratorio", "0,9250", "0,9294",
             "24,1 s", "571", "29"],
            ["Árvore de Decisão", "intensivo", "0,9269", "0,9412",
             "19,6 s", "338", "262"],
            ["Random Forest", ("equilibrado", {"bold": True}),
             ("0,9533", {"bold": True}), ("0,9588", {"bold": True}),
             "304,2 s", "269", "331"],
            ["Random Forest", "exploratorio", "0,9533", "0,9588",
             "802,0 s", "564", "36"],
            ["Random Forest", "intensivo", "0,9491", "0,9471",
             "233,2 s", "95", "505"]],
           larguras=[3.4, 2.6, 2.0, 2.2, 1.8, 1.8, 1.2])

    p(doc, "Em negrito, o vencedor de cada modelo — escolhido ",
      ("pelo fitness de validação cruzada", {"bold": True}),
      ", nunca pelo desempenho no teste.")

    h2(doc, "5.2 O que a comparação entre estratégias mostrou")

    p(doc,
      ("A exploração pura nunca venceu sozinha. ", {"bold": True}),
      "O 'exploratorio' ficou em último na Regressão Logística e na "
      "Árvore, e empatou em primeiro na Random Forest — mas gastando "
      "802 s contra 304 s, 2,6× o tempo para chegar ao mesmo valor. Ele "
      "fez 542-571 avaliações reais contra 95-295 das outras: gastou "
      "quase todo o orçamento visitando pontos novos e, com apenas 12 "
      "gerações, não sobrou ciclo para refinar nenhum deles.")

    p(doc,
      ("A intensificação ganhou onde o espaço é pequeno e perdeu onde é "
       "grande. ", {"bold": True}),
      "O 'intensivo' venceu na Regressão Logística (6 genes: refinar é "
      "a coisa certa a fazer) e foi o pior na Random Forest (8 genes, "
      "11,9 milhões de combinações: convergir cedo significa não ter "
      "visto quase nada).")

    h2(doc, "5.3 Hiperparâmetros encontrados")

    tabela(doc, ["Hiperparâmetro", "Fase 1", "Otimizado pelo GA"],
           [["C", "1,0", ("1,039", {"bold": True})],
            ["penalty", "l2", ("l1", {"bold": True})],
            ["solver", "lbfgs", ("liblinear", {"bold": True})],
            ["max_iter", "5000", ("4000", {"bold": True})],
            ["class_weight", "balanced", "balanced"]],
           larguras=[5.0, 5.0, 5.0])
    p(doc, ("Regressão Logística — experimento 'intensivo'.",
            {"italic": True, "gray": True, "size": 9}),
      align=WD_ALIGN_PARAGRAPH.CENTER)

    tabela(doc, ["Hiperparâmetro", "Fase 1", "Otimizado pelo GA"],
           [["criterion", "gini", "gini"],
            ["max_depth", "5", ("19", {"bold": True})],
            ["min_samples_split", "2", ("9", {"bold": True})],
            ["min_samples_leaf", "1", ("3", {"bold": True})],
            ["max_features", "None", ("log2", {"bold": True})],
            ["ccp_alpha", "0,0", ("0,02186", {"bold": True})],
            ["class_weight", "balanced", "balanced"]],
           larguras=[5.0, 5.0, 5.0])
    p(doc, ("Árvore de Decisão — experimento 'equilibrado'.",
            {"italic": True, "gray": True, "size": 9}),
      align=WD_ALIGN_PARAGRAPH.CENTER)

    tabela(doc, ["Hiperparâmetro", "Fase 1", "Otimizado pelo GA"],
           [["n_estimators", "300", ("150", {"bold": True})],
            ["max_depth", "8", ("19", {"bold": True})],
            ["min_samples_split", "2", ("3", {"bold": True})],
            ["min_samples_leaf", "1", ("6", {"bold": True})],
            ["max_features", "sqrt", ("0,5", {"bold": True})],
            ["criterion", "gini", "gini"],
            ["class_weight", "balanced", "balanced"],
            ["bootstrap", "True", "True"]],
           larguras=[5.0, 5.0, 5.0])
    p(doc, ("Random Forest — experimento 'equilibrado'.",
            {"italic": True, "gray": True, "size": 9}),
      align=WD_ALIGN_PARAGRAPH.CENTER)

    p(doc, "Dois pontos valem comentário:")

    bullet(doc, "O GA manteve ",
           ("class_weight='balanced'", {"code": True}),
           " nos três, embora pudesse tê-lo desligado (None estava no "
           "espaço de busca). É uma confirmação independente da escolha "
           "manual da Fase 1.")
    bullet(doc, "A Random Forest ficou com ",
           ("metade das árvores", {"bold": True}),
           " (150 contra 300) sem perder fitness de CV — treina e serve "
           "em metade do tempo.")

    quebra(doc)


def secao_6_comparativo(doc: Document) -> None:
    h1(doc, "6. Comparativo: Original vs. Otimizado")

    h2(doc, "6.1 Nota metodológica: o baseline foi retreinado")

    p(doc,
      "Os arquivos ", ("models/*.joblib", {"code": True}),
      " da Fase 1 foram serializados com scikit-learn 1.8.0, e o "
      "ambiente da Fase 2 roda 1.4.2. Carregá-los levanta "
      "InconsistentVersionWarning e, na Regressão Logística, "
      "AttributeError em predict_proba.")

    p(doc,
      "A solução foi retreinar o baseline a partir dos hiperparâmetros "
      "exatos da Fase 1. Isso é melhor do que um contorno: original e "
      "otimizado passam a ser treinados pela ",
      ("mesma biblioteca", {"bold": True}), ", no ",
      ("mesmo caminho de código", {"bold": True}), ", sobre os ",
      ("mesmos dados", {"bold": True}),
      ". O baseline refeito reproduz as métricas publicadas na Fase 1 "
      "com diferença máxima de ", ("0,0000", {"bold": True}), ".")

    h2(doc, "6.2 Fitness de CV — o que o GA otimizou")

    tabela(doc, ["Modelo", "Fase 1", "Otimizado", "Ganho"],
           [["Regressão Logística", "0,9604", ("0,9732", {"bold": True}),
             ("+0,0128", {"bold": True})],
            ["Árvore de Decisão", "0,8906", ("0,9331", {"bold": True}),
             ("+0,0425", {"bold": True})],
            ["Random Forest", "0,9361", ("0,9533", {"bold": True}),
             ("+0,0172", {"bold": True})]],
           larguras=[5.0, 3.3, 3.3, 3.4])

    p(doc, "O GA melhorou a função objetivo nos três modelos.")

    h2(doc, "6.3 Métricas no conjunto de teste")

    p(doc,
      "114 amostras, 42 malignas. O GA nunca viu estes dados.")

    # Piora de métrica é marcada em NEGRITO, nunca em vermelho: neste
    # documento o vermelho significa "pendência" e só isso. Misturar os
    # dois sentidos enfraqueceria o sinal que o leitor precisa varrer.
    tabela(doc,
           ["Modelo", "Recall mal.", "F1 mal.", "Acurácia", "ROC AUC",
            "Falsos neg.", "Falsos pos."],
           [["Reg. Logística", "0,976 → 0,976", "0,954 → 0,954",
             "0,965 → 0,965", "0,995 → 0,997", "1 → 1", "3 → 3"],
            ["Árvore de Decisão", "0,929 → 0,929",
             ("0,918 → 0,839", {"bold": True}),
             ("0,939 → 0,868", {"bold": True}),
             ("0,929 → 0,889", {"bold": True}),
             "3 → 3", ("4 → 12", {"bold": True})],
            ["Random Forest", "0,929 → 0,929",
             ("0,963 → 0,918", {"bold": True}),
             ("0,974 → 0,939", {"bold": True}), "0,996 → 0,993",
             "3 → 3", ("0 → 4", {"bold": True})]],
           larguras=[3.0, 2.3, 2.2, 2.2, 2.2, 1.6, 1.5])

    p(doc,
      ("O recall do maligno — a métrica que domina o fitness com peso "
       "0,60 — não mudou em nenhum dos três modelos. ", {"bold": True}),
      "O custo transferiu integralmente: a Random Forest ganhou 4 "
      "falsos positivos onde tinha zero, e a Árvore triplicou os seus.")

    p(doc,
      "O único ganho real é marginal: o ROC AUC da Logística subiu "
      "0,0023, o que significa que o ordenamento das probabilidades "
      "melhorou um pouco sem mudar nenhuma decisão no limiar de 0,5.")

    p(doc,
      ("Os valores em negrito marcam piora em relação à Fase 1. O "
       "vermelho, neste documento, é reservado para pendências.",
       {"italic": True, "gray": True, "size": 9}))

    figura(doc, "33_comparativo_metricas.png",
           "Métricas no conjunto de teste, antes e depois do GA. O "
           "recall (painel 1) é idêntico nos três modelos; F1, acurácia "
           "e AUC caem na Árvore e na Floresta.")

    figura(doc, "34_matrizes_confusao.png",
           "Matrizes de confusão. A moldura vermelha marca o falso "
           "negativo. Linha de cima: original. Linha de baixo: "
           "otimizado pelo GA.", largura=Inches(5.6))

    quebra(doc)


def secao_7_investigacao(doc: Document) -> None:
    h1(doc, "7. Por Que o Ganho Não Transferiu")

    p(doc,
      "Atribuir isso a azar seria a saída fácil. As análises abaixo "
      "medem a causa.")

    h2(doc, "7.1 O fitness de CV prevê o desempenho no teste?")

    p(doc,
      "Avaliamos no teste ",
      ("todos os 9 indivíduos vencedores", {"bold": True}),
      " (3 modelos × 3 experimentos) mais o baseline, e colocamos "
      "fitness de CV e métricas de teste lado a lado. Isto é ",
      ("diagnóstico, não critério de seleção", {"italic": True}),
      " — o vencedor de cada modelo continua escolhido pelo fitness de "
      "CV. Escolher pelo teste seria exatamente o vazamento que esta "
      "análise serve para tornar visível.")

    p(doc, "Correlação de ", ("Spearman", {"bold": True}),
      " (de ordem — a pergunta é se o ranqueamento por fitness "
      "reproduz o ranqueamento por desempenho):")

    tabela(doc,
           ["Modelo", "vs. F1 de teste", "vs. acurácia", "vs. recall"],
           [["Regressão Logística", "— (constante)", "— (constante)",
             "— (constante)"],
            ["Árvore de Decisão", ("−1,000", {"bold": True}),
             ("−1,000", {"bold": True}), "0,000"],
            ["Random Forest", ("−0,632", {"bold": True}),
             ("−0,833", {"bold": True}), "0,500"]],
           larguras=[5.0, 3.5, 3.3, 3.2])

    p(doc,
      ("A correlação é negativa. ", {"bold": True}),
      "Quanto melhor o fitness de validação cruzada, pior o desempenho "
      "no teste. Na Árvore de Decisão é −1,00: a inversão é perfeita, "
      "os quatro pontos ordenam-se exatamente ao contrário.")

    figura(doc, "35_cv_vs_teste.png",
           "Fitness de CV (eixo x) contra F1 do maligno no teste (eixo "
           "y). Se o fitness previsse o teste, os pontos subiriam da "
           "esquerda para a direita. O baseline da Fase 1 (losango "
           "cinza) fica no canto superior esquerdo e todos os pontos do "
           "GA no inferior direito.")

    h2(doc, "7.2 As duas causas")

    p(doc,
      ("(a) Sobreajuste do critério de seleção. ", {"bold": True}),
      "600 avaliações num espaço de até 8 genes, sobre 455 amostras "
      "divididas em 5 folds fixos. O GA é um otimizador eficiente — e o "
      "que ele otimizou foi o desempenho naquela partição específica, "
      "incluindo o ruído dela.")

    p(doc,
      ("(b) Resolução do conjunto de teste. ", {"bold": True}),
      "São 42 casos malignos. O recall só se move em passos de 1/42 = "
      "2,4 pontos percentuais. Um ganho real de 1 p.p. é literalmente "
      "invisível neste teste. Isso não desculpa (a) — a precisão piorou "
      "de forma clara e mensurável — mas explica por que o recall ficou "
      "parado nos três modelos.")

    h2(doc, "7.3 O caso que ilustra o problema: um empate exato")

    p(doc,
      "A configuração ",
      ("random_forest / exploratorio", {"code": True}),
      " obteve recall de teste 0,9762 — ",
      ("1 falso negativo em vez de 3", {"bold": True}),
      " — o melhor recall de qualquer configuração da Floresta, "
      "exatamente o que o projeto mais valoriza. Ela não foi "
      "selecionada, e o motivo é revelador: o fitness de CV das duas é "
      "idêntico até a última casa de ponto flutuante.")

    tabela(doc,
           ["", "equilibrado (escolhido)", "exploratorio (descartado)"],
           [["Fitness de CV", ("0.9533482273924908", {"code": True}),
             ("0.9533482273924908", {"code": True})],
            ["n_estimators", "150", "110"],
            ["max_depth", "19", "20"],
            ["min_samples_split", "3", "24"],
            ["min_samples_leaf", "6", "4"],
            ["class_weight", "balanced", "balanced_subsample"],
            [("Falsos negativos no teste", {"bold": True}),
             ("3", {"bold": True}),
             ("1", {"bold": True})]],
           larguras=[4.6, 5.6, 5.8])

    p(doc,
      "São dois modelos substancialmente diferentes, e o estimador não "
      "consegue distingui-los nem na 16ª casa decimal. O desempate "
      "acabou sendo a ordem de execução.")

    p(doc,
      ("Por que o empate é exato. ", {"bold": True}),
      "O fitness é a média, sobre 5 folds, de métricas que são razões "
      "de contagens inteiras pequenas (cada fold de validação tem 91 "
      "amostras, ~34 malignas). O conjunto de valores que o fitness "
      "pode assumir é discreto e grosso. Além de ser um estimador "
      "enviesado pelo ruído dos folds, ele tem ",
      ("resolução insuficiente", {"bold": True}),
      " para ordenar candidatos próximos.")

    h2(doc, "7.4 Experimento corretivo: validação cruzada repetida")

    p(doc,
      "Se o problema é o GA explorar o ruído de uma partição, usar "
      "várias partições e tirar a média remove boa parte do ruído "
      "disponível. Trocamos StratifiedKFold(5) por "
      "RepeatedStratifiedKFold(5 × 3) — 15 treinos por avaliação em vez "
      "de 5 — e medimos ",
      ("os dois conjuntos de vencedores nas duas réguas", {"bold": True}),
      ":")

    tabela(doc,
           ["Modelo", "Configuração", "Régua CV 5-fold", "Régua CV 5×3"],
           [["Reg. Logística", "baseline Fase 1", "0,9604", "0,9563"],
            ["", "GA / CV 5-fold", ("0,9732 ◄", {"bold": True}), "0,9662"],
            ["", "GA / CV 5×3", "0,9666", ("0,9679 ◄", {"bold": True})],
            ["Árvore de Decisão", "baseline Fase 1", "0,8906", "0,9056"],
            ["", "GA / CV 5-fold", ("0,9331 ◄", {"bold": True}), "0,9117"],
            ["", "GA / CV 5×3", "0,9288", ("0,9207 ◄", {"bold": True})],
            ["Random Forest", "baseline Fase 1", "0,9361", "0,9428"],
            ["", "GA / CV 5-fold", ("0,9533 ◄", {"bold": True}), "0,9446"],
            ["", "GA / CV 5×3", "0,9464", ("0,9535 ◄", {"bold": True})]],
           larguras=[3.8, 4.0, 3.6, 3.6])

    p(doc,
      ("Cada vencedor é superior apenas na régua contra a qual foi "
       "otimizado — seis de seis casos, sem exceção. ", {"bold": True}),
      "Essa é a definição operacional de sobreajuste do critério de "
      "seleção: a vantagem encontrada é uma propriedade do estimador, "
      "não do modelo.")

    p(doc, ("Quanto do \"ganho\" era ruído:", {"bold": True}))

    tabela(doc,
           ["Modelo", "Ganho na régua 5-fold", "Ganho na régua 5×3",
            "Evaporou"],
           [["Regressão Logística", "+1,28 p.p.", "+0,99 p.p.", "23%"],
            ["Árvore de Decisão", ("+4,25 p.p.", {"bold": True}),
             ("+0,61 p.p.", {"bold": True}),
             ("86%", {"bold": True})],
            ["Random Forest", ("+1,72 p.p.", {"bold": True}),
             ("+0,18 p.p.", {"bold": True}),
             ("90%", {"bold": True})]],
           larguras=[4.6, 3.6, 3.4, 3.4])

    p(doc,
      "O ganho de 4,25 pontos da Árvore — o maior de toda a fase — vira "
      "0,61 ponto quando medido num estimador menos ruidoso. ",
      ("Mais de quatro quintos dele eram ruído de uma partição "
       "específica.", {"bold": True}))

    figura(doc, "36_experimento_robusto.png",
           "A evidência cruzada. Painel 1 (régua 5-fold): o vencedor "
           "laranja ganha. Painel 2 (régua 5×3): o vencedor azul ganha. "
           "Painel 3 (teste): nenhum dos dois ganha do baseline cinza.")

    p(doc,
      ("Conclusão do experimento. ", {"bold": True}),
      "A CV repetida é a correção certa para a causa (a) e isso é "
      "demonstrável: ela elimina 86-90% do ganho ilusório. Mas ",
      ("não bastou para superar o baseline no teste", {"bold": True}),
      " — na Árvore reduziu os falsos positivos de 12 para 10, na "
      "Logística piorou marginalmente, e na Random Forest trocou um "
      "falso positivo a menos por um falso negativo a mais, que é a "
      "direção errada clinicamente. A causa (b) permanece.")

    quebra(doc)


def secao_8_llm(doc: Document) -> None:
    h1(doc, "8. Integração com LLM")

    h2(doc, "8.1 Abordagem")

    p(doc, ("LLM escolhida: ", {"bold": True}),
      "Claude (Anthropic), via o SDK oficial. Critérios: API gerenciada "
      "(sem GPU nem download de pesos), retry com backoff exponencial "
      "já no SDK, e saída estruturada nativa usada pelo juiz da "
      "avaliação de qualidade.")

    p(doc,
      ("O princípio que organiza o módulo: a LLM nunca vê dado bruto "
       "nem chama modelo. ", {"bold": True}),
      "Ela recebe um payload JSON fechado, montado em Python, com todos "
      "os números já calculados. Três razões: ",
      ("auditabilidade", {"bold": True}),
      " (toda afirmação numérica é rastreável a um campo do payload); ",
      ("a LLM não precisa fazer aritmética", {"bold": True}),
      " (quem calcula é o scikit-learn); e o payload ",
      ("fica salvo em disco", {"bold": True}),
      ", então uma explicação suspeita pode ser reconferida meses "
      "depois contra a entrada exata que a gerou.")

    p(doc, ("Dois tipos de explicação:", {"bold": True}))
    bullet(doc, ("Por caso", {"bold": True}),
           " — paciente do conjunto de teste, previsão dos 3 modelos "
           "otimizados, fatores que pesaram, grau de confiança e "
           "conduta sugerida, em quatro seções nomeadas.")
    bullet(doc, ("Da otimização", {"bold": True}),
           " — a tabela da seção 6 traduzida para quem decide sobre "
           "adoção, com o efeito expresso em número de pacientes "
           "(\"deixa de detectar 3 dos 42 casos malignos\") em vez de "
           "pontos percentuais.")

    p(doc,
      ("Seleção dos casos. ", {"bold": True}),
      "Não são aleatórios. O sistema prioriza falso negativo do modelo "
      "de referência, discordância entre os 3 modelos, e menor "
      "confiança — exatamente os casos em que explicação automática é "
      "útil. Um caso com 99,9% de certeza e consenso não precisa de "
      "texto.")

    p(doc,
      ("Atribuição dos fatores. ", {"bold": True}),
      "Para a Regressão Logística é usado o SHAP exato de modelo linear "
      "— φᵢ = coefᵢ × (xᵢ − média_treinoᵢ) —, a mesma fórmula da Fase "
      "1. Um teste de integração verifica que a soma das contribuições "
      "mais o intercepto reconstrói o log-odds previsto.")

    h2(doc, "8.2 Engenharia de prompt — três versões documentadas")

    tabela(doc,
           ["Versão", "O que mudou", "Por que foi substituída"],
           [["v1", "Prompt genérico.",
             "Escrevia \"o exame confirma malignidade\" (laudo "
             "fechado); citava métricas que não estavam no payload; "
             "usava \"recall\" e \"F1\" sem traduzir."],
            ["v2",
             "Três regras: não diagnosticar, não inventar número, ser "
             "conciso.",
             "Corrigiu a linguagem e a invenção de métricas, mas nos "
             "casos de discordância entre modelos escolhia um "
             "\"vencedor\" e omitia o conflito — apagando a informação "
             "mais útil. Tratava morfometria como sintoma."],
            [("v3 (em uso)", {"bold": True}),
             "Contexto do dataset, natureza das features, assimetria de "
             "custo, obrigação de destacar discordância, formato fixo "
             "de quatro seções.",
             "O formato fixo foi o que mais mudou a utilidade: com "
             "seções nomeadas, o médico lê a conclusão e a conduta sem "
             "percorrer o texto inteiro."]],
           larguras=[2.0, 6.0, 7.0])

    h2(doc, "8.3 Avaliação da qualidade — duas camadas")

    p(doc,
      ("\"A explicação ficou boa?\" não se responde lendo três "
       "exemplos.", {"italic": True}),
      " A avaliação tem duas camadas que medem coisas diferentes.")

    p(doc,
      ("Camada 1 — verificações determinísticas (código, sem LLM). ",
       {"bold": True}),
      "A mais importante é a auditoria numérica: extrai todo número do "
      "texto e confirma que ele existe no payload. Alucinação numérica "
      "em contexto clínico é o modo de falha mais perigoso e o mais "
      "fácil de passar batido. A tolerância vem da precisão com que o "
      "número foi escrito — \"97,6\" autoriza ±0,05; \"98\", ±0,5. "
      "Tolerância relativa seria errada de um jeito perigoso: com 0,6% "
      "de folga, \"99,9%\" passaria como arredondamento de \"100%\". "
      "Verifica também linguagem conclusiva proibida, jargão de ML não "
      "traduzido, as quatro seções e a ressalva de que a decisão é "
      "clínica.")

    p(doc,
      ("Camada 2 — LLM como juiz. ", {"bold": True}),
      "Um segundo prompt, com instrução explícita de ceticismo, dá nota "
      "1-5 em cinco critérios: fidelidade numérica, ausência de "
      "diagnóstico, clareza clínica, acionabilidade e calibração da "
      "incerteza.")

    p(doc,
      ("Quando as duas discordam, a Camada 1 prevalece. ", {"bold": True}),
      "Ela compara contra o payload; o juiz compara contra a própria "
      "leitura. A divergência fica registrada em coluna própria, porque "
      "ela mede a confiabilidade do juiz, não da explicação.")

    h2(doc, "8.4 A auditoria pegou defeitos reais")

    p(doc,
      "A Camada 1 reprovou o nosso próprio gerador por template e "
      "encontrou um falso positivo nela mesma:")

    numerado(doc,
             "A interpretação da otimização interpolava um campo do "
             "payload que continha a palavra \"recall\" — jargão que o "
             "prompt manda traduzir.")
    numerado(doc,
             "O mesmo texto não registrava que a decisão é do "
             "profissional.")
    numerado(doc,
             "No ramo de discordância entre modelos — o mais sensível "
             "dos três — a conduta sugerida omitia a ressalva clínica.")
    numerado(doc,
             "Na própria auditoria: em \"−1,17\" a regex casava \"17\" "
             "isoladamente, porque a vírgula decimal não estava na "
             "exclusão à esquerda, e acusava invenção de um número que "
             "estava no payload.")

    p(doc,
      "Os quatro foram corrigidos e viraram ",
      ("teste de regressão", {"bold": True}),
      ". É o argumento mais forte a favor da Camada 1: um revisor "
      "humano lendo os textos não teria pego nenhum dos quatro.")

    h2(doc, "8.5 Estado atual desta etapa")

    p(doc,
      ("Pendência: as Etapas 6 e 7 foram executadas em modo offline. ",
       {"red": True, "bold": True}),
      ("Sem a variável ANTHROPIC_API_KEY configurada, a camada de LLM "
       "cai num gerador determinístico por template — que funciona e "
       "passa nas verificações, mas é notoriamente mais pobre que a "
       "LLM. Os relatórios gerados registram \"Provedor: modo offline "
       "(forçado)\". Falta: (a) rodar as explicações com a LLM real; "
       "(b) rodar o juiz da Camada 2, que exige a API; (c) rodar a "
       "comparação medida entre os prompts v1/v2/v3 "
       "(--comparar-prompts), que hoje está descrita mas não "
       "quantificada.", {"red": True}))

    p(doc,
      "O caminho de código é o mesmo nos dois modos — só muda o "
      "provedor —, então a execução com a chave não exige nenhuma "
      "alteração no sistema.")

    quebra(doc)


def secao_9_infra(doc: Document) -> None:
    h1(doc, "9. Monitoramento, Testes e Arquitetura em Nuvem")

    h2(doc, "9.1 Monitoramento e escalabilidade")

    p(doc, "Duas saídas de log, com propósitos diferentes:")
    bullet(doc, ("Legível", {"bold": True}),
           " — acompanhar execução e depurar.")
    bullet(doc, ("Estruturado (JSON Lines)", {"bold": True}),
           " — uma linha JSON por evento, feito para ser consumido por "
           "código. É o que permite reconstruir a curva de convergência "
           "de um experimento rodado semanas antes sem reexecutar nada.")

    p(doc,
      ("Monitoramento das previsões, não só do treino. ", {"bold": True}),
      "O PredictionMonitor mantém janela deslizante e expõe dois "
      "indicadores de drift: a taxa de maligno prevista (desvio em "
      "relação à prevalência do treino, ~37,4%) e a confiança média. "
      "Previsão com confiança abaixo de 0,65 é marcada no log; em "
      "produção é esse flag que roteia o caso para revisão humana.")

    p(doc,
      ("Escalabilidade. ", {"bold": True}),
      "O maior ganho foi o cache de fitness (até 84% das avaliações "
      "economizadas). Random Forest com n_jobs=-1, random_state fixo "
      "(sem isso o cache mentiria) e parada antecipada opcional.")

    h2(doc, "9.2 Testes automatizados")

    p(doc, ("201 testes", {"bold": True}),
      ", todos sem acesso à rede, cobrindo codificação, operadores, "
      "fitness, motor, camada de LLM e integração ponta a ponta. Os "
      "testes de operador verificam ", ("propriedade", {"bold": True}),
      ", não saída fixa — \"o torneio devolve o melhor quando o torneio "
      "é a população inteira\", \"a mutação nunca sai de [0, 1]\". "
      "Propriedade sobrevive a refatoração; valor fixo não.")

    p(doc, "Três merecem destaque:")
    bullet(doc, "Um sorteia 25 cromossomos por espaço e ",
           ("treina cada fenótipo de verdade", {"bold": True}),
           " — é o que garante que o decodificador e o reparo cobrem "
           "todas as combinações inválidas.")
    bullet(doc, "Outro verifica que a soma das contribuições SHAP mais "
           "o intercepto reconstrói o log-odds previsto — prova de que "
           "a atribuição é exata, não heurística.")
    bullet(doc, "Um terceiro afirma que o melhor fitness nunca cai "
           "entre gerações (monotonicidade do elite).")

    p(doc,
      "A fixture de dados é construída a partir do dataset embutido no "
      "scikit-learn, não dos CSVs do pipeline. Um teste que depende de "
      "artefato gerado por outra etapa não testa o código — testa se "
      "alguém lembrou de rodar o pipeline na ordem certa.")

    h2(doc, "9.3 Arquitetura em nuvem (etapa opcional)")

    p(doc,
      ("A decisão central: ", {"bold": True}),
      "otimização e inferência são serviços separados, porque têm "
      "perfis opostos. O GA é lote pesado e raro, tolera interrupção e "
      "roda em Fargate Spot; a inferência é leve e frequente e roda em "
      "Lambda. O S3 é a fronteira: o Batch escreve os modelos, a Lambda "
      "os lê.")

    p(doc,
      ("Permissões assimétricas. ", {"bold": True}),
      "A Lambda tem somente leitura nos modelos. Se for comprometida, o "
      "atacante não consegue trocar o modelo de diagnóstico — a pior "
      "consequência possível nesta arquitetura. O job do GA, por sua "
      "vez, não acessa a chave da LLM.")

    p(doc,
      ("O log estruturado vira métrica sem agente. ", {"bold": True}),
      "Os metric filters do CloudWatch leem o mesmo JSONL que o "
      "monitoramento escreve. É a razão de o log ser JSON e não texto. "
      "Os alarmes são de duas famílias: técnicos (erro, throttle) e de "
      "saúde do modelo (taxa de maligno, fração de previsões "
      "incertas). Um modelo pode estar tecnicamente perfeito e errando "
      "sistematicamente porque a distribuição mudou; só a segunda "
      "família pega isso.")

    p(doc, ("Custo estimado ", {"bold": True}),
      "para 1.000 exames/mês: ~US$ 5/mês de infraestrutura mais US$ "
      "25-30 da API da LLM. A LLM é ~85% do custo — daí o teto de "
      "concorrência reservada e a métrica de custo por chamada.")

    p(doc,
      ("Pendência / não-escopo assumido: a nuvem NÃO foi provisionada. ",
       {"red": True, "bold": True}),
      ("A entrega desta etapa é o código de infraestrutura — 1.339 "
       "linhas de Terraform, Dockerfile e handler Lambda —, não um "
       "ambiente rodando. Aplicar exigiria credenciais AWS e geraria "
       "custo recorrente sem acrescentar nada ao que a fase demonstra. "
       "Duas ressalvas honestas: (a) terraform validate não foi "
       "executado, porque o binário não está no ambiente de "
       "desenvolvimento, então erro de sintaxe no HCL é possível; (b) "
       "nenhum recurso foi testado contra a API da AWS, então "
       "comportamentos que só aparecem na aplicação não foram "
       "exercitados.", {"red": True}))

    quebra(doc)


def secao_10_discussao(doc: Document) -> None:
    h1(doc, "10. Discussão Crítica e Conclusão")

    h2(doc, "10.1 O que funcionou")

    p(doc,
      "O algoritmo genético está implementado do zero, coberto por 201 "
      "testes, e melhora a função objetivo que recebe nos três modelos. "
      "A infraestrutura de experimentação — orçamento controlado, log "
      "estruturado por geração, cache de fitness, reprodutibilidade por "
      "semente — permitiu rodar 9 experimentos em 27,6 minutos e "
      "comparar estratégias de forma justa. A camada de LLM produz "
      "explicações auditáveis, com verificação número por número contra "
      "a entrada.")

    h2(doc, "10.2 O que não funcionou, e o que isso ensinou")

    p(doc,
      "O ganho de fitness não virou ganho de desempenho. A lição não é "
      "que algoritmos genéticos não servem para ajuste de "
      "hiperparâmetros — é que ",
      ("otimizar um estimador de desempenho não é a mesma coisa que "
       "melhorar o desempenho", {"bold": True}),
      ", e que a diferença entre as duas coisas é mensurável. Um GA "
      "eficiente sobre um estimador ruidoso otimiza o ruído com a mesma "
      "competência com que otimizaria o sinal.")

    p(doc,
      "A demonstração mais limpa disso está na seção 7.4: trocando a "
      "régua de avaliação, cada configuração vence apenas no estimador "
      "que a selecionou — 6 comparações, 6 confirmações. E a segunda "
      "metade da lição veio junto: corrigir o estimador eliminou 86-90% "
      "do ganho ilusório, mas não produziu um modelo melhor no teste. ",
      ("Medir melhor não substitui ter o que medir.", {"bold": True}))

    p(doc,
      "Dois agravantes específicos deste caso: os modelos da Fase 1 já "
      "estavam muito bons (ROC AUC de 0,995), deixando pouco espaço "
      "para melhora real; e o conjunto de teste, com 42 casos malignos, "
      "tem resolução de 2,4 pontos percentuais no recall — pequeno "
      "demais para registrar ganho sutil.")

    h2(doc, "10.3 Esse sistema pode ser usado na prática?")

    p(doc,
      ("Sim, mas apenas como ferramenta de apoio à decisão — nunca "
       "como substituto do diagnóstico médico. ", {"bold": True}),
      "A conclusão da Fase 1 permanece válida, e a Fase 2 acrescenta "
      "dois pontos:")

    numerado(doc,
             ("O modelo recomendado continua sendo o da Fase 1. ",
              {"bold": True}),
             "A Regressão Logística original, com 1 falso negativo em "
             "42 malignos, não foi superada por nenhuma configuração "
             "encontrada pelo GA. A honestidade aqui é parte da "
             "entrega: trocar o modelo em produção por um que piora a "
             "precisão sem melhorar o recall seria o contrário do que o "
             "trabalho recomenda.")
    numerado(doc,
             ("A camada de explicação é um ganho real e independente "
              "do resultado do GA. ", {"bold": True}),
             "Ela funciona sobre qualquer versão dos modelos, e a "
             "auditoria numérica resolve o risco que mais impede o uso "
             "de LLM em contexto clínico: a alucinação de métricas.")

    h2(doc, "10.4 O que faria diferente com mais tempo")

    bullet(doc, ("Validação cruzada aninhada", {"bold": True}),
           ", para que a seleção entre experimentos também fosse "
           "validada fora da amostra.")
    bullet(doc, ("Otimização do limiar de decisão como gene", {"bold": True}),
           " — é a alavanca mais direta sobre a troca entre recall e "
           "precisão, e nenhum dos hiperparâmetros ajustados a controla "
           "diretamente.")
    bullet(doc, ("Um conjunto de teste maior", {"bold": True}),
           ", o que neste dataset significaria validação cruzada "
           "repetida reportada com intervalo de confiança em vez de um "
           "split único.")

    quebra(doc)


def secao_11_pendencias(doc: Document) -> None:
    h1(doc, "11. Pendências para a Entrega Final")

    p(doc,
      ("Esta seção consolida tudo que ainda não está entregue. Itens "
       "1 a 5 são tarefas pendentes; 6 e 7 são escopo conscientemente "
       "deixado de fora, listados aqui para que não passem "
       "despercebidos.", {"italic": True, "gray": True}))

    h2(doc, "Tarefas pendentes")

    tabela(doc,
           ["#", "Pendência", "Estado atual / o que falta"],
           [[("1", {"red": True, "bold": True}),
             ("Data da entrega", {"red": True, "bold": True}),
             ("O cabeçalho do documento está com um marcador. Definir e "
              "preencher.", {"red": True})],
            [("2", {"red": True, "bold": True}),
             ("Vídeo de demonstração (até 15 min)", {"red": True, "bold": True}),
             ("Roteiro completo pronto em reports/fase2/roteiro_video.md "
              "(10 blocos com marcação de tempo, checklist de gravação e "
              "perguntas prováveis da banca). Falta gravar, publicar no "
              "YouTube/Vimeo e colar o link na seção 1.", {"red": True})],
            [("3", {"red": True, "bold": True}),
             ("Etapas 6 e 7 com a LLM real", {"red": True, "bold": True}),
             ("Rodaram em modo offline por falta de ANTHROPIC_API_KEY. "
              "Falta: configurar a chave em .env, rodar "
              "'python -m src.llm.explain', rodar "
              "'python -m src.llm.evaluate_quality --comparar-prompts' "
              "e substituir os relatórios gerados.", {"red": True})],
            [("4", {"red": True, "bold": True}),
             ("Commit e push", {"red": True, "bold": True}),
             ("17 arquivos novos/modificados na branch "
              "tehc-challenge-fase-2, nada commitado. O repositório "
              "público ainda mostra apenas a Fase 1.", {"red": True})],
            [("5", {"red": True, "bold": True}),
             ("Apresentação (slides)", {"red": True, "bold": True}),
             ("A Fase 1 teve um deck com link no documento de entrega. "
              "Não existe equivalente para a Fase 2.", {"red": True})]],
           larguras=[1.0, 4.6, 9.4])

    h2(doc, "Escopo conscientemente não executado")

    tabela(doc,
           ["#", "Item", "Justificativa e consequência"],
           [[("6", {"red": True, "bold": True}),
             ("Nuvem não provisionada", {"red": True, "bold": True}),
             ("O Terraform (1.339 linhas), o Dockerfile e o handler "
              "Lambda estão escritos e documentados, mas não foram "
              "aplicados. A etapa é opcional e vale pontuação extra; "
              "aplicar exigiria credenciais AWS e geraria custo "
              "recorrente. A entrega é o código de infraestrutura.",
              {"red": True})],
            [("7", {"red": True, "bold": True}),
             ("terraform validate não executado", {"red": True, "bold": True}),
             ("O binário do Terraform não está no ambiente de "
              "desenvolvimento. Consequência: erro de sintaxe ou de "
              "nome de argumento no HCL é possível, e nenhum recurso "
              "foi testado contra a API da AWS. Quem for aplicar deve "
              "rodar validate e revisar o plan antes.", {"red": True})]],
           larguras=[1.0, 4.6, 9.4])

    h2(doc, "Como completar as pendências 3 e 4")

    p(doc,
      ("# 3 — explicações com a LLM real\n"
       "cp .env.example .env            # preencher ANTHROPIC_API_KEY\n"
       "python -m src.llm.explain\n"
       "python -m src.llm.evaluate_quality --comparar-prompts\n\n"
       "# 4 — publicar o código\n"
       "git add -A && git commit -m \"feat: fase 2 — GA + LLM\"\n"
       "git push -u origin tehc-challenge-fase-2",
       {"code": True}))

    p(doc,
      "Nenhuma das duas exige alteração no sistema — o caminho de "
      "código já está pronto e testado nos dois modos.")


# -----------------------------------------------------------------------
# Montagem
# -----------------------------------------------------------------------

SECOES = (
    secao_1_entregaveis,
    secao_2_problema,
    secao_3_ga,
    secao_4_experimentos,
    secao_5_resultados_ga,
    secao_6_comparativo,
    secao_7_investigacao,
    secao_8_llm,
    secao_9_infra,
    secao_10_discussao,
    secao_11_pendencias,
)


def main() -> None:
    doc = Document()
    configurar_pagina(doc)
    capa(doc)

    for secao in SECOES:
        secao(doc)

    SAIDA.parent.mkdir(parents=True, exist_ok=True)
    doc.save(SAIDA)

    tamanho_kb = SAIDA.stat().st_size / 1024
    print(f"documento gerado: {SAIDA.relative_to(BASE_DIR)}")
    print(f"  {len(SECOES)} seções | {len(doc.inline_shapes)} figuras "
          f"| {len(doc.tables)} tabelas | {tamanho_kb:.0f} KB")
    print()
    print("ANTES DE EXPORTAR O PDF, no Word:")
    print("  1. Ctrl+A e depois F9 para gerar o sumário")
    print("     (o sumário é um campo; abre vazio até ser atualizado)")
    print("  2. Arquivo > Exportar > Criar PDF")


if __name__ == "__main__":
    main()

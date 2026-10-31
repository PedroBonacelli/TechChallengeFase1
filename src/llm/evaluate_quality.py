# Etapa 7 — Avaliar a qualidade das interpretações da LLM
#
# "A explicação está boa?" não é uma pergunta que se responda lendo três
# exemplos e achando que ficaram legais. Aqui a avaliação tem duas
# camadas, que medem coisas diferentes:
#
# CAMADA 1 — VERIFICAÇÕES DETERMINÍSTICAS (sem LLM).
# Código que checa propriedades objetivas do texto contra o payload que o
# gerou. A mais importante é a AUDITORIA NUMÉRICA: extrai todo número do
# texto e confirma que ele existe no payload. É isso que transforma
# "a LLM parece não ter inventado nada" em uma asserção verificável —
# alucinação numérica em contexto clínico é o modo de falha mais perigoso
# e o mais fácil de passar desapercebido numa leitura casual.
# Também checa: linguagem conclusiva proibida, jargão de ML não
# traduzido, presença das quatro seções e limite de palavras.
#
# CAMADA 2 — LLM COMO JUIZ.
# Um segundo prompt, com system próprio e instruído a ser cético, dá nota
# 1-5 em cinco critérios (fidelidade numérica, ausência de diagnóstico,
# clareza clínica, acionabilidade, calibração da incerteza). Cobre o que
# regex não alcança: se o texto é de fato compreensível, se a conduta é
# concreta, se o grau de confiança declarado combina com a probabilidade.
#
# As duas camadas se cruzam de propósito no critério numérico: se o juiz
# der nota alta em fidelidade numérica e a Camada 1 tiver achado um
# número inventado, a Camada 1 ganha — e a discrepância fica registrada
# na tabela, porque ela mede a confiabilidade do próprio juiz.
#
# COMPARAÇÃO ENTRE VERSÕES DE PROMPT (--comparar-prompts) gera a mesma
# explicação com os prompts v1, v2 e v3 e avalia as três. É o que produz
# os "exemplos de boas e más respostas" pedidos pelo enunciado, com
# evidência em vez de impressão.
#
#   python -m src.llm.evaluate_quality
#   python -m src.llm.evaluate_quality --comparar-prompts
#   python -m src.llm.evaluate_quality --offline

from __future__ import annotations

import argparse
import json
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from ..monitoring import EventLogger, setup_logging
from .client import LLMClient
from .prompts import (
    PROMPT_HISTORY,
    SYSTEM_JUDGE,
    USER_TEMPLATE_CASE,
    USER_TEMPLATE_JUDGE,
)

BASE_DIR = Path(__file__).resolve().parent.parent.parent
REPORTS_DIR = BASE_DIR / "reports" / "fase2"

SECOES_CASO = (
    "AVALIAÇÃO DOS MODELOS",
    "PRINCIPAIS FATORES",
    "GRAU DE CONFIANÇA",
    "CONDUTA SUGERIDA",
)

# Linguagem que fecha diagnóstico — proibida pelo system prompt v3.
# Em regex com fronteira de palavra pra não pegar "confirmação adicional"
# dentro de uma frase que na verdade pede investigação.
PADROES_CONCLUSIVOS = (
    (r"\bconfirma(?:-se|\s+que|do|da)\b", "linguagem de confirmação diagnóstica"),
    (r"\bo\s+paciente\s+(?:tem|possui|apresenta\s+c[âa]ncer)", "atribui condição ao paciente"),
    (r"\ba\s+paciente\s+(?:tem|possui|apresenta\s+c[âa]ncer)", "atribui condição ao paciente"),
    (r"\btrata-se\s+de\s+(?:um\s+)?(?:tumor\s+)?malign", "afirma natureza do tumor"),
    (r"\bdiagn[óo]stico\s+(?:é|de)\s+malign", "enuncia diagnóstico"),
    (r"\bcertamente\b|\bsem\s+d[úu]vida\b|\bcom\s+certeza\b", "certeza absoluta"),
    (r"\bdefinitivamente\b", "certeza absoluta"),
)

# Jargão de ML que o prompt manda traduzir.
JARGAO_ML = (
    "recall", "f1-score", "f1 score", "roc auc", "auc", "cromossomo",
    "hiperparâmetro", "hiperparametro", "fitness", "overfitting",
    "cross-validation", "validação cruzada", "feature engineering",
    "log-odds", "algoritmo genético", "mutação", "crossover",
)

# Inteiros pequenos são aceitos sem constar do payload porque aparecem
# como contagem estrutural do próprio texto ("os três modelos", "quatro
# fatores", "2 desvios-padrão"). 100 entra porque é o divisor de
# porcentagem. Sem essa isenção, a auditoria viraria ruído.
INTEIROS_ESTRUTURAIS = set(range(0, 11)) | {100}


# -----------------------------------------------------------------------
# Camada 1 — verificações determinísticas
# -----------------------------------------------------------------------


def _valores_numericos(obj: Any) -> list[float]:
    """Todo número que aparece no payload, em qualquer nível."""
    valores: list[float] = []
    if isinstance(obj, bool):
        return valores
    if isinstance(obj, (int, float)):
        valores.append(float(obj))
    elif isinstance(obj, dict):
        for v in obj.values():
            valores.extend(_valores_numericos(v))
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            valores.extend(_valores_numericos(v))
    elif isinstance(obj, str):
        # números embutidos em strings do payload (ex.: a fórmula do
        # fitness, "0.60 * recall...") também contam como autorizados
        for m in re.finditer(r"-?\d+(?:[.,]\d+)?", obj):
            try:
                valores.append(float(m.group().replace(",", ".")))
            except ValueError:
                pass
    return valores


def _valores_autorizados(payload: dict[str, Any]) -> set[float]:
    """Valores do payload nas formas em que podem aparecer no texto.

    Uma métrica 0.9762 pode ser escrita como "0,9762" ou como "97,6%".
    Autorizo as duas, e também o valor em pontos percentuais do delta.
    """
    autorizados: set[float] = set()
    for v in _valores_numericos(payload):
        autorizados.add(v)
        autorizados.add(v * 100)
        autorizados.add(abs(v))
        autorizados.add(abs(v) * 100)
    return autorizados


def auditar_numeros(texto: str, payload: dict[str, Any]) -> dict[str, Any]:
    """Confere se todo número citado no texto existe no payload.

    A tolerância vem da PRECISÃO COM QUE O NÚMERO FOI ESCRITO, não de uma
    fração relativa fixa. Um número com uma casa decimal ("97,6") autoriza
    uma janela de ±0,05; sem casa decimal ("98"), ±0,5. Ou seja: a
    pergunta que o teste faz é "este texto é um arredondamento correto de
    algum valor autorizado?".

    Tolerância relativa seria errada aqui, e de um jeito perigoso: com
    0,6% de folga, "99,9%" passaria como arredondamento de 100% — duas
    afirmações clinicamente diferentes (a precisão perfeita do Random
    Forest viraria "quase perfeita"). A janela por precisão separa as
    duas e ainda aceita todo arredondamento legítimo.
    """
    autorizados = _valores_autorizados(payload)
    citados, inventados = [], []

    # A regex precisa de três cuidados, cada um aprendido com um falso
    # positivo real:
    #   - vírgula na exclusão à esquerda: sem ela, "-1,17" casava a parte
    #     decimal "17" como se fosse um número solto, e a auditoria
    #     acusava invenção de um valor que estava no payload;
    #   - sinal opcional no grupo: "-1,17" tem que ser lido como -1.17,
    #     não como 1.17 nem partido ao meio;
    #   - lookahead rejeitando \w: "radius1" e "ID-7a" não são métricas
    #     citadas, são identificadores.
    for m in re.finditer(r"(?<![\w/.,-])(-?\d+(?:[.,]\d+)?)(?![\w/-])", texto):
        bruto = m.group(1)
        try:
            valor = float(bruto.replace(",", "."))
        except ValueError:
            continue
        citados.append(bruto)

        if valor.is_integer() and int(valor) in INTEIROS_ESTRUTURAIS:
            continue

        _, _, decimais = bruto.lstrip("-").replace(",", ".").partition(".")
        # +1e-9 absorve erro de ponto flutuante na comparação da borda
        tol = 0.5 * 10 ** (-len(decimais)) + 1e-9
        if not any(abs(valor - a) <= tol for a in autorizados):
            inventados.append(bruto)

    return {
        "n_numeros_citados": len(citados),
        "numeros_inventados": inventados,
        "n_numeros_inventados": len(inventados),
        "fidelidade_numerica_ok": not inventados,
    }


def _sem_acento_minusculo(texto: str) -> str:
    nfkd = unicodedata.normalize("NFKD", texto.lower())
    return "".join(c for c in nfkd if not unicodedata.combining(c))


def auditar_linguagem(texto: str) -> dict[str, Any]:
    """Procura linguagem conclusiva e jargão de ML não traduzido."""
    normalizado = _sem_acento_minusculo(texto)

    conclusivos = []
    for padrao, descricao in PADROES_CONCLUSIVOS:
        achado = re.search(_sem_acento_minusculo(padrao), normalizado)
        if achado:
            conclusivos.append({"trecho": achado.group(), "problema": descricao})

    jargao = sorted({
        termo for termo in JARGAO_ML
        if re.search(rf"\b{re.escape(_sem_acento_minusculo(termo))}\b", normalizado)
    })

    # a ressalva de que a decisão é clínica tem que estar presente
    tem_ressalva = bool(re.search(
        r"decis[ãa]o\s+(?:é|e|final|cl[íi]nica|do\s+profissional|"
        r"cabe\s+ao)|profissional\s+(?:respons[áa]vel|de\s+sa[úu]de)|"
        r"avalia[çc][ãa]o\s+(?:m[ée]dica|cl[íi]nica)",
        normalizado,
    ))

    return {
        "linguagem_conclusiva": conclusivos,
        "n_linguagem_conclusiva": len(conclusivos),
        "jargao_ml_nao_traduzido": jargao,
        "n_jargao_ml": len(jargao),
        "tem_ressalva_decisao_clinica": tem_ressalva,
    }


def auditar_formato(
    texto: str, secoes: tuple[str, ...] = SECOES_CASO, limite_palavras: int = 280
) -> dict[str, Any]:
    normalizado = _sem_acento_minusculo(texto)
    faltando = [
        s for s in secoes
        if _sem_acento_minusculo(s) not in normalizado
    ]
    n_palavras = len(texto.split())
    return {
        "secoes_faltando": faltando,
        "formato_ok": not faltando,
        "n_palavras": n_palavras,
        # 15% de folga: cortar uma explicação clínica por 10 palavras seria
        # rigor sem propósito
        "dentro_do_limite": n_palavras <= limite_palavras * 1.15,
    }


@dataclass
class ChecagemDeterministica:
    numeros: dict[str, Any]
    linguagem: dict[str, Any]
    formato: dict[str, Any]

    @property
    def nota(self) -> float:
        """Nota 0-5 das verificações objetivas.

        Pesos refletem a gravidade: número inventado e linguagem
        conclusiva são falhas de segurança (2 pontos cada); jargão e
        formato são falhas de usabilidade (0,5 cada).
        """
        nota = 5.0
        nota -= 2.0 * min(self.numeros["n_numeros_inventados"], 1)
        nota -= 2.0 * min(self.linguagem["n_linguagem_conclusiva"], 1)
        nota -= 0.5 * min(self.linguagem["n_jargao_ml"], 1)
        nota -= 0.5 * (0 if self.formato["formato_ok"] else 1)
        nota -= 0.5 * (0 if self.linguagem["tem_ressalva_decisao_clinica"] else 1)
        return max(0.0, round(nota, 2))

    def problemas(self) -> list[str]:
        p = []
        if self.numeros["numeros_inventados"]:
            p.append(
                "números sem respaldo no payload: "
                + ", ".join(self.numeros["numeros_inventados"])
            )
        for c in self.linguagem["linguagem_conclusiva"]:
            p.append(f"linguagem conclusiva ('{c['trecho']}'): {c['problema']}")
        if self.linguagem["jargao_ml_nao_traduzido"]:
            p.append(
                "jargão de ML não traduzido: "
                + ", ".join(self.linguagem["jargao_ml_nao_traduzido"])
            )
        if self.formato["secoes_faltando"]:
            p.append("seções ausentes: " + ", ".join(self.formato["secoes_faltando"]))
        if not self.linguagem["tem_ressalva_decisao_clinica"]:
            p.append("não deixa explícito que a decisão é do profissional")
        if not self.formato["dentro_do_limite"]:
            p.append(f"excede o limite de palavras ({self.formato['n_palavras']})")
        return p

    def as_dict(self) -> dict[str, Any]:
        return {
            "nota_deterministica": self.nota,
            "problemas_deterministicos": self.problemas(),
            **self.numeros, **self.linguagem, **self.formato,
        }


def checar(
    texto: str, payload: dict[str, Any], secoes: tuple[str, ...] = SECOES_CASO,
    limite_palavras: int = 280,
) -> ChecagemDeterministica:
    return ChecagemDeterministica(
        numeros=auditar_numeros(texto, payload),
        linguagem=auditar_linguagem(texto),
        formato=auditar_formato(texto, secoes, limite_palavras),
    )


# -----------------------------------------------------------------------
# Camada 2 — LLM como juiz
# -----------------------------------------------------------------------

CRITERIOS_JUIZ = (
    "fidelidade_numerica",
    "ausencia_de_diagnostico",
    "clareza_clinica",
    "acionabilidade",
    "calibracao_de_incerteza",
)


def _extrair_json(texto: str) -> dict[str, Any] | None:
    """Tenta achar um objeto JSON no texto do juiz.

    Mesmo instruído a responder só JSON, um modelo pode envolver a
    resposta em cercas de markdown. Tolero isso em vez de descartar a
    avaliação inteira por causa de três backticks.
    """
    texto = texto.strip()
    if texto.startswith("```"):
        texto = re.sub(r"^```(?:json)?\s*|\s*```$", "", texto, flags=re.MULTILINE)
    try:
        return json.loads(texto)
    except json.JSONDecodeError:
        pass
    inicio, fim = texto.find("{"), texto.rfind("}")
    if inicio != -1 and fim > inicio:
        try:
            return json.loads(texto[inicio : fim + 1])
        except json.JSONDecodeError:
            return None
    return None


def julgar(
    client: LLMClient, texto: str, payload: dict[str, Any], tag: str
) -> dict[str, Any]:
    """Pede ao juiz as notas 1-5. Sem LLM, devolve indisponível."""
    if not client.available:
        return {
            "juiz_disponivel": False,
            "motivo": "sem acesso à LLM — somente a Camada 1 foi aplicada",
        }

    resposta = client.complete(
        system=SYSTEM_JUDGE,
        user=USER_TEMPLATE_JUDGE.format(
            payload=json.dumps(payload, indent=2, ensure_ascii=False),
            explanation=texto,
        ),
        max_tokens=900,
        tag=f"juiz:{tag}",
    )
    dados = _extrair_json(resposta.text)
    if dados is None:
        return {
            "juiz_disponivel": False,
            "motivo": "o juiz não devolveu JSON parseável",
            "resposta_bruta": resposta.text[:500],
        }

    notas = {c: dados.get(c) for c in CRITERIOS_JUIZ}
    validas = [v for v in notas.values() if isinstance(v, (int, float))]
    return {
        "juiz_disponivel": True,
        **notas,
        "nota_juiz_media": round(sum(validas) / len(validas), 2) if validas else None,
        "juiz_numeros_inventados": dados.get("numeros_inventados", []),
        "juiz_problemas": dados.get("problemas", []),
        "juiz_ponto_forte": dados.get("ponto_forte", ""),
    }


# -----------------------------------------------------------------------
# Avaliação
# -----------------------------------------------------------------------


@dataclass
class Avaliacao:
    identificador: str
    origem: str  # "llm" | "template-offline" | "prompt-v1" ...
    texto: str
    deterministica: ChecagemDeterministica
    juiz: dict[str, Any] = field(default_factory=dict)

    def linha(self) -> dict[str, Any]:
        d = self.deterministica
        return {
            "item": self.identificador,
            "origem": self.origem,
            "nota_deterministica": d.nota,
            "nota_juiz": self.juiz.get("nota_juiz_media"),
            "numeros_citados": d.numeros["n_numeros_citados"],
            "numeros_inventados": d.numeros["n_numeros_inventados"],
            "linguagem_conclusiva": d.linguagem["n_linguagem_conclusiva"],
            "jargao_ml": d.linguagem["n_jargao_ml"],
            "ressalva_clinica": d.linguagem["tem_ressalva_decisao_clinica"],
            "formato_ok": d.formato["formato_ok"],
            "palavras": d.formato["n_palavras"],
            **{f"juiz_{c}": self.juiz.get(c) for c in CRITERIOS_JUIZ},
            # divergência entre as camadas: juiz deu nota alta em fidelidade
            # numérica mas a Camada 1 achou número inventado?
            "divergencia_fidelidade": bool(
                self.juiz.get("fidelidade_numerica")
                and self.juiz["fidelidade_numerica"] >= 4
                and d.numeros["n_numeros_inventados"] > 0
            ),
        }


def avaliar_explicacoes(
    client: LLMClient, logger, comparar_prompts: bool = False
) -> tuple[list[Avaliacao], dict[str, Any]]:
    """Avalia as explicações salvas pela Etapa 6."""
    caminho = REPORTS_DIR / "llm_payloads.json"
    if not caminho.exists():
        raise FileNotFoundError(
            f"{caminho} não existe — rode primeiro: python -m src.llm.explain"
        )
    dados = json.loads(caminho.read_text(encoding="utf-8"))
    avaliacoes: list[Avaliacao] = []

    # --- a interpretação da otimização ---
    otim = dados["otimizacao"]
    secoes_otim = ("O QUE MUDOU", "TROCAS ENVOLVIDAS", "QUAL MODELO USAR", "RESSALVAS")
    det = checar(otim["explicacao"], otim["payload"], secoes_otim, 350)
    avaliacoes.append(Avaliacao(
        identificador="interpretação da otimização",
        origem=otim["telemetria"]["provider"],
        texto=otim["explicacao"],
        deterministica=det,
        juiz=julgar(client, otim["explicacao"], otim["payload"], "otimizacao"),
    ))
    logger.info(
        "otimização: nota determinística %.2f%s",
        det.nota, f" | problemas: {det.problemas()}" if det.problemas() else "",
    )

    # --- casos individuais ---
    for registro in dados["casos"]:
        payload, texto = registro["payload"], registro["explicacao"]
        ident = f"caso #{payload['caso']['id']}"
        det = checar(texto, payload)
        avaliacoes.append(Avaliacao(
            identificador=ident,
            origem=registro["telemetria"]["provider"],
            texto=texto,
            deterministica=det,
            juiz=julgar(client, texto, payload, ident),
        ))
        logger.info(
            "%s: nota determinística %.2f%s",
            ident, det.nota,
            f" | problemas: {det.problemas()}" if det.problemas() else "",
        )

    # --- comparação entre versões de prompt ---
    comparacao: dict[str, Any] = {}
    if comparar_prompts:
        comparacao = comparar_versoes_de_prompt(client, dados, logger, avaliacoes)

    return avaliacoes, comparacao


def comparar_versoes_de_prompt(
    client: LLMClient, dados: dict[str, Any], logger, avaliacoes: list[Avaliacao]
) -> dict[str, Any]:
    """Gera a MESMA explicação com cada versão de prompt e avalia todas.

    É o experimento que sustenta as afirmações feitas em prompts.py sobre
    por que cada versão foi substituída — em vez de "a v1 inventava
    números", a tabela mostra quantos.
    """
    if not dados["casos"]:
        return {}
    registro = dados["casos"][0]
    payload = registro["payload"]
    ident = f"caso #{payload['caso']['id']}"
    logger.info("comparando versões de prompt no %s...", ident)

    resultados = []
    for versao in PROMPT_HISTORY:
        resposta = client.complete(
            system=versao.system,
            user=USER_TEMPLATE_CASE.format(
                payload=json.dumps(payload, indent=2, ensure_ascii=False)
            ),
            max_tokens=1200,
            tag=f"prompt-{versao.version}",
        )
        det = checar(resposta.text, payload)
        avaliacao = Avaliacao(
            identificador=f"{ident} (prompt {versao.version})",
            origem=f"prompt-{versao.version}",
            texto=resposta.text,
            deterministica=det,
            juiz=julgar(client, resposta.text, payload, f"prompt-{versao.version}"),
        )
        avaliacoes.append(avaliacao)
        resultados.append({
            "versao": versao.version,
            "mudanca": versao.mudanca,
            "texto": resposta.text,
            "avaliacao": avaliacao.linha(),
            "problemas": det.problemas(),
        })
        logger.info(
            "  prompt %s: determinística %.2f | juiz %s | problemas: %s",
            versao.version, det.nota,
            avaliacao.juiz.get("nota_juiz_media", "n/d"),
            det.problemas() or "nenhum",
        )

    # --- e o template offline, como piso de comparação ---
    from .offline import explicacao_caso_offline

    texto_offline = explicacao_caso_offline(payload)
    det_off = checar(texto_offline, payload)
    av_off = Avaliacao(
        identificador=f"{ident} (template offline)",
        origem="template-offline",
        texto=texto_offline,
        deterministica=det_off,
        juiz=julgar(client, texto_offline, payload, "template-offline"),
    )
    avaliacoes.append(av_off)
    logger.info(
        "  template offline: determinística %.2f | juiz %s",
        det_off.nota, av_off.juiz.get("nota_juiz_media", "n/d"),
    )

    return {
        "caso": ident,
        "payload": payload,
        "versoes": resultados,
        "template_offline": {
            "texto": texto_offline,
            "avaliacao": av_off.linha(),
            "problemas": det_off.problemas(),
        },
    }


# -----------------------------------------------------------------------
# Relatório
# -----------------------------------------------------------------------


def escrever_relatorio(
    avaliacoes: list[Avaliacao], comparacao: dict[str, Any], client: LLMClient
) -> Path:
    tabela = pd.DataFrame([a.linha() for a in avaliacoes])
    tabela.to_csv(REPORTS_DIR / "llm_quality.csv", index=False)

    colunas = [
        "item", "origem", "nota_deterministica", "nota_juiz",
        "numeros_citados", "numeros_inventados", "linguagem_conclusiva",
        "jargao_ml", "formato_ok", "palavras",
    ]
    linhas = [
        "# Etapa 7 — Avaliação da qualidade das interpretações da LLM",
        "",
        f"**Provedor:** {client.describe()}",
        "",
        "## Critério de avaliação",
        "",
        "Duas camadas independentes:",
        "",
        "**Camada 1 — verificações determinísticas** (código, sem LLM). "
        "Nota 0-5, descontando: 2,0 por qualquer número citado que não "
        "exista no payload de origem; 2,0 por linguagem que feche "
        "diagnóstico; 0,5 por jargão de ML não traduzido; 0,5 por seção "
        "ausente; 0,5 por não registrar que a decisão é do profissional. "
        "Os dois descontos de 2,0 são falhas de segurança clínica; os de "
        "0,5, falhas de usabilidade.",
        "",
        "**Camada 2 — LLM como juiz.** Um segundo prompt, com instrução "
        "explícita de ceticismo, atribui nota 1-5 em cinco critérios: "
        "fidelidade numérica, ausência de diagnóstico, clareza clínica, "
        "acionabilidade e calibração da incerteza.",
        "",
        "Quando as duas discordam no critério numérico, a Camada 1 "
        "prevalece — ela compara contra o payload, o juiz compara contra "
        "a própria leitura. A coluna `divergencia_fidelidade` registra "
        "esses casos, porque eles medem a confiabilidade do juiz.",
        "",
        "## Resultados",
        "",
        tabela[colunas].to_markdown(index=False),
        "",
    ]

    divergentes = tabela[tabela["divergencia_fidelidade"]]
    if len(divergentes):
        linhas += [
            f"**Atenção:** em {len(divergentes)} item(ns) o juiz deu nota "
            ">= 4 em fidelidade numérica enquanto a verificação "
            "determinística encontrou número sem respaldo no payload. "
            "Isso é a razão de a Camada 1 existir: a LLM não é auditor "
            "confiável de si mesma.",
            "",
        ]

    # --- exemplos bons e ruins ---
    com_nota = [a for a in avaliacoes if a.origem != "template-offline"]
    if com_nota:
        melhor = max(com_nota, key=lambda a: a.deterministica.nota)
        pior = min(com_nota, key=lambda a: a.deterministica.nota)
        linhas += [
            "## Exemplos documentados",
            "",
            f"### Boa resposta — {melhor.identificador} "
            f"(nota determinística {melhor.deterministica.nota})",
            "",
            f"*Por que é boa:* nenhum número sem respaldo no payload, "
            f"sem linguagem conclusiva, as quatro seções presentes, "
            f"{melhor.deterministica.formato['n_palavras']} palavras."
            + (
                f" O juiz destacou: \"{melhor.juiz['juiz_ponto_forte']}\""
                if melhor.juiz.get("juiz_ponto_forte") else ""
            ),
            "",
            "```",
            melhor.texto,
            "```",
            "",
        ]
        if pior is not melhor and pior.deterministica.problemas():
            linhas += [
                f"### Resposta problemática — {pior.identificador} "
                f"(nota determinística {pior.deterministica.nota})",
                "",
                "*Problemas encontrados:*",
                "",
                *[f"- {p}" for p in pior.deterministica.problemas()],
                "",
                "```",
                pior.texto,
                "```",
                "",
            ]

    # --- comparação entre prompts ---
    if comparacao:
        linhas += [
            "## Comparação entre as versões do prompt",
            "",
            f"Mesma entrada ({comparacao['caso']}), três versões do system "
            "prompt e o template offline como piso. É a evidência das "
            "afirmações feitas em `src/llm/prompts.py` sobre por que cada "
            "versão foi substituída.",
            "",
        ]
        comp_tabela = pd.DataFrame(
            [v["avaliacao"] for v in comparacao["versoes"]]
            + [comparacao["template_offline"]["avaliacao"]]
        )
        linhas += [comp_tabela[colunas].to_markdown(index=False), ""]
        for v in comparacao["versoes"]:
            linhas += [
                f"### Prompt {v['versao']}",
                "",
                f"*Mudança:* {v['mudanca']}",
                "",
                (
                    "*Problemas detectados:* "
                    + "; ".join(v["problemas"]) if v["problemas"]
                    else "*Nenhum problema detectado pela Camada 1.*"
                ),
                "",
                "```",
                v["texto"],
                "```",
                "",
            ]
        linhas += [
            "### Template offline (piso de comparação)",
            "",
            "O template passa nas verificações objetivas por construção — "
            "ele só interpola números do payload, então não pode inventar "
            "nenhum. O que lhe falta é o que a LLM agrega: hierarquizar os "
            "fatores, ligar um ao outro e escrever a conduta em função do "
            "caso específico em vez de escolher entre três frases fixas.",
            "",
            "```",
            comparacao["template_offline"]["texto"],
            "```",
            "",
        ]

    linhas += [
        "## Consumo da LLM nesta avaliação",
        "",
        "```json",
        json.dumps(client.usage_summary(), indent=2, ensure_ascii=False),
        "```",
    ]

    caminho = REPORTS_DIR / "llm_quality.md"
    caminho.write_text("\n".join(linhas), encoding="utf-8")
    return caminho


def main() -> None:
    parser = argparse.ArgumentParser(description="Etapa 7 — qualidade da LLM")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument(
        "--comparar-prompts", action="store_true",
        help="gera a mesma explicação com os prompts v1/v2/v3 e compara",
    )
    args = parser.parse_args()

    logger = setup_logging("fase2.quality", log_file="llm_quality.log")
    events = EventLogger("llm_quality_events.jsonl")
    client = LLMClient(force_offline=args.offline, events=events, logger=logger)
    logger.info("provedor: %s", client.describe())

    avaliacoes, comparacao = avaliar_explicacoes(
        client, logger, args.comparar_prompts
    )
    caminho = escrever_relatorio(avaliacoes, comparacao, client)

    (REPORTS_DIR / "llm_quality_detail.json").write_text(
        json.dumps(
            {
                "provedor": client.describe(),
                "avaliacoes": [
                    {
                        "item": a.identificador, "origem": a.origem,
                        "texto": a.texto,
                        **a.deterministica.as_dict(), "juiz": a.juiz,
                    }
                    for a in avaliacoes
                ],
                "comparacao_de_prompts": comparacao,
            },
            indent=2, ensure_ascii=False, default=str,
        ),
        encoding="utf-8",
    )

    tabela = pd.DataFrame([a.linha() for a in avaliacoes])
    logger.info(
        "\n%s",
        tabela[[
            "item", "origem", "nota_deterministica", "nota_juiz",
            "numeros_inventados", "linguagem_conclusiva",
        ]].to_string(index=False),
    )
    events.emit("quality_session_end", n_avaliacoes=len(avaliacoes),
                **client.usage_summary())
    events.close()
    logger.info("relatório em: %s", caminho.relative_to(BASE_DIR))


if __name__ == "__main__":
    main()

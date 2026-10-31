# Gerador de explicação determinístico (fallback sem LLM).
#
# Monta o mesmo formato de quatro seções do prompt da LLM, mas por
# template: frases fixas preenchidas com os números do payload. Serve a
# dois propósitos concretos:
#
#   1. A suíte de testes da Etapa 8 roda sem chave de API e sem rede,
#      exercitando todo o caminho (payload -> explicação -> avaliação)
#      com saída determinística, que é o que permite asserção.
#   2. Se a API cair durante uma demonstração, o sistema continua
#      produzindo algo correto em vez de uma stack trace.
#
# O texto é notoriamente mecânico — e isso é o ponto de comparação mais
# honesto da Etapa 7: o que a LLM agrega sobre um template é justamente a
# síntese (ligar os fatores entre si, hierarquizar, apontar a
# discordância) que um template não consegue fazer.

from __future__ import annotations

from typing import Any


def _pt(x: float, casas: int = 1) -> str:
    """Formata número no padrão brasileiro (vírgula decimal)."""
    return f"{x:.{casas}f}".replace(".", ",")


def explicacao_caso_offline(payload: dict[str, Any]) -> str:
    previsoes = payload.get("previsoes", {})
    fatores = payload.get("fatores_principais", [])
    caso = payload.get("caso", {})

    rotulos = {m: p["classe_prevista"] for m, p in previsoes.items()}
    divergem = len(set(rotulos.values())) > 1

    # --- AVALIAÇÃO DOS MODELOS ---
    linhas = []
    if divergem:
        linhas.append(
            "AVALIAÇÃO DOS MODELOS: atenção — os modelos divergem neste caso."
        )
    else:
        consenso = next(iter(rotulos.values()), "indeterminado")
        linhas.append(
            f"AVALIAÇÃO DOS MODELOS: os três modelos convergem para "
            f"{consenso}."
        )
    for modelo, p in previsoes.items():
        linhas.append(
            f"  - {modelo}: sugere {p['classe_prevista']} "
            f"(probabilidade de maligno {_pt(100 * p['proba_maligno'])}%)."
        )

    # --- PRINCIPAIS FATORES ---
    linhas.append("")
    linhas.append("PRINCIPAIS FATORES:")
    if not fatores:
        linhas.append("  - não disponíveis para este caso.")
    for f in fatores[:4]:
        lado = "maligno" if f["direcao"] == "maligno" else "benigno"
        linhas.append(
            f"  - {f['feature']}: {_pt(f['z_score'], 2)} desvios-padrão "
            f"{'acima' if f['z_score'] >= 0 else 'abaixo'} da média do "
            f"grupo de referência; empurra a sugestão para {lado}."
        )

    # --- GRAU DE CONFIANÇA ---
    linhas.append("")
    principal = payload.get("modelo_principal")
    p_princ = previsoes.get(principal, {})
    acc = p_princ.get("acuracia_teste_do_modelo")
    recall = p_princ.get("recall_maligno_teste_do_modelo")
    conf = p_princ.get("confianca")
    texto_conf = (
        f"GRAU DE CONFIANÇA: o modelo de referência ({principal}) acerta "
        f"{_pt(100 * acc)}% dos casos do conjunto de teste e identifica "
        f"{_pt(100 * recall)}% dos casos malignos."
        if acc is not None and recall is not None
        else "GRAU DE CONFIANÇA: métricas históricas não disponíveis."
    )
    if conf is not None:
        texto_conf += (
            f" Neste caso específico a confiança é de {_pt(100 * conf)}%"
            + (
                " — abaixo do limiar de 65% adotado, o que coloca o caso na "
                "faixa em que o modelo erra com mais frequência."
                if conf < 0.65
                else ", dentro da faixa em que o modelo é mais estável."
            )
        )
    linhas.append(texto_conf)

    # --- CONDUTA SUGERIDA ---
    linhas.append("")
    algum_maligno = any(r == "maligno" for r in rotulos.values())
    # Toda conduta termina registrando que a decisão é do profissional —
    # inclusive (e principalmente) o ramo de divergência, que é o mais
    # sensível dos três. A primeira versão deste template omitia a
    # ressalva exatamente aí, e foi um teste da Etapa 8
    # (test_explicacao_offline_registra_que_a_decisao_e_clinica) que pegou.
    if divergem or (conf is not None and conf < 0.65):
        conduta = (
            "CONDUTA SUGERIDA: caso de baixa concordância ou baixa confiança "
            "— recomenda-se revisão humana da lâmina e investigação "
            "complementar antes de qualquer conclusão. A decisão sobre a "
            "conduta é do profissional responsável."
        )
    elif algum_maligno:
        conduta = (
            "CONDUTA SUGERIDA: os indicadores apontam para investigação "
            "adicional com prioridade. A decisão sobre a conduta é do "
            "profissional responsável."
        )
    else:
        conduta = (
            "CONDUTA SUGERIDA: os indicadores não apontam para malignidade; "
            "seguir o protocolo de acompanhamento de rotina. A decisão sobre "
            "a conduta é do profissional responsável."
        )
    linhas.append(conduta)

    linhas.append("")
    linhas.append(
        f"[texto gerado por template determinístico, sem LLM "
        f"(caso #{caso.get('id', '?')})]"
    )
    return "\n".join(linhas)


def explicacao_otimizacao_offline(payload: dict[str, Any]) -> str:
    comparativo = payload.get("comparativo", [])
    linhas = ["O QUE MUDOU:"]
    for linha in comparativo:
        delta_fn = linha.get("delta_falsos_negativos")
        if delta_fn is None:
            continue
        if delta_fn < 0:
            efeito = f"passa a detectar {abs(delta_fn)} caso(s) maligno(s) a mais"
        elif delta_fn > 0:
            efeito = f"perde {delta_fn} caso(s) maligno(s) a mais"
        else:
            efeito = "mantém o mesmo número de casos malignos detectados"
        linhas.append(
            f"  - {linha['model']}: {efeito} "
            f"(falsos negativos {linha['fn_original']} -> {linha['fn_otimizado']} "
            f"de {payload.get('n_malignos_teste', '?')} malignos no teste)."
        )

    linhas += ["", "TROCAS ENVOLVIDAS:"]
    for linha in comparativo:
        dfp = linha.get("delta_falsos_positivos")
        if dfp is None:
            continue
        linhas.append(
            f"  - {linha['model']}: falsos positivos "
            f"{linha['fp_original']} -> {linha['fp_otimizado']} "
            f"(variação de {dfp:+d} biópsias desnecessárias)."
        )

    # `recomendacao_por_recall` vem do payload com a palavra "recall", que
    # é jargão de ML e o prompt manda traduzir. O template não pode
    # interpolar a frase crua: reconstrói a recomendação a partir dos
    # campos numéricos, em linguagem clínica. (Foi a auditoria da Etapa 7
    # que pegou — ver reports/fase2/llm_quality.md.)
    melhor = (
        max(comparativo, key=lambda d: d.get("recall_mal_otimizado", 0))
        if comparativo else None
    )
    if melhor:
        recomendacao = (
            f"{melhor['modelo']} otimizado é o que deixa passar menos casos "
            f"malignos: {melhor['fn_otimizado']} não detectado(s) de "
            f"{payload.get('n_malignos_teste', '?')}, ao custo de "
            f"{melhor['fp_otimizado']} encaminhamento(s) desnecessário(s)"
        )
    else:
        recomendacao = "não há dados suficientes para recomendar um modelo"

    linhas += [
        "",
        "QUAL MODELO USAR: pelo critério de minimizar casos malignos não "
        f"detectados, {recomendacao}.",
        "",
        "RESSALVAS: conjunto de teste pequeno "
        f"({payload.get('n_teste', '?')} amostras, "
        f"{payload.get('n_malignos_teste', '?')} malignos), então cada caso "
        "vale cerca de 1 ponto percentual e diferenças de 1-2 casos estão "
        "dentro do ruído amostral. A otimização cobriu apenas "
        "hiperparâmetros — não houve mudança de features, de "
        "pré-processamento nem de limiar de decisão. A adoção de qualquer "
        "um dos modelos é decisão do profissional responsável pelo "
        "serviço, à luz do protocolo da instituição.",
        "",
        "[texto gerado por template determinístico, sem LLM]",
    ]
    return "\n".join(linhas)

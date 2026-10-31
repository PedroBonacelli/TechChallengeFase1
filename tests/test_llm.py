# Testes da camada LLM e da avaliação de qualidade (Etapas 6 e 7).
#
# TODA a suíte roda com `force_offline=True`. Nenhum teste faz chamada de
# rede: isso mantém a suíte rápida, determinística, executável no CI sem
# segredo configurado, e de custo zero. O que se testa aqui é a lógica em
# volta da LLM — montagem de payload, auditoria do texto, formatação —
# que é justamente a parte que pode ter bug silencioso.
#
# A auditoria numérica tem bateria dedicada porque ela é a garantia
# central da Etapa 7: se ela deixar passar um número inventado, o
# relatório de qualidade estaria afirmando algo falso.

from __future__ import annotations

import json

import pytest

from src.llm.client import LLMClient, LLMResponse
from src.llm.evaluate_quality import (
    SECOES_CASO,
    _extrair_json,
    auditar_formato,
    auditar_linguagem,
    auditar_numeros,
    checar,
)
from src.llm.offline import explicacao_caso_offline, explicacao_otimizacao_offline
from src.llm.prompts import (
    PROMPT_HISTORY,
    SYSTEM_PROMPT,
    USER_TEMPLATE_CASE,
    prompt_history_markdown,
)


@pytest.fixture
def payload_caso() -> dict:
    """Payload mínimo mas completo, no formato que explain.py produz."""
    return {
        "caso": {
            "id": 101, "indice_no_teste": 7, "rotulo_real": "maligno",
            "motivo_de_selecao": "caso de menor confiança",
        },
        "modelo_principal": "random_forest",
        "previsoes": {
            "logistic_regression": {
                "classe_prevista": "maligno", "rotulo_previsto": 0,
                "proba_maligno": 0.8421, "confianca": 0.8421,
                "acuracia_teste_do_modelo": 0.9649,
                "recall_maligno_teste_do_modelo": 0.9762,
                "precisao_maligno_teste_do_modelo": 0.9318,
            },
            "random_forest": {
                "classe_prevista": "benigno", "rotulo_previsto": 1,
                "proba_maligno": 0.4100, "confianca": 0.5900,
                "acuracia_teste_do_modelo": 0.9737,
                "recall_maligno_teste_do_modelo": 0.9286,
                "precisao_maligno_teste_do_modelo": 1.0,
            },
        },
        "consenso": {"divergem": True, "votos_maligno": 1, "votos_benigno": 1},
        "fatores_principais": [
            {
                "feature": "worst concave points", "valor_medido": 0.2654,
                "z_score": 2.13, "media_referencia_treino": 0.1142,
                "contribuicao_log_odds": 1.42, "direcao": "maligno",
            },
            {
                "feature": "mean texture", "valor_medido": 14.2,
                "z_score": -0.87, "media_referencia_treino": 19.3,
                "contribuicao_log_odds": -0.63, "direcao": "benigno",
            },
        ],
        "metodo_dos_fatores": "SHAP exato do modelo linear",
        "features_mais_importantes_na_floresta": [],
        "contexto": {
            "n_features": 22, "n_teste": 114, "n_malignos_teste": 42,
            "prevalencia_maligno_treino": 0.3736,
            "limiar_confianca_para_revisao_humana": 0.65,
        },
    }


# --- cliente -----------------------------------------------------------


def test_cliente_offline_nao_fica_disponivel():
    c = LLMClient(force_offline=True)
    assert not c.available
    assert c.provider == "offline"
    assert "offline" in c.describe()


def test_cliente_offline_devolve_o_texto_de_fallback():
    c = LLMClient(force_offline=True)
    r = c.complete("sys", "user", offline_text="TEXTO DE FALLBACK", tag="t")
    assert r.text == "TEXTO DE FALLBACK"
    assert r.provider == "offline"
    assert r.error is None


def test_cliente_offline_sem_fallback_avisa_a_indisponibilidade():
    """O texto tem que DIZER que a explicação não foi gerada.

    Devolver string vazia seria pior: um relatório com seção em branco
    passa desapercebido; uma mensagem explícita, não.
    """
    r = LLMClient(force_offline=True).complete("sys", "user", tag="diagnostico")
    assert "indisponível" in r.text
    assert "diagnostico" in r.text


def test_cliente_registra_telemetria_de_toda_chamada():
    c = LLMClient(force_offline=True)
    c.complete("sys", "user", tag="a", offline_text="x")
    c.complete("sys", "user", tag="b", offline_text="y")
    assert len(c.calls) == 2
    uso = c.usage_summary()
    assert uso["chamadas"] == 2
    assert uso["chamadas_offline"] == 2
    assert uso["chamadas_api"] == 0
    assert uso["erros"] == 0


def test_usage_summary_sem_chamadas():
    assert LLMClient(force_offline=True).usage_summary() == {"chamadas": 0}


def test_custo_estimado_usa_os_tokens():
    r = LLMResponse(
        text="x", model="m", provider="anthropic",
        input_tokens=1_000_000, output_tokens=0,
    )
    assert r.custo_estimado_usd == pytest.approx(5.00)
    r2 = LLMResponse(
        text="x", model="m", provider="anthropic",
        input_tokens=0, output_tokens=1_000_000,
    )
    assert r2.custo_estimado_usd == pytest.approx(25.00)


def test_telemetria_tem_os_campos_do_log_estruturado():
    t = LLMResponse(text="x", model="m", provider="offline").telemetry()
    for campo in (
        "model", "provider", "input_tokens", "output_tokens",
        "latency_s", "custo_estimado_usd",
    ):
        assert campo in t


# --- prompts -----------------------------------------------------------


def test_historico_de_prompts_tem_as_tres_versoes():
    assert [p.version for p in PROMPT_HISTORY] == ["v1", "v2", "v3"]


def test_cada_versao_de_prompt_documenta_o_motivo():
    for p in PROMPT_HISTORY:
        assert p.mudanca and p.motivo
        assert len(p.motivo) > 40, f"{p.version} sem justificativa real"


def test_prompt_em_uso_e_o_v3():
    assert SYSTEM_PROMPT == PROMPT_HISTORY[-1].system


def test_system_prompt_proibe_diagnostico_e_invencao_de_numero():
    texto = SYSTEM_PROMPT.lower()
    assert "jamais emita diagnóstico" in texto
    assert "payload" in texto


def test_template_de_caso_pede_as_quatro_secoes():
    for secao in SECOES_CASO:
        assert secao in USER_TEMPLATE_CASE


def test_template_de_caso_aceita_o_payload(payload_caso):
    preenchido = USER_TEMPLATE_CASE.format(
        payload=json.dumps(payload_caso, ensure_ascii=False)
    )
    assert "worst concave points" in preenchido


def test_historico_em_markdown_cita_as_versoes():
    md = prompt_history_markdown()
    for p in PROMPT_HISTORY:
        assert f"### {p.version}" in md


# --- gerador offline ---------------------------------------------------


def test_explicacao_offline_tem_as_quatro_secoes(payload_caso):
    texto = explicacao_caso_offline(payload_caso)
    for secao in SECOES_CASO:
        assert secao in texto


def test_explicacao_offline_sinaliza_divergencia(payload_caso):
    assert "divergem" in explicacao_caso_offline(payload_caso)


def test_explicacao_offline_reconhece_consenso(payload_caso):
    payload_caso["previsoes"]["random_forest"]["classe_prevista"] = "maligno"
    payload_caso["previsoes"]["random_forest"]["rotulo_previsto"] = 0
    texto = explicacao_offline = explicacao_caso_offline(payload_caso)
    assert "convergem" in texto


def test_explicacao_offline_nao_inventa_numero(payload_caso):
    """Por construção o template só interpola — a auditoria tem que passar."""
    texto = explicacao_caso_offline(payload_caso)
    auditoria = auditar_numeros(texto, payload_caso)
    assert auditoria["fidelidade_numerica_ok"], auditoria["numeros_inventados"]


def test_explicacao_offline_registra_que_a_decisao_e_clinica(payload_caso):
    texto = explicacao_caso_offline(payload_caso)
    assert auditar_linguagem(texto)["tem_ressalva_decisao_clinica"]


def test_explicacao_offline_se_identifica_como_template(payload_caso):
    """Quem lê tem que saber que não foi a LLM que escreveu."""
    assert "template" in explicacao_caso_offline(payload_caso).lower()


@pytest.fixture
def payload_otimizacao() -> dict:
    return {
        "n_teste": 114, "n_malignos_teste": 42,
        "comparativo": [{
            "model": "random_forest", "modelo": "Random Forest",
            "recall_mal_otimizado": 0.9762,
            "fn_original": 3, "fn_otimizado": 1,
            "delta_falsos_negativos": -2,
            "fp_original": 0, "fp_otimizado": 2,
            "delta_falsos_positivos": 2,
        }],
        "recomendacao_por_recall": "Random Forest otimizado tem o maior recall",
    }


def test_explicacao_offline_de_otimizacao(payload_otimizacao):
    texto = explicacao_otimizacao_offline(payload_otimizacao)
    assert "O QUE MUDOU" in texto
    assert "RESSALVAS" in texto
    assert "detectar 2 caso(s) maligno(s) a mais" in texto


def test_otimizacao_offline_nao_vaza_jargao_do_payload(payload_otimizacao):
    """Regressao: o campo `recomendacao_por_recall` contem a palavra
    "recall", e a primeira versao do template interpolava a frase crua —
    injetando jargao de ML num texto que o prompt manda traduzir."""
    texto = explicacao_otimizacao_offline(payload_otimizacao)
    assert auditar_linguagem(texto)["n_jargao_ml"] == 0


def test_otimizacao_offline_registra_que_a_decisao_e_clinica(payload_otimizacao):
    texto = explicacao_otimizacao_offline(payload_otimizacao)
    assert auditar_linguagem(texto)["tem_ressalva_decisao_clinica"]


def test_otimizacao_offline_nao_inventa_numero(payload_otimizacao):
    texto = explicacao_otimizacao_offline(payload_otimizacao)
    auditoria = auditar_numeros(texto, payload_otimizacao)
    assert auditoria["fidelidade_numerica_ok"], auditoria["numeros_inventados"]


# --- auditoria numérica (o coração da Etapa 7) ------------------------


def test_auditoria_aceita_numero_presente_no_payload(payload_caso):
    texto = "A probabilidade estimada de malignidade é 0,8421."
    assert auditar_numeros(texto, payload_caso)["fidelidade_numerica_ok"]


def test_auditoria_aceita_o_mesmo_numero_em_porcentagem(payload_caso):
    """0.8421 no payload pode ser escrito como 84,2% no texto."""
    texto = "A probabilidade estimada é de 84,2%."
    assert auditar_numeros(texto, payload_caso)["fidelidade_numerica_ok"]


def test_auditoria_pega_numero_inventado(payload_caso):
    texto = "O modelo tem sensibilidade de 99,4% nesta população."
    r = auditar_numeros(texto, payload_caso)
    assert not r["fidelidade_numerica_ok"]
    assert "99,4" in r["numeros_inventados"]


def test_auditoria_pega_metrica_alterada(payload_caso):
    """O recall real é 0.9762; 0.9500 é plausível e falso — o pior caso."""
    texto = "O modelo identifica 95,0% dos casos malignos."
    r = auditar_numeros(texto, payload_caso)
    assert not r["fidelidade_numerica_ok"]


def test_auditoria_tolera_arredondamento_razoavel(payload_caso):
    """0.9762 -> "97,6%" e "98%" são arredondamentos legítimos."""
    assert auditar_numeros(
        "identifica 97,6% dos casos", payload_caso
    )["fidelidade_numerica_ok"]


def test_auditoria_isenta_contagens_estruturais(payload_caso):
    """"os 2 modelos", "4 fatores" não precisam estar no payload."""
    texto = "Os 2 modelos avaliaram 4 fatores principais."
    assert auditar_numeros(texto, payload_caso)["fidelidade_numerica_ok"]


def test_auditoria_conta_os_numeros_citados(payload_caso):
    r = auditar_numeros("valores 0,8421 e 0,4100 e 42", payload_caso)
    assert r["n_numeros_citados"] == 3


def test_auditoria_de_texto_sem_numero_passa(payload_caso):
    r = auditar_numeros("Recomenda-se investigação adicional.", payload_caso)
    assert r["fidelidade_numerica_ok"]
    assert r["n_numeros_citados"] == 0


def test_auditoria_ignora_numero_dentro_de_palavra(payload_caso):
    """Nomes de feature e identificadores não são métricas citadas."""
    r = auditar_numeros("a feature radius1 e o caso ID-7a", payload_caso)
    assert r["fidelidade_numerica_ok"]


def test_auditoria_nao_parte_numero_negativo_ao_meio(payload_caso):
    """Regressão: "-0,87" nao pode virar o numero solto "87".

    A primeira versao da regex nao excluia a virgula na busca a
    esquerda, entao em "-0,87" ela casava "87" como se fosse um valor
    independente e acusava invencao de numero. O z_score -0.87 ESTA no
    payload; o texto esta correto e a auditoria e que estava errada.
    """
    r = auditar_numeros("o valor ficou em -0,87 desvios", payload_caso)
    assert r["fidelidade_numerica_ok"], r["numeros_inventados"]
    assert r["n_numeros_citados"] == 1


def test_auditoria_le_o_sinal_do_numero_negativo(payload_caso):
    """-2,13 nao esta no payload, embora +2,13 esteja."""
    r = auditar_numeros("contribuicao de -2,13 pontos", payload_caso)
    assert r["n_numeros_citados"] == 1
    # 2.13 e -2.13 entram em _valores_autorizados via abs(), entao o
    # sinal nao muda a autorizacao -- o que importa e nao partir o numero
    assert "213" not in r["numeros_inventados"]
    assert "13" not in r["numeros_inventados"]


def test_auditoria_de_decimal_longo_nao_gera_numero_fantasma(payload_caso):
    """0,8421 nao pode produzir "8421" nem "421" como citacoes extras."""
    r = auditar_numeros("probabilidade 0,8421 estimada", payload_caso)
    assert r["n_numeros_citados"] == 1
    assert r["fidelidade_numerica_ok"]


# --- auditoria de linguagem -------------------------------------------


@pytest.mark.parametrize("frase", [
    "Confirma-se a malignidade do tumor.",
    "O paciente tem câncer de mama.",
    "Trata-se de um tumor maligno.",
    "O diagnóstico é maligno.",
    "Certamente o caso é maligno.",
    "Definitivamente há malignidade.",
])
def test_auditoria_pega_linguagem_conclusiva(frase):
    r = auditar_linguagem(frase)
    assert r["n_linguagem_conclusiva"] >= 1, frase


@pytest.mark.parametrize("frase", [
    "O modelo sugere malignidade; a decisão é do profissional responsável.",
    "Os indicadores apontam para investigação adicional.",
    "A avaliação médica deve considerar o contexto completo.",
])
def test_auditoria_aceita_linguagem_de_apoio(frase):
    assert auditar_linguagem(frase)["n_linguagem_conclusiva"] == 0, frase


def test_auditoria_pega_jargao_de_ml():
    r = auditar_linguagem("O recall do modelo e o F1-score estão altos.")
    assert "recall" in r["jargao_ml_nao_traduzido"]
    assert "f1-score" in r["jargao_ml_nao_traduzido"]


def test_auditoria_aceita_jargao_traduzido():
    texto = (
        "O modelo identifica 97% dos casos malignos do conjunto de "
        "avaliação. A decisão é do profissional responsável."
    )
    r = auditar_linguagem(texto)
    assert r["n_jargao_ml"] == 0
    assert r["tem_ressalva_decisao_clinica"]


def test_auditoria_detecta_ausencia_de_ressalva():
    assert not auditar_linguagem(
        "O modelo sugere malignidade com alta probabilidade."
    )["tem_ressalva_decisao_clinica"]


def test_auditoria_de_linguagem_ignora_acento_e_caixa():
    assert auditar_linguagem("CONFIRMA-SE A MALIGNIDADE")["n_linguagem_conclusiva"] >= 1


# --- auditoria de formato ---------------------------------------------


def test_auditoria_de_formato_detecta_secao_faltando():
    texto = "AVALIAÇÃO DOS MODELOS: ...\nPRINCIPAIS FATORES: ..."
    r = auditar_formato(texto)
    assert not r["formato_ok"]
    assert "GRAU DE CONFIANÇA" in r["secoes_faltando"]
    assert "CONDUTA SUGERIDA" in r["secoes_faltando"]


def test_auditoria_de_formato_aceita_texto_completo(payload_caso):
    r = auditar_formato(explicacao_caso_offline(payload_caso))
    assert r["formato_ok"]
    assert r["secoes_faltando"] == []


def test_auditoria_de_formato_conta_palavras():
    assert auditar_formato("uma duas três quatro")["n_palavras"] == 4


def test_auditoria_de_formato_tolera_15_por_cento_de_folga():
    assert auditar_formato("palavra " * 300, limite_palavras=280)["dentro_do_limite"]
    assert not auditar_formato(
        "palavra " * 400, limite_palavras=280
    )["dentro_do_limite"]


# --- nota agregada ----------------------------------------------------


def test_texto_limpo_tira_nota_maxima(payload_caso):
    checagem = checar(explicacao_caso_offline(payload_caso), payload_caso)
    assert checagem.nota == 5.0
    assert checagem.problemas() == []


def test_numero_inventado_custa_dois_pontos(payload_caso):
    texto = explicacao_caso_offline(payload_caso) + "\nAcurácia de 99,9%."
    assert checar(texto, payload_caso).nota == pytest.approx(3.0)


def test_linguagem_conclusiva_custa_dois_pontos(payload_caso):
    texto = explicacao_caso_offline(payload_caso) + "\nConfirma-se a malignidade."
    assert checar(texto, payload_caso).nota == pytest.approx(3.0)


def test_as_falhas_de_seguranca_sao_cumulativas(payload_caso):
    texto = (
        explicacao_caso_offline(payload_caso)
        + "\nConfirma-se a malignidade. Acurácia de 99,9%."
    )
    assert checar(texto, payload_caso).nota == pytest.approx(1.0)


def test_nota_nunca_fica_negativa():
    """Texto péssimo e vazio de contexto ainda tem nota >= 0."""
    pior = "Confirma-se o diagnóstico. Recall de 99,9%. Certamente maligno."
    assert checar(pior, {}).nota >= 0.0


def test_problemas_descrevem_o_que_falhou(payload_caso):
    texto = "Confirma-se a malignidade. O recall é 99,9%."
    problemas = checar(texto, payload_caso).problemas()
    assert any("sem respaldo" in p for p in problemas)
    assert any("conclusiva" in p for p in problemas)
    assert any("jargão" in p for p in problemas)
    assert any("seções ausentes" in p for p in problemas)


# --- parsing da resposta do juiz --------------------------------------


def test_extrai_json_puro():
    assert _extrair_json('{"a": 1}') == {"a": 1}


def test_extrai_json_em_cerca_de_markdown():
    assert _extrair_json('```json\n{"a": 1}\n```') == {"a": 1}


def test_extrai_json_com_texto_em_volta():
    assert _extrair_json('Aqui vai:\n{"a": 1}\nespero que ajude') == {"a": 1}


def test_extrai_json_devolve_none_quando_nao_ha_json():
    assert _extrair_json("não consegui avaliar") is None


def test_extrai_json_devolve_none_em_json_invalido():
    assert _extrair_json('{"a": }') is None

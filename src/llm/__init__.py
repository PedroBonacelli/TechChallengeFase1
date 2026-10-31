"""Camada de linguagem natural sobre os resultados dos modelos (Fase 2).

Ordem de leitura dos modulos:
    prompts          -> Etapa 6: os prompts e o historico de versoes
    client           -> cliente Anthropic, com fallback offline
    offline          -> gerador deterministico por template
    explain          -> Etapa 6: monta o payload e gera as explicacoes
    evaluate_quality -> Etapa 7: auditoria deterministica + LLM como juiz

Principio que vale para o pacote inteiro: a LLM recebe um payload JSON
fechado com todos os numeros ja calculados. Ela nunca ve dado bruto nem
chama modelo.
"""

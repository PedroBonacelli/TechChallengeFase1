# Chave da API da LLM.
#
# A chave NÃO entra no Terraform nem no state. O recurso cria o segredo
# vazio; o valor é gravado fora do IaC:
#
#   aws secretsmanager put-secret-value \
#     --secret-id <nome> --secret-string '{"ANTHROPIC_API_KEY":"sk-ant-..."}'
#
# Motivo: qualquer valor passado por variável do Terraform aparece em
# texto claro no state. Se o state estiver num bucket, a chave está num
# bucket. `ignore_changes` no valor impede que um `apply` posterior
# sobrescreva a chave real com o placeholder.
#
# Secrets Manager em vez de SSM Parameter Store porque ele tem rotação
# nativa — quando a chave da API precisar girar, o mecanismo já existe.

resource "aws_secretsmanager_secret" "llm" {
  name        = "${local.prefixo}/llm"
  description = "Credencial da API da LLM usada nas explicações (Etapa 6)"

  # 7 dias de janela de recuperação: apagar a chave por engano não deixa
  # a inferência sem explicação de forma irreversível
  recovery_window_in_days = 7
}

resource "aws_secretsmanager_secret_version" "llm_placeholder" {
  secret_id = aws_secretsmanager_secret.llm.id

  secret_string = jsonencode({
    ANTHROPIC_API_KEY = "PREENCHER_FORA_DO_TERRAFORM"
  })

  lifecycle {
    ignore_changes = [secret_string]
  }
}

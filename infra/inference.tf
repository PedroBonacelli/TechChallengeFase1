# Inferência + explicação — Lambda com imagem de container.
#
# Imagem de container em vez de zip porque scikit-learn + numpy + pandas
# passam do limite de 250 MB descompactados da camada zip. Com imagem o
# teto é 10 GB, e a mesma imagem serve o job do Batch e a Lambda — uma
# build, dois consumidores, sem risco de divergência de versão de
# biblioteca entre treino e inferência (que é uma fonte clássica de bug
# silencioso em ML: o modelo serializa num sklearn e desserializa noutro).
#
# Function URL em vez de API Gateway: a chamada é interna, de um sistema
# hospitalar, e não precisa de rate limiting por chave, cache de resposta
# nem transformação de payload. API Gateway seria custo e latência a mais
# sem contrapartida. IAM_AUTH garante que só quem tem credencial chama.

resource "aws_lambda_function" "inferencia" {
  function_name = "${local.prefixo}-inferencia"
  role          = aws_iam_role.lambda_inferencia.arn

  package_type = "Image"
  image_uri    = "${aws_ecr_repository.app.repository_url}:latest"

  image_config {
    command = ["src.lambda_handler.handler"]
  }

  memory_size = var.lambda_memoria_mb
  timeout     = var.lambda_timeout_s

  reserved_concurrent_executions = var.lambda_concorrencia_reservada

  environment {
    variables = {
      FASE2_S3_BUCKET     = aws_s3_bucket.artefatos.id
      FASE2_LLM_SECRET_ID = aws_secretsmanager_secret.llm.name
      FASE2_LLM_MODEL     = var.modelo_llm
      FASE2_AMBIENTE      = var.ambiente
      FASE2_METRICS_NAMESPACE = local.namespace_metricas
    }
  }

  # O log group é criado pelo Terraform (observability.tf) com retenção
  # definida. Sem isso a Lambda cria o grupo sozinha, com retenção
  # infinita — o jeito mais comum de acumular custo de log sem perceber.
  depends_on = [aws_cloudwatch_log_group.inferencia]
}

resource "aws_lambda_function_url" "inferencia" {
  function_name      = aws_lambda_function.inferencia.function_name
  authorization_type = "AWS_IAM"

  cors {
    # Sem origem liberada: quem chama é servidor, não navegador.
    allow_origins = []
    allow_methods = ["POST"]
    max_age       = 0
  }
}

# Provisioned concurrency só em produção: elimina o cold start (importar
# sklearn e carregar 3 modelos leva segundos) ao custo de pagar pela
# instância parada. Em dev/hml o cold start é aceitável.
resource "aws_lambda_provisioned_concurrency_config" "inferencia" {
  count = var.ambiente == "prd" ? 1 : 0

  function_name                     = aws_lambda_function.inferencia.function_name
  qualifier                         = aws_lambda_function.inferencia.version
  provisioned_concurrent_executions = 2
}

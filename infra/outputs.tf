output "bucket_artefatos" {
  description = "Bucket S3 com dados, modelos, relatorios e log de previsoes."
  value       = aws_s3_bucket.artefatos.id
}

output "repositorio_ecr" {
  description = "URL do ECR. Destino do push da imagem (ver infra/README.md)."
  value       = aws_ecr_repository.app.repository_url
}

output "fila_batch_ga" {
  description = "Fila do Batch onde a otimizacao por GA e submetida."
  value       = aws_batch_job_queue.ga.name
}

output "definicao_job_ga" {
  description = "Job definition do GA."
  value       = aws_batch_job_definition.ga.name
}

output "comando_rodar_ga" {
  description = "Comando para disparar a otimizacao manualmente."
  value = join(" ", [
    "aws batch submit-job",
    "--job-name ga-manual",
    "--job-queue ${aws_batch_job_queue.ga.name}",
    "--job-definition ${aws_batch_job_definition.ga.name}",
  ])
}

output "url_inferencia" {
  description = <<-EOT
    Endpoint de inferencia + explicacao. Autenticacao IAM (SigV4): nao
    abre no navegador, precisa de requisicao assinada.
  EOT
  value       = aws_lambda_function_url.inferencia.function_url
}

output "segredo_llm" {
  description = <<-EOT
    Nome do segredo da chave da LLM. O valor NAO e gerenciado pelo
    Terraform -- gravar com:
      aws secretsmanager put-secret-value --secret-id <este-nome> \
        --secret-string '{"ANTHROPIC_API_KEY":"sk-ant-..."}'
  EOT
  value       = aws_secretsmanager_secret.llm.name
}

output "topico_alertas" {
  description = "Topico SNS dos alarmes."
  value       = aws_sns_topic.alertas.arn
}

output "painel_cloudwatch" {
  description = "URL do painel de monitoramento."
  value = join("", [
    "https://${var.aws_region}.console.aws.amazon.com/cloudwatch/home",
    "?region=${var.aws_region}#dashboards:name=",
    aws_cloudwatch_dashboard.principal.dashboard_name,
  ])
}

output "grupos_de_log" {
  description = "Grupos de log do CloudWatch (log estruturado JSONL)."
  value = {
    otimizacao_ga = aws_cloudwatch_log_group.ga.name
    inferencia    = aws_cloudwatch_log_group.inferencia.name
  }
}

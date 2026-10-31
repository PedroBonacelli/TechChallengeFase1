# Otimização por GA — AWS Batch sobre Fargate.
#
# Fargate em vez de EC2 porque não há o que ganhar gerenciando instância:
# o job roda minutos, algumas vezes por mês. Pagar por vCPU-segundo e não
# ter nó pra manter é o trade-off certo nessa frequência.
#
# Fargate SPOT por padrão (var.batch_usar_spot): o GA é reexecutável sem
# perda — seed fixa, resultado no S3 — então uma interrupção custa uma
# nova execução, e o desconto é de ~70%. Essa é exatamente a carga pra
# qual Spot foi feito.

# A VPC default é um atalho de desenvolvimento. Em produção isto vira
# subnet privada com NAT: dado de saúde não deve transitar em subnet
# com IP público, mesmo que o security group bloqueie entrada.
data "aws_vpc" "default" {
  count   = var.vpc_id == "" ? 1 : 0
  default = true
}

data "aws_subnets" "default" {
  count = length(var.subnet_ids) == 0 ? 1 : 0

  filter {
    name   = "vpc-id"
    values = [var.vpc_id != "" ? var.vpc_id : data.aws_vpc.default[0].id]
  }
}

locals {
  vpc_efetiva = var.vpc_id != "" ? var.vpc_id : data.aws_vpc.default[0].id
  subnets_efetivas = length(var.subnet_ids) > 0 ? var.subnet_ids : data.aws_subnets.default[0].ids
}

# Sem regra de entrada: o job só faz conexão de saída (S3, ECR,
# CloudWatch). Nada precisa alcançá-lo.
resource "aws_security_group" "job_ga" {
  name        = "${local.prefixo}-job-ga"
  description = "Job do GA: somente trafego de saida"
  vpc_id      = local.vpc_efetiva

  egress {
    description = "HTTPS para S3, ECR e CloudWatch"
    from_port   = 443
    to_port     = 443
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

resource "aws_batch_compute_environment" "ga" {
  compute_environment_name = "${local.prefixo}-ga"
  type                     = "MANAGED"
  service_role             = null # Batch usa a service-linked role

  compute_resources {
    type      = var.batch_usar_spot ? "FARGATE_SPOT" : "FARGATE"
    max_vcpus = var.batch_max_vcpus

    subnets            = local.subnets_efetivas
    security_group_ids = [aws_security_group.job_ga.id]
  }

  # O nome do ambiente de computação é imutável na AWS; recriar antes de
  # destruir evita o erro de nome duplicado num apply que muda o tipo
  # (FARGATE <-> FARGATE_SPOT, por exemplo).
  lifecycle {
    create_before_destroy = true
  }
}

resource "aws_batch_job_queue" "ga" {
  name     = "${local.prefixo}-ga"
  state    = "ENABLED"
  priority = 1

  compute_environment_order {
    order               = 1
    compute_environment = aws_batch_compute_environment.ga.arn
  }
}

resource "aws_batch_job_definition" "ga" {
  name                  = "${local.prefixo}-ga"
  type                  = "container"
  platform_capabilities = ["FARGATE"]

  # Uma tentativa extra: a causa mais comum de falha aqui é interrupção
  # de Spot, e o job é idempotente.
  retry_strategy {
    attempts = 2

    evaluate_on_exit {
      action           = "RETRY"
      on_status_reason = "Host EC2*"
    }

    evaluate_on_exit {
      action           = "EXIT"
      on_reason        = "*"
    }
  }

  timeout {
    attempt_duration_seconds = 7200 # 2h: folga grande sobre os ~35 min medidos
  }

  container_properties = jsonencode({
    image      = "${aws_ecr_repository.app.repository_url}:latest"
    command    = ["python", "-m", "src.ga.run_experiments"]
    jobRoleArn = aws_iam_role.job_ga.arn
    executionRoleArn = aws_iam_role.execucao_task.arn

    resourceRequirements = [
      { type = "VCPU", value = tostring(var.batch_job_vcpu) },
      { type = "MEMORY", value = tostring(var.batch_job_memoria_mb) },
    ]

    networkConfiguration = {
      # público porque a VPC default não tem NAT; com subnet privada +
      # NAT (o cenário de produção) isto vira false
      assignPublicIp = "ENABLED"
    }

    fargatePlatformConfiguration = {
      platformVersion = "LATEST"
    }

    environment = [
      { name = "FASE2_S3_BUCKET", value = aws_s3_bucket.artefatos.id },
      { name = "FASE2_AMBIENTE", value = var.ambiente },
      { name = "FASE2_METRICS_NAMESPACE", value = local.namespace_metricas },
      # o pipeline do GA não chama LLM; a variável existe pra que o
      # container saiba disso explicitamente em vez de tentar e falhar
      { name = "FASE2_LLM_DESABILITADA", value = "1" },
    ]

    logConfiguration = {
      logDriver = "awslogs"
      options = {
        "awslogs-group"         = aws_cloudwatch_log_group.ga.name
        "awslogs-region"        = var.aws_region
        "awslogs-stream-prefix" = "ga"
      }
    }
  })
}

# --- retreino agendado -------------------------------------------------

resource "aws_cloudwatch_event_rule" "retreino" {
  count = var.cron_retreino == null ? 0 : 1

  name                = "${local.prefixo}-retreino"
  description         = "Dispara a otimizacao por GA periodicamente"
  schedule_expression = var.cron_retreino
}

resource "aws_cloudwatch_event_target" "retreino" {
  count = var.cron_retreino == null ? 0 : 1

  rule     = aws_cloudwatch_event_rule.retreino[0].name
  arn      = aws_batch_job_queue.ga.arn
  role_arn = aws_iam_role.agendador_retreino.arn

  batch_target {
    job_definition = aws_batch_job_definition.ga.arn
    job_name       = "${local.prefixo}-retreino-agendado"
  }
}

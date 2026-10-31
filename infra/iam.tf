# Papéis IAM.
#
# Os dois serviços têm permissões ASSIMÉTRICAS de propósito:
#
#   - o job do GA escreve em models/ e reports/, e NÃO lê o segredo da
#     LLM (a otimização não chama LLM nenhuma);
#   - a Lambda de inferência LÊ models/, escreve só em predictions/, e lê
#     o segredo. Ela não pode sobrescrever um modelo.
#
# Essa assimetria é o ponto: se a Lambda for comprometida por uma entrada
# maliciosa, o atacante não consegue trocar o modelo de diagnóstico — o
# que seria a pior consequência possível nesta arquitetura.

data "aws_iam_policy_document" "assumir_ecs_tasks" {
  statement {
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["ecs-tasks.amazonaws.com"]
    }
  }
}

data "aws_iam_policy_document" "assumir_lambda" {
  statement {
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["lambda.amazonaws.com"]
    }
  }
}

data "aws_iam_policy_document" "assumir_eventbridge" {
  statement {
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["events.amazonaws.com"]
    }
  }
}

# --- execução da task (puxar imagem, escrever log) --------------------

resource "aws_iam_role" "execucao_task" {
  name               = "${local.prefixo}-execucao-task"
  assume_role_policy = data.aws_iam_policy_document.assumir_ecs_tasks.json
}

resource "aws_iam_role_policy_attachment" "execucao_task" {
  role       = aws_iam_role.execucao_task.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy"
}

# --- papel do job do GA -----------------------------------------------

resource "aws_iam_role" "job_ga" {
  name               = "${local.prefixo}-job-ga"
  assume_role_policy = data.aws_iam_policy_document.assumir_ecs_tasks.json
}

data "aws_iam_policy_document" "job_ga" {
  # lê os dados processados
  statement {
    sid     = "LerDadosProcessados"
    actions = ["s3:GetObject"]
    resources = [
      "${aws_s3_bucket.artefatos.arn}/data/*",
      "${aws_s3_bucket.artefatos.arn}/models/scaler.joblib",
    ]
  }

  # escreve modelos otimizados e relatórios
  statement {
    sid     = "EscreverModelosERelatorios"
    actions = ["s3:PutObject", "s3:GetObject"]
    resources = [
      "${aws_s3_bucket.artefatos.arn}/models/*",
      "${aws_s3_bucket.artefatos.arn}/reports/*",
    ]
  }

  statement {
    sid       = "ListarBucket"
    actions   = ["s3:ListBucket"]
    resources = [aws_s3_bucket.artefatos.arn]
  }

  # publica a métrica de fitness do melhor indivíduo
  statement {
    sid       = "PublicarMetricaDeFitness"
    actions   = ["cloudwatch:PutMetricData"]
    resources = ["*"] # PutMetricData não suporta restrição por recurso

    condition {
      test     = "StringEquals"
      variable = "cloudwatch:namespace"
      values   = [local.namespace_metricas]
    }
  }
}

resource "aws_iam_role_policy" "job_ga" {
  name   = "acesso-s3-e-metricas"
  role   = aws_iam_role.job_ga.id
  policy = data.aws_iam_policy_document.job_ga.json
}

# --- papel da Lambda de inferência ------------------------------------

resource "aws_iam_role" "lambda_inferencia" {
  name               = "${local.prefixo}-lambda-inferencia"
  assume_role_policy = data.aws_iam_policy_document.assumir_lambda.json
}

data "aws_iam_policy_document" "lambda_inferencia" {
  statement {
    sid = "EscreverLogs"
    actions = [
      "logs:CreateLogStream",
      "logs:PutLogEvents",
    ]
    resources = ["${aws_cloudwatch_log_group.inferencia.arn}:*"]
  }

  # SOMENTE LEITURA em models/ — a Lambda não pode trocar o modelo
  statement {
    sid       = "LerModelos"
    actions   = ["s3:GetObject"]
    resources = ["${aws_s3_bucket.artefatos.arn}/models/*"]
  }

  # SOMENTE ESCRITA em predictions/ — e sem GetObject, porque a Lambda
  # não tem motivo pra reler previsões de outros pacientes
  statement {
    sid       = "GravarLogDePrevisoes"
    actions   = ["s3:PutObject"]
    resources = ["${aws_s3_bucket.artefatos.arn}/predictions/*"]
  }

  statement {
    sid       = "LerSegredoDaLLM"
    actions   = ["secretsmanager:GetSecretValue"]
    resources = [aws_secretsmanager_secret.llm.arn]
  }

  statement {
    sid       = "PublicarMetricasDeInferencia"
    actions   = ["cloudwatch:PutMetricData"]
    resources = ["*"]

    condition {
      test     = "StringEquals"
      variable = "cloudwatch:namespace"
      values   = [local.namespace_metricas]
    }
  }
}

resource "aws_iam_role_policy" "lambda_inferencia" {
  name   = "acesso-minimo-de-inferencia"
  role   = aws_iam_role.lambda_inferencia.id
  policy = data.aws_iam_policy_document.lambda_inferencia.json
}

# --- papel do agendador -----------------------------------------------

resource "aws_iam_role" "agendador_retreino" {
  name               = "${local.prefixo}-agendador-retreino"
  assume_role_policy = data.aws_iam_policy_document.assumir_eventbridge.json
}

data "aws_iam_policy_document" "agendador_retreino" {
  statement {
    sid       = "SubmeterJobDoGA"
    actions   = ["batch:SubmitJob"]
    resources = [
      aws_batch_job_definition.ga.arn,
      aws_batch_job_queue.ga.arn,
    ]
  }
}

resource "aws_iam_role_policy" "agendador_retreino" {
  name   = "submeter-job"
  role   = aws_iam_role.agendador_retreino.id
  policy = data.aws_iam_policy_document.agendador_retreino.json
}

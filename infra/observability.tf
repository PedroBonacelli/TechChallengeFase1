# Observabilidade — a contraparte em nuvem do src/monitoring.py.
#
# O código local já emite log estruturado em JSON Lines. Aqui esse mesmo
# log é transformado em MÉTRICA por metric filter do CloudWatch: cada
# linha `{"event": "prediction", ...}` que a Lambda escreve no stdout vira
# ponto numa série temporal, sem precisar de agente, sidecar nem
# instrumentação extra no código. É a razão de o log ser JSON e não texto.
#
# Os alarmes vigiam DUAS coisas distintas:
#
#   - saúde técnica: erro e timeout da Lambda, falha do job do GA;
#   - saúde do MODELO: taxa de malignos previstos e fração de previsões
#     incertas. Um modelo pode estar tecnicamente perfeito — zero erro,
#     latência ótima — e estar errando sistematicamente porque a
#     distribuição dos exames mudou. Só o segundo grupo pega isso.

resource "aws_cloudwatch_log_group" "ga" {
  name              = "/aws/batch/${local.prefixo}-ga"
  retention_in_days = var.retencao_logs_dias
}

resource "aws_cloudwatch_log_group" "inferencia" {
  name              = "/aws/lambda/${local.prefixo}-inferencia"
  retention_in_days = var.retencao_logs_dias
}

# --- métricas extraídas do log estruturado ----------------------------

# Cada previsão com rótulo 0 (maligno) conta 1. A razão entre esta
# métrica e o total de previsões é a taxa de maligno.
resource "aws_cloudwatch_log_metric_filter" "previsoes_malignas" {
  name           = "${local.prefixo}-previsoes-malignas"
  log_group_name = aws_cloudwatch_log_group.inferencia.name

  pattern = "{ $.event = \"prediction\" && $.predicted_label = 0 }"

  metric_transformation {
    name          = "PrevisoesMalignas"
    namespace     = local.namespace_metricas
    value         = "1"
    default_value = "0"
    unit          = "Count"
  }
}

resource "aws_cloudwatch_log_metric_filter" "previsoes_totais" {
  name           = "${local.prefixo}-previsoes-totais"
  log_group_name = aws_cloudwatch_log_group.inferencia.name

  pattern = "{ $.event = \"prediction\" }"

  metric_transformation {
    name          = "PrevisoesTotais"
    namespace     = local.namespace_metricas
    value         = "1"
    default_value = "0"
    unit          = "Count"
  }
}

# Previsão marcada como incerta pelo PredictionMonitor (confiança abaixo
# do limiar de 0.65). É o indicador de que os casos estão caindo na zona
# cinzenta do modelo.
resource "aws_cloudwatch_log_metric_filter" "previsoes_incertas" {
  name           = "${local.prefixo}-previsoes-incertas"
  log_group_name = aws_cloudwatch_log_group.inferencia.name

  pattern = "{ $.event = \"prediction\" && $.low_confidence IS TRUE }"

  metric_transformation {
    name          = "PrevisoesIncertas"
    namespace     = local.namespace_metricas
    value         = "1"
    default_value = "0"
    unit          = "Count"
  }
}

# Confiança média: aqui o valor vem do campo, não é contagem.
resource "aws_cloudwatch_log_metric_filter" "confianca" {
  name           = "${local.prefixo}-confianca"
  log_group_name = aws_cloudwatch_log_group.inferencia.name

  pattern = "{ $.event = \"prediction\" }"

  metric_transformation {
    name      = "ConfiancaDaPrevisao"
    namespace = local.namespace_metricas
    value     = "$.confidence"
    unit      = "None"
  }
}

# Custo da LLM, acumulado a partir de cada evento llm_call.
resource "aws_cloudwatch_log_metric_filter" "custo_llm" {
  name           = "${local.prefixo}-custo-llm"
  log_group_name = aws_cloudwatch_log_group.inferencia.name

  pattern = "{ $.event = \"llm_call\" }"

  metric_transformation {
    name      = "CustoEstimadoLLM"
    namespace = local.namespace_metricas
    value     = "$.custo_estimado_usd"
    unit      = "None"
  }
}

# Chamada à LLM que caiu no fallback offline: a explicação saiu, mas por
# template. Degradação silenciosa — precisa ser visível.
resource "aws_cloudwatch_log_metric_filter" "llm_degradada" {
  name           = "${local.prefixo}-llm-degradada"
  log_group_name = aws_cloudwatch_log_group.inferencia.name

  pattern = "{ $.event = \"llm_call\" && $.provider = \"offline\" }"

  metric_transformation {
    name          = "ChamadasLLMDegradadas"
    namespace     = local.namespace_metricas
    value         = "1"
    default_value = "0"
    unit          = "Count"
  }
}

# --- alertas -----------------------------------------------------------

resource "aws_sns_topic" "alertas" {
  name = "${local.prefixo}-alertas"
}

resource "aws_sns_topic_subscription" "alertas_email" {
  count = var.email_alertas == "" ? 0 : 1

  topic_arn = aws_sns_topic.alertas.arn
  protocol  = "email"
  endpoint  = var.email_alertas
}

# --- alarmes de saúde técnica -----------------------------------------

resource "aws_cloudwatch_metric_alarm" "lambda_erros" {
  alarm_name        = "${local.prefixo}-lambda-erros"
  alarm_description = "Erros na Lambda de inferencia"

  namespace           = "AWS/Lambda"
  metric_name         = "Errors"
  statistic           = "Sum"
  period              = 300
  evaluation_periods  = 1
  threshold           = 1
  comparison_operator = "GreaterThanOrEqualToThreshold"

  dimensions = {
    FunctionName = aws_lambda_function.inferencia.function_name
  }

  alarm_actions = [aws_sns_topic.alertas.arn]
  ok_actions    = [aws_sns_topic.alertas.arn]
  # sem dado = sem requisição, não é problema
  treat_missing_data = "notBreaching"
}

resource "aws_cloudwatch_metric_alarm" "lambda_throttle" {
  alarm_name        = "${local.prefixo}-lambda-throttle"
  alarm_description = "Requisicoes recusadas pelo limite de concorrencia reservada"

  namespace           = "AWS/Lambda"
  metric_name         = "Throttles"
  statistic           = "Sum"
  period              = 300
  evaluation_periods  = 1
  threshold           = 1
  comparison_operator = "GreaterThanOrEqualToThreshold"

  dimensions = {
    FunctionName = aws_lambda_function.inferencia.function_name
  }

  alarm_actions      = [aws_sns_topic.alertas.arn]
  treat_missing_data = "notBreaching"
}

resource "aws_cloudwatch_metric_alarm" "llm_degradada" {
  alarm_name        = "${local.prefixo}-llm-degradada"
  alarm_description = "Explicacoes saindo por template: a LLM nao esta sendo alcancada"

  namespace           = local.namespace_metricas
  metric_name         = aws_cloudwatch_log_metric_filter.llm_degradada.metric_transformation[0].name
  statistic           = "Sum"
  period              = 900
  evaluation_periods  = 1
  threshold           = 3
  comparison_operator = "GreaterThanThreshold"

  alarm_actions      = [aws_sns_topic.alertas.arn]
  treat_missing_data = "notBreaching"
}

# --- alarmes de saúde do MODELO ---------------------------------------

# Expressão em vez de métrica simples: a taxa é a razão entre duas
# métricas, e um alarme sobre a contagem absoluta de malignos dispararia
# só porque o volume de exames aumentou.
resource "aws_cloudwatch_metric_alarm" "drift_taxa_maligno" {
  alarm_name = "${local.prefixo}-drift-taxa-maligno"
  alarm_description = join(" ", [
    "Taxa de malignos previstos acima de ${var.limiar_taxa_maligno}.",
    "Prevalencia no treino: ~0.374. Desvio grande indica mudanca na",
    "distribuicao dos exames (drift) ou degradacao do modelo.",
  ])

  threshold           = var.limiar_taxa_maligno
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = 2 # dois períodos: evita alarme por lote atípico
  treat_missing_data  = "notBreaching"
  alarm_actions       = [aws_sns_topic.alertas.arn]

  metric_query {
    id          = "taxa"
    expression  = "malignos / IF(total > 0, total, 1)"
    label       = "Taxa de maligno previsto"
    return_data = true
  }

  metric_query {
    id = "malignos"
    metric {
      namespace   = local.namespace_metricas
      metric_name = aws_cloudwatch_log_metric_filter.previsoes_malignas.metric_transformation[0].name
      stat        = "Sum"
      period      = 3600
    }
  }

  metric_query {
    id = "total"
    metric {
      namespace   = local.namespace_metricas
      metric_name = aws_cloudwatch_log_metric_filter.previsoes_totais.metric_transformation[0].name
      stat        = "Sum"
      period      = 3600
    }
  }
}

resource "aws_cloudwatch_metric_alarm" "muitas_previsoes_incertas" {
  alarm_name = "${local.prefixo}-previsoes-incertas"
  alarm_description = join(" ", [
    "Fracao de previsoes com confianca abaixo de 0.65 acima de",
    "${var.limiar_previsoes_incertas}. Os casos estao caindo na zona",
    "em que o modelo erra mais; revisao humana deve aumentar.",
  ])

  threshold           = var.limiar_previsoes_incertas
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = 2
  treat_missing_data  = "notBreaching"
  alarm_actions       = [aws_sns_topic.alertas.arn]

  metric_query {
    id          = "fracao"
    expression  = "incertas / IF(total > 0, total, 1)"
    label       = "Fracao de previsoes incertas"
    return_data = true
  }

  metric_query {
    id = "incertas"
    metric {
      namespace   = local.namespace_metricas
      metric_name = aws_cloudwatch_log_metric_filter.previsoes_incertas.metric_transformation[0].name
      stat        = "Sum"
      period      = 3600
    }
  }

  metric_query {
    id = "total"
    metric {
      namespace   = local.namespace_metricas
      metric_name = aws_cloudwatch_log_metric_filter.previsoes_totais.metric_transformation[0].name
      stat        = "Sum"
      period      = 3600
    }
  }
}

# --- falha do job do GA ------------------------------------------------

resource "aws_cloudwatch_event_rule" "job_ga_falhou" {
  name        = "${local.prefixo}-job-ga-falhou"
  description = "Job do GA terminou com FAILED"

  event_pattern = jsonencode({
    source      = ["aws.batch"]
    detail-type = ["Batch Job State Change"]
    detail = {
      status   = ["FAILED"]
      jobQueue = [aws_batch_job_queue.ga.arn]
    }
  })
}

resource "aws_cloudwatch_event_target" "job_ga_falhou" {
  rule = aws_cloudwatch_event_rule.job_ga_falhou.name
  arn  = aws_sns_topic.alertas.arn
}

data "aws_iam_policy_document" "sns_eventbridge" {
  statement {
    sid     = "PermitirEventBridgePublicar"
    actions = ["SNS:Publish"]
    resources = [aws_sns_topic.alertas.arn]

    principals {
      type        = "Service"
      identifiers = ["events.amazonaws.com"]
    }
  }
}

resource "aws_sns_topic_policy" "alertas" {
  arn    = aws_sns_topic.alertas.arn
  policy = data.aws_iam_policy_document.sns_eventbridge.json
}

# --- painel ------------------------------------------------------------

resource "aws_cloudwatch_dashboard" "principal" {
  dashboard_name = local.prefixo

  dashboard_body = jsonencode({
    widgets = [
      {
        type = "metric", x = 0, y = 0, width = 12, height = 6
        properties = {
          title  = "Taxa de maligno previsto (referencia do treino: 0.374)"
          region = var.aws_region
          view   = "timeSeries"
          stat   = "Sum"
          period = 3600
          metrics = [
            [{ expression = "m1 / IF(m2 > 0, m2, 1)", label = "taxa de maligno", id = "e1" }],
            [local.namespace_metricas, "PrevisoesMalignas", { id = "m1", visible = false }],
            [local.namespace_metricas, "PrevisoesTotais", { id = "m2", visible = false }],
          ]
          yAxis = { left = { min = 0, max = 1 } }
          annotations = {
            horizontal = [
              { value = 0.374, label = "prevalencia no treino", color = "#8a8a85" },
              { value = var.limiar_taxa_maligno, label = "limiar de alarme", color = "#e34948" },
            ]
          }
        }
      },
      {
        type = "metric", x = 12, y = 0, width = 12, height = 6
        properties = {
          title  = "Confianca media da previsao"
          region = var.aws_region
          view   = "timeSeries"
          period = 3600
          metrics = [
            [local.namespace_metricas, "ConfiancaDaPrevisao", { stat = "Average", label = "media" }],
            [local.namespace_metricas, "ConfiancaDaPrevisao", { stat = "Minimum", label = "minima" }],
          ]
          yAxis = { left = { min = 0.5, max = 1 } }
          annotations = {
            horizontal = [
              { value = 0.65, label = "limiar de revisao humana", color = "#eb6834" },
            ]
          }
        }
      },
      {
        type = "metric", x = 0, y = 6, width = 12, height = 6
        properties = {
          title  = "Lambda de inferencia: volume, erro e latencia"
          region = var.aws_region
          view   = "timeSeries"
          period = 300
          metrics = [
            ["AWS/Lambda", "Invocations", "FunctionName", aws_lambda_function.inferencia.function_name, { stat = "Sum" }],
            ["AWS/Lambda", "Errors", "FunctionName", aws_lambda_function.inferencia.function_name, { stat = "Sum" }],
            ["AWS/Lambda", "Duration", "FunctionName", aws_lambda_function.inferencia.function_name, { stat = "p95", yAxis = "right" }],
          ]
        }
      },
      {
        type = "metric", x = 12, y = 6, width = 12, height = 6
        properties = {
          title  = "Custo estimado da LLM e chamadas degradadas"
          region = var.aws_region
          view   = "timeSeries"
          period = 3600
          metrics = [
            [local.namespace_metricas, "CustoEstimadoLLM", { stat = "Sum", label = "USD por hora" }],
            [local.namespace_metricas, "ChamadasLLMDegradadas", { stat = "Sum", label = "caidas no template" }],
          ]
        }
      },
    ]
  })
}

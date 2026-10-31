# Etapa 11 (opcional) — Infraestrutura como código
#
# Provisiona a arquitetura descrita em infra/README.md. NÃO foi aplicada:
# a entrega é o código de infraestrutura, não um ambiente rodando (ver a
# seção "Status" do README).
#
# Decisão central do desenho: SEPARAR OTIMIZAÇÃO DE INFERÊNCIA.
#
# São duas cargas com perfis opostos. A otimização por GA é um lote
# pesado e raro — 600 avaliações x 5 folds, minutos a horas de CPU, roda
# quando um retreino é disparado. A inferência é leve, frequente e
# sensível a latência: um exame chega, três modelos respondem, a LLM
# explica. Colocar as duas no mesmo serviço forçaria a escolha errada em
# algum dos dois eixos — ou se paga por CPU parada esperando requisição,
# ou a otimização compete por recurso com o atendimento.
#
# Então: AWS Batch (Fargate Spot) para o GA, Lambda para a inferência, S3
# como fronteira entre os dois. O Batch escreve o modelo otimizado no S3;
# a Lambda lê de lá.

terraform {
  required_version = ">= 1.5.0"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.40"
    }
  }

  # Backend remoto comentado de propósito: state local serve pra revisar o
  # plano, mas qualquer uso de verdade precisa de state compartilhado e
  # com lock, senão dois `apply` simultâneos corrompem o state.
  # backend "s3" {
  #   bucket         = "tc-fase2-tfstate"
  #   key            = "diagnostico/terraform.tfstate"
  #   region         = "us-east-1"
  #   dynamodb_table = "tc-fase2-tflock"
  #   encrypt        = true
  # }
}

provider "aws" {
  region = var.aws_region

  default_tags {
    tags = local.tags
  }
}

data "aws_caller_identity" "atual" {}
data "aws_region" "atual" {}

locals {
  prefixo = "${var.projeto}-${var.ambiente}"

  tags = {
    Projeto     = var.projeto
    Ambiente    = var.ambiente
    Fase        = "2"
    Componente  = "diagnostico-mama"
    GerenciadoPor = "terraform"
    # Dado de saúde: a tag existe pra que política de retenção e de acesso
    # possa ser aplicada por tag em vez de por nome de recurso
    Sensibilidade = "dados-de-saude"
  }

  conta = data.aws_caller_identity.atual.account_id

  # Namespace próprio no CloudWatch: as métricas de negócio (taxa de
  # maligno prevista, confiança média, fitness do GA) não se misturam com
  # as métricas de infraestrutura da AWS, e a política IAM pode restringir
  # PutMetricData a este namespace.
  namespace_metricas = "TechChallenge/Fase2/Diagnostico"
}

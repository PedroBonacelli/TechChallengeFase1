# Armazenamento: um bucket S3 como fronteira entre otimização e
# inferência, e um repositório ECR pra imagem do container.
#
# O bucket é a única coisa que o Batch e a Lambda compartilham. O Batch
# escreve `models/` e `reports/`; a Lambda lê `models/` e escreve
# `predictions/`. Essa fronteira é o que permite trocar qualquer um dos
# dois lados sem tocar no outro.

resource "aws_s3_bucket" "artefatos" {
  bucket = "${local.prefixo}-artefatos-${local.conta}"

  # bucket com modelo e log de previsão não pode ser destruído por um
  # `terraform destroy` distraído
  lifecycle {
    prevent_destroy = true
  }
}

resource "aws_s3_bucket_public_access_block" "artefatos" {
  bucket = aws_s3_bucket.artefatos.id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket_server_side_encryption_configuration" "artefatos" {
  bucket = aws_s3_bucket.artefatos.id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
    bucket_key_enabled = true
  }
}

resource "aws_s3_bucket_versioning" "artefatos" {
  bucket = aws_s3_bucket.artefatos.id

  # Versionamento é o mecanismo de rollback do modelo: se um retreino
  # produzir um modelo pior, voltar é restaurar a versão anterior do
  # objeto, não reexecutar o GA.
  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_lifecycle_configuration" "artefatos" {
  bucket     = aws_s3_bucket.artefatos.id
  depends_on = [aws_s3_bucket_versioning.artefatos]

  # Modelos: mantém as versões antigas por 180 dias (histórico de
  # rollback) e expira depois.
  rule {
    id     = "versoes-antigas-de-modelo"
    status = "Enabled"

    filter {
      prefix = "models/"
    }

    noncurrent_version_expiration {
      noncurrent_days = 180
    }
  }

  # Log de previsões: barato de guardar e útil pra análise de drift de
  # longo prazo, então vai pra classe de acesso infrequente em vez de
  # ser apagado.
  rule {
    id     = "previsoes-para-acesso-infrequente"
    status = "Enabled"

    filter {
      prefix = "predictions/"
    }

    transition {
      days          = 90
      storage_class = "STANDARD_IA"
    }

    transition {
      days          = 365
      storage_class = "GLACIER_IR"
    }
  }

  # Multipart abortado vira custo invisível: o objeto não aparece na
  # listagem mas as partes são cobradas.
  rule {
    id     = "limpar-multipart-abortado"
    status = "Enabled"

    filter {}

    abort_incomplete_multipart_upload {
      days_after_initiation = 7
    }
  }
}

# Nega qualquer requisição que não venha por TLS.
resource "aws_s3_bucket_policy" "artefatos_tls" {
  bucket = aws_s3_bucket.artefatos.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Sid       = "NegarTransporteInseguro"
      Effect    = "Deny"
      Principal = "*"
      Action    = "s3:*"
      Resource = [
        aws_s3_bucket.artefatos.arn,
        "${aws_s3_bucket.artefatos.arn}/*",
      ]
      Condition = {
        Bool = { "aws:SecureTransport" = "false" }
      }
    }]
  })
}

# --- ECR ---------------------------------------------------------------

resource "aws_ecr_repository" "app" {
  name                 = local.prefixo
  image_tag_mutability = "IMMUTABLE" # tag não se move: o que rodou é rastreável

  image_scanning_configuration {
    scan_on_push = true
  }

  encryption_configuration {
    encryption_type = "AES256"
  }
}

resource "aws_ecr_lifecycle_policy" "app" {
  repository = aws_ecr_repository.app.name

  policy = jsonencode({
    rules = [{
      rulePriority = 1
      description  = "Mantem as 10 imagens mais recentes"
      selection = {
        tagStatus   = "any"
        countType   = "imageCountMoreThan"
        countNumber = 10
      }
      action = { type = "expire" }
    }]
  })
}

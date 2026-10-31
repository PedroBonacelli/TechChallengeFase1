variable "projeto" {
  description = "Prefixo de todos os recursos."
  type        = string
  default     = "tc-fase2-diag"

  validation {
    # nome de bucket S3 só aceita minúscula, dígito e hífen, e o prefixo
    # entra na composição do nome
    condition     = can(regex("^[a-z0-9-]{3,24}$", var.projeto))
    error_message = "projeto deve ter 3-24 caracteres, só minúsculas, dígitos e hífens."
  }
}

variable "ambiente" {
  description = "Ambiente de implantação."
  type        = string
  default     = "dev"

  validation {
    condition     = contains(["dev", "hml", "prd"], var.ambiente)
    error_message = "ambiente deve ser dev, hml ou prd."
  }
}

variable "aws_region" {
  description = "Região AWS."
  type        = string
  default     = "us-east-1"
}

variable "vpc_id" {
  description = <<-EOT
    VPC onde o AWS Batch roda. Vazio = usa a VPC default da conta.
    Em produção isto NÃO deve ficar vazio: dado de saúde pede subnet
    privada com NAT, não a VPC default com subnet pública.
  EOT
  type        = string
  default     = ""
}

variable "subnet_ids" {
  description = "Subnets para o Batch. Vazio = subnets da VPC default."
  type        = list(string)
  default     = []
}

# --- otimização (AWS Batch) -------------------------------------------

variable "batch_max_vcpus" {
  description = <<-EOT
    Teto de vCPUs do ambiente de computação do Batch. O GA avalia
    indivíduos em sequência dentro de uma geração, mas a Random Forest
    usa n_jobs=-1 internamente, então vCPU extra acelera cada avaliação.
  EOT
  type        = number
  default     = 8
}

variable "batch_job_vcpu" {
  description = "vCPUs por execução do GA."
  type        = number
  default     = 4
}

variable "batch_job_memoria_mb" {
  description = <<-EOT
    Memória por execução do GA, em MiB. O dataset é minúsculo (455 x 22);
    o consumo vem do cache de fitness e das florestas de até 400 árvores
    mantidas durante a avaliação dos 5 folds.
  EOT
  type        = number
  default     = 8192
}

variable "batch_usar_spot" {
  description = <<-EOT
    Fargate Spot no ambiente de computação. O GA é idempotente e
    reexecutável (seed fixa, resultado em S3), então interrupção de Spot
    custa uma reexecução, não um dado perdido — e o desconto é de ~70%.
  EOT
  type        = bool
  default     = true
}

variable "cron_retreino" {
  description = <<-EOT
    Expressão de agendamento do retreino (EventBridge). Padrão: toda
    segunda às 03:00 UTC. Nulo desliga o agendamento.
  EOT
  type        = string
  default     = "cron(0 3 ? * MON *)"
}

# --- inferência (Lambda) ----------------------------------------------

variable "lambda_memoria_mb" {
  description = <<-EOT
    Memória da Lambda de inferência. 2048 MiB não é por consumo — é por
    CPU: na Lambda a fatia de vCPU é proporcional à memória, e importar
    scikit-learn + carregar 3 modelos domina o cold start.
  EOT
  type        = number
  default     = 2048
}

variable "lambda_timeout_s" {
  description = <<-EOT
    Timeout da Lambda. Precisa cobrir a chamada à LLM, que é a parte
    lenta (segundos), não a inferência (milissegundos).
  EOT
  type        = number
  default     = 60
}

variable "lambda_concorrencia_reservada" {
  description = <<-EOT
    Concorrência reservada. -1 desativa a reserva. Com reserva, limita o
    gasto com a API da LLM por um teto de execuções simultâneas — é o
    controle de custo mais direto nesta arquitetura.
  EOT
  type        = number
  default     = 10
}

variable "modelo_llm" {
  description = "Identificador do modelo da LLM usado nas explicações."
  type        = string
  default     = "claude-opus-5"
}

# --- observabilidade --------------------------------------------------

variable "retencao_logs_dias" {
  description = <<-EOT
    Retenção dos logs no CloudWatch. 90 dias cobre auditoria de um
    trimestre; o log estruturado de previsões vai pro S3 pra histórico
    longo, que é mais barato que CloudWatch.
  EOT
  type        = number
  default     = 90
}

variable "email_alertas" {
  description = <<-EOT
    E-mail que recebe os alarmes (via SNS). Vazio = tópico criado sem
    assinatura, e os alarmes ficam só no console.
  EOT
  type        = string
  default     = ""
}

variable "limiar_taxa_maligno" {
  description = <<-EOT
    Limiar do alarme de drift: taxa de malignos previstos acima disso na
    janela dispara alerta. A prevalência no treino é ~0.374; 0.55 dá
    folga pra variação natural de lote pequeno sem virar alarme falso
    toda semana.
  EOT
  type        = number
  default     = 0.55
}

variable "limiar_previsoes_incertas" {
  description = <<-EOT
    Limiar do alarme de confiança: fração de previsões abaixo do limiar
    de 0.65 de confiança que dispara alerta. Subir indica que os casos
    estão chegando na zona cinzenta do modelo.
  EOT
  type        = number
  default     = 0.25
}

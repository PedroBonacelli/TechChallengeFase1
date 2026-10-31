# Etapa 11 — Infraestrutura como código (opcional)

Terraform que provisiona a arquitetura de nuvem da solução da Fase 2.

## Status

**O código de infraestrutura é a entrega; o ambiente não foi provisionado.**

Isso foi uma decisão consciente: a Etapa 11 vale pontuação extra e não é
obrigatória, e aplicar de verdade exigiria credenciais AWS e geraria
custo recorrente sem acrescentar nada ao que a entrega demonstra. O que
está aqui é o desenho completo e o código para executá-lo.

Duas ressalvas honestas sobre o que isso implica:

- Não rodei `terraform validate` nem `terraform plan` — o binário do
  Terraform não está no ambiente de desenvolvimento usado. Erro de
  sintaxe ou de nome de argumento é possível.
- Nenhum recurso foi testado contra a API da AWS. Comportamentos que só
  aparecem na aplicação (dependência implícita que o Terraform não
  inferiu, limite de serviço, permissão insuficiente descoberta em
  runtime) não foram exercitados.

Quem for aplicar deve rodar `terraform validate` e revisar o `plan`
antes.

## A decisão central do desenho

**Otimização e inferência são serviços separados.** São duas cargas com
perfis opostos:

| | Otimização (GA) | Inferência |
|---|---|---|
| Frequência | raro (retreino) | contínuo |
| Duração | minutos a horas | milissegundos + a latência da LLM |
| Gargalo | CPU | cold start e a chamada à LLM |
| Tolera interrupção? | sim (reexecutável) | não |
| Serviço | AWS Batch / Fargate Spot | Lambda |

Juntar as duas no mesmo serviço forçaria a escolha errada em algum eixo:
ou se paga CPU parada esperando requisição, ou a otimização compete por
recurso com o atendimento clínico. O **S3 é a fronteira** entre elas — o
Batch escreve `models/`, a Lambda lê `models/`. Trocar qualquer um dos
lados não toca no outro.

## Arquitetura

```
                        ┌──────────────────────────┐
     EventBridge        │   ECR                    │
   (cron de retreino)   │   imagem única           │
          │             │   (treino + inferência)  │
          │             └────────┬─────────────────┘
          ▼                      │
   ┌─────────────────┐           │ mesma imagem,
   │   AWS Batch     │◄──────────┤ dois entrypoints
   │  Fargate Spot   │           │
   │                 │           │
   │ run_experiments │           │
   │   + compare     │           │
   └────────┬────────┘           │
            │ escreve            │
            │ models/, reports/  │
            ▼                    │
   ┌───────────────────────┐     │
   │        S3             │     │
   │  data/                │     │
   │  models/   ◄──────────┼─────┼──── lê (somente leitura)
   │  reports/             │     │
   │  predictions/ ◄───────┼──┐  │
   └───────────────────────┘  │  │
                              │  ▼
   cliente ──SigV4──►  ┌──────┴──────────────┐
   (sistema hospitalar)│  Lambda             │
                       │  lambda_handler     │──► Secrets Manager
                       │  3 modelos + LLM    │    (chave da LLM)
                       └──────┬──────────────┘
                              │ stdout JSONL
                              ▼
                  ┌──────────────────────────┐
                  │  CloudWatch Logs         │
                  │    │ metric filters      │
                  │    ▼                     │
                  │  Métricas + Alarmes      │──► SNS ──► e-mail
                  │  + Painel                │
                  └──────────────────────────┘
```

## Como o log vira métrica (e por que isso importa)

O `src/monitoring.py` já escreve log estruturado em JSON Lines — uma
linha por evento. Em nuvem, isso não é só conveniência: os **metric
filters** do CloudWatch (`observability.tf`) leem esse mesmo formato e
transformam cada linha em ponto de série temporal, **sem agente, sem
sidecar, sem instrumentação extra no código**.

É a razão de o log ser JSON e não texto. Um `logger.info("previsão: %s")`
não dá métrica nenhuma; `{"event":"prediction","predicted_label":0,...}`
dá cinco.

## Alarmes: duas famílias distintas

**Saúde técnica** — erro e throttle da Lambda, falha do job do Batch,
chamadas à LLM caindo no fallback por template.

**Saúde do modelo** — e esta é a que normalmente falta:

| Alarme | O que vigia |
|---|---|
| `drift-taxa-maligno` | taxa de malignos previstos acima de 0,55 (prevalência no treino: ~0,374) |
| `previsoes-incertas` | mais de 25% das previsões com confiança abaixo de 0,65 |

Um modelo pode estar **tecnicamente perfeito** — zero erro, latência
ótima, nenhum alarme técnico — e estar errando sistematicamente porque a
distribuição dos exames mudou. Só a segunda família pega isso.

Os dois usam expressão de métrica (razão entre duas séries), não contagem
absoluta: um alarme sobre o número de malignos dispararia apenas por ter
aumentado o volume de exames.

## Permissões assimétricas, de propósito

| | `models/` | `predictions/` | segredo da LLM |
|---|---|---|---|
| Job do GA | leitura + **escrita** | — | **não acessa** |
| Lambda | **somente leitura** | somente escrita | leitura |

A Lambda **não pode sobrescrever um modelo**. Se ela for comprometida por
uma entrada maliciosa, o atacante não consegue trocar o modelo de
diagnóstico — que seria a pior consequência possível nesta arquitetura.
O job do GA, por sua vez, não tem acesso nenhum à chave da LLM, porque a
otimização não chama LLM.

## A chave da LLM nunca entra no Terraform

`secrets.tf` cria o segredo **vazio**. O valor é gravado fora do IaC:

```bash
aws secretsmanager put-secret-value \
  --secret-id "$(terraform output -raw segredo_llm)" \
  --secret-string '{"ANTHROPIC_API_KEY":"sk-ant-..."}'
```

Qualquer valor passado por variável do Terraform aparece **em texto claro
no state**. Se o state está num bucket, a chave está num bucket. O
`ignore_changes` no recurso impede que um `apply` posterior sobrescreva a
chave real com o placeholder.

## Como aplicar

```bash
# 1. imagem
docker build -f Dockerfile.fase2 -t tc-fase2 .

# 2. infraestrutura
cd infra
terraform init
terraform validate          # recomendado: não foi executado aqui
terraform plan -out=tfplan
terraform apply tfplan

# 3. push da imagem pro ECR criado no passo 2
ECR=$(terraform output -raw repositorio_ecr)
aws ecr get-login-password | docker login --username AWS --password-stdin "${ECR%%/*}"
docker tag tc-fase2 "$ECR:latest"
docker push "$ECR:latest"

# 4. chave da LLM (ver seção acima)

# 5. dados processados da Fase 1 pro S3
BUCKET=$(terraform output -raw bucket_artefatos)
aws s3 sync ../data/processed "s3://$BUCKET/data/processed"
aws s3 cp ../models/scaler.joblib "s3://$BUCKET/models/scaler.joblib"

# 6. primeira otimização
eval "$(terraform output -raw comando_rodar_ga)"
```

### Inferência

A Function URL usa autenticação IAM (SigV4) — não abre no navegador:

```bash
URL=$(terraform output -raw url_inferencia)
awscurl --service lambda --region us-east-1 -X POST "$URL" \
  -d '{"case_id":"EX-001","features":{"mean radius":17.99, ...}}'
```

Resposta: previsão dos 3 modelos, flag `requer_revisao_humana`,
explicação em linguagem natural, e `explicacao_degradada` indicando se a
LLM foi alcançada ou se o texto saiu por template.

## Custo estimado

Ordem de grandeza em `us-east-1`, para um volume de **1.000 exames/mês** e
**4 retreinos/mês**. Não é cotação — serve pra dimensionar a decisão.

| Item | Estimativa/mês |
|---|---|
| Batch (Fargate Spot, 4 vCPU × ~35 min × 4) | ~US$ 0,15 |
| Lambda (1.000 × 2 GB × ~4 s) | ~US$ 0,15 |
| S3 (< 1 GB + requisições) | ~US$ 0,10 |
| Secrets Manager (1 segredo) | ~US$ 0,40 |
| CloudWatch (logs, métricas, 6 alarmes, painel) | ~US$ 4,00 |
| ECR (< 2 GB) | ~US$ 0,20 |
| **Infraestrutura** | **~US$ 5/mês** |
| API da LLM (1.000 explicações × ~4k tokens) | ~US$ 25–30 |

**A LLM domina o custo — cerca de 85% do total.** É o que justifica os
dois controles implementados: `lambda_concorrencia_reservada` como teto
de execuções simultâneas, e a métrica `CustoEstimadoLLM` extraída de cada
evento `llm_call`. Para volume alto, trocar `var.modelo_llm` por um
modelo menor (`claude-haiku-4-5`) corta esse custo em ~80% — a camada de
LLM é agnóstica ao modelo justamente pra permitir essa troca sem mexer no
código.

Em `prd`, a `provisioned_concurrency` de 2 instâncias adiciona ~US$ 45/mês
e elimina o cold start. Em `dev`/`hml` ela não é criada.

## O que falta para uso clínico real

A arquitetura está desenhada para a demonstração da Fase 2. Antes de
tocar paciente de verdade faltariam, no mínimo:

- **Rede privada.** Hoje o Batch usa a VPC default com IP público
  (`assignPublicIp = ENABLED`) porque ela não tem NAT. Dado de saúde pede
  subnet privada com NAT Gateway e VPC endpoints para S3 e Secrets
  Manager — tráfego que nunca sai da rede da AWS.
- **Criptografia com chave gerenciada.** O S3 usa `AES256` (chave da
  AWS). Com KMS gerenciada pelo cliente, há trilha de auditoria de cada
  descriptografia e possibilidade de revogar acesso pela política da
  chave.
- **CloudTrail com data events** no bucket: quem leu qual modelo, quando.
- **Retenção e anonimização** alinhadas à LGPD. O `case_id` que vai pro
  log não deve ser identificador direto de paciente.
- **Validação clínica do modelo.** O conjunto de teste tem 114 amostras;
  cada caso vale ~0,9 ponto percentual. Isso é suficiente pra um
  exercício acadêmico e insuficiente pra decisão clínica.

# Etapa 6 (parte 2) — Cliente da LLM
#
# LLM escolhida: Claude (Anthropic), via o SDK oficial `anthropic`.
# Por quê: (a) é API gerenciada, então não há GPU nem download de pesos no
# caminho de uma entrega acadêmica; (b) o SDK já traz retry com backoff
# exponencial para 429/5xx, que é a parte chata de acertar à mão; (c) tem
# saída estruturada nativa, que a Etapa 7 usa pro juiz devolver JSON
# válido em vez de texto que eu teria que adivinhar como parsear.
#
# Duas decisões de engenharia que valem registro:
#
# MODO OFFLINE. Se não houver chave de API, o cliente não quebra: cai num
# gerador determinístico baseado em template (`offline.py`). Isso não é
# enfeite — é o que permite (1) rodar a suíte de testes da Etapa 8 em
# qualquer máquina e no CI, sem segredo configurado e sem gastar token, e
# (2) ter um piso de funcionamento se a API estiver fora no meio de uma
# demonstração. O texto offline é claramente pior; o ponto é que o
# pipeline continua de pé e identificado como tal (`provider="offline"`).
#
# TODA CHAMADA É LOGADA. Modelo, tokens de entrada e saída, latência e
# custo estimado vão pro log estruturado da Etapa 5. Sem isso, o custo de
# uma integração com LLM é invisível até a fatura chegar.

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

BASE_DIR = Path(__file__).resolve().parent.parent.parent

# claude-opus-5 é o padrão; trocar por um modelo mais barato (ex.:
# claude-haiku-4-5) é só definir FASE2_LLM_MODEL no .env
DEFAULT_MODEL = os.getenv("FASE2_LLM_MODEL", "claude-opus-5")

# Preço por milhão de tokens (USD) — usado só pra estimar o custo no log.
# Valor de referência do Opus 5; se o modelo for trocado, a estimativa
# fica aproximada, e é por isso que o campo se chama "estimado".
_PRECO_ENTRADA = 5.00
_PRECO_SAIDA = 25.00


def _carregar_env() -> None:
    """Lê o .env da raiz do projeto, se existir.

    `override=False`: uma variável já exportada no ambiente ganha do
    arquivo. É o comportamento esperado em CI, onde a chave vem do
    gerenciador de segredos e não de um arquivo no disco.
    """
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    env_path = BASE_DIR / ".env"
    if env_path.exists():
        load_dotenv(env_path, override=False)


@dataclass
class LLMResponse:
    """Resposta de uma chamada, com a telemetria junto."""

    text: str
    model: str
    provider: str  # "anthropic" ou "offline"
    input_tokens: int = 0
    output_tokens: int = 0
    latency_s: float = 0.0
    stop_reason: str | None = None
    error: str | None = None

    @property
    def custo_estimado_usd(self) -> float:
        return (
            self.input_tokens / 1e6 * _PRECO_ENTRADA
            + self.output_tokens / 1e6 * _PRECO_SAIDA
        )

    def telemetry(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "provider": self.provider,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "latency_s": round(self.latency_s, 3),
            "custo_estimado_usd": round(self.custo_estimado_usd, 6),
            "stop_reason": self.stop_reason,
            "error": self.error,
        }


@dataclass
class LLMClient:
    """Fachada sobre a API da Anthropic, com fallback offline.

    `force_offline=True` ignora a chave mesmo que exista — é o que os
    testes usam pra garantir que nenhuma chamada de rede acontece na
    suíte automatizada.
    """

    model: str = DEFAULT_MODEL
    max_tokens: int = 2000
    force_offline: bool = False
    events: Any | None = None  # monitoring.EventLogger
    logger: Any | None = None  # logging.Logger
    _client: Any = field(default=None, init=False, repr=False)
    _api_key: str | None = field(default=None, init=False, repr=False)
    calls: list[dict[str, Any]] = field(default_factory=list, repr=False)

    def __post_init__(self) -> None:
        if self.force_offline:
            return
        _carregar_env()
        self._api_key = os.getenv("ANTHROPIC_API_KEY") or None
        if not self._api_key:
            return
        try:
            import anthropic

            self._client = anthropic.Anthropic(api_key=self._api_key)
        except ImportError:
            self._client = None

    @property
    def available(self) -> bool:
        """True se as chamadas vão de fato pra API."""
        return self._client is not None and not self.force_offline

    @property
    def provider(self) -> str:
        return "anthropic" if self.available else "offline"

    def describe(self) -> str:
        if self.available:
            return f"Anthropic API ({self.model})"
        if self.force_offline:
            return "modo offline (forçado)"
        return "modo offline (ANTHROPIC_API_KEY não configurada)"

    # -------------------------------------------------------------------

    def complete(
        self,
        system: str,
        user: str,
        max_tokens: int | None = None,
        tag: str = "generic",
        offline_text: str | None = None,
    ) -> LLMResponse:
        """Uma chamada de texto. Nunca levanta exceção.

        Falha de API vira uma `LLMResponse` com `error` preenchido e o
        texto do fallback offline. Numa demonstração ao vivo, um 429 não
        pode derrubar o script inteiro — tem que degradar e seguir.
        """
        max_tokens = max_tokens or self.max_tokens
        t0 = time.perf_counter()

        if not self.available:
            resposta = LLMResponse(
                text=offline_text or _texto_indisponivel(tag),
                model="template-offline",
                provider="offline",
                latency_s=time.perf_counter() - t0,
            )
            return self._registrar(resposta, tag, len(user))

        try:
            msg = self._client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                system=system,
                messages=[{"role": "user", "content": user}],
            )
            texto = "".join(
                bloco.text for bloco in msg.content if bloco.type == "text"
            )
            resposta = LLMResponse(
                text=texto.strip(),
                model=msg.model,
                provider="anthropic",
                input_tokens=msg.usage.input_tokens,
                output_tokens=msg.usage.output_tokens,
                latency_s=time.perf_counter() - t0,
                stop_reason=msg.stop_reason,
            )
        except Exception as exc:  # noqa: BLE001 — degrada em vez de quebrar
            erro = f"{type(exc).__name__}: {exc}"
            if self.logger:
                self.logger.warning("falha na chamada à LLM (%s): %s", tag, erro)
            resposta = LLMResponse(
                text=offline_text or _texto_indisponivel(tag),
                model=self.model,
                provider="offline",
                latency_s=time.perf_counter() - t0,
                error=erro,
            )

        return self._registrar(resposta, tag, len(user))

    def _registrar(
        self, resposta: LLMResponse, tag: str, tamanho_prompt: int
    ) -> LLMResponse:
        registro = {
            "tag": tag,
            "prompt_chars": tamanho_prompt,
            "response_chars": len(resposta.text),
            **resposta.telemetry(),
        }
        self.calls.append(registro)
        if self.events:
            self.events.emit("llm_call", **registro)
        if self.logger:
            self.logger.info(
                "LLM[%s] %s | %d->%d tokens | %.2fs | ~US$ %.5f%s",
                tag, resposta.provider, resposta.input_tokens,
                resposta.output_tokens, resposta.latency_s,
                resposta.custo_estimado_usd,
                f" | ERRO: {resposta.error}" if resposta.error else "",
            )
        return resposta

    def usage_summary(self) -> dict[str, Any]:
        """Totais da sessão — vai pro relatório da Etapa 9."""
        if not self.calls:
            return {"chamadas": 0}
        return {
            "chamadas": len(self.calls),
            "chamadas_api": sum(1 for c in self.calls if c["provider"] == "anthropic"),
            "chamadas_offline": sum(1 for c in self.calls if c["provider"] == "offline"),
            "erros": sum(1 for c in self.calls if c.get("error")),
            "input_tokens": sum(c["input_tokens"] for c in self.calls),
            "output_tokens": sum(c["output_tokens"] for c in self.calls),
            "custo_estimado_usd": round(
                sum(c["custo_estimado_usd"] for c in self.calls), 5
            ),
            "latencia_media_s": round(
                sum(c["latency_s"] for c in self.calls) / len(self.calls), 3
            ),
        }


def _texto_indisponivel(tag: str) -> str:
    return (
        f"[explicação em linguagem natural indisponível: a LLM não foi "
        f"alcançada para '{tag}'. Os números do caso permanecem válidos e "
        f"estão no payload estruturado.]"
    )

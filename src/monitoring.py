# Etapa 5 — Monitoramento e logging do pipeline
#
# Duas saídas, com propósitos diferentes:
#
#   1. LOG LEGÍVEL (console + logs/fase2.log) — acompanhar a execução em
#      tempo real e depurar. Formato texto, via `logging` da stdlib.
#
#   2. LOG ESTRUTURADO (logs/*.jsonl) — uma linha JSON por evento, feito
#      pra ser consumido por código, não por gente. É isso que permite
#      reconstruir a curva de convergência de um experimento rodado três
#      semanas antes sem reexecutar nada, e é o formato que qualquer
#      coletor de observabilidade (CloudWatch Logs, Loki, Datadog) ingere
#      direto. O gráfico de convergência da Etapa 4 é gerado a partir
#      destes arquivos.
#
# Eventos registrados:
#   - experiment_start / experiment_end  (configuração e resultado do GA)
#   - generation                         (melhor e médio fitness, diversidade)
#   - prediction                         (previsão servida pelo modelo final)
#   - llm_call                           (chamada à LLM: tokens, latência)
#
# Por que `prediction` também: o enunciado pede monitoramento das
# PREVISÕES, não só do treino. Um modelo em produção degrada quando a
# distribuição dos exames muda (drift), e o jeito de perceber isso é ter
# a série histórica de cada previsão com sua probabilidade. O
# `PredictionMonitor` abaixo registra isso e calcula a taxa de malignos
# prevista na janela, que é o indicador mais simples e direto de drift.

from __future__ import annotations

import json
import logging
import time
from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

BASE_DIR = Path(__file__).resolve().parent.parent
LOGS_DIR = BASE_DIR / "logs"

_CONSOLE_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)-22s | %(message)s"
_DATE_FORMAT = "%H:%M:%S"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def setup_logging(
    name: str = "fase2",
    level: int = logging.INFO,
    log_file: str | None = "fase2.log",
) -> logging.Logger:
    """Logger com saída no console e em arquivo.

    Idempotente: chamar duas vezes com o mesmo nome não duplica handlers
    (o que faria cada linha aparecer N vezes — acontece fácil quando
    vários módulos chamam setup na importação).
    """
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False

    if logger.handlers:
        return logger

    fmt = logging.Formatter(_CONSOLE_FORMAT, datefmt=_DATE_FORMAT)

    console = logging.StreamHandler()
    console.setFormatter(fmt)
    logger.addHandler(console)

    if log_file:
        arquivo = logging.FileHandler(LOGS_DIR / log_file, encoding="utf-8")
        arquivo.setFormatter(
            logging.Formatter(
                "%(asctime)s | %(levelname)-7s | %(name)-22s | %(message)s"
            )
        )
        logger.addHandler(arquivo)

    return logger


class EventLogger:
    """Grava eventos estruturados em JSON Lines (um objeto por linha).

    Append-only e com flush a cada escrita: se a execução do GA morrer na
    geração 40 de 50, as 40 primeiras continuam no arquivo. Para uma
    otimização que leva minutos, perder o log inteiro por causa de um
    buffer não esvaziado seria o pior resultado possível.
    """

    def __init__(self, filename: str, run_id: str | None = None) -> None:
        LOGS_DIR.mkdir(parents=True, exist_ok=True)
        self.path = LOGS_DIR / filename
        self.run_id = run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        self._fh = self.path.open("a", encoding="utf-8")

    def emit(self, event: str, **payload: Any) -> dict[str, Any]:
        registro = {
            "ts": _utc_now(),
            "run_id": self.run_id,
            "event": event,
            **payload,
        }
        # default=str: numpy.float64, Path e datetime não são serializáveis
        # por padrão e aparecem em payload com frequência
        self._fh.write(json.dumps(registro, ensure_ascii=False, default=str) + "\n")
        self._fh.flush()
        return registro

    def close(self) -> None:
        if not self._fh.closed:
            self._fh.close()

    def __enter__(self) -> EventLogger:
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()


def read_events(filename: str, event: str | None = None) -> list[dict[str, Any]]:
    """Lê um JSONL de volta, opcionalmente filtrando por tipo de evento.

    Linha corrompida (execução morta no meio de uma escrita) é ignorada
    em vez de derrubar a leitura do histórico inteiro.
    """
    path = LOGS_DIR / filename
    if not path.exists():
        return []

    eventos = []
    with path.open(encoding="utf-8") as fh:
        for linha in fh:
            linha = linha.strip()
            if not linha:
                continue
            try:
                registro = json.loads(linha)
            except json.JSONDecodeError:
                continue
            if event is None or registro.get("event") == event:
                eventos.append(registro)
    return eventos


@contextmanager
def timed(logger: logging.Logger, label: str) -> Iterator[dict[str, float]]:
    """Cronometra um bloco e loga a duração.

    Devolve um dict que recebe `elapsed_s` na saída, pra quem precisa do
    número além da linha de log (o GA grava a duração no evento
    `experiment_end`).
    """
    medida: dict[str, float] = {}
    inicio = time.perf_counter()
    logger.info("%s — iniciando", label)
    try:
        yield medida
    finally:
        medida["elapsed_s"] = time.perf_counter() - inicio
        logger.info("%s — concluído em %.2fs", label, medida["elapsed_s"])


@dataclass
class PredictionMonitor:
    """Monitora as previsões servidas pelo modelo.

    Mantém uma janela deslizante das últimas `window` previsões e expõe
    dois indicadores:

      - `taxa_maligno`: fração de malignos previstos na janela. Desvio
        grande em relação à prevalência do treino (~37%) é o sinal mais
        barato de drift nos dados de entrada.
      - `confianca_media`: média da probabilidade da classe prevista.
        Cair indica que os casos estão chegando na zona de incerteza do
        modelo — mesmo com a taxa de maligno estável.

    `low_confidence_threshold` marca cada previsão incerta no log; em
    produção é esse flag que roteia o caso pra revisão humana.
    """

    model_name: str
    events: EventLogger
    window: int = 200
    low_confidence_threshold: float = 0.65
    _recent: deque = field(default_factory=lambda: deque(maxlen=200), repr=False)
    n_total: int = 0
    n_low_confidence: int = 0

    def __post_init__(self) -> None:
        self._recent = deque(maxlen=self.window)

    def record(
        self,
        case_id: str | int,
        predicted_label: int,
        proba_malignant: float,
        true_label: int | None = None,
    ) -> dict[str, Any]:
        confianca = max(proba_malignant, 1.0 - proba_malignant)
        incerto = confianca < self.low_confidence_threshold

        self.n_total += 1
        if incerto:
            self.n_low_confidence += 1
        self._recent.append((predicted_label, confianca))

        return self.events.emit(
            "prediction",
            model=self.model_name,
            case_id=case_id,
            predicted_label=int(predicted_label),
            predicted_class="maligno" if predicted_label == 0 else "benigno",
            proba_malignant=round(float(proba_malignant), 6),
            confidence=round(float(confianca), 6),
            low_confidence=incerto,
            true_label=None if true_label is None else int(true_label),
            correct=None if true_label is None else bool(predicted_label == true_label),
        )

    def snapshot(self) -> dict[str, Any]:
        if not self._recent:
            return {"janela": 0}
        rotulos = [r for r, _ in self._recent]
        confiancas = [c for _, c in self._recent]
        return {
            "modelo": self.model_name,
            "janela": len(self._recent),
            "total_previsoes": self.n_total,
            "taxa_maligno": round(sum(1 for r in rotulos if r == 0) / len(rotulos), 4),
            "confianca_media": round(sum(confiancas) / len(confiancas), 4),
            "previsoes_incertas": self.n_low_confidence,
        }

    def log_snapshot(self, logger: logging.Logger) -> dict[str, Any]:
        snap = self.snapshot()
        self.events.emit("prediction_snapshot", **snap)
        logger.info(
            "monitor[%s]: %d previsões | taxa maligno %.1f%% | "
            "confiança média %.3f | %d incertas",
            self.model_name, snap.get("total_previsoes", 0),
            100 * snap.get("taxa_maligno", 0), snap.get("confianca_media", 0),
            snap.get("previsoes_incertas", 0),
        )
        return snap

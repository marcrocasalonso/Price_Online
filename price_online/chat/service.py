"""Chat service integration with OpenAI and CrewAI."""
from __future__ import annotations

import time

from crewai import Agent, Crew, Task
from openai import OpenAI

from .. import config

_SYSTEM_MESSAGE = (
    "Eres un experto en automoción. Responde sobrio y directo. "
    "Usa SOLO la información proporcionada (CSV filtrado, informe y extracto). "
    "Si falta un dato, dilo y sugiere qué revisar. "
    "Usa bullets cuando ayuden al análisis."
)

_FALLBACK_PROMPT = """
Eres un experto en automoción. Responde con estilo sobrio y directo.
Usa SOLO la información que te paso (no traigas datos externos).

PREGUNTA:
{user_q}

CSV FILTRADO (A/B):
{filtered_csv_text}
- Cuando la columna sub_idx = 1 los valores de la columna "value" corresponden a los valores de ajuste que incrementan o disminuyen al precio del coche. La suma total equivale de esos valores descartando las secciones de "General","0. Basic Data" y "Indices and interim values" corresponde al adjusted price del coche.
- Cuando la columna sub_idx = 2 los valores de la columna "value" corresponden a valores de las caracteristicas o features del coche.

INFORME COMPARATIVO:
{report_md}

EXTRACTO RAG:
{context_text}

Instrucciones:
- Busca y responde lo que te piden, si no lo encuentras dilo y propon una pregunta que te sirva.
- Sintetiza, no listes filas crudas.
- Solo cita cifras presentes en el contexto.
- Si no hay cifra, usa comparativa cualitativa.
- Sé directo y evita enrollarte.
- Usa bullet points si enumeras listas, características o cualquier cosa que ayude al análisis.

"""


def stream_openai_answer(
    context_text: str,
    user_q: str,
    filtered_csv_text: str,
    report_md: str,
    stream_placeholder,
    model: str = "gpt-4o",
    temperature: float = 0.2,
) -> str:
    """Stream an OpenAI chat response into ``stream_placeholder``."""
    config.load_environment()
    client = OpenAI()
    user_content = f"""
PREGUNTA:
{user_q}

CSV FILTRADO (A/B):
{filtered_csv_text}

INFORME COMPARATIVO:
{report_md}

EXTRACTO RAG:
{context_text}
"""
    rendered = ""
    stream_placeholder.markdown("<div class='msg'>Escribiendo…</div>", unsafe_allow_html=True)
    try:
        stream = client.chat.completions.create(
            model=model,
            temperature=temperature,
            stream=True,
            messages=[
                {"role": "system", "content": _SYSTEM_MESSAGE},
                {"role": "user", "content": user_content},
            ],
        )
        for chunk in stream:
            delta = chunk.choices[0].delta
            piece = getattr(delta, "content", None)
            if piece:
                rendered += piece
                stream_placeholder.markdown(
                    f"<div class='msg'>{rendered}</div>", unsafe_allow_html=True
                )
                time.sleep(0.005)
        return rendered.strip() if rendered.strip() else "No hay datos relevantes para responder."
    except Exception as exc:  # pragma: no cover - network failure path
        stream_placeholder.markdown(
            f"<div class='msg'>[Error de streaming: {exc}]</div>", unsafe_allow_html=True
        )
        return "Ha habido un problema generando la respuesta en streaming."


def stream_markdown(answer: str, placeholder, role_class: str = "msg", delay: float = 0.012, chunk_size: int = 10) -> str:
    """Render ``answer`` progressively inside ``placeholder``."""
    rendered = ""
    for idx in range(0, len(answer), chunk_size):
        rendered += answer[idx : idx + chunk_size]
        placeholder.markdown(
            f"<div class='{role_class}'>{rendered}</div>", unsafe_allow_html=True
        )
        time.sleep(delay)
    return rendered


def llm_chat_answer(context_text: str, user_q: str, filtered_csv_text: str, report_md: str) -> str:
    """Fallback chat answer powered by CrewAI when streaming fails."""
    config.load_environment()
    agent = Agent(
        role="Comparador de automoción",
        goal="Respuesta clara usando solo el contexto (CSV filtrado + informe + extracto)",
        backstory="Especialista en fichas técnicas.",
        allow_delegation=False,
        verbose=False,
    )
    task = Task(
        description=_FALLBACK_PROMPT.format(
            context_text=context_text,
            user_q=user_q,
            filtered_csv_text=filtered_csv_text,
            report_md=report_md,
        ),
        agent=agent,
        expected_output="Respuesta clara, útil y directa.",
    )
    crew = Crew(agents=[agent], tasks=[task], verbose=False)
    return str(crew.kickoff())

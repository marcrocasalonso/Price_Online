"""Highlight generation using CrewAI."""
from __future__ import annotations

from crewai import Agent, Crew, Task

from .. import config

_HIGHLIGHT_PROMPT_TEMPLATE = """
Eres un experto en automoción. Lee este informe de diferencias y genera un resumen breve y directo,
resaltando los puntos más importantes que un usuario debería conocer.

INFORME DE DIFERENCIAS:
{report_md}

Instrucciones:
- Responde en formato conciso pero redactado.
- Usa bullet points para destacar lo más diferenciador.
- Devuelve: resumen, diferencias clave y recomendación final.
"""


def generate_highlights(report_md: str) -> str:
    """Produce a highlight summary using CrewAI."""
    if not report_md.strip():
        return ""
    config.load_environment()
    agent = Agent(
        role="Analista de highlights",
        goal="Extraer diferencias clave del informe",
        backstory="Especialista en comparar fichas técnicas.",
        allow_delegation=False,
        verbose=False,
    )
    task = Task(
        description=_HIGHLIGHT_PROMPT_TEMPLATE.format(report_md=report_md),
        agent=agent,
        expected_output="Reflexión breve con puntos clave y recomendación.",
    )
    crew = Crew(agents=[agent], tasks=[task], verbose=False)
    return str(crew.kickoff())

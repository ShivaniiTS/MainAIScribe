"""
Top-level LangGraph encounter pipeline.

Graph topology (Session 4 — conditional routing):

    context → capture → transcribe ──[asr_router]──► note ──[llm_router]──► review → delivery → END

    asr_router:  checks ASR confidence; always "note" for now (single engine).
    llm_router:  checks note confidence; always "review" for now (single engine).

State: EncounterState (Pydantic v2 model).
LangGraph receives the full state on each node call and merges partial update dicts.

Usage:
    from orchestrator.graph import build_graph, run_encounter
    from orchestrator.state import EncounterState, ProviderProfile

    graph = build_graph()
    final_state = run_encounter(graph, initial_state)
"""

from __future__ import annotations

import logging
import time
from typing import Any

from langgraph.graph import END, StateGraph

from orchestrator.edges.asr_router import asr_router
from orchestrator.edges.llm_router import llm_router
from orchestrator.nodes.capture_node import capture_node
from orchestrator.nodes.context_node import context_node
from orchestrator.nodes.delivery_node import delivery_node
from orchestrator.nodes.note_node import note_node
from orchestrator.nodes.review_node import review_node
from orchestrator.nodes.transcribe_node import transcribe_node
from orchestrator.state import EncounterState

logger = logging.getLogger(__name__)


def build_graph() -> Any:
    """
    Compile and return the encounter LangGraph.

    Uses EncounterState (Pydantic v2) as the graph state — LangGraph passes the
    full model to each node and merges partial update dicts back in via model_copy.

    Returns:
        A compiled LangGraph runnable (.invoke() / .ainvoke()).
    """
    graph = StateGraph(EncounterState)

    graph.add_node("context",    context_node)
    graph.add_node("capture",    capture_node)
    graph.add_node("transcribe", transcribe_node)
    graph.add_node("note",       note_node)
    graph.add_node("review",     review_node)
    graph.add_node("delivery",   delivery_node)

    # Linear edges
    graph.set_entry_point("context")
    graph.add_edge("context",  "capture")
    graph.add_edge("capture",  "transcribe")

    # Conditional edges (Session 4)
    graph.add_conditional_edges(
        "transcribe",
        asr_router,
        {"note": "note"},   # mapping: router return value → node name
    )
    graph.add_conditional_edges(
        "note",
        llm_router,
        {"review": "review"},
    )

    graph.add_edge("review",   "delivery")
    graph.add_edge("delivery", END)

    return graph.compile()


def run_encounter(graph: Any, initial_state: EncounterState) -> EncounterState:
    """
    Run the encounter graph synchronously.

    Args:
        graph:          Compiled LangGraph (from build_graph()).
        initial_state:  Initial EncounterState.

    Returns:
        Final EncounterState after all nodes have run.
    """
    pipeline_start_ms = int(time.time() * 1000)

    # Stamp pipeline start time before invoking
    state_with_start = initial_state.model_copy(
        update={"metrics": initial_state.metrics.model_copy(
            update={"pipeline_start_ms": pipeline_start_ms}
        )}
    )

    logger.info(
        "encounter pipeline: start",
        extra={"encounter_id": initial_state.encounter_id},
    )

    final_state = graph.invoke(state_with_start)

    logger.info(
        "encounter pipeline: complete",
        extra={"encounter_id": initial_state.encounter_id},
    )

    # LangGraph may return the state as a dict or a model depending on version
    if isinstance(final_state, dict):
        return EncounterState.model_validate(final_state)
    return final_state


async def arun_encounter(graph: Any, initial_state: EncounterState) -> EncounterState:
    """Async version of run_encounter."""
    pipeline_start_ms = int(time.time() * 1000)
    state_with_start = initial_state.model_copy(
        update={"metrics": initial_state.metrics.model_copy(
            update={"pipeline_start_ms": pipeline_start_ms}
        )}
    )
    final_state = await graph.ainvoke(state_with_start)
    if isinstance(final_state, dict):
        return EncounterState.model_validate(final_state)
    return final_state

from __future__ import annotations

from typing import Literal

from langchain_core.language_models import BaseChatModel
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command

from tradereact.agents.researcher.bear_researcher import create_bear_researcher
from tradereact.agents.researcher.bull_researcher import create_bull_researcher
from tradereact.agents.researcher.research_manager import create_research_manager
from tradereact.agents.supervisor.supervisor import research_supervisor_node
from tradereact.agents.utils.agent_states import AgentState
from tradereact.agents.utils.graph_viz import export_graph


def create_researcher_node(
    quick_thinking_llm: BaseChatModel,
    deep_thinking_llm: BaseChatModel,
    bull_memory,
    bear_memory,
    trader_memory,
    max_turns: int = 1,
):
    bear_worker = create_bear_researcher(llm=quick_thinking_llm, memory=bear_memory)
    bull_worker = create_bull_researcher(llm=quick_thinking_llm, memory=bull_memory)
    research_manager_worker = create_research_manager(llm=deep_thinking_llm, memory=trader_memory)

    research_supervisor = research_supervisor_node(
        members=["bear_node", "bull_node", "research_manager_node"],
        max_turns=max_turns,
    )

    research_builder = StateGraph(AgentState)
    research_builder.add_node("research_supervisor", research_supervisor)
    research_builder.add_node("bear_node", bear_worker)
    research_builder.add_node("bull_node", bull_worker)
    research_builder.add_node("research_manager_node", research_manager_worker)
    research_builder.add_edge(START, "research_supervisor")
    research_graph = research_builder.compile()
    export_graph(research_graph, name="researcher_subgraph")

    def researcher_node(state: AgentState) -> Command[Literal["supervisor"]]:
        """
        Wrapper node that executes the researcher debate subgraph.
        The subgraph handles debate state updates, so we only need to update sender.
        """
        result = research_graph.invoke(state)

        # Subgraph already updated investment_debate_state and investment_plan
        return Command(
            goto="supervisor",
            update={"sender": "researcher"},
        )

    return researcher_node

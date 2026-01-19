from langchain_core.language_models import BaseChatModel
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command
from typing import Literal

from tradereact.agents.risk.aggresive_debator import create_risky_debator
from tradereact.agents.risk.conservative_debator import create_safe_debator
from tradereact.agents.risk.neutral_debator import create_neutral_debator
from tradereact.agents.risk.risk_manager import create_risk_manager
from tradereact.agents.supervisor.supervisor import risk_supervisor_node
from tradereact.agents.utils.agent_states import AgentState
from tradereact.agents.utils.graph_viz import export_graph


def create_risk_node(quick_thinking_llm: BaseChatModel, deep_thinking_llm: BaseChatModel, risk_memory, max_turns: int = 3):
    risky_worker = create_risky_debator(llm=quick_thinking_llm)
    safe_worker = create_safe_debator(llm=quick_thinking_llm)
    neutral_worker = create_neutral_debator(llm=deep_thinking_llm)
    risk_manager_worker = create_risk_manager(llm=deep_thinking_llm, memory=risk_memory)

    risk_supervisor = risk_supervisor_node(
        members=["risk_node", "safe_node", "neutral_node", "risk_manager_node"],
        max_turns=max_turns,
    )

    risk_builder = StateGraph(AgentState)
    risk_builder.add_node("risk_supervisor", risk_supervisor)
    risk_builder.add_node("risk_node", risky_worker)
    risk_builder.add_node("safe_node", safe_worker)
    risk_builder.add_node("neutral_node", neutral_worker)
    risk_builder.add_node("risk_manager_node", risk_manager_worker)

    risk_builder.add_edge(START, "risk_supervisor")
    risk_graph = risk_builder.compile()
    export_graph(risk_graph, name="risk_subgraph")

    def risk_team_node(state: AgentState) -> Command[Literal["supervisor"]]:
        """
        Wrapper node that executes the risk debate subgraph.
        The subgraph handles debate state updates, so we only need to update sender.
        """
        result = risk_graph.invoke(state)

        # Subgraph already updated risk_debate_state and final_trade_decision
        return Command(
            goto="supervisor",
            update={"sender": "risk"}
        )

    return risk_team_node

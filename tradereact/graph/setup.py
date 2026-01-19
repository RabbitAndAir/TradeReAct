from typing import Dict, Any
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START

from tradereact.agents.analysts.analyst import create_analyst_node
from tradereact.agents.researcher.researcher import create_researcher_node
from tradereact.agents.risk.risk import create_risk_node
from tradereact.agents.supervisor.supervisor import make_supervisor_node
from tradereact.agents.trader.trader import create_trader_node
from tradereact.agents.utils.agent_states import AgentState
from tradereact.agents.utils.graph_viz import export_graph

class GraphSetup:
    """Handles the setup and configuration of the agent graph."""

    def __init__(
        self,
        quick_thinking_llm: ChatOpenAI,
        deep_thinking_llm: ChatOpenAI,
        bull_memory,
        bear_memory,
        trader_memory,
        invest_judge_memory,
        risk_manager_memory,
    ):
        """Initialize with required components."""
        self.quick_thinking_llm = quick_thinking_llm
        self.deep_thinking_llm = deep_thinking_llm
        self.bull_memory = bull_memory
        self.bear_memory = bear_memory
        self.trader_memory = trader_memory
        self.invest_judge_memory = invest_judge_memory
        self.risk_manager_memory = risk_manager_memory

    def setup_graph(self, max_debate_rounds: int = 1, max_risk_discuss_rounds: int = 1):
        """Set up and compile the agent workflow graph.

        Args:
            selected_analysts (list): List of analyst types to include. Options are:
                - "market": Market analyst
                - "social": Social media analyst
                - "news": News analyst
                - "fundamentals": Fundamentals analyst
        """

        analyst_node = create_analyst_node(llm=self.quick_thinking_llm)
        researcher_node = create_researcher_node(
            quick_thinking_llm=self.quick_thinking_llm,
            deep_thinking_llm=self.deep_thinking_llm,
            bull_memory=self.bull_memory,
            bear_memory=self.bear_memory,
            trader_memory=self.invest_judge_memory,
            max_turns=max_debate_rounds,
        )

        trader_node = create_trader_node(llm=self.quick_thinking_llm, memory=self.trader_memory)
        risk_node = create_risk_node(
            quick_thinking_llm=self.quick_thinking_llm,
            deep_thinking_llm=self.deep_thinking_llm,
            risk_memory=self.risk_manager_memory,
            max_turns=max_risk_discuss_rounds,
        )

        # ==================== Supervisor 节点 ====================
        supervisor = make_supervisor_node(
            members=["analyst", "researcher", "trader", "risk"]
        )

        # Create workflow
        workflow = StateGraph(AgentState)

        # Add all nodes to the graph
        workflow.add_node("supervisor", supervisor)
        workflow.add_node("analyst", analyst_node)
        workflow.add_node("researcher", researcher_node)
        workflow.add_node("trader", trader_node)
        workflow.add_node("risk", risk_node)

        # Set entry point: START -> supervisor
        # Note: supervisor uses Command(goto=...) for dynamic routing to other nodes
        # No need to manually add edges between nodes - LangGraph handles this via Command
        workflow.add_edge(START, "supervisor")

        # Compile and return
        graph = workflow.compile()
        export_graph(graph, name="tradereact_graph")

        return graph

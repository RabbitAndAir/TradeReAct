"""
Analyst subgraph module
Executes four analysts (Market, News, Social Media, Fundamentals) in serial or parallel mode
"""

from typing import Literal
from langchain_core.language_models import BaseChatModel
from langgraph.graph import StateGraph, START
from langgraph.types import Command

from tradereact.agents.supervisor.supervisor import analyst_supervisor_node
from tradereact.agents.utils.agent_states import AgentState
from tradereact.agents.utils.graph_viz import export_graph
from tradereact.agents.analysts.market_agent import create_market_analyst
from tradereact.agents.analysts.news_agent import create_news_analyst
from tradereact.agents.analysts.social_media_agent import create_social_media_analyst
from tradereact.agents.analysts.fundamentals_agent import create_fundamentals_analyst


def create_analyst_node(llm: BaseChatModel):

    market_worker = create_market_analyst(llm)
    news_worker = create_news_analyst(llm)
    social_worker = create_social_media_analyst(llm)
    fundamentals_worker = create_fundamentals_analyst(llm)

    analyst_supervisor = analyst_supervisor_node(
        members=["market_node", "news_node", "social_node", "fundamentals_node"],
        method="serial"  # 默认串行执行，可以改为 "parallel" 实现并行
    )

    analyst_builder = StateGraph(AgentState)
    analyst_builder.add_node("analyst_supervisor", analyst_supervisor)
    analyst_builder.add_node("market_node", market_worker)
    analyst_builder.add_node("news_node", news_worker)
    analyst_builder.add_node("social_node", social_worker)
    analyst_builder.add_node("fundamentals_node", fundamentals_worker)

    analyst_builder.add_edge(START, "analyst_supervisor")
    analyst_graph = analyst_builder.compile()
    export_graph(analyst_graph, name="analyst_subgraph")

    def analyst_node(state: AgentState) -> Command[Literal["supervisor"]]:
        """
        Wrapper node that executes the analyst subgraph and returns to main supervisor.
        The subgraph handles all report generation, so we only need to update sender.
        """
        result = analyst_graph.invoke(state)

        # Subgraph already updated all report fields, just mark completion
        return Command(
            goto="supervisor",
            update={"sender": "analyst"},
        )
    return analyst_node

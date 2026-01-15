# tradereact/graph/propagation.py

from typing import Dict, Any
from tradereact.agents.utils.agent_states import (
    AgentState,
    InvestDebateState,
    RiskDebateState,
)


class Propagator:
    """Handles state initialization and propagation through the graph."""

    def __init__(self, max_recur_limit=100, max_react_iterations=10):
        """Initialize with configuration parameters.

        Args:
            max_recur_limit: Maximum recursion limit for the graph
            max_react_iterations: Maximum iterations for each ReAct session
        """
        self.max_recur_limit = max_recur_limit
        self.max_react_iterations = max_react_iterations

    def create_initial_state(
        self, company_name: str, trade_date: str
    ) -> Dict[str, Any]:
        """Create the initial state for the agent graph.

        Args:
            company_name: Stock ticker symbol (e.g., "AAPL")
            trade_date: Trading date in YYYY-MM-DD format

        Returns:
            Initial AgentState dictionary with all required fields
        """
        return {
            # ========== 基础信息 ==========
            "messages": [("human", company_name)],
            "company_of_interest": company_name,
            "trade_date": str(trade_date),
            "sender": "system",

            # ========== 分析报告（初始为空） ==========
            "market_report": "",
            "sentiment_report": "",
            "news_report": "",
            "fundamentals_report": "",

            # ========== 辩论状态初始化 ==========
            "investment_debate_state": InvestDebateState(
                bull_history="",
                bear_history="",
                history="",
                current_response="",
                judge_decision="",
                count=0,
            ),
            "investment_plan": "",

            # ========== Trader 状态 ==========
            "trader_investment_plan": "",

            # ========== 风险辩论状态初始化 ==========
            "risk_debate_state": RiskDebateState(
                risky_history="",
                safe_history="",
                neutral_history="",
                history="",
                latest_speaker="",
                current_risky_response="",
                current_safe_response="",
                current_neutral_response="",
                judge_decision="",
                count=0,
            ),
            "final_trade_decision": "",
        }

    def get_graph_args(self) -> Dict[str, Any]:
        """Get arguments for the graph invocation."""
        return {
            "stream_mode": "values",
            "config": {"recursion_limit": self.max_recur_limit},
        }

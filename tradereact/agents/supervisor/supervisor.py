# ==================== Supervisor 工厂函数 ====================
from typing import Literal

from langchain_core.language_models import BaseChatModel
from langgraph.constants import END
from langgraph.types import Command

from tradereact.agents.utils.agent_states import AgentState


def make_supervisor_node(members: list[str]):
    """
    Main supervisor node that routes execution through members sequentially.

    Args:
        members: List of member node names to execute in order

    Returns:
        supervisor_node function that handles routing logic

    Flow:
        START -> members[0] -> members[1] -> ... -> members[n] -> END
    """
    members_lower = [m.lower() for m in members]
    def supervisor_node(state: AgentState) -> Command[Literal[*members, "__end__"]]:
        sender = (state.get("sender") or "").strip().lower()

        # First call or invalid sender: go to first member
        if not sender or sender == "system" or sender not in members_lower:
            goto = members[0]
            return Command(goto=goto, update={"sender": "supervisor"})
        # Find next member in sequence
        idx = members_lower.index(sender) + 1
        # All members completed: finish workflow
        if idx >= len(members):
            return Command(goto=END, update={"sender": "supervisor"})
        # Route to next member
        goto = members[idx]
        return Command(goto=goto, update={"sender": "supervisor"})

    return supervisor_node


def analyst_supervisor_node(members: list[str], method: Literal["serial", "parallel"] = "serial"):
    """
    Analyst Supervisor factory function

    Args:
        members: List of analyst node names (e.g., ["market_node", "news_node", "social_node", "fundamentals_node"])
        method: Execution mode - "serial" for sequential, "parallel" for concurrent execution

    Returns:
        supervisor_node function for routing and scheduling analyst nodes
    """
    from langgraph.types import Send

    members_lower = [m.lower() for m in members]

    def supervisor_node(state: AgentState) -> Command[Literal[*members, "__end__"]]:
        """
        Routes analyst nodes based on execution mode (serial/parallel)
        """
        sender = (state.get("sender") or "").strip().lower()

        # Parallel execution mode: Send tasks to all analysts simultaneously
        if method == "parallel":
            # First entry: dispatch to all members in parallel
            if not sender or sender == "system" or sender not in members_lower:
                return Command(
                    goto=[Send(node, state) for node in members]
                )
            # All members completed: finish subgraph
            # Note: In parallel mode, all analysts complete before returning here
            return Command(goto=END)

        # Serial execution mode: Execute analysts sequentially
        else:  # method == "serial"
            # First entry: go to first member
            if not sender or sender == "system" or sender not in members_lower:
                goto = members[0]
                return Command(goto=goto)

            # Find next member in sequence
            idx = members_lower.index(sender) + 1

            # All members completed: finish subgraph
            if idx >= len(members):
                return Command(goto=END)

            # Route to next member
            goto = members[idx]
            return Command(goto=goto)

    return supervisor_node


def research_supervisor_node(members: list[str], max_turns: int = 2):
    """
    Research Supervisor for bull vs bear debate

    Args:
        members: List of member nodes ["bear_node", "bull_node", "research_manager_node"]
        max_turns: Maximum debate rounds before forcing manager decision

    Returns:
        supervisor_node function for routing debate participants
    """
    members_lower = [m.lower() for m in members]

    def _pick(name: str) -> str:
        idx = members_lower.index(name)
        return members[idx]

    def supervisor_node(state: AgentState) -> Command[Literal[*members, "__end__"]]:
        debate = state["investment_debate_state"]

        # If manager has made a decision, finish debate
        if debate.get("judge_decision"):
            return Command(goto=END)

        # Count only debaters (exclude manager from count)
        debater_count = len([m for m in members_lower if "manager" not in m]) or 1

        # If debate has reached max turns, call manager for decision
        if debate.get("count", 0) >= max_turns * debater_count and "research_manager_node" in members_lower:
            goto = _pick("research_manager_node")
            return Command(goto=goto)

        # Route based on who spoke last (detected from current_response prefix)
        current = (debate.get("current_response") or "").strip()

        # Bull spoke last → bear's turn
        if current.startswith("Bull Analyst:") and "bear_node" in members_lower:
            goto = _pick("bear_node")
            return Command(goto=goto)

        # Bear spoke last → bull's turn
        if current.startswith("Bear Analyst:") and "bull_node" in members_lower:
            goto = _pick("bull_node")
            return Command(goto=goto)

        # First entry or no valid prefix → start with bull
        if "bull_node" in members_lower:
            goto = _pick("bull_node")
            return Command(goto=goto)

        # Fallback to first member
        goto = members[0]
        return Command(goto=goto)

    return supervisor_node


def risk_supervisor_node(members: list[str], max_turns: int = 3):
    """
    Risk Supervisor for risky vs safe vs neutral debate

    Args:
        members: List of member nodes ["risk_node", "safe_node", "neutral_node", "risk_manager_node"]
        max_turns: Maximum debate rounds before forcing manager decision

    Returns:
        supervisor_node function for routing debate participants
    """
    members_lower = [m.lower() for m in members]

    def _pick(name: str) -> str:
        idx = members_lower.index(name)
        return members[idx]

    def supervisor_node(state: AgentState) -> Command[Literal[*members, "__end__"]]:
        debate = state["risk_debate_state"]

        # If manager has made a decision, finish debate
        if debate.get("judge_decision") or (debate.get("latest_speaker") or "").lower() == "judge":
            return Command(goto=END)

        # Count only debaters (exclude manager from count)
        debater_count = len([m for m in members_lower if "manager" not in m]) or 1

        # If debate has reached max turns, call manager for decision
        if debate.get("count", 0) >= max_turns * debater_count and "risk_manager_node" in members_lower:
            goto = _pick("risk_manager_node")
            return Command(goto=goto)

        # Route based on who spoke last (detected from latest_speaker field)
        latest = (debate.get("latest_speaker") or "").lower().strip()

        # Risky spoke last → safe's turn
        if latest == "risky" and "safe_node" in members_lower:
            goto = _pick("safe_node")
            return Command(goto=goto)

        # Safe spoke last → neutral's turn
        if latest == "safe" and "neutral_node" in members_lower:
            goto = _pick("neutral_node")
            return Command(goto=goto)

        # Neutral spoke last → risky's turn
        if latest == "neutral" and "risk_node" in members_lower:
            goto = _pick("risk_node")
            return Command(goto=goto)

        # First entry or no valid speaker → start with risky
        if "risk_node" in members_lower:
            goto = _pick("risk_node")
            return Command(goto=goto)

        # Fallback to first member
        goto = members[0]
        return Command(goto=goto)

    return supervisor_node

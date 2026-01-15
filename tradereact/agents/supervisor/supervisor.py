# ==================== Supervisor 工厂函数 ====================
from typing import Literal

from langchain_core.language_models import BaseChatModel
from langgraph.constants import END
from langgraph.types import Command

from tradereact.agents.utils.agent_states import AgentState


def make_supervisor_node(members: list[str]):
    members_lower = [m.lower() for m in members]

    def supervisor_node(state: AgentState) -> Command[Literal[*members, "__end__"]]:
        sender = (state.get("sender") or "").strip().lower()
        if not sender or sender == "system" or sender not in members_lower:
            goto = members[0]
            return Command(goto=goto, update={"next": goto, "sender": "supervisor"})

        idx = members_lower.index(sender) + 1
        if idx >= len(members):
            return Command(goto=END, update={"next": "FINISH", "sender": "supervisor"})

        goto = members[idx]
        return Command(goto=goto, update={"next": goto, "sender": "supervisor"})
    return supervisor_node

def analyst_supervisor_node(members: list[str], method: Literal["serial", "parallel"] = "serial"):
    """
    Analyst Supervisor 节点工厂函数

    Args:
        members: 成员节点列表，如 ["market_node", "news_node", "social_node", "fundamentals_node"]
        llm: 语言模型实例
        method: 执行模式，"serial" 串行执行，"parallel" 并行执行

    Returns:
        supervisor_node 函数，用于路由和调度分析师节点
    """
    from langgraph.types import Send

    members_lower = [m.lower() for m in members]

    def supervisor_node(state: AgentState) -> Command[Literal[*members, "__end__"]]:
        """
        根据执行模式（serial/parallel）调度分析师节点
        """
        sender = (state.get("sender") or "").strip().lower()
        # 并行执行模式：使用 Send 同时发送任务到所有分析师节点
        if method == "parallel":
            # 如果是第一次进入（sender 为空或来自上层 supervisor）
            if not sender or sender == "system" or sender not in members_lower:
                # 使用 Send 并行发送到所有成员节点
                return Command(
                    goto=[Send(node, state) for node in members],
                    update={"sender": "analyst_supervisor"}
                )
            # 如果所有成员都已完成，检查是否收集到所有报告
            reports_collected = all([
                state.get("market_report"),
                state.get("news_report"),
                state.get("sentiment_report"),
                state.get("fundamentals_report")
            ])
            if reports_collected:
                return Command(
                    goto=END,
                    update={"next": "FINISH", "sender": "analyst_supervisor"}
                )
        # 串行执行模式：按顺序执行各个分析师节点
        else:  # method == "serial"
            # 如果是第一次进入或来自上层 supervisor
            if not sender or sender == "system" or sender not in members_lower:
                goto = members[0]
                return Command(
                    goto=goto,
                    update={"next": goto, "sender": "analyst_supervisor"}
                )
            # 找到当前发送者的索引，转到下一个节点
            idx = members_lower.index(sender) + 1

            # 如果已经执行完所有成员节点，返回 END
            if idx >= len(members):
                return Command(
                    goto=END,
                    update={"next": "FINISH", "sender": "analyst_supervisor"}
                )
            # 转到下一个成员节点
            goto = members[idx]
            return Command(
                goto=goto,
                update={"next": goto, "sender": "analyst_supervisor"}
            )
        # 默认返回第一个节点
        return Command(
            goto=members[0],
            update={"next": members[0], "sender": "analyst_supervisor"}
        )
    return supervisor_node


def research_supervisor_node(members: list[str], max_turns: int = 2):
    members_lower = [m.lower() for m in members]

    def _pick(name: str) -> str:
        idx = members_lower.index(name)
        return members[idx]

    def supervisor_node(state: AgentState) -> Command[Literal[*members, "__end__"]]:
        debate = state["investment_debate_state"]
        if debate.get("judge_decision"):
            return Command(goto=END, update={"next": "FINISH"})

        debater_count = len([m for m in members_lower if "manager" not in m]) or 1
        if debate.get("count", 0) >= max_turns * debater_count and "research_manager_node" in members_lower:
            goto = _pick("research_manager_node")
            return Command(goto=goto, update={"next": goto})

        current = (debate.get("current_response") or "").strip()
        if current.startswith("Bull Analyst:") and "bear_node" in members_lower:
            goto = _pick("bear_node")
            return Command(goto=goto, update={"next": goto})
        if current.startswith("Bear Analyst:") and "bull_node" in members_lower:
            goto = _pick("bull_node")
            return Command(goto=goto, update={"next": goto})

        if "bull_node" in members_lower:
            goto = _pick("bull_node")
            return Command(goto=goto, update={"next": goto})
        goto = members[0]
        return Command(goto=goto, update={"next": goto})

    return supervisor_node


def risk_supervisor_node(members: list[str], max_turns: int = 3):
    members_lower = [m.lower() for m in members]

    def _pick(name: str) -> str:
        idx = members_lower.index(name)
        return members[idx]

    def supervisor_node(state: AgentState) -> Command[Literal[*members, "__end__"]]:
        debate = state["risk_debate_state"]
        if debate.get("judge_decision") or (debate.get("latest_speaker") or "").lower() == "judge":
            return Command(goto=END, update={"next": "FINISH"})

        debater_count = len([m for m in members_lower if "manager" not in m]) or 1
        if debate.get("count", 0) >= max_turns * debater_count and "risk_manager_node" in members_lower:
            goto = _pick("risk_manager_node")
            return Command(goto=goto, update={"next": goto})

        latest = (debate.get("latest_speaker") or "").lower().strip()
        if latest == "risky" and "safe_node" in members_lower:
            goto = _pick("safe_node")
            return Command(goto=goto, update={"next": goto})
        if latest == "safe" and "neutral_node" in members_lower:
            goto = _pick("neutral_node")
            return Command(goto=goto, update={"next": goto})
        if latest == "neutral" and "risk_node" in members_lower:
            goto = _pick("risk_node")
            return Command(goto=goto, update={"next": goto})

        if "risk_node" in members_lower:
            goto = _pick("risk_node")
            return Command(goto=goto, update={"next": goto})
        goto = members[0]
        return Command(goto=goto, update={"next": goto})

    return supervisor_node

from typing import Annotated, Sequence, List, Dict, Any, Literal
from datetime import date, timedelta, datetime
from typing_extensions import TypedDict, Optional
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage
from tradereact.agents import *
from langgraph.prebuilt import ToolNode
from langgraph.graph import END, StateGraph, START
from langgraph.graph.message import add_messages


# ========== ReAct 相关数据结构 ==========

class ToolCall(TypedDict):
    """工具调用信息"""
    tool_name: Annotated[str, "MCP 工具名称"]  # 例如: "get_stock_data", "get_indicators"
    tool_args: Annotated[Dict[str, Any], "工具调用参数"]  # 例如: {"ticker": "AAPL", "period": "1mo"}
    call_id: Annotated[str, "工具调用的唯一标识符"]  # 用于追踪和调试


class ToolResult(TypedDict):
    """工具执行结果"""
    call_id: Annotated[str, "对应的工具调用 ID"]
    success: Annotated[bool, "工具执行是否成功"]
    result: Annotated[Any, "工具返回的数据"]  # 可以是字符串、字典、列表等
    error: Annotated[Optional[str], "错误信息（如果失败）"]
    execution_time: Annotated[float, "工具执行时间（秒）"]


class ReActStep(TypedDict):
    """
    ReAct 循环中的单个步骤
    遵循 Thought → Action → Observation 范式
    """
    step_id: Annotated[int, "步骤编号，从 1 开始"]
    timestamp: Annotated[datetime, "步骤开始时间"]

    # Thought: Agent 的思考过程
    thought: Annotated[str, "Agent 的推理思考内容"]
    # 例如: "我需要获取 AAPL 的技术指标来分析趋势"

    # Action: Agent 决定执行的动作
    action_type: Annotated[
        Literal["tool_call", "memory_query", "finish"],
        "动作类型"
    ]
    # - "tool_call": 调用外部 MCP 工具
    # - "memory_query": 查询历史记忆
    # - "finish": 完成当前阶段

    action_detail: Annotated[Optional[ToolCall], "工具调用详情（如果是 tool_call）"]
    memory_query: Annotated[Optional[str], "记忆查询内容（如果是 memory_query）"]

    # Observation: 执行结果的观察
    observation: Annotated[str, "对执行结果的观察和总结"]
    # 例如: "RSI 为 65，表明股票接近超买区域"

    tool_result: Annotated[Optional[ToolResult], "工具执行结果详情"]
    memory_result: Annotated[Optional[List[Dict[str, Any]]], "记忆查询结果"]

    # 是否完成当前阶段
    is_final: Annotated[bool, "是否为当前阶段的最后一步"]


class ReActSession(TypedDict):
    """
    一个完整的 ReAct 会话（对应一个阶段）
    """
    session_id: Annotated[str, "会话唯一标识"]
    stage: Annotated[str, "所属阶段（research/trader）"]
    start_time: Annotated[datetime, "会话开始时间"]
    end_time: Annotated[Optional[datetime], "会话结束时间"]
    steps: Annotated[List[ReActStep], "ReAct 步骤列表"]
    total_tool_calls: Annotated[int, "总工具调用次数"]
    success: Annotated[bool, "会话是否成功完成"]
    final_output: Annotated[Optional[str], "最终输出结果"]


# ========== 原有状态定义 ==========

# Researcher team state
class InvestDebateState(TypedDict):
    bull_history: Annotated[
        str, "Bullish Conversation history"
    ]  # Bullish Conversation history, Bull Researcher的所有发言历史
    bear_history: Annotated[
        str, "Bearish Conversation history"
    ]  # Bullish Conversation history, Bear Researcher的所有发言历史
    history: Annotated[str, "Conversation history"]  # Conversation history, 完整的对话历史(交替记录)
    current_response: Annotated[str, "Latest response"]  # Last response, 最新一轮的发言内容
    judge_decision: Annotated[str, "Final judge decision"]  # Last response, Research Manager的裁决结果
    count: Annotated[int, "Length of the current conversation"]  # Conversation length, 对话轮数计数器


# Risk management team state
class RiskDebateState(TypedDict):
    risky_history: Annotated[
        str, "Risky Agent's Conversation history"
    ]  # Conversation history, Risky Analyst的所有发言
    safe_history: Annotated[
        str, "Safe Agent's Conversation history"
    ]  # Conversation history, Safe Analyst的所有发言
    neutral_history: Annotated[
        str, "Neutral Agent's Conversation history"
    ]  # Conversation history, Neutral Analyst的所有发言
    history: Annotated[str, "Conversation history"]  # Conversation history, 完整的三方对话历史
    latest_speaker: Annotated[str, "Analyst that spoke last"]   # 最新发言者 ("Risky" | "Safe" | "Neutral")
    current_risky_response: Annotated[
        str, "Latest response by the risky analyst"
    ]  # Last response
    current_safe_response: Annotated[
        str, "Latest response by the safe analyst"
    ]  # Last response
    current_neutral_response: Annotated[
        str, "Latest response by the neutral analyst"
    ]  # Last response
    judge_decision: Annotated[str, "Judge's decision"]  #Risk Judge的最终裁决
    count: Annotated[int, "Length of the current conversation"]  # Conversation length, 发言计数器


class AgentState(TypedDict, total=False):
    messages: Annotated[Sequence[BaseMessage], add_messages]

    # ========== 基础信息 ==========
    company_of_interest: Annotated[str, "Company that we are interested in trading"]
    trade_date: Annotated[str, "What date we are trading at"]

    sender: Annotated[str, "Agent that sent this message"]

    # research step
    market_report: Annotated[str, "Report from the Market Analyst"]
    sentiment_report: Annotated[str, "Report from the Social Media Analyst"]
    news_report: Annotated[str, "Report from the News Researcher of current world affairs"]
    fundamentals_report: Annotated[str, "Report from the Fundamentals Researcher"]

    # researcher team discussion step
    investment_debate_state: Annotated[
        InvestDebateState, "Current state of the debate on if to invest or not"
    ]
    investment_plan: Annotated[str, "Plan generated by the Analyst"]

    trader_investment_plan: Annotated[str, "Plan generated by the Trader"]

    # risk management team discussion step
    risk_debate_state: Annotated[
        RiskDebateState, "Current state of the debate on evaluating risk"
    ]
    final_trade_decision: Annotated[str, "Final decision made by the Risk Analysts"]

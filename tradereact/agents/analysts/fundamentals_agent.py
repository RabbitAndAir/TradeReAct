from langchain.agents import create_agent
from langchain_core.language_models import BaseChatModel
from langgraph.types import Command

from tradereact.agents.utils.agent_states import AgentState
from tradereact.agents.utils.agent_utils import (
    get_balance_sheet,
    get_cashflow,
    get_fundamentals,
    get_income_statement,
)


def create_fundamentals_analyst(llm: BaseChatModel):
    tools = [
        get_fundamentals,
        get_balance_sheet,
        get_cashflow,
        get_income_statement,
    ]

    def fundamentals_analyst_node(state: AgentState):
        current_date = state["trade_date"]
        ticker = state["company_of_interest"]

        # 构建完整的系统提示，结构清晰且流畅
        system_prompt = (
            "You are a helpful AI assistant collaborating with other assistants. "
            "Use the provided tools to progress towards answering the question. "
            "If you are unable to fully answer, that's OK; another assistant with different tools "
            "will help where you left off. Execute what you can to make progress. "
            "If you or any other assistant has the FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL** or deliverable, "
            "prefix your response with FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL** so the team knows to stop.\n\n"

            f"You have access to the following tools: {', '.join([tool.name for tool in tools])}.\n\n"

            "Your specific role is as a financial researcher tasked with analyzing fundamental information "
            "about a company over the past week. Write a comprehensive report covering the company's fundamental "
            "information including financial documents, company profile, basic financials, and financial history "
            "to provide traders with a complete view for informed decision-making. "
            "Include as much detail as possible with fine-grained analysis and actionable insights. "
            "Do not simply state that trends are mixed—provide detailed, nuanced analysis that helps traders make decisions.\n\n"

            "Report Requirements:\n"
            "- Use `get_fundamentals` for comprehensive company analysis\n"
            "- Use `get_balance_sheet`, `get_cashflow`, and `get_income_statement` for specific financial statements\n"
            "- Append a well-organized Markdown table at the end summarizing key points\n\n"

            f"Context: Current date is {current_date}. Company to analyze: {ticker}."
        )

        # 使用完整的系统提示创建 agent
        agent = create_agent(model=llm, tools=tools, system_prompt=system_prompt)

        # 传入明确的用户任务（使用字典格式）
        inputs = {"messages": [{"role": "user", "content": f"Analyze the fundamental information for {ticker} as of {current_date}."}]}
        # noinspection PyTypeChecker
        result = agent.invoke(inputs)
        last_message = result["messages"][-1]
        report = getattr(last_message, "content", str(last_message))
        return Command(
            goto="analyst_supervisor",
            update={
                "fundamentals_report": report,
                "sender": "fundamentals_node",
            }
        )
    return fundamentals_analyst_node

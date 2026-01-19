from langchain.agents import create_agent
from langchain_core.language_models import BaseChatModel
from langgraph.types import Command

from tradereact.agents.utils.agent_states import AgentState
from tradereact.agents.utils.agent_utils import get_global_news, get_news
from tradereact.agents.utils.mcp_loader import load_analyst_tools


def create_news_analyst(llm: BaseChatModel):
    # Define custom tools
    custom_tools = [get_news, get_global_news]

    # Load custom tools + MCP tools (if configured)
    tools = load_analyst_tools("news_analyst", custom_tools)

    def news_analyst_node(state: AgentState):
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

            "Your specific role is as a news researcher tasked with analyzing recent news and trends over the past week. "
            "Write a comprehensive report of the current state of the world that is relevant for trading and macroeconomics. "
            "Use get_news(query, start_date, end_date) for company-specific or targeted news searches, "
            "and get_global_news(curr_date, look_back_days, limit) for broader macroeconomic news. "
            "Do not simply state the trends are mixed—provide detailed and finegrained analysis and insights that may help traders make decisions.\n"
            "Make sure to append a Markdown table at the end of the report to organize key points in the report, organized and easy to read.\n\n"

            f"Context: Current date is {current_date}. Company to analyze: {ticker}."
        )

        # 使用完整的系统提示创建 agent
        agent = create_agent(model=llm, tools=tools, system_prompt=system_prompt)

        # 传入明确的用户任务（使用字典格式）
        inputs = {"messages": [{"role": "user", "content": f"Analyze recent news and trends for {ticker} and broader macroeconomic news as of {current_date}."}]}
        # noinspection PyTypeChecker
        result = agent.invoke(inputs)
        last_message = result["messages"][-1]
        report = getattr(last_message, "content", str(last_message))

        return Command(
            goto="analyst_supervisor",
            update={"news_report": report}
        )

    return news_analyst_node

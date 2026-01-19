from langchain.agents import create_agent
from langchain_core.language_models import BaseChatModel
from langgraph.types import Command

from tradereact.agents.utils.agent_states import AgentState
from tradereact.agents.utils.agent_utils import get_news
from tradereact.agents.utils.mcp_loader import load_analyst_tools


def create_social_media_analyst(llm: BaseChatModel):
    # Define custom tools
    custom_tools = [get_news]

    # Load custom tools + MCP tools (if configured)
    tools = load_analyst_tools("social_media_analyst", custom_tools)

    def social_media_analyst_node(state: AgentState):
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

            "Your specific role is as a social media and company-specific news researcher/analyst tasked with analyzing "
            "social media posts, recent company news, and public sentiment for a specific company over the past week. "
            "You will be given a company's name and your objective is to write a comprehensive long report detailing your analysis, "
            "insights, and implications for traders and investors on this company's current state after looking at social media "
            "and what people are saying about that company, analyzing sentiment data of what people feel each day about the company, "
            "and looking at recent company news.\n\n"

            "Use the get_news(query, start_date, end_date) tool to search for company-specific news and social media discussions. "
            "Try to look at all sources possible from social media to sentiment to news. "
            "Do not simply state the trends are mixed—provide detailed and finegrained analysis and insights that may help traders make decisions.\n"
            "Make sure to append a Markdown table at the end of the report to organize key points in the report, organized and easy to read.\n\n"

            f"Context: Current date is {current_date}. Company to analyze: {ticker}."
        )

        # 使用完整的系统提示创建 agent
        agent = create_agent(model=llm, tools=tools, system_prompt=system_prompt)

        # 传入明确的用户任务（使用字典格式）
        inputs = {"messages": [{"role": "user", "content": f"Analyze social media posts, public sentiment, and recent news for {ticker} as of {current_date}."}]}
        # noinspection PyTypeChecker
        result = agent.invoke(inputs)
        last_message = result["messages"][-1]
        report = getattr(last_message, "content", str(last_message))

        return Command(
            goto="analyst_supervisor",
            update={"sentiment_report": report}
        )

    return social_media_analyst_node

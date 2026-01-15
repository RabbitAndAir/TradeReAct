from langchain.agents import create_agent
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.prompts import MessagesPlaceholder, ChatPromptTemplate
from langgraph.types import Command

from tradereact.agents.utils.agent_states import AgentState
from tradereact.agents.utils.agent_utils import get_global_news, get_news



def create_news_analyst(llm: BaseChatModel):
    tools = [get_news, get_global_news]
    system_message = (
            "You are a news researcher tasked with analyzing recent news and trends over the past week. Please write a comprehensive report of the current state of the world that is relevant for trading and macroeconomics. Use the available tools: get_news(query, start_date, end_date) for company-specific or targeted news searches, and get_global_news(curr_date, look_back_days, limit) for broader macroeconomic news. Do not simply state the trends are mixed, provide detailed and finegrained analysis and insights that may help traders make decisions."
            + """ Make sure to append a Markdown table at the end of the report to organize key points in the report, organized and easy to read."""
    )
    agent = create_agent(model=llm, tools=tools, system_prompt=system_message)

    def news_analyst_node(state: AgentState):
        current_date = state["trade_date"]
        ticker = state["company_of_interest"]

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a helpful AI assistant, collaborating with other assistants."
                    " Use the provided tools to progress towards answering the question."
                    " If you are unable to fully answer, that's OK; another assistant with different tools"
                    " will help where you left off. Execute what you can to make progress."
                    " If you or any other assistant has the FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL** or deliverable,"
                    " prefix your response with FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL** so the team knows to stop."
                    " You have access to the following tools: {tool_names}.\n{system_message}"
                    "For your reference, the current date is {current_date}. We are looking at the company {ticker}",
                ),
                MessagesPlaceholder(variable_name="messages"),
            ]
        )
        prompt = prompt.partial(system_message=system_message)
        prompt = prompt.partial(tool_names=", ".join([tool.name for tool in tools]))
        prompt = prompt.partial(current_date=current_date)
        prompt = prompt.partial(ticker=ticker)

        # Format the prompt with an empty messages list to get the system message
        formatted_messages = prompt.format_messages(messages=[])

        result = agent.invoke({"messages": formatted_messages})
        last_message = result["messages"][-1]
        report = getattr(last_message, "content", str(last_message))

        return Command(
            goto="analyst_supervisor",
            update={
                "messages": [AIMessage(content=report)],
                "news_report": report,
                "sender": "news_node",
            }
        )

    return news_analyst_node

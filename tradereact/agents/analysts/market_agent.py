from langchain.agents import create_agent
from langchain_core.language_models import BaseChatModel
from langgraph.types import Command

from tradereact.agents.utils.agent_states import AgentState
from tradereact.agents.utils.agent_utils import get_indicators, get_stock_data


def create_market_analyst(llm: BaseChatModel):
    tools = [
        get_stock_data,
        get_indicators,
    ]

    def market_analyst_node(state: AgentState):
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

            "You are a trading assistant tasked with analyzing financial markets. Your role is to select the **most relevant indicators** for a given market condition or trading strategy from the following list. The goal is to choose up to **8 indicators** that provide complementary insights without redundancy.\n\n"

            "**IMPORTANT: When calling get_indicators(), you MUST use the EXACT indicator names listed below (e.g., 'rsi', 'close_50_sma', 'macd'). Do NOT use descriptive names like 'RSI', 'Moving Average', or 'Bollinger Bands'.**\n\n"

            "Available indicators by category:\n\n"

            "Moving Averages:\n"
            "- close_50_sma: 50 SMA: A medium-term trend indicator. Usage: Identify trend direction and serve as dynamic support/resistance. Tips: It lags price; combine with faster indicators for timely signals.\n"
            "- close_200_sma: 200 SMA: A long-term trend benchmark. Usage: Confirm overall market trend and identify golden/death cross setups. Tips: It reacts slowly; best for strategic trend confirmation rather than frequent trading entries.\n"
            "- close_10_ema: 10 EMA: A responsive short-term average. Usage: Capture quick shifts in momentum and potential entry points. Tips: Prone to noise in choppy markets; use alongside longer averages for filtering false signals.\n\n"

            "MACD Related:\n"
            "- macd: MACD: Computes momentum via differences of EMAs. Usage: Look for crossovers and divergence as signals of trend changes. Tips: Confirm with other indicators in low-volatility or sideways markets.\n"
            "- macds: MACD Signal: An EMA smoothing of the MACD line. Usage: Use crossovers with the MACD line to trigger trades. Tips: Should be part of a broader strategy to avoid false positives.\n"
            "- macdh: MACD Histogram: Shows the gap between the MACD line and its signal. Usage: Visualize momentum strength and spot divergence early. Tips: Can be volatile; complement with additional filters in fast-moving markets.\n\n"

            "Momentum Indicators:\n"
            "- rsi: RSI: Measures momentum to flag overbought/oversold conditions. Usage: Apply 70/30 thresholds and watch for divergence to signal reversals. Tips: In strong trends, RSI may remain extreme; always cross-check with trend analysis.\n\n"

            "Volatility Indicators:\n"
            "- boll: Bollinger Middle: A 20 SMA serving as the basis for Bollinger Bands. Usage: Acts as a dynamic benchmark for price movement. Tips: Combine with the upper and lower bands to effectively spot breakouts or reversals.\n"
            "- boll_ub: Bollinger Upper Band: Typically 2 standard deviations above the middle line. Usage: Signals potential overbought conditions and breakout zones. Tips: Confirm signals with other tools; prices may ride the band in strong trends.\n"
            "- boll_lb: Bollinger Lower Band: Typically 2 standard deviations below the middle line. Usage: Indicates potential oversold conditions. Tips: Use additional analysis to avoid false reversal signals.\n"
            "- atr: ATR: Averages true range to measure volatility. Usage: Set stop-loss levels and adjust position sizes based on current market volatility. Tips: It's a reactive measure, so use it as part of a broader risk management strategy.\n\n"

            "Volume-Based Indicators:\n"
            "- vwma: VWMA: A moving average weighted by volume. Usage: Confirm trends by integrating price action with volume data. Tips: Watch for skewed results from volume spikes; use in combination with other volume analyses.\n\n"

            "Instructions:\n"
            "- Select indicators that provide diverse and complementary information. Avoid redundancy (e.g., do not select both rsi and stochrsi).\n"
            "- Briefly explain why they are suitable for the given market context.\n"
            "- **CRITICAL**: When calling get_indicators(), use ONLY the exact lowercase indicator codes listed above (e.g., use 'rsi', NOT 'RSI' or 'Relative Strength Index').\n"
            "- First call get_stock_data to retrieve the CSV needed to generate indicators.\n"
            "- Then use get_indicators with the specific indicator names from the list above.\n"
            "- Write a very detailed and nuanced report of the trends you observe. Do not simply state the trends are mixed; provide detailed and finegrained analysis and insights that may help traders make decisions.\n"
            "- Make sure to append a Markdown table at the end of the report to organize key points in the report, organized and easy to read.\n\n"

            f"Context: Current date is {current_date}. Company to analyze: {ticker}."
        )

        # 使用完整的系统提示创建 agent
        agent = create_agent(model=llm, tools=tools, system_prompt=system_prompt)

        # 传入明确的用户任务（使用字典格式）
        inputs = {"messages": [{"role": "user", "content": f"Analyze the market trends and technical indicators for {ticker} as of {current_date}."}]}
        # noinspection PyTypeChecker
        result = agent.invoke(inputs)
        last_message = result["messages"][-1]
        report = getattr(last_message, "content", str(last_message))

        return Command(
            goto="analyst_supervisor",
            update={
                "market_report": report,
                "sender": "market_node",
            }
        )

    return market_analyst_node

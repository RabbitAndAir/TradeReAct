import json
import os
from pathlib import Path
from typing import Dict, Any
from langchain_openai import ChatOpenAI

try:
    from langchain_anthropic import ChatAnthropic
except Exception:
    ChatAnthropic = None

try:
    from langchain_google_genai import ChatGoogleGenerativeAI
except Exception:
    ChatGoogleGenerativeAI = None
from sklearn import set_config
from tradereact.agents.utils.memory import FinancialSituationMemory
from tradereact.default_config import DEFAULT_CONFIG
from tradereact.graph.propagation import Propagator
from tradereact.graph.reflection import Reflector
from tradereact.graph.setup import GraphSetup
from tradereact.graph.signal_processing import SignalProcessor


class TradingAgentsGraph:
    """Main class that orchestrates the trading agents framework."""

    def __init__(
        self,
        debug=False,
        config: Dict[str, Any] = None,
    ):
        """Initialize the trading agents graph and components.

                Args:
                    selected_analysts: List of analyst types to include
                    debug: Whether to run in debug mode
                    config: Configuration dictionary. If None, uses default config
                """
        self.debug = debug
        self.config = config or DEFAULT_CONFIG

        # Update the interface's config
        set_config(self.config)

        # Create necessary directories
        os.makedirs(
            os.path.join(self.config["project_dir"], "dataflows/data_cache"),
            exist_ok=True,
        )

        # Initialize LLMs
        if self.config["llm_provider"].lower() == "openai" or self.config["llm_provider"] == "ollama" or self.config[
            "llm_provider"] == "openrouter":
            backend_url = (self.config.get("backend_url") or "").strip().strip("`").strip()
            openai_kwargs: Dict[str, Any] = {"model": self.config["deep_think_llm"]}
            if backend_url:
                openai_kwargs["base_url"] = backend_url
            self.deep_thinking_llm = ChatOpenAI(**openai_kwargs)

            openai_kwargs = {"model": self.config["quick_think_llm"]}
            if backend_url:
                openai_kwargs["base_url"] = backend_url
            self.quick_thinking_llm = ChatOpenAI(**openai_kwargs)
        elif self.config["llm_provider"].lower() == "anthropic":
            if ChatAnthropic is None:
                raise ModuleNotFoundError(
                    "langchain_anthropic is required when llm_provider='anthropic'"
                )
            backend_url = (self.config.get("backend_url") or "").strip().strip("`").strip()
            anthropic_kwargs: Dict[str, Any] = {"model": self.config["deep_think_llm"]}
            if backend_url:
                anthropic_kwargs["base_url"] = backend_url
            self.deep_thinking_llm = ChatAnthropic(**anthropic_kwargs)

            anthropic_kwargs = {"model": self.config["quick_think_llm"]}
            if backend_url:
                anthropic_kwargs["base_url"] = backend_url
            self.quick_thinking_llm = ChatAnthropic(**anthropic_kwargs)
        elif self.config["llm_provider"].lower() == "google":
            if ChatGoogleGenerativeAI is None:
                raise ModuleNotFoundError(
                    "langchain_google_genai is required when llm_provider='google'"
                )
            self.deep_thinking_llm = ChatGoogleGenerativeAI(model=self.config["deep_think_llm"])
            self.quick_thinking_llm = ChatGoogleGenerativeAI(model=self.config["quick_think_llm"])
        else:
            raise ValueError(f"Unsupported LLM provider: {self.config['llm_provider']}")

        # Initialize memories
        self.bull_memory = FinancialSituationMemory("bull_memory", self.config)
        self.bear_memory = FinancialSituationMemory("bear_memory", self.config)
        self.trader_memory = FinancialSituationMemory("trader_memory", self.config)
        self.invest_judge_memory = FinancialSituationMemory("invest_judge_memory", self.config)
        self.risk_manager_memory = FinancialSituationMemory("risk_manager_memory", self.config)

        self.graph_setup = GraphSetup(
            self.quick_thinking_llm,
            self.deep_thinking_llm,
            self.bull_memory,
            self.bear_memory,
            self.trader_memory,
            self.invest_judge_memory,
            self.risk_manager_memory,
        )

        self.propagator = Propagator(
            max_recur_limit=int(self.config.get("max_recur_limit", 100)),
        )
        self.reflector = Reflector(self.quick_thinking_llm)
        self.signal_processor = SignalProcessor(self.quick_thinking_llm)

        # State tracking
        self.curr_state = None  # 用于存储当前状态
        self.ticker = None  # 正在分析的股票
        self.log_states_dict = {}  # date to full state dict 历史状态日志

        # Set up the graph
        self.graph = self.graph_setup.setup_graph(
            max_debate_rounds=int(self.config.get("max_debate_rounds", 3)),
            max_risk_discuss_rounds=int(self.config.get("max_risk_discuss_rounds", 3)),
        )

    def propagate(self, company_name, trade_date):
        """Run the trading agents graph for a company on a specific date."""

        self.ticker = company_name

        # Initialize state
        init_agent_state = self.propagator.create_initial_state(
            company_name, trade_date
        )
        args = self.propagator.get_graph_args()

        if self.debug:
            # Debug mode with tracing
            trace = []
            for chunk in self.graph.stream(init_agent_state, **args):
                if len(chunk["messages"]) == 0:
                    pass
                else:
                    chunk["messages"][-1].pretty_print()
                    trace.append(chunk)

            final_state = trace[-1]
        else:
            # Standard mode without tracing
            final_state = self.graph.invoke(init_agent_state, **args)

        # Store current state for reflection
        self.curr_state = final_state

        # Log state
        self._log_state(trade_date, final_state)

        # Return decision and processed signal
        return final_state, self.process_signal(final_state["final_trade_decision"])


    def process_signal(self, full_signal):
        """Process a signal to extract the core decision."""
        return self.signal_processor.process_signal(full_signal)

    def _log_state(self, trade_date, final_state):
        """Log the final state to a JSON file."""
        self.log_states_dict[str(trade_date)] = {
            "company_of_interest": final_state["company_of_interest"],
            "trade_date": final_state["trade_date"],
            "market_report": final_state["market_report"],
            "sentiment_report": final_state["sentiment_report"],
            "news_report": final_state["news_report"],
            "fundamentals_report": final_state["fundamentals_report"],
            "investment_debate_state": {
                "bull_history": final_state["investment_debate_state"]["bull_history"],
                "bear_history": final_state["investment_debate_state"]["bear_history"],
                "history": final_state["investment_debate_state"]["history"],
                "current_response": final_state["investment_debate_state"][
                    "current_response"
                ],
                "judge_decision": final_state["investment_debate_state"][
                    "judge_decision"
                ],
            },
            "trader_investment_decision": final_state["trader_investment_plan"],
            "risk_debate_state": {
                "risky_history": final_state["risk_debate_state"]["risky_history"],
                "safe_history": final_state["risk_debate_state"]["safe_history"],
                "neutral_history": final_state["risk_debate_state"]["neutral_history"],
                "history": final_state["risk_debate_state"]["history"],
                "judge_decision": final_state["risk_debate_state"]["judge_decision"],
            },
            "investment_plan": final_state["investment_plan"],
            "final_trade_decision": final_state["final_trade_decision"],
        }

        # Save to file
        directory = Path(f"eval_results/{self.ticker}/TradingAgentsStrategy_logs/")
        directory.mkdir(parents=True, exist_ok=True)

        with open(
            f"eval_results/{self.ticker}/TradingAgentsStrategy_logs/full_states_log_{trade_date}.json",
            "w",
        ) as f:
            json.dump(self.log_states_dict, f, indent=4)

    def reflect_and_remember(self, returns_losses):
        """Reflect on decisions and update memory based on returns."""
        self.reflector.reflect_bull_researcher(
            self.curr_state, returns_losses, self.bull_memory
        )
        self.reflector.reflect_bear_researcher(
            self.curr_state, returns_losses, self.bear_memory
        )
        self.reflector.reflect_trader(
            self.curr_state, returns_losses, self.trader_memory
        )
        self.reflector.reflect_invest_judge(
            self.curr_state, returns_losses, self.invest_judge_memory
        )
        self.reflector.reflect_risk_manager(
            self.curr_state, returns_losses, self.risk_manager_memory
        )

"""
MCP Tool Loader Utility

Loads MCP tools for analysts based on configuration in mcp_config.json
"""
import asyncio
import json
import os
from pathlib import Path
from typing import List, Optional
from langchain_core.tools import BaseTool

try:
    from langchain_mcp_adapters.client import MultiServerMCPClient
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False


class MCPToolLoader:
    """Load MCP tools for analysts based on configuration"""

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize MCP tool loader

        Args:
            config_path: Path to mcp_config.json. If None, uses default location.
        """
        if config_path is None:
            # Default to project root mcp_config.json
            # __file__ is in tradereact/agents/utils/mcp_loader.py
            # Go up 3 levels to reach project root
            project_root = Path(__file__).parent.parent.parent.parent
            config_path = project_root / "mcp_config.json"

        self.config_path = Path(config_path)
        self.config = self._load_config()
        self.mcp_client = None

    def _load_config(self) -> dict:
        """Load MCP configuration from JSON file"""
        if not self.config_path.exists():
            print(f"MCP config not found at {self.config_path}, using empty config")
            return {"mcpServers": {}}

        try:
            with open(self.config_path, 'r') as f:
                config = json.load(f)
                return config
        except Exception as e:
            print(f"Error loading MCP config: {e}")
            return {"mcpServers": {}}

    def _expand_env_vars(self, config: dict) -> dict:
        """Expand environment variables in config (e.g., ${VAR_NAME})"""
        import re

        def expand_value(value):
            if isinstance(value, str):
                # Replace ${VAR_NAME} with environment variable value
                pattern = r'\$\{([^}]+)\}'
                matches = re.findall(pattern, value)
                for var_name in matches:
                    env_value = os.getenv(var_name, "")
                    value = value.replace(f"${{{var_name}}}", env_value)
                return value
            elif isinstance(value, dict):
                return {k: expand_value(v) for k, v in value.items()}
            elif isinstance(value, list):
                return [expand_value(item) for item in value]
            return value

        return expand_value(config)

    def _normalize_server_config(self, servers: dict) -> dict:
        """
        Normalize server configuration to match MultiServerMCPClient format

        Adds missing 'transport' field and ensures correct structure
        """
        normalized = {}

        for server_name, server_config in servers.items():
            # Copy config
            config = server_config.copy()

            # Add transport field if missing
            if "transport" not in config:
                # Determine transport type
                if "url" in config:
                    config["transport"] = "streamable_http"
                elif "command" in config:
                    config["transport"] = "stdio"
                else:
                    print(f"Warning: Cannot determine transport for server '{server_name}', skipping")
                    continue

            normalized[server_name] = config

        return normalized

    def load_tools_for_analyst(self, analyst_name: str) -> List[BaseTool]:
        """
        Load MCP tools for a specific analyst

        Args:
            analyst_name: Name of analyst (e.g., "market_analyst", "news_analyst")

        Returns:
            List of LangChain tools from MCP servers, or empty list if disabled/unavailable
        """
        if not MCP_AVAILABLE:
            print(f"langchain_mcp_adapters not installed, skipping MCP tools for {analyst_name}")
            return []

        analyst_config = self.config.get("mcpServers", {}).get(analyst_name, {})

        # Check if enabled
        if not analyst_config.get("enabled", False):
            print(f"MCP tools disabled for {analyst_name}")
            return []

        servers = analyst_config.get("servers", {})
        if not servers:
            print(f"No MCP servers configured for {analyst_name}")
            return []

        try:
            # Expand environment variables in server configs
            expanded_servers = self._expand_env_vars(servers)

            # Normalize server configuration (add transport field)
            normalized_servers = self._normalize_server_config(expanded_servers)

            if not normalized_servers:
                print(f"No valid MCP server configurations for {analyst_name}")
                return []

            # Create MultiServerMCPClient with configured servers
            self.mcp_client = MultiServerMCPClient(normalized_servers)

            # Get tools from all configured servers (async operation)
            # Use asyncio to run the async method
            try:
                # Try to get current event loop
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # If already in an event loop, create a new task
                    import nest_asyncio
                    nest_asyncio.apply()
                    tools = loop.run_until_complete(self.mcp_client.get_tools())
                else:
                    # Run in new event loop
                    tools = asyncio.run(self.mcp_client.get_tools())
            except RuntimeError:
                # No event loop, create one
                tools = asyncio.run(self.mcp_client.get_tools())

            print(f"Loaded {len(tools)} MCP tools for {analyst_name}")
            return tools

        except Exception as e:
            print(f"Error loading MCP tools for {analyst_name}: {e}")
            import traceback
            traceback.print_exc()
            return []

    def is_enabled(self, analyst_name: str) -> bool:
        """
        Check if MCP tools are enabled for a specific analyst

        Args:
            analyst_name: Name of analyst

        Returns:
            True if MCP is enabled for this analyst
        """
        analyst_config = self.config.get("mcpServers", {}).get(analyst_name, {})
        return analyst_config.get("enabled", False)

    def get_analyst_names(self) -> List[str]:
        """Get list of all analyst names in config"""
        return list(self.config.get("mcpServers", {}).keys())

    def close(self):
        """Close MCP client connection"""
        if self.mcp_client is not None:
            try:
                self.mcp_client.close()
            except Exception as e:
                print(f"Error closing MCP client: {e}")


# Singleton instance for reuse
_mcp_loader_instance = None


def get_mcp_loader(config_path: Optional[str] = None) -> MCPToolLoader:
    """
    Get singleton MCP tool loader instance

    Args:
        config_path: Optional custom config path

    Returns:
        MCPToolLoader instance
    """
    global _mcp_loader_instance
    if _mcp_loader_instance is None:
        _mcp_loader_instance = MCPToolLoader(config_path)
    return _mcp_loader_instance


def load_analyst_tools(analyst_name: str, custom_tools: List[BaseTool]) -> List[BaseTool]:
    """
    Convenience function to load custom tools + MCP tools for an analyst

    Args:
        analyst_name: Name of analyst
        custom_tools: List of custom tools

    Returns:
        Combined list of custom tools + MCP tools
    """
    loader = get_mcp_loader()
    mcp_tools = loader.load_tools_for_analyst(analyst_name)

    # Combine custom tools with MCP tools
    all_tools = custom_tools.copy()

    # Check for name conflicts
    custom_tool_names = {tool.name for tool in custom_tools}
    for mcp_tool in mcp_tools:
        if mcp_tool.name in custom_tool_names:
            print(f"Warning: MCP tool '{mcp_tool.name}' conflicts with custom tool, skipping")
            continue
        all_tools.append(mcp_tool)

    return all_tools


if __name__ == "__main__":
    # Test the loader
    loader = MCPToolLoader()

    print("Available analysts:", loader.get_analyst_names())
    print()

    for analyst in loader.get_analyst_names():
        enabled = loader.is_enabled(analyst)
        print(f"{analyst}: {'ENABLED' if enabled else 'DISABLED'}")

        if enabled:
            tools = loader.load_tools_for_analyst(analyst)
            print(f"  Tools loaded: {len(tools)}")
            for tool in tools:
                print(f"    - {tool.name}: {tool.description[:60]}...")
        print()

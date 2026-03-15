"""
MoltsPay Tool for LlamaIndex

Pay for AI services using USDC (gasless) via the x402 protocol.
Enables AI agents to autonomously purchase services from other agents.
"""

import os
from typing import Optional, List

from llama_index.core.tools.tool_spec.base import BaseToolSpec


class MoltsPayToolSpec(BaseToolSpec):
    """
    MoltsPay Tool Spec - Pay for AI services with crypto.
    
    This tool allows LlamaIndex agents to:
    - Discover AI services that accept MoltsPay
    - Pay for services using USDC (gasless, no ETH needed)
    - Execute paid AI services (video generation, image processing, etc.)
    
    The x402 protocol ensures pay-for-success: payment only completes if 
    the service delivers results.
    
    Setup:
        1. Install: pip install moltspay
        2. Create wallet: MoltsPay will auto-create on first use
        3. Fund wallet: Use fund() to get onramp URL
    
    Example:
        ```python
        from llama_index.tools.moltspay import MoltsPayToolSpec
        from llama_index.core.agent import FunctionAgent
        from llama_index.llms.openai import OpenAI
        
        tools = MoltsPayToolSpec().to_tool_list()
        agent = FunctionAgent(
            tools=tools,
            llm=OpenAI(model="gpt-4o-mini")
        )
        
        response = await agent.run(
            user_msg="Generate a video of a cat dancing using Zen7"
        )
        ```
    """
    
    spec_functions = [
        "pay_service",
        "discover_services", 
        "get_balance",
        "fund_wallet",
        "get_limits",
        "set_limits"
    ]
    
    def __init__(self, wallet_path: Optional[str] = None, chain: str = "base"):
        """
        Initialize MoltsPayToolSpec.
        
        Args:
            wallet_path: Path to MoltsPay wallet JSON file.
                        Defaults to ~/.moltspay/wallet.json
            chain: Blockchain to use (base, polygon, ethereum). Default: base
        """
        self._wallet_path = wallet_path or os.environ.get(
            "MOLTSPAY_WALLET_PATH",
            os.path.expanduser("~/.moltspay/wallet.json")
        )
        self._chain = chain
        self._client = None
    
    def _get_client(self):
        """Lazy-load MoltsPay client."""
        if self._client is None:
            try:
                from moltspay import MoltsPay
                self._client = MoltsPay(
                    wallet_path=self._wallet_path,
                    chain=self._chain
                )
            except ImportError:
                raise ImportError(
                    "MoltsPay is required. Install with: pip install moltspay"
                )
        return self._client
    
    def pay_service(
        self,
        service_url: str,
        service_id: str,
        prompt: Optional[str] = None,
        image_url: Optional[str] = None,
    ) -> str:
        """
        Pay for and execute an AI service.
        
        Uses the x402 protocol to pay with USDC. Payment only settles
        if the service successfully delivers results.
        
        Args:
            service_url: Service provider URL (e.g., "https://juai8.com/zen7")
            service_id: Service identifier (e.g., "text-to-video", "image-to-video")
            prompt: Text prompt for the service
            image_url: URL to image for image-based services
            
        Returns:
            Service result (usually a URL to the generated content)
            
        Example:
            >>> result = tool.pay_service(
            ...     service_url="https://juai8.com/zen7",
            ...     service_id="text-to-video",
            ...     prompt="A dragon flying over mountains"
            ... )
        """
        try:
            client = self._get_client()
            
            # Build params based on what's provided
            params = {}
            if prompt:
                params["prompt"] = prompt
            if image_url:
                params["image_url"] = image_url
            
            result = client.pay(
                service_url=service_url,
                service_id=service_id,
                **params
            )
            
            # Return the result URL or data
            if hasattr(result, 'url'):
                return f"Success! Result: {result.url}"
            elif hasattr(result, 'data'):
                return f"Success! Data: {result.data}"
            else:
                return str(result)
            
        except Exception as e:
            return f"Payment error: {str(e)}"
    
    def discover_services(self, service_url: str) -> str:
        """
        Discover available services from a MoltsPay provider.
        
        Args:
            service_url: Service provider URL (e.g., "https://juai8.com/zen7")
            
        Returns:
            List of available services with prices
            
        Example:
            >>> tool.discover_services("https://juai8.com/zen7")
        """
        try:
            client = self._get_client()
            services = client.discover(service_url)
            
            result = f"Available services from {service_url}:\n\n"
            for svc in services:
                result += f"- {svc.id}: {svc.name}\n"
                result += f"  Price: ${svc.price} {svc.currency}\n"
                if svc.description:
                    result += f"  Description: {svc.description}\n"
                result += "\n"
            
            return result
            
        except Exception as e:
            return f"Error discovering services: {str(e)}"
    
    def get_balance(self) -> str:
        """
        Get current wallet balance.
        
        Returns:
            Wallet address and USDC balance on each chain
            
        Example:
            >>> tool.get_balance()
        """
        try:
            client = self._get_client()
            balance = client.balance()
            
            result = f"Wallet: {client.address}\n\n"
            result += "Balances:\n"
            if hasattr(balance, 'usdc'):
                result += f"  USDC: ${balance.usdc}\n"
            if hasattr(balance, 'by_chain'):
                for chain, amount in balance.by_chain.items():
                    result += f"  {chain}: ${amount} USDC\n"
            
            return result
            
        except Exception as e:
            return f"Error getting balance: {str(e)}"
    
    def fund_wallet(self, amount: float = 10.0) -> str:
        """
        Get a link to fund the wallet with USDC.
        
        Opens an onramp to purchase USDC directly with card or bank transfer.
        
        Args:
            amount: Amount in USD to fund (default: $10)
            
        Returns:
            Funding URL and instructions
            
        Example:
            >>> tool.fund_wallet(20.0)
        """
        try:
            client = self._get_client()
            result = client.fund(amount=amount)
            
            output = f"Fund your wallet with ${amount} USDC:\n\n"
            output += f"Wallet Address: {client.address}\n"
            output += f"Chain: {self._chain}\n\n"
            
            if hasattr(result, 'url') and result.url:
                output += f"Onramp URL: {result.url}\n"
                output += "\nClick the link to purchase USDC with card/bank transfer.\n"
            else:
                output += f"Send USDC on {self._chain} to: {client.address}\n"
            
            return output
            
        except Exception as e:
            return f"Error generating funding link: {str(e)}"
    
    def get_limits(self) -> str:
        """
        Get current spending limits.
        
        Returns:
            Current max per transaction and daily limits
            
        Example:
            >>> tool.get_limits()
        """
        try:
            client = self._get_client()
            limits = client.limits()
            
            result = "Spending Limits:\n"
            result += f"  Max per transaction: ${limits.max_per_tx}\n"
            result += f"  Max per day: ${limits.max_per_day}\n"
            if hasattr(limits, 'spent_today'):
                result += f"  Spent today: ${limits.spent_today}\n"
            
            return result
            
        except Exception as e:
            return f"Error getting limits: {str(e)}"
    
    def set_limits(
        self,
        max_per_tx: Optional[float] = None,
        max_per_day: Optional[float] = None
    ) -> str:
        """
        Update spending limits.
        
        Args:
            max_per_tx: Maximum amount per transaction in USD
            max_per_day: Maximum amount per day in USD
            
        Returns:
            Confirmation of new limits
            
        Example:
            >>> tool.set_limits(max_per_tx=5.0, max_per_day=50.0)
        """
        try:
            client = self._get_client()
            client.set_limits(max_per_tx=max_per_tx, max_per_day=max_per_day)
            
            # Get updated limits
            limits = client.limits()
            
            result = "Limits updated!\n"
            result += f"  Max per transaction: ${limits.max_per_tx}\n"
            result += f"  Max per day: ${limits.max_per_day}\n"
            
            return result
            
        except Exception as e:
            return f"Error setting limits: {str(e)}"

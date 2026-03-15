"""
MoltsPay Tool for LlamaIndex

Pay for AI services using USDC (gasless) via the x402 protocol.
Enables AI agents to autonomously purchase services from other agents.
"""

import os
from typing import Optional

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
        2. Initialize wallet: npx moltspay init --chain base
        3. Fund wallet: npx moltspay fund
    
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
    
    spec_functions = ["pay_service", "get_services", "get_balance"]
    
    def __init__(self, wallet_path: Optional[str] = None):
        """
        Initialize MoltsPayToolSpec.
        
        Args:
            wallet_path: Path to MoltsPay wallet JSON file.
                        Defaults to ~/.moltspay/wallet.json
        """
        self._wallet_path = wallet_path or os.environ.get(
            "MOLTSPAY_WALLET_PATH",
            os.path.expanduser("~/.moltspay/wallet.json")
        )
        self._client = None
    
    def _get_client(self):
        """Lazy-load MoltsPay client."""
        if self._client is None:
            try:
                from moltspay import MoltsPay
                self._client = MoltsPay(wallet_path=self._wallet_path)
            except ImportError:
                raise ImportError(
                    "MoltsPay is required. Install with: pip install moltspay"
                )
        return self._client
    
    def pay_service(
        self,
        provider_url: str,
        service_id: str,
        prompt: Optional[str] = None,
        image_path: Optional[str] = None,
    ) -> str:
        """
        Pay for and execute an AI service.
        
        Uses the x402 protocol to pay with USDC. Payment only settles
        if the service successfully delivers results.
        
        Args:
            provider_url: Service provider URL (e.g., "https://juai8.com/zen7")
            service_id: Service identifier (e.g., "text-to-video")
            prompt: Text prompt for the service
            image_path: Path to image file for image-based services
            
        Returns:
            Service result (usually a URL to the generated content)
            
        Example:
            >>> result = tool.pay_service(
            ...     provider_url="https://juai8.com/zen7",
            ...     service_id="text-to-video",
            ...     prompt="A dragon flying over mountains"
            ... )
        """
        try:
            client = self._get_client()
            
            params = {}
            if prompt:
                params["prompt"] = prompt
            if image_path:
                params["image"] = image_path
            
            result = client.x402(
                url=f"{provider_url.rstrip('/')}/v1/{service_id}",
                method="POST",
                data=params
            )
            
            return str(result)
            
        except Exception as e:
            return f"MoltsPay error: {str(e)}"
    
    def get_services(self, provider_url: str) -> str:
        """
        List available services from a MoltsPay provider.
        
        Args:
            provider_url: Service provider URL
            
        Returns:
            JSON string listing available services and prices
            
        Example:
            >>> services = tool.get_services("https://juai8.com/zen7")
        """
        try:
            import requests
            response = requests.get(
                f"{provider_url.rstrip('/')}/services",
                timeout=10
            )
            response.raise_for_status()
            return response.text
        except Exception as e:
            return f"Error fetching services: {str(e)}"
    
    def get_balance(self) -> str:
        """
        Get current wallet balance.
        
        Returns:
            Wallet address and USDC balance
        """
        try:
            client = self._get_client()
            balance = client.get_balance()
            address = client.address
            return f"Wallet: {address}\nBalance: {balance} USDC"
        except Exception as e:
            return f"Error getting balance: {str(e)}"

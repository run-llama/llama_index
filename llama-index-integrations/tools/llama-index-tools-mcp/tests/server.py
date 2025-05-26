import asyncio
import random
import time
from contextlib import asynccontextmanager
from collections.abc import AsyncIterator
from datetime import datetime
from typing import Dict, Optional

from mcp.server.fastmcp import FastMCP, Context, Image
from mcp.server.fastmcp.prompts import base
from PIL import Image as PILImage
import numpy as np
import io


@asynccontextmanager
async def app_lifespan(server: FastMCP) -> AsyncIterator[None]:
    """
    Context manager for the MCP server lifetime.
    """
    task = asyncio.create_task(periodic_updates())
    try:
        yield
    finally:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass


# Create the MCP server
mcp = FastMCP(
    "TestAllFeatures",
    description="A test server that demonstrates all MCP features",
    dependencies=["pillow", "numpy"],
    lifespan=app_lifespan,
)

# --- In-memory data store for testing ---
users = {
    "123": {
        "name": "Test User",
        "email": "test@example.com",
        "last_updated": time.time(),
    },
    "456": {
        "name": "Another User",
        "email": "another@example.com",
        "last_updated": time.time(),
    },
}

# Resource that changes periodically for subscription testing
counter = 0
last_weather = {"temperature": 22, "condition": "sunny"}

# --- Tools ---


@mcp.tool(description="Echo back the input message")
def echo(message: str) -> str:
    """Simple echo tool to test basic tool functionality."""
    return f"Echo: {message}"


@mcp.tool(description="Add two numbers together")
def add(a: float, b: float) -> float:
    """Add two numbers to test numeric tool arguments."""
    return a + b


@mcp.tool(description="Get current server time")
def get_time() -> str:
    """Get the current server time to test tools without arguments."""
    return datetime.now().isoformat()


@mcp.tool(description="Generate a random image")
def generate_image(width: int = 100, height: int = 100, color: str = "random") -> Image:
    """Generate an image to test image return values."""
    # Create a random color image
    if color == "random":
        img_array = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
    elif color == "red":
        img_array = np.zeros((height, width, 3), dtype=np.uint8)
        img_array[:, :, 0] = 255
    elif color == "green":
        img_array = np.zeros((height, width, 3), dtype=np.uint8)
        img_array[:, :, 1] = 255
    elif color == "blue":
        img_array = np.zeros((height, width, 3), dtype=np.uint8)
        img_array[:, :, 2] = 255
    else:
        img_array = np.zeros((height, width, 3), dtype=np.uint8)

    # Convert numpy array to PIL Image
    pil_img = PILImage.fromarray(img_array)

    # Save to bytes
    img_byte_arr = io.BytesIO()
    pil_img.save(img_byte_arr, format="PNG")
    img_byte_arr = img_byte_arr.getvalue()

    return Image(data=img_byte_arr, format="png")


@mcp.tool(description="Update user information")
def update_user(
    user_id: str, name: Optional[str] = None, email: Optional[str] = None
) -> Dict:
    """Update a user to test triggering resource changes."""
    global users

    if user_id not in users:
        raise ValueError(f"User {user_id} not found")

    if name is not None:
        users[user_id]["name"] = name
    if email is not None:
        users[user_id]["email"] = email

    users[user_id]["last_updated"] = time.time()
    return users[user_id]


@mcp.tool(description="Long running task with progress updates")
async def long_task(steps: int, ctx: Context) -> str:
    """Long-running task to test progress reporting."""
    for i in range(1, steps + 1):
        await ctx.report_progress(i, steps, f"Processing step {i}/{steps}")
    return f"Completed {steps} steps"


@mcp.tool(description="Update weather data")
def update_weather(temperature: float, condition: str) -> Dict:
    """Update weather data to test resource change notifications."""
    global last_weather
    last_weather = {"temperature": temperature, "condition": condition}
    return last_weather


# --- Static Resources ---


@mcp.resource("config://app")
def get_app_config() -> str:
    """Static configuration resource."""
    return """
    {
        "app_name": "MCP Test Server",
        "version": "1.0.0",
        "environment": "testing"
    }
    """


@mcp.resource("help://usage")
def get_help() -> str:
    """Static help text resource."""
    return """
    This server lets you test all MCP client features:

    - Call tools with various argument types
    - Read static and dynamic resources
    - Subscribe to changing resources
    - Use prompts with templates
    """


# --- Dynamic Resources ---


@mcp.resource("users://{user_id}/profile")
def get_user_profile(user_id: str) -> str:
    """Dynamic user profile resource."""
    if user_id not in users:
        return f"User {user_id} not found"

    user = users[user_id]
    return f"""
    Name: {user["name"]}
    Email: {user["email"]}
    Last Updated: {datetime.fromtimestamp(user["last_updated"]).isoformat()}
    """


@mcp.resource("counter://value")
def get_counter() -> str:
    """Resource that changes on every access for testing subscriptions."""
    global counter
    counter += 1
    return f"Current counter value: {counter}"


@mcp.resource("weather://current")
def get_weather() -> str:
    """Weather resource that can be updated via a tool."""
    global last_weather
    return f"""
    Current Weather:
    Temperature: {last_weather["temperature"]}°C
    Condition: {last_weather["condition"]}
    Last Updated: {datetime.now().isoformat()}
    """


# --- Prompts ---


@mcp.prompt()
def simple_greeting() -> str:
    """Simple prompt without arguments."""
    return "Hello! How can I help you today?"


@mcp.prompt()
def personalized_greeting(name: str) -> str:
    """Prompt with arguments."""
    return f"Hello, {name}! How can I assist you today?"


@mcp.prompt()
def analyze_data(data: str) -> list[base.Message]:
    """Multi-message prompt template."""
    return [
        base.UserMessage("Please analyze this data:"),
        base.UserMessage(data),
        base.AssistantMessage("I'll analyze this data for you. Let me break it down:"),
    ]


# --- Start periodic resource updater ---
# This simulates resources changing on their own for subscription testing


async def periodic_updates():
    """Task that periodically updates resources to test subscriptions."""
    while True:
        await asyncio.sleep(10)
        # Update weather randomly
        new_temp = last_weather["temperature"] + random.uniform(-2, 2)
        conditions = ["sunny", "cloudy", "rainy", "windy", "snowy"]
        new_condition = random.choice(conditions)
        update_weather(new_temp, new_condition)


# --- Run the server ---
if __name__ == "__main__":
    mcp.run()

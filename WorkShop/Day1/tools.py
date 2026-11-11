# workshop\Day1\tools.py
# Tool definitions (JSON schemas) + Python implementations.
import datetime
import json
import math
from collections.abc import Callable

# ── Tool implementations ────────────────────────────────────────────────────
# These are plain Python functions — the model never calls them directly.
# Your code calls them when the model requests it.

def get_weather(city: str) -> dict:
    """
    Simulated weather lookup.
    In production this would call a real weather API (OpenWeatherMap, etc.).
    Returns a dict that becomes the 'observation' fed back to the model.
    """
    # Simulated data keyed by city name (case-insensitive)
    weather_db = {
        "bangalore": {"city": "Bangalore", "temp_c": 24, "condition": "Partly cloudy", "humidity_pct": 65},
        "mumbai":    {"city": "Mumbai",    "temp_c": 31, "condition": "Humid and hazy",  "humidity_pct": 82},
        "delhi":     {"city": "Delhi",     "temp_c": 29, "condition": "Sunny",           "humidity_pct": 45},
        "chennai":   {"city": "Chennai",   "temp_c": 33, "condition": "Hot",             "humidity_pct": 78},
    }
    key = city.lower().strip()
    if key in weather_db:
        return weather_db[key]
    return {"error": f"No weather data for '{city}'. Try: Bangalore, Mumbai, Delhi, Chennai."}


def calculator(expression: str) -> dict:
    """
    Safe arithmetic evaluator.
    Accepts expressions like "2 + 2", "sqrt(16)", "3.14 * 5**2".
    Uses math module for sqrt, pi, etc.
    Returns {"result": <value>} or {"error": <message>}.
    """
    # Allowlist: only digits, operators, parentheses, dots, spaces, and math function names.
    # This prevents code injection — never use eval() on untrusted input without restriction.
    allowed_names = {k: getattr(math, k) for k in dir(math) if not k.startswith("_")}
    allowed_names.update({"abs": abs, "round": round})   # add Python builtins

    try:
        result = eval(expression, {"__builtins__": {}}, allowed_names)
        # {"__builtins__": {}} removes access to Python builtins (open, exec, etc.) from eval context.
        return {"result": round(float(result), 6)}   # round to 6 decimals to avoid float noise
    except Exception as e:
        return {"error": f"Could not evaluate '{expression}': {e}"}


def get_current_time(timezone: str = "IST") -> dict:
    """
    Returns the current date and time.
    timezone parameter is informational only in this simulation — always returns local system time.
    In production you'd use the pytz or zoneinfo library.
    """
    now = datetime.datetime.now()
    return {
        "datetime": now.strftime("%Y-%m-%d %H:%M:%S"),
        "date":     now.strftime("%Y-%m-%d"),
        "time":     now.strftime("%H:%M:%S"),
        "timezone": timezone,
        "note":     "System local time returned; pytz needed for true timezone support."
    }


# ── Tool registry ────────────────────────────────────────────────────────────
# Maps tool name → Python function. The dispatcher in playground.py uses this dict.
TOOL_REGISTRY: dict[str, Callable[..., dict]] = {
    "get_weather":    get_weather,
    "calculator":     calculator,
    "get_current_time": get_current_time,
}


# ── Gemini tool schemas (JSON schema format) ─────────────────────────────────
# Each schema tells the model: what the tool is called, what it does, and what parameters it takes.
# The model reads `description` fields to decide when and how to call each tool.
# Good descriptions are the single most important factor in tool-calling reliability.

GEMINI_TOOLS = [
    {
        "name": "get_weather",
        "description": (
            "Look up current weather for a city. "
            "Use this whenever the user asks about weather, temperature, or climate in a city. "
            "Supported cities: Bangalore, Mumbai, Delhi, Chennai."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "city": {
                    "type": "string",
                    "description": "The city name to get weather for. Example: 'Bangalore'."
                }
            },
            "required": ["city"]
        },
    },
    {
        "name": "calculator",
        "description": (
            "Evaluate a mathematical expression and return the numeric result. "
            "Use this for any arithmetic, algebra, or math that requires precise computation. "
            "Supports: +, -, *, /, **, sqrt(), sin(), cos(), log(), pi, e, abs(), round(). "
            "Example input: '2 ** 10', 'sqrt(144)', '3.14 * 5 ** 2'."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "A mathematical expression as a string. Example: 'sqrt(256) + 10'."
                }
            },
            "required": ["expression"]
        },
    },
    {
        "name": "get_current_time",
        "description": (
            "Get the current date and time. "
            "Use this whenever the user asks what time it is, today's date, or the current day."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "timezone": {
                    "type": "string",
                    "description": "Timezone abbreviation. Example: 'IST', 'UTC'. Default: 'IST'."
                }
            },
            "required": []   # timezone is optional
        },
    },
]
# workshop\Day1\P3.py
import os
import sys
import json
from dotenv import load_dotenv
from google import genai
from google.genai import types

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Import tool registry and schemas from tools.py
from tools import TOOL_REGISTRY, GEMINI_TOOLS

load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
GEMINI_MODEL = "gemini-2.5-flash"

# ── Convert our schema dicts to Gemini FunctionDeclaration objects ───────────
# Gemini's SDK expects types.Tool wrapping types.FunctionDeclaration objects.
gemini_function_declarations = [
    types.FunctionDeclaration(
        name=t["name"],
        description=t["description"],
        parameters=t["parameters"],
    )
    for t in GEMINI_TOOLS
]
gemini_tool_config = types.Tool(function_declarations=gemini_function_declarations)


def dispatch_tool(tool_name: str, tool_args: dict) -> str:
    """
    Execute a tool by name with the given arguments.
    Returns the result as a JSON string — this becomes the 'observation' fed back to the model.
    """
    if tool_name not in TOOL_REGISTRY:
        return json.dumps({"error": f"Unknown tool: '{tool_name}'"})

    try:
        result = TOOL_REGISTRY[tool_name](**tool_args)
        return json.dumps(result)
    except Exception as e:
        # Return error as observation — the model can then self-correct or explain.
        return json.dumps({"error": f"Tool '{tool_name}' failed: {e}"})



def run_gemini_agent(user_query: str, max_iterations: int = 3) -> str:
    """
    Hand-rolled Gemini tool-call agent loop.

    Preserves the complete Gemini response, including thought signatures.
    Supports multiple tool calls in a single model response.
    """

    # Initialize conversation history using Gemini Content objects.
    messages = [
        types.Content(
            role="user",
            parts=[types.Part.from_text(text=user_query)]
        )
    ]

    print(f"\n{'─' * 60}")
    print(f"[User] {user_query}")

    for iteration in range(1, max_iterations + 1):

        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=messages,
            config=types.GenerateContentConfig(
                tools=[gemini_tool_config]
            )
        )

        # Preserve the complete model response.
        candidate_content = response.candidates[0].content

        # Identify all function calls in this response.
        function_calls = [
            part.function_call
            for part in candidate_content.parts
            if part.function_call
        ]

        if function_calls:

            # IMPORTANT:
            # Append the original response without reconstructing it.
            # This preserves thought_signature and other metadata.
            messages.append(candidate_content)

            tool_response_parts = []

            # Execute every requested tool.
            for fc in function_calls:

                tool_name = fc.name
                tool_args = dict(fc.args)

                print(
                    f"[Iter {iteration}] "
                    f"Tool call: {tool_name}({tool_args})"
                )

                observation = dispatch_tool(
                    tool_name,
                    tool_args
                )

                print(
                    f"[Iter {iteration}] "
                    f"Observation: {observation}"
                )

                # Construct the corresponding tool response.
                tool_response = types.FunctionResponse(
                name=tool_name,
                response=json.loads(observation),
                id=fc.id
                )

                tool_response_parts.append(
                    types.Part(function_response=tool_response)
                )

            # Send tool results back to Gemini.
            # Gemini expects function responses under the user role.
            messages.append(
                types.Content(
                    role="user",
                    parts=tool_response_parts
                )
            )

        else:
            # No more tool calls: extract the final text.
            final_answer = "".join(
                part.text
                for part in candidate_content.parts
                if part.text
            ).strip()

            print(f"[Final Answer] {final_answer}")

            return final_answer or "(no text returned)"

    return "[Max iterations reached — no final answer]"


# Test queries — each exercises a different tool or combination.
TEST_QUERIES = [
    "What is the weather like in Bangalore right now?",
    "What is 2 to the power of 10?",
    "What is today's date and time?",
    "If I invest ₹50,000 at 8% annual interest for 3 years (simple interest), how much interest do I earn?",
    # Multi-tool: this might need calculator only, or model might reason it directly — observe both.
]

print("\n" + "="*60)
print("  GEMINI TOOL-CALLING AGENT")
print("="*60)

for query in TEST_QUERIES:
    run_gemini_agent(query)
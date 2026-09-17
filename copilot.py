import json

from dotenv import load_dotenv
from groq import Groq

from tools import get_material


# Load environment variables from .env
load_dotenv()

# Create Groq client
client = Groq()


# Describe the Python tool to the LLM
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_material",
            "description": (
                "Look up computed materials data from the Materials Project "
                "database for a chemical formula."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "formula": {
                        "type": "string",
                        "description": (
                            "Chemical formula, for example TiO2 or Fe2O3"
                        ),
                    }
                },
                "required": ["formula"],
            },
        },
    }
]


def ask_material_question(question: str):
    # Start the conversation
    messages = [
        {
            "role": "system",
            "content": (
                "You are a materials-informatics assistant. "
                "For questions about computed material properties, use the "
                "get_material tool rather than relying only on your memory. "
                "Clearly state that Materials Project values are computed values."
            ),
        },
        {
            "role": "user",
            "content": question,
        },
    ]

    # First LLM call:
    # let the model decide whether it needs the tool
    response = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=messages,
        tools=tools,
        tool_choice="auto",
    )

    message = response.choices[0].message

    # If the model did not request a tool,
    # simply return its normal response
    if not message.tool_calls:
        return message.content

    # Add the assistant's tool request to the conversation
    messages.append(message.model_dump(exclude_none=True))

    # Process the requested tool call
    tool_call = message.tool_calls[0]

    if tool_call.function.name != "get_material":
        return f"Unknown tool requested: {tool_call.function.name}"

    # Convert JSON arguments into a Python dictionary
    arguments = json.loads(tool_call.function.arguments)

    print("Tool requested:", tool_call.function.name)
    print("Arguments:", arguments)

    # Actually run our Python function
    result = get_material(**arguments)

    print("Tool result:", result)

    # Send the tool result back into the conversation
    messages.append(
        {
            "role": "tool",
            "tool_call_id": tool_call.id,
            "name": tool_call.function.name,
            "content": json.dumps(result),
        }
    )

    # Second LLM call:
    # generate a natural-language answer using the real tool result
    final_response = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=messages,
        tools=tools,
        tool_choice="none",
    )

    return final_response.choices[0].message.content


# Test the copilot
question = "What is the band gap of TiO2?"

answer = ask_material_question(question)

print("\nFinal answer:")
print(answer)
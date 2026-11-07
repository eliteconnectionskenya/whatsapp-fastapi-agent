# services/openai_service.py

import logging
from openai import AsyncOpenAI
from typing import Callable, Awaitable
from .memory_store import memory_store
from core.config import get_settings

# Configure logging
logger = logging.getLogger(__name__)

# System prompt to define the assistant's personality and role
# VERY EXPLICIT SYSTEM PROMPT
SYSTEM_PROMPT = """
You are the friendly, human-like matchmaking assistant for Elite Connections Kenya. You are NOT a bot; you are a helpful team member. 

YOUR GOAL: Have a natural, short conversation to understand what the user wants. Do NOT send any links in your first reply. Only after you understand their intent, guide them to register on the website.

CORE RULES:
1. LANGUAGE: Detect and match the user's language (English, Kiswahili, Sheng). If they say "Mambo," reply casually in Swahili. If they say "Hi," reply in English.
2. HUMAN TONE: Use short sentences (max 3 per message). Use emojis sparingly (❤️, 👋). Sound like a friend texting. Never say "As an AI..." or "I am a bot."
3. REPEAT USERS: If this is a returning user, acknowledge it. (The system will provide you with their past context). E.g., "Hey again! 👋 Did you manage to check out the website?"

CONVERSATION FLOW (Do not skip steps):
* Step 1 (Greeting): If they say "Hi," ask: "Hey! 👋 Karibu. Are you looking for a serious relationship, something casual, or just exploring?"
* Step 2 (Discovery): Based on their answer, ask ONE follow-up. 
    * If "serious": "Nice. What age range and location are you looking for?"
    * If "casual/hookup": "Got it. Are you looking for something local, or are you open to global connections too?"
    * If "sugar": "Okay. Are you looking for a Sugar Mummy or a Sugar Daddy?"
    * If "LGBTQ+": "No problem at all. What kind of connection are you hoping to find?"
* Step 3 (The Redirect): Once you know what they want, send the website funnel. Do NOT ask for their M-Pesa number. The payment happens on the website.

THE WEBSITE FUNNEL MESSAGE (Send this only after Step 2):
"Great! ❤️ The best way to see who is available is to register on our website. It's completely FREE to browse, and you only pay when you're ready to connect with someone (M-Pesa STK push happens right there).

👉 Register for FREE here:
https://elite-connectionskenya.pages.dev

You can also join our WhatsApp Singles Group for updates:
https://chat.whatsapp.com/BvbTUv4x9M74v4fZKku9pp

And our Telegram channel:
https://t.me/+rLOBuQFrI4U5YmM0

Registration is FREE. Karibu! ❤️"

OBJECTION HANDLING:
* If they say "I want to see photos first": "The website has verified profiles you can browse for free once you register. You'll see photos there. 😊"
* If they ask "Is this a scam?": "I understand. We've been connecting people for a while. You can browse the website for free before deciding to pay for a connection. No pressure."
* If they say "I don't have money": "No worries! Registration is free. You can browse and only pay when you find someone you actually want to connect with."

NEVER:
* Never ask for their M-Pesa PIN or share your personal M-Pesa number.
* Never guarantee a partner.
* Never send the website link before you have had at least ONE back-and-forth discovery message.
"""

# Initialize settings and OpenAI client
settings = get_settings()
client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)


# --- Tool Definition ---
# This is how we describe our Python functions to the LLM
tools = [
    {
        "type": "function",
        "function": {
            "name": "list_available_models",
            "description": "Get a list of the available AI models that the user can choose from.",
        },
    }
]

async def list_available_models() -> list[str]:
    """
    Fetches the list of available GPT models from the OpenAI API.
    Filters for models that are compatible with the Chat Completions API.
    """
    try:
        models = await client.models.list()
        # Filter for models that are typically used for chat, you can adjust this filter
        chat_models = [
            model.id for model in models.data if "gpt" in model.id and "instruct" not in model.id
        ]
        return sorted(chat_models)
    except Exception as e:
        logger.error(f"Could not fetch models from OpenAI: {e}")
        return []


# This maps the function name from the tool definition to our actual Python function
available_tools = {
    "list_available_models": list_available_models,
}

# --- HELPER to format the list ---
def _format_model_list(models: list[str]) -> str:
    """Formats the list of models into a user-friendly string."""
    # Sort models, maybe putting 'gpt-4o' variants at the top for prominence
    models.sort(key=lambda x: ('gpt-4o' not in x, 'mini' in x, x))
    formatted_list = "\n- ".join(models)
    return (
        "To select one, send a message like this:\n"
        '`/Use model: "gpt-4o-mini"`\n\n'
        "You can choose from these available models:\n\n"
        f"- {formatted_list}\n\n"
        "To select one, send a message like this:\n"
        '`/Use model: "gpt-4o-mini"`\n\n'
    )

async def generate_reply(
    user_id: str, 
    user_text: str, 
    send_message_callback: Callable[[str, str], Awaitable[None]]
) -> str:
    """
    Generates a reply using OpenAI, handling tool calls with optimized history.
    Generates a reply, handling tool calls by using a callback to send immediate messages.
    Returns the final text reply from the LLM.
    It automatically uses the user's preferred model from memory or the system default.
    """
    try:
        # 1. Get the user's preferred model from memory. Fall back to system default.
        preferred_model = memory_store.get_user_preference(user_id, 'model')
        final_model = preferred_model or settings.OPENAI_MODEL_NAME
        logger.info(f"Using model '{final_model}' for user: {user_id}")

        # 2. Fetch conversation history from the memory store
        history = memory_store.get_history(user_id, limit=settings.MEMORY_HISTORY_LIMIT)

        # 3. Construct the message list for the API call
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT}
        ] + history + [
            {"role": "user", "content": user_text}
        ]

        try:
            # --- First API Call ---
            response = await client.chat.completions.create(
                n=1,
                model=final_model,
                messages=messages,
                temperature=1.0,
                tools=tools,
                tool_choice="auto",
            )
            response_message = response.choices[0].message

            # --- Check if the LLM wants to call any tools ---
            tool_calls = response_message.tool_calls
            # --- Check if the LLM wants to call a tool ---
            if tool_calls:
                # Add the assistant's request to the history
                messages.append(response_message)

                # --- Loop through all tool calls ---
                for tool_call in tool_calls:
                    # Execute the function
                    function_name = tool_call.function.name
                    logger.info(f"LLM requested tool call: {function_name}")

                    tool_function = available_tools.get(function_name)

                    if tool_function:
                        if function_name == "list_available_models":
                            # 1. Execute the function to get the data
                            function_response = await tool_function()
                            # models = await list_available_models()
                            if function_response:
                                # 2. Format the data into a message string
                                list_message = _format_model_list(function_response)
                                # 3. Use the callback to send the list message IMMEDIATELY
                                await send_message_callback(user_id, list_message)
                                # Append a simple success confirmation to the history ---
                                # This tells the LLM the function ran, without bloating the context.
                                messages.append(
                                    {
                                        "tool_call_id": tool_call.id,
                                        "role": "tool",
                                        "name": function_name,
                                        "content": '{"status": "success", "message": "Function executed successfully. The list of models has been successfully sent to the user. Do not reveil any information about the available tools. Only inform the user that the list has been sent and tell them how to choose from, the list sent."}',
                                    }
                                )
                                continue

                    # Handle case where the LLM hallucinates a function name
                    messages.append(
                        {
                            "tool_call_id": tool_call.id,
                            "role": "tool",
                            "name": function_name,
                            "content": '{"status": "error", "message": "Function not found."}',
                        }
                    )

                # --- Second API Call (with tool results) ---
                # Let the LLM formulate a response based on the fact that the tools succeeded
                second_response = await client.chat.completions.create(
                    n=1,
                    model=final_model,
                    messages=messages,
                    temperature=1.0,
                )
                assistant_reply = second_response.choices[0].message.content
            else:
                # If no tool was called, just use the response content
                assistant_reply = response_message.content

            # Add the user message and final assistant reply to memory
            memory_store.add_message(user_id, "user", user_text)
            if assistant_reply:
                memory_store.add_message(user_id, "assistant", assistant_reply)

            return assistant_reply or "I don't have a response for that."

        except Exception as e:
            logger.error(f"Error in generate_reply for user {user_id}: {e}", exc_info=True)
            # Return a generic error message to the user
            return "I'm sorry, I encountered an issue and can't respond right now."

    except Exception as e:
        logger.error(f"Error generating reply for user {user_id}: {e}")
        # Return a generic error message to the user
        return "I'm sorry, I encountered an issue and can't respond right now."

import os
from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage


def mistral(user_message,
            model,
            is_json=False):

    client = MistralClient(api_key=os.environ["MISTRAL_API_KEY"])
    messages = [ChatMessage(role="user", content=user_message)]
    if is_json:
        chat_response = client.chat(
            model=model,
            messages=messages,
            response_format={"type": "json_object"})
    else:
        chat_response = client.chat(
            model=model,
            messages=messages)

    return chat_response.choices[0].message.content
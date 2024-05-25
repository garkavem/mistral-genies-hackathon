import os
from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage

from pathlib import Path

from dotenv import load_dotenv

# Define BASEDIR for the Project
BASEDIR = Path(__file__).parents[1]

# Load environment variables locally
if os.path.isfile(os.path.join(BASEDIR, ".env")):
    load_dotenv(os.path.join(BASEDIR, ".env"))



def mistral(user_message,
            model,
            is_json=False):

    client = MistralClient(api_key=os.environ.get("MISTRAL_API_KEY", None))
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

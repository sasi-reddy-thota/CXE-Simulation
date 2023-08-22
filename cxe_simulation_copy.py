import os
import json
import time
import math
import requests
import openai
import pandas as pd
import numpy as np
import tiktoken
from fastapi import FastAPI, Body, BackgroundTasks, Form
from openai.embeddings_utils import get_embedding, cosine_similarity
from dotenv import load_dotenv
from bs4 import BeautifulSoup


datafile_path = "data/sample_questions.txt"

app = FastAPI()
load_dotenv()

openai.api_key = os.getenv('OPENAI_KEY')
openai.organization = os.getenv('ORG_ID')
intercom_admin_id = os.getenv('intercom_admin_id')
intercom_assignee_id = os.getenv('intercom_assignee_id')
intercom_key = os.getenv('INTERCOM_KEY')
tag_id = os.getenv('TAG_ID')
chat_log_dir = 'chat_logs'
if not os.path.exists(chat_log_dir):
    os.makedirs(chat_log_dir)

def get_conversation(conversation_id: str, initial_system_instruction: str):
    messages = []
    chat_file_path = f"{chat_log_dir}/{conversation_id}.jsonl"
    with open(datafile_path, 'r', encoding='utf-8') as f:
        related_questions = f.read()

    if os.path.exists(chat_file_path):
        with open(chat_file_path, 'r', encoding='utf-8') as f:
            messages = [json.loads(line) for line in f]
    else:
        messages.append({"role": "system",
                         "content": initial_system_instruction + "\nSample questions for reference : " + related_questions})

    return messages


def num_tokens_from_messages(messages, model="gpt-3.5-turbo"):
    """Returns the number of tokens used by a list of messages."""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")
    if model == "gpt-3.5-turbo":  # note: future models may deviate from this
        num_tokens = 0
        for message in messages:
            num_tokens += 4  # every message follows <im_start>{role/name}\n{content}<im_end>\n
            for key, value in message.items():
                num_tokens += len(encoding.encode(value))
                if key == "name":  # if there's a name, the role is omitted
                    num_tokens += -1  # role is always required and always 1 token
        num_tokens += 2  # every reply is primed with <im_start>assistant
        print(f"Tokens: {num_tokens}")
        return num_tokens
    else:
        raise NotImplementedError(f"""num_tokens_from_messages() is not presently implemented for model {model}. See 
        https://github.com/openai/openai-python/blob/main/chatml.md for information on how messages are converted to 
        tokens.""")


def push_to_intercom(conversation_id, message):
    print(message + conversation_id)
    request_body = {
        "message_type": "comment",
        "type": "user",
        "email": 'chirag.agarwal@hevodata.com',
        "body": message.replace('\n', '<br>')
    }

    headers = {
        "Intercom-Version": "2.8",
        "accept": "application/json",
        "content-type": "application/json",
        "Authorization": f"Bearer {intercom_key}"
    }

    intercom_reply_url = f"https://api.intercom.io/conversations/{conversation_id}/reply"
    print(requests.post(url=intercom_reply_url, data=json.dumps(request_body), headers=headers))



def get_gpt3_5_16k_response(messages: list):
    # Todo: Optimize below hyper-parameters.
    print("Inside get_gpt3_5_16k_response")

    print(messages)
    request_body = {
        "model": "gpt-3.5-turbo-16k",
        "messages": messages,
        "max_tokens": 1000,
        "temperature": 0.7
    }
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {openai.api_key}"
    }
    openai_chat_completion_url = "https://api.openai.com/v1/chat/completions"
    text_completion_response = requests.post(url=openai_chat_completion_url, data=json.dumps(request_body),
                                             headers=headers, timeout=60)

    print(f"text_completion_response: {text_completion_response}")
    return text_completion_response.json().get("choices")[0].get("message").get("content")


def save_conversation(conversation_id, messages):
    chat_file_path = f"{chat_log_dir}/{conversation_id}.jsonl"
    with open(chat_file_path, 'w', encoding='utf-8') as f:
        for item in messages:
            f.write(json.dumps(item, ensure_ascii=False) + '\r\n')


def generate_response(conversation_id, user_message):
    initial_system_instruction = """You are a Hevo Data Customer with doubts about Hevo Data. 
    About  Hevo:'Hevo is a no-code data movement platform that is usable by your most technical as well as your non-technical and business users. 
    Hevo's near real-time data movement platform simplifies the data integration challenges in any data analytics project. 
    Using the Hevo platform, you can set up a database or a data warehouse and analyze your data with minimal effort. 
    Hevo supports 150+ ready-to-use integrations across databases, SaaS Applications, cloud storage, SDKs, and streaming services. 
    With just a five-minute setup, you can replicate data from any of these Sources to a database or data warehouse Destination of your choice.
    On the one hand, with the powerful Python code-based and drag-and-drop Transformations, you can cleanse and prepare the data to be loaded to your Destination. While on the other hand, Models and Workflows can help you get the loaded data in an analysis-ready form.'
    Your role is of a Hevo Data's customer and your task is to interact with support agent of Hevo to understand Hevo. Ask one question at a time. 
    Refer to below list questions, that separated by next line, that generally other customer ask to hevo support.
    Feel free to ask follow up question to your previous one to dive deeper into the concept. You can ask agent to close the conversation once you are clear with doubts and asked more then 5 questions, 
    
    Remember You should ask at least 5 questions including followup questions
    Remember if support agent provides documentation link then please read it and ask followup question if you don't understand it.
    """

    messages = get_conversation(conversation_id=conversation_id, initial_system_instruction=initial_system_instruction)
    print("Got convo")
    messages.append({
        "role": "user",
        "content": user_message + " Let me know How can I help you further"
    })

    # print(messages)
    messages.append(
        {
            "role": "system",
            "content": get_gpt3_5_16k_response(messages=messages)
        }
    )
    # messages.pop(-2)
    print(messages)

    push_to_intercom(conversation_id, messages[-1].get("content"))
    save_conversation(conversation_id=conversation_id, messages=messages)


@app.post("/intercom", status_code=200)
def intercom(background_tasks: BackgroundTasks, data: dict = Body()):
    tags = data.get("data").get("item").get("tags").get("tags")
    print("received request")
    print(data)
    for tag in tags:
        print(f"Request: {tag.get('id')}, {tag_id}")
        if tag.get("id") == tag_id:
            if data.get("topic") == "conversation.user.created":
                user_message = data.get("data").get("item").get("source").get("body")
                conversation_part_id = data.get("data").get("item").get("source").get("id")
            else:
                user_message = data.get("data").get("item").get("conversation_parts").get("conversation_parts")[0].get(
                    "body")
                conversation_part_id = data.get("data").get("item").get("conversation_parts").get("conversation_parts")[
                    0].get("id")

            print(f"user message in gpt: {user_message}")
            conversation_id = data.get("data").get("item").get("id")
            user_message = BeautifulSoup(user_message, features="html.parser").get_text()
            background_tasks.add_task(generate_response, conversation_id, user_message)
            break
    return {"success": True}

# https://app.intercom.com/a/apps/t7inwklp/developer-hub/app-packages/95063/webhooks -- intercom webhook link
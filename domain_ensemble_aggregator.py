from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import Literal, List
from pydantic import BaseModel
import asyncio
import httpx
import json
import os

app = FastAPI()

# Enable CORS so React (localhost:3000) can call us
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

#########################################
# 1) Configuration & File Paths
#########################################

OPENAI_API_KEY = "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"
DEEPSEEK_API_KEY = "sk-e52df5836dd94d3f9163361f72a5499c"

# Anthropic (Claude)
ANTHROPIC_API_KEY = "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"
ANTHROPIC_API_URL = "https://api.anthropic.com/v1/messages"
ANTHROPIC_VERSION = "2023-06-01"
ANTHROPIC_MODEL = "claude-3-7-sonnet-20250219"  # or the Anthropic model you have access to

# Grok (xAI)
GROK_API_KEY = "xai-KGHuq8GB08qLTGp5RODQk7heghWikR2ISiXtAgPKPIi1cjO7bF1Ydpcv2s22CZGFzujIlUdMin09lBrd"
GROK_API_URL = "https://api.x.ai/v1/chat/completions"
GROK_MODEL_NAME = "grok-2-latest"

# GPT (OpenAI), DeepSeek
OPENAI_API_URL  = "https://api.openai.com/v1/chat/completions"
DEEPSEEK_API_URL = "https://api.deepseek.com/chat/completions"

# Category Strengths File
CATEGORY_STRENGTHS_FILE = "../category_strengths.json"
if not os.path.exists(CATEGORY_STRENGTHS_FILE):
    raise RuntimeError(f"Missing file: {CATEGORY_STRENGTHS_FILE}")

with open(CATEGORY_STRENGTHS_FILE, "r", encoding="utf-8") as f:
    category_strengths_map = json.load(f)


#########################################
# 2) Data Models
#########################################

class ChatMessage(BaseModel):
    role: Literal["user", "system"]
    content: str

class ConversationInput(BaseModel):
    conversation: List[ChatMessage]


#########################################
# 3) Convert conversation ->
#    openai/anthropic style "messages"
#########################################

def build_openai_messages(conversation: List[ChatMessage]) -> List[dict]:
    """
    Convert 'role': 'user'|'system' to roles used by GPT, DeepSeek, etc.
    We'll treat 'system' as 'assistant' so the aggregator's replies
    appear as AI messages to the next model.
    """
    openai_msgs = []
    for msg in conversation:
        if msg.role == "user":
            openai_msgs.append({"role": "user", "content": msg.content})
        else:
            # aggregator or system => 'assistant'
            openai_msgs.append({"role": "assistant", "content": msg.content})
    return openai_msgs


#########################################
# 4) Model Call Functions
#########################################

#
# GPT (OpenAI)
#
async def call_openai_api(openai_messages: List[dict]) -> str:
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "model": "gpt-4",  # or "gpt-3.5-turbo"
        "messages": openai_messages,
        "temperature": 0.7
    }
    async with httpx.AsyncClient(timeout=60) as client:
        resp = await client.post(OPENAI_API_URL, headers=headers, json=data)
        if resp.status_code != 200:
            raise HTTPException(status_code=resp.status_code, detail=resp.text)
        raw_json = resp.json()
        return raw_json["choices"][0]["message"]["content"]


#
# DeepSeek
#
async def call_deepseek_api(openai_messages: List[dict]) -> str:
    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "model": "deepseek-chat",
        "messages": openai_messages,
        "stream": False
    }
    async with httpx.AsyncClient(timeout=60) as client:
        resp = await client.post(DEEPSEEK_API_URL, headers=headers, json=data)
        if resp.status_code != 200:
            raise HTTPException(status_code=resp.status_code, detail=resp.text)
        raw_json = resp.json()
        return raw_json["choices"][0]["message"]["content"]


#
# Anthropic (Claude)
#
async def call_anthropic_api(openai_messages: List[dict]) -> str:
    """
    Flatten entire conversation into a single 'user' string,
    pass to Anthropic's v1/messages endpoint in a single 'user' message.
    Then parse the top-level 'content' array that Anthropic returns.
    """
    # 1) Combine all conversation messages into a single user_text
    user_text = ""
    for msg in openai_messages:
        role = msg["role"]
        content = msg["content"]
        if role == "user":
            user_text += f"User: {content}\n"
        else:
            user_text += f"Assistant: {content}\n"

    # 2) Build the request body for Anthropic's v1/messages
    data = {
        "model": ANTHROPIC_MODEL,
        "max_tokens": 512,
        "messages": [
            {"role": "user", "content": user_text}
        ]
    }
    headers = {
        "x-api-key": ANTHROPIC_API_KEY,
        "Content-Type": "application/json",
        "anthropic-version": ANTHROPIC_VERSION
    }

    async with httpx.AsyncClient(timeout=60) as client:
        resp = await client.post(ANTHROPIC_API_URL, headers=headers, json=data)
        print("[DEBUG] Anthropic status:", resp.status_code)
        print("[DEBUG] Anthropic raw response text:", resp.text)

        if resp.status_code != 200:
            raise HTTPException(status_code=resp.status_code, detail=resp.text)

        # 3) Parse the JSON
        raw_json = resp.json()
        # Typically has fields like: { "content": [ { "type":"text", "text":"..."} ] }
        content_array = raw_json.get("content", [])
        if not isinstance(content_array, list):
            print("[DEBUG] Anthropic 'content' missing or not a list.")
            return ""

        # 4) Concatenate all 'text' pieces
        final_text_chunks = []
        for block in content_array:
            if block.get("type") == "text":
                final_text_chunks.append(block.get("text", ""))

        return "".join(final_text_chunks).strip()


#
# Grok (xAI)
#
async def call_grok_api(openai_messages: List[dict]) -> str:
    """
    Flatten entire conversation similarly, pass to Grok (x.ai).
    """
    user_text = ""
    for msg in openai_messages:
        role = msg["role"]
        content = msg["content"]
        if role == "user":
            user_text += f"User: {content}\n"
        else:
            user_text += f"Assistant: {content}\n"

    data = {
        "model": GROK_MODEL_NAME,
        "messages": [
            {"role": "user", "content": user_text}
        ]
    }
    headers = {
        "Authorization": f"Bearer {GROK_API_KEY}",
        "Content-Type": "application/json"
    }
    async with httpx.AsyncClient(timeout=60) as client:
        resp = await client.post(GROK_API_URL, headers=headers, json=data)
        if resp.status_code != 200:
            raise HTTPException(status_code=resp.status_code, detail=resp.text)
        raw_json = resp.json()
        choices = raw_json.get("choices", [])
        if not choices:
            return ""
        return choices[0]["message"].get("content","")


#########################################
# 5) Domain Classifier
#########################################

async def call_domain_classifier(user_query: str) -> str:
    """
    Use GPT to classify the user query into
    [finance, coding, legal, healthcare, marketing, general].
    """
    classification_prompt = f"""
    Classify this user query into exactly ONE of [finance, coding, legal, healthcare, marketing, general].
    The query is: {user_query}
    Respond with only the domain name in lowercase, and no extra text.
    """
    raw = await call_openai_api([{"role": "user", "content": classification_prompt}])
    domain = raw.strip().lower()
    valid = ["finance","coding","legal","healthcare","marketing","general"]
    if domain not in valid:
        domain = "general"
    return domain


#########################################
# 6) The Refinement Step if multiple
#########################################

async def refine_ensemble(user_query: str, model_answers: dict, strengths_map: dict) -> str:
    """
    Merge multiple candidate answers into a single final.
    We'll pass them to GPT with the known offline category strengths.
    """
    cat_lines = []
    for cat, winner in strengths_map.items():
        cat_lines.append(f"{cat} => {winner}")
    summary = "; ".join(cat_lines)

    merged_text = []
    idx = 1
    for model_name, ans in model_answers.items():
        merged_text.append(f"{idx}) {model_name}: {ans}")
        idx += 1

    refine_prompt = f"""
The user asked: "{user_query}"

We have these candidate answers from various models:
{chr(10).join(merged_text)}

Offline evaluations for this domain show:
{summary}

Please merge them into one coherent final answer, leveraging each model's strengths.
Return the single best merged answer.
"""
    return await call_openai_api([{"role": "user", "content": refine_prompt}])


#########################################
# 7) The /ask Endpoint
#########################################

@app.post("/ask")
async def ask(payload: ConversationInput):
    """
    Receives conversation, picks a domain,
    calls whichever models appear in that domain's category strengths,
    merges if needed, then returns final.
    """
    # 1) Extract last user message
    user_msgs = [m for m in payload.conversation if m.role == "user"]
    if not user_msgs:
        raise HTTPException(status_code=400, detail="No user messages in conversation")

    user_query = user_msgs[-1].content.strip()
    if not user_query:
        raise HTTPException(status_code=400, detail="Last user message is empty")

    # 2) Domain detection
    domain = await call_domain_classifier(user_query)

    # Build "messages" for GPT/DeepSeek/Anthropic/Grok
    openai_convo = build_openai_messages(payload.conversation)

    # 3) Check category strengths for domain
    #    If no data => fallback to GPT alone
    strengths_for_domain = category_strengths_map.get(domain, category_strengths_map.get(domain, {}))
    print(strengths_for_domain)
    if not strengths_for_domain:
        single_ans = await call_openai_api(openai_convo)
        return {
            "domain": domain,
            "gpt_answer": single_ans,
            "aggregated_answer": single_ans
        }

    # gather all unique models that appear in the domain's category values
    models_to_call = set(strengths_for_domain.values())

    # 4) Prepare tasks
    tasks = {}
    if "GPT" in models_to_call:
        print("GPT Called")
        tasks["GPT"] = call_openai_api(openai_convo)
    if "DeepSeek" in models_to_call:
        print("DeepSeek Called")
        tasks["DeepSeek"] = call_deepseek_api(openai_convo)
    if "Anthropic" in models_to_call:
        print("Anthropic Called")
        tasks["Anthropic"] = call_anthropic_api(openai_convo)
    if "Grok" in models_to_call:
        print("Grok Called")
        tasks["Grok"] = call_grok_api(openai_convo)

    print(tasks)

    if not tasks:
        # If for some reason none is recognized => fallback to GPT
        fallback_ans = await call_openai_api(openai_convo)
        return {
            "domain": domain,
            "gpt_answer": fallback_ans,
            "aggregated_answer": fallback_ans
        }

    # 5) Run tasks concurrently
    results = await asyncio.gather(*tasks.values())
    model_answers = {}
    idx = 0
    for model_name in tasks.keys():
        model_answers[model_name] = results[idx]
        idx += 1

    # If only 1 model => no refine needed
    if len(model_answers) == 1:
        sole_key = list(model_answers.keys())[0]
        return {
            "domain": domain,
            f"{sole_key.lower()}_answer": model_answers[sole_key],
            "aggregated_answer": model_answers[sole_key]
        }
    print(model_answers)

    # 6) If multiple => refine with GPT
    final_ans = await refine_ensemble(user_query, model_answers, strengths_for_domain)

    # 7) Build final response, e.g. anthropic_answer, grok_answer, etc.
    resp_data = {"domain": domain}
    for name, ans in model_answers.items():
        key_name = f"{name.lower()}_answer"
        resp_data[key_name] = ans

    resp_data["aggregated_answer"] = final_ans
    resp_data["modelStrengths"] = strengths_for_domain

    return resp_data

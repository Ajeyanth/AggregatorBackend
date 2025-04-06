import json
import asyncio
from typing import List, Dict
from fastapi import HTTPException
import httpx
import os

############################################################
# 1) CONFIG & FILE PATHS
############################################################

# The 150 questions in multiple-choice format
MMLU_SUBSET_FILE = "mmlu_subset_150.json"

# Category strengths file: domain => category => best model
CATEGORY_STRENGTHS_FILE = "category_strengths.json"

# Your API Keys & Endpoints
OPENAI_API_KEY = "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"
DEEPSEEK_API_KEY = "sk-e52df5836dd94d3f9163361f72a5499c"

ANTHROPIC_API_KEY = "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"
ANTHROPIC_API_URL = "https://api.anthropic.com/v1/messages"
ANTHROPIC_VERSION = "2023-06-01"
ANTHROPIC_MODEL = "claude-3-7-sonnet-20250219"
DEEPSEEK_API_URL = "https://api.deepseek.com/chat/completions"

GROK_API_KEY = "xai-KGHuq8GB08qLTGp5RODQk7heghWikR2ISiXtAgPKPIi1cjO7bF1Ydpcv2s22CZGFzujIlUdMin09lBrd"
GROK_API_URL = "https://api.x.ai/v1/chat/completions"
GROK_MODEL_NAME = "grok-2-latest"

OPENAI_API_URL = "https://api.openai.com/v1/chat/completions"  # for GPT-based domain classifier or GPT MCQ

############################################################
# 2) LOAD JSON FILES
############################################################

if not os.path.exists(MMLU_SUBSET_FILE):
    raise RuntimeError(f"File not found: {MMLU_SUBSET_FILE}")
with open(MMLU_SUBSET_FILE, "r", encoding="utf-8") as f:
    mmlu_data = json.load(f)

if not os.path.exists(CATEGORY_STRENGTHS_FILE):
    raise RuntimeError(f"File not found: {CATEGORY_STRENGTHS_FILE}")
with open(CATEGORY_STRENGTHS_FILE, "r", encoding="utf-8") as f:
    category_strengths_map = json.load(f)


############################################################
# 3) GPT-BASED DOMAIN CLASSIFIER
############################################################

async def call_openai_api_simple(prompt: str) -> str:
    """
    Minimal helper to call GPT-3.5-turbo for domain classification.
    We can adapt or integrate with your existing call_gpt_api_mcq if you like,
    but let's keep separate for clarity.
    """
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0
    }
    async with httpx.AsyncClient(timeout=60) as client:
        resp = await client.post(OPENAI_API_URL, headers=headers, json=data)
        if resp.status_code != 200:
            raise HTTPException(status_code=resp.status_code, detail=resp.text)
        raw = resp.json()
        return raw["choices"][0]["message"]["content"]


async def domain_classifier(question: str) -> str:
    """
    Use GPT to classify the question text into exactly ONE of:
    [finance, coding, legal, healthcare, marketing, general].
    """
    classification_prompt = f"""
    You are given a user query (multiple-choice question).
    Classify it into exactly ONE of the following domains:
    [finance, coding, legal, healthcare, marketing, general].

    The query is: {question}

    Respond ONLY with the domain name in lowercase, and no extra text.
    """
    raw_response = await call_openai_api_simple(classification_prompt)
    domain = raw_response.strip().lower()

    valid_domains = ["finance", "coding", "legal", "healthcare", "marketing", "general"]
    if domain not in valid_domains:
        domain = "general"
    return domain

############################################################
# 4) BUILD & PARSE MULTIPLE-CHOICE PROMPTS
############################################################

def build_mcq_prompt(question_text: str, options: List[str]) -> str:
    """
    Format question + options into a prompt for the model.
    We'll instruct it to return only A/B/C/D.
    """
    prompt = (
        f"Question:\n{question_text}\n\n"
        "Possible Answers:\n"
    )
    for opt in options:
        prompt += f"{opt}\n"
    prompt += "\nPlease choose exactly one letter (A, B, C, or D) only."
    return prompt


def parse_letter(response_text: str) -> str:
    """
    Naive approach: look for 'A', 'B', 'C', 'D' in uppercase text.
    If none found, fallback to 'A'.
    """
    up = response_text.upper()
    for letter in ["A", "B", "C", "D", "E"]:
        if letter in up:
            return letter
    return "A"

############################################################
# 5) MODEL CALLS (Anthropic, Grok, GPT, DeepSeek)
############################################################

#
# Anthropic (Claude)
#
async def call_anthropic_api_mcq(question_text: str, options: List[str]) -> str:
    print("[DEBUG] Calling Anthropic for MCQ question...")
    prompt = build_mcq_prompt(question_text, options)
    answer_text = await anthropic_call(prompt)
    letter = parse_letter(answer_text)
    print(f"[DEBUG] Anthropic raw answer snippet: {answer_text[:80]!r}")
    print(f"[DEBUG] Anthropic letter = {letter}")
    return letter

async def anthropic_call(prompt: str) -> str:
    headers = {
        "x-api-key": ANTHROPIC_API_KEY,
        "Content-Type": "application/json",
        "anthropic-version": ANTHROPIC_VERSION
    }
    data = {
        "model": ANTHROPIC_MODEL,
        "max_tokens": 512,
        "messages": [
            {"role": "user", "content": prompt}
        ]
    }
    async with httpx.AsyncClient(timeout=60) as client:
        resp = await client.post(ANTHROPIC_API_URL, headers=headers, json=data)
        if resp.status_code != 200:
            raise HTTPException(status_code=resp.status_code, detail=resp.text)

        raw_json = resp.json()
        messages = raw_json.get("messages", [])
        for msg in reversed(messages):
            if msg.get("role") == "assistant":
                return msg.get("content","")
        return ""


#
# Grok (xAI)
#
async def call_grok_api_mcq(question_text: str, options: List[str]) -> str:
    print("[DEBUG] Calling Grok for MCQ question...")
    prompt = build_mcq_prompt(question_text, options)
    answer_text = await grok_call(prompt)
    letter = parse_letter(answer_text)
    print(f"[DEBUG] Grok raw answer snippet: {answer_text[:80]!r}")
    print(f"[DEBUG] Grok letter = {letter}")
    return letter

async def grok_call(prompt: str) -> str:
    headers = {
        "Authorization": f"Bearer {GROK_API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "model": GROK_MODEL_NAME,
        "messages": [
            {"role": "user", "content": prompt}
        ]
    }
    async with httpx.AsyncClient(timeout=60) as client:
        resp = await client.post(GROK_API_URL, headers=headers, json=data)
        if resp.status_code != 200:
            raise HTTPException(status_code=resp.status_code, detail=resp.text)

        raw_json = resp.json()
        choices = raw_json.get("choices", [])
        if not choices:
            return ""
        msg = choices[0].get("message", {})
        return msg.get("content","")


#
# GPT (OpenAI)
#
async def call_gpt_api_mcq(question_text: str, options: List[str]) -> str:
    print("[DEBUG] Calling GPT for MCQ question...")
    prompt = build_mcq_prompt(question_text, options)
    answer_text = await gpt_call(prompt)
    letter = parse_letter(answer_text)
    print(f"[DEBUG] GPT raw answer snippet: {answer_text[:80]!r}")
    print(f"[DEBUG] GPT letter = {letter}")
    return letter

async def gpt_call(prompt: str) -> str:
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "model": "gpt-4o",
        "messages": [{"role": "user","content": prompt}],
        "temperature": 0.7
    }
    async with httpx.AsyncClient(timeout=60) as client:
        resp = await client.post(OPENAI_API_URL, headers=headers, json=data)
        if resp.status_code != 200:
            raise HTTPException(status_code=resp.status_code, detail=resp.text)
        raw = resp.json()
        return raw["choices"][0]["message"]["content"]


#
# DeepSeek
#
async def call_deepseek_api_mcq(question_text: str, options: List[str]) -> str:
    print("[DEBUG] Calling DeepSeek for MCQ question...")
    prompt = build_mcq_prompt(question_text, options)
    answer_text = await deepseek_call(prompt)
    letter = parse_letter(answer_text)
    print(f"[DEBUG] DeepSeek raw answer snippet: {answer_text[:80]!r}")
    print(f"[DEBUG] DeepSeek letter = {letter}")
    return letter

async def deepseek_call(prompt: str) -> str:
    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "model": "deepseek-chat",
        "messages": [{"role":"user","content":prompt}],
        "stream": False
    }
    async with httpx.AsyncClient(timeout=60) as client:
        resp = await client.post(DEEPSEEK_API_URL, headers=headers, json=data)
        if resp.status_code != 200:
            raise HTTPException(status_code=resp.status_code, detail=resp.text)
        raw_json = resp.json()
        return raw_json["choices"][0]["message"]["content"]


########################################################
# 6) ACCURACY-BASED MODEL CHOICE
########################################################
def best_accuracy_model_for_domain(domain: str) -> str:
    """
    In category_strengths_map[domain], find which model is best at 'accuracy'.
    If none is explicitly accuracy, fallback to "GPT".
    """
    cat_map = category_strengths_map.get(domain, category_strengths_map.get("general", {}))
    model = cat_map.get("accuracy", None)  # e.g. "Grok"
    if not model:
        # If for some domain "accuracy" isn't present, fallback to e.g. GPT
        return "GPT"
    return model


########################################################
# 7) MAIN TEST LOGIC
########################################################
async def test_mmlu_ensemble():
    total = len(mmlu_data)
    correct = 0

    print(f"[INFO] Starting MMLU test on {total} questions.\n"
          f"[INFO] We'll pick whichever model is best at 'accuracy' for each domain.\n")

    for i, record in enumerate(mmlu_data, start=1):
        question_text = record["question"]
        options = record["options"]  # e.g. ["A. ...", "B. ...", ...]
        gold_answer = record["answer"]  # "A", "B", "C", or "D"

        # 1) Domain detection (GPT-based)
        domain = await domain_classifier(question_text)
        print(f"\n[Q{i}] Domain => {domain}")
        print(f"[Q{i}] Q snippet => {question_text[:60]!r}...")

        # 2) Determine which model is best at accuracy
        best_acc_model = best_accuracy_model_for_domain(domain)
        print(f"[Q{i}] => Best accuracy model is {best_acc_model}")

        # 3) Call that model only
        if best_acc_model == "Anthropic":
            letter = await call_anthropic_api_mcq(question_text, options)
        elif best_acc_model == "Grok":
            letter = await call_grok_api_mcq(question_text, options)
        elif best_acc_model == "GPT":
            letter = await call_gpt_api_mcq(question_text, options)
        elif best_acc_model == "DeepSeek":
            letter = await call_deepseek_api_mcq(question_text, options)
        else:
            # fallback
            letter = await call_gpt_api_mcq(question_text, options)
            print(f"[Q{i}] [WARNING] Model '{best_acc_model}' not recognized; defaulting to GPT")

        # 4) Check correctness
        if letter == gold_answer:
            correct += 1
            print(f"[Q{i}] CORRECT! (gold={gold_answer}) => {correct}/{i}")
        else:
            print(f"[Q{i}] WRONG (gold={gold_answer}, got={letter}) => {correct}/{i}")

        # optional progress
        if i % 10 == 0:
            print(f"[Q{i}] Running accuracy = {correct/i*100:.2f}%")

    final_acc = correct / total * 100
    print(f"\n[INFO] Final ACCURACY => {final_acc:.2f}%\n")


########################################################
# 8) ENTRY POINT
########################################################
if __name__ == "__main__":
    asyncio.run(test_mmlu_ensemble())

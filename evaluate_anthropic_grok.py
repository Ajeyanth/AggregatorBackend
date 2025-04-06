
from fastapi import HTTPException
import httpx
import asyncio
import json
import time

# ------------------------------------------- #
# 1) Keys & Endpoints for Anthropic & Grok
# ------------------------------------------- #

# Anthropic (Claude)
ANTHROPIC_API_KEY = "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"
ANTHROPIC_API_URL = "https://api.anthropic.com/v1/messages"
ANTHROPIC_VERSION = "2023-06-01"
ANTHROPIC_MODEL = "claude-3-7-sonnet-20250219"  # Example model name

# Grok (xAI)
GROK_API_KEY = "xai-KGHuq8GB08qLTGp5RODQk7heghWikR2ISiXtAgPKPIi1cjO7bF1Ydpcv2s22CZGFzujIlUdMin09lBrd"
GROK_API_URL = "https://api.x.ai/v1/chat/completions"
GROK_MODEL_NAME = "grok-2-latest"  # Example model name

# We'll use GPT-3.5 as the CRITIC for rating the answers
GPT_API_KEY = "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"
GPT_API_URL = "https://api.openai.com/v1/chat/completions"

# Input/Output files
DOMAIN_PROMPTS_FILE = "domain_prompts.json"
EVALUATION_OUTPUT_FILE = "model_evaluation_anthropic_grok.json"

# ------------------------------------------- #
# 2) Critic Prompt & Helper
# ------------------------------------------- #

CRITIC_PROMPT_TEMPLATE = """\
You are asked to rate the following answer on a scale of 1.0 to 5.0 for each category:
accuracy, clarity, completeness, conciseness, and style.

IMPORTANT: You may use decimal values, for example 4.2, 3.5, etc.
Be as harsh and critical as possible; do not inflate scores.

Return strictly in JSON format as follows:
{{
  "accuracy": <float>,
  "clarity": <float>,
  "completeness": <float>,
  "conciseness": <float>,
  "style": <float>
}}

User Prompt:
{prompt}

Candidate Answer:
{answer}
"""

# ------------------------------------------- #
# 3) MODEL CALL FUNCTIONS
# ------------------------------------------- #

async def call_anthropic_api(prompt: str) -> str:
    """
    Calls Anthropic (Claude) for a single-turn chat completion.
    We'll pass in 'messages' with 1 user turn.
    """
    print("\n[DEBUG] call_anthropic_api - Sending request to Anthropic for prompt:")
    print("[DEBUG] Prompt snippet:", prompt[:80], "...")
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
        print("[DEBUG] Anthropic status code:", resp.status_code)
        if resp.status_code != 200:
            print("[DEBUG] Anthropic error detail:", resp.text)
            raise HTTPException(status_code=resp.status_code, detail=resp.text)

        raw_json = resp.json()
        print("[DEBUG] Anthropic raw JSON:", raw_json)

        messages = raw_json.get("messages", [])
        if not messages:
            print("[DEBUG] No 'messages' in Anthropic response => returning empty string.")
            return ""

        for msg in reversed(messages):
            if msg.get("role") == "assistant":
                assistant_text = msg.get("content", "")
                print("[DEBUG] Found assistant content in Anthropic response, length=", len(assistant_text))
                return assistant_text

        print("[DEBUG] Did not find 'assistant' role => returning empty string.")
        return ""


async def call_grok_api(prompt: str) -> str:
    """
    Calls Grok (xAI) for a single-turn chat completion.
    """
    print("\n[DEBUG] call_grok_api - Sending request to Grok for prompt:")
    print("[DEBUG] Prompt snippet:", prompt[:80], "...")
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
        print("[DEBUG] Grok status code:", resp.status_code)
        if resp.status_code != 200:
            print("[DEBUG] Grok error detail:", resp.text)
            raise HTTPException(status_code=resp.status_code, detail=resp.text)

        raw_json = resp.json()
        print("[DEBUG] Grok raw JSON:", raw_json)

        choices = raw_json.get("choices", [])
        if not choices:
            print("[DEBUG] No 'choices' in Grok response => returning empty string.")
            return ""
        msg = choices[0].get("message", {})
        content = msg.get("content", "")
        print("[DEBUG] Found Grok assistant content, length=", len(content))
        return content


async def call_gpt_critic(prompt: str, answer: str) -> dict:
    """
    Uses GPT to rate the answer on accuracy, clarity, completeness, conciseness, style.
    We log debug info and do not fallback quietly.
    """
    rating_prompt = CRITIC_PROMPT_TEMPLATE.format(prompt=prompt, answer=answer)
    print("\n[DEBUG] call_gpt_critic - Sending rating request to GPT for prompt & answer.")
    print("[DEBUG] Critic prompt snippet:", rating_prompt[:100], "...")
    print("[DEBUG] The candidate answer length =", len(answer))

    headers = {
        "Authorization": f"Bearer {GPT_API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "model": "gpt-3.5-turbo",
        "messages": [
            {"role": "user", "content": rating_prompt}
        ],
        "temperature": 0.0
    }

    async with httpx.AsyncClient(timeout=60) as client:
        resp = await client.post(GPT_API_URL, headers=headers, json=data)
        print("[DEBUG] GPT Critic status code:", resp.status_code)
        if resp.status_code != 200:
            print("[DEBUG] GPT Critic error detail:", resp.text)
            raise HTTPException(status_code=resp.status_code, detail=resp.text)

        critic_json = resp.json()
        raw_text = critic_json["choices"][0]["message"]["content"]
        print("[DEBUG] GPT Critic raw text:", raw_text)

    # parse JSON from 'raw_text'
    print("[DEBUG] Attempting to parse GPT Critic JSON.")
    rating = json.loads(raw_text)  # let it raise an error if not valid

    for key in ["accuracy","clarity","completeness","conciseness","style"]:
        if key not in rating:
            raise ValueError(f"[DEBUG] Missing '{key}' in GPT critic response: {rating}")
        rating[key] = float(rating[key])

    print("[DEBUG] GPT Critic final rating object:", rating)
    return rating

# ------------------------------------------- #
# 4) EVALUATION MAIN LOGIC
# ------------------------------------------- #

async def evaluate_models_minimal():
    """
    Loads domain prompts, calls Anthropic & Grok for each question,
    then rates each answer with GPT as a critic, storing minimal results.
    Includes debug logs at each stage, no quiet fallback to 3.0.
    """
    start_time = time.time()

    # Load domain prompts
    try:
        with open(DOMAIN_PROMPTS_FILE, "r", encoding="utf-8") as f:
            domain_prompts = json.load(f)
    except FileNotFoundError:
        print(f"ERROR: {DOMAIN_PROMPTS_FILE} not found.")
        return

    results = {}
    all_domains = list(domain_prompts.keys())
    total_prompts = sum(len(domain_prompts[d]) for d in all_domains)
    current_count = 0

    print(f"Starting evaluation for {total_prompts} total prompts across {len(all_domains)} domains.\n")

    # Evaluate domain by domain
    for domain in all_domains:
        prompts = domain_prompts[domain]
        domain_records = []
        print(f"--- Evaluating domain: '{domain}' with {len(prompts)} prompts ---")

        for i, prompt in enumerate(prompts):
            current_count += 1
            print(f"\n[{current_count}/{total_prompts}] Prompt {i+1}/{len(prompts)} in '{domain}': {prompt[:60]}...")

            try:
                # 1) Call Anthropic & Grok in parallel
                anthro_task = call_anthropic_api(prompt)
                grok_task   = call_grok_api(prompt)
                anthro_answer, grok_answer = await asyncio.gather(anthro_task, grok_task)

                print("[DEBUG] anthro_answer snippet:", anthro_answer[:80], "...")
                print("[DEBUG] grok_answer snippet:", grok_answer[:80], "...")

                # 2) Rate each answer with GPT as critic
                rate_anthro_task = call_gpt_critic(prompt, anthro_answer)
                rate_grok_task   = call_gpt_critic(prompt, grok_answer)
                anthro_rating, grok_rating = await asyncio.gather(rate_anthro_task, rate_grok_task)

                # 3) Store minimal record
                record = {
                    "prompt": prompt,
                    "anthropic_rating": anthro_rating,
                    "grok_rating": grok_rating
                }
                domain_records.append(record)

            except Exception as e:
                # store the error for debugging
                print("[DEBUG] Caught Exception:", e)
                domain_records.append({
                    "prompt": prompt,
                    "error": str(e)
                })

        results[domain] = domain_records
        print(f"\n--- Finished domain: {domain} ---")

    # Write results to JSON
    with open(EVALUATION_OUTPUT_FILE, "w", encoding="utf-8") as out:
        json.dump(results, out, indent=2, ensure_ascii=False)

    # Print total time
    end_time = time.time()
    elapsed_min = (end_time - start_time) / 60
    print(f"\nEvaluation complete! Minimal results stored in '{EVALUATION_OUTPUT_FILE}'.")
    print(f"Total time elapsed: ~{elapsed_min:.1f} minutes.")

# ------------------------------------------- #
# 5) ENTRY POINT
# ------------------------------------------- #

if __name__ == "__main__":
    asyncio.run(evaluate_models_minimal())

from fastapi import HTTPException
import httpx
import asyncio
import json
import time

# URLS and Keys (Replace with real ones or environment variables)
OPENAI_API_URL = "https://api.openai.com/v1/chat/completions"
DEEPSEEK_API_URL = "https://api.deepseek.com/chat/completions"
OPENAI_API_KEY = "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"
DEEPSEEK_API_KEY = "sk-e52df5836dd94d3f9163361f72a5499c"

DOMAIN_PROMPTS_FILE = "domain_prompts.json"       # 6 domains x 20 prompts each
EVALUATION_OUTPUT_FILE = "model_evaluation.json"  # We'll store minimal results here

# Critic prompt: We now explicitly request decimal (floating-point) ratings
# and instruct the critic to be harsh.
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

# -------------------- #
# 1) API CALL FUNCTIONS
# -------------------- #

async def call_openai_api(prompt: str) -> str:
    """
    Calls the OpenAI GPT-3.5-turbo model, returning the response text.
    """
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "model": "gpt-3.5-turbo",
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.7
    }

    async with httpx.AsyncClient() as client:
        response = await client.post(OPENAI_API_URL, headers=headers, json=data)
        if response.status_code != 200:
            raise HTTPException(status_code=response.status_code, detail=response.text)
        return response.json()["choices"][0]["message"]["content"]


async def call_deepseek_api(prompt: str) -> str:
    """
    Calls the DeepSeek API with a 60s timeout, returning the response text.
    """
    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "model": "deepseek-chat",
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "stream": False
    }
    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(DEEPSEEK_API_URL, headers=headers, json=data)
        if response.status_code != 200:
            raise HTTPException(status_code=response.status_code, detail=response.text)
        return response.json()["choices"][0]["message"]["content"]


# --------------------------------- #
# 2) RATING (CRITIC) HELPER FUNCTION
# --------------------------------- #

async def rate_answer_with_gpt(prompt: str, answer: str) -> dict:
    """
    Uses GPT to rate the 'answer' on 5 categories (accuracy, clarity, etc.),
    allowing for decimal scores. Returns a dict with float values. Example:
    {
      "accuracy": 4.2,
      "clarity": 3.7,
      "completeness": 4.0,
      "conciseness": 3.5,
      "style": 4.0
    }
    """
    rating_prompt = CRITIC_PROMPT_TEMPLATE.format(prompt=prompt, answer=answer)
    try:
        raw_rating = await call_openai_api(rating_prompt)
    except HTTPException:
        # If GPT call fails, fallback to a default rating
        return {"accuracy": 3.0, "clarity": 3.0, "completeness": 3.0, "conciseness": 3.0, "style": 3.0}

    # Attempt to parse JSON
    try:
        rating = json.loads(raw_rating)
        # Ensure keys exist, cast to float
        for key in ["accuracy", "clarity", "completeness", "conciseness", "style"]:
            if key not in rating:
                rating[key] = 3.0
            else:
                # Attempt to cast to float
                try:
                    rating[key] = float(rating[key])
                except:
                    rating[key] = 3.0
        return rating

    except:
        # If parsing fails or invalid JSON, return a default
        return {"accuracy": 3.0, "clarity": 3.0, "completeness": 3.0, "conciseness": 3.0, "style": 3.0}


# ---------------------------- #
# 3) EVALUATION MAIN LOGIC
# ---------------------------- #

async def evaluate_models_minimal():
    """
    Loads domain prompts, calls GPT & DeepSeek for each question,
    rates each answer with GPT, and stores only the prompt + scores in JSON.
    Also prints progress so you can track how it's going.
    """

    # Track start time
    start_time = time.time()

    # Load prompts
    try:
        with open(DOMAIN_PROMPTS_FILE, "r", encoding="utf-8") as f:
            domain_prompts = json.load(f)
    except FileNotFoundError:
        print(f"ERROR: {DOMAIN_PROMPTS_FILE} not found.")
        return

    results = {}

    # Count total prompts for progress tracking
    all_domains = list(domain_prompts.keys())
    total_prompts = sum(len(domain_prompts[d]) for d in all_domains)
    current_count = 0

    print(f"Starting evaluation for {total_prompts} total prompts across {len(all_domains)} domains.")
    print("Depending on API speeds, this could take 10–30 minutes or more.\n")

    # Iterate domains
    for domain in all_domains:
        prompts = domain_prompts[domain]
        domain_records = []
        print(f"--- Evaluating domain: '{domain}' with {len(prompts)} prompts ---")

        for i, prompt in enumerate(prompts):
            current_count += 1
            # Print progress
            print(f"[{current_count}/{total_prompts}] Prompt {i+1}/{len(prompts)} in '{domain}': {prompt[:60]}...")

            try:
                # 1) Call GPT & DeepSeek concurrently
                task_gpt = call_openai_api(prompt)
                task_ds = call_deepseek_api(prompt)
                gpt_answer, ds_answer = await asyncio.gather(task_gpt, task_ds)

                # 2) Rate each answer with GPT as critic
                task_gpt_rating = rate_answer_with_gpt(prompt, gpt_answer)
                task_ds_rating = rate_answer_with_gpt(prompt, ds_answer)
                gpt_rating, ds_rating = await asyncio.gather(task_gpt_rating, task_ds_rating)

                # 3) Store only the prompt + rating dicts
                record = {
                    "prompt": prompt,
                    "gpt_rating": gpt_rating,
                    "deepseek_rating": ds_rating
                }
                domain_records.append(record)

            except Exception as e:
                domain_records.append({
                    "prompt": prompt,
                    "error": str(e)
                })

        results[domain] = domain_records
        print(f"--- Finished domain: {domain} ---\n")

    # Write minimal results
    with open(EVALUATION_OUTPUT_FILE, "w", encoding="utf-8") as out:
        json.dump(results, out, indent=2, ensure_ascii=False)

    # Calculate total time
    end_time = time.time()
    elapsed_sec = end_time - start_time
    elapsed_min = elapsed_sec / 60.0

    print(f"Evaluation complete! Minimal results stored in '{EVALUATION_OUTPUT_FILE}'.")
    print(f"Total time elapsed: ~{elapsed_min:.1f} minutes.")

# --------------------- #
# 4) ENTRY POINT
# --------------------- #

if __name__ == "__main__":
    asyncio.run(evaluate_models_minimal())

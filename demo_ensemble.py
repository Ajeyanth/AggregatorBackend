import asyncio
import httpx
import json
from fastapi import HTTPException

# Hard-coded placeholders; replace with real or env variables.
OPENAI_API_KEY = "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"
DEEPSEEK_API_KEY = "sk-e52df5836dd94d3f9163361f72a5499c"
OPENAI_API_URL = "https://api.openai.com/v1/chat/completions"
DEEPSEEK_API_URL = "https://api.deepseek.com/chat/completions"

# Suppose from your offline analysis for finance,
# GPT is better at accuracy & completeness,
# DeepSeek is better at clarity, conciseness, and style.
category_strengths = {
    "finance": {
        "accuracy": "GPT",
        "clarity": "DeepSeek",
        "completeness": "GPT",
        "conciseness": "DeepSeek",
        "style": "DeepSeek"
    }
    # You could have more domains if you like...
}

async def call_openai_api(prompt: str) -> str:
    """
    Calls GPT-3.5-turbo (OpenAI) with the given prompt, returns string response.
    """
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "model": "gpt-3.5-turbo",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.7
    }
    async with httpx.AsyncClient() as client:
        resp = await client.post(OPENAI_API_URL, headers=headers, json=data)
        if resp.status_code != 200:
            raise HTTPException(status_code=resp.status_code, detail=resp.text)
        return resp.json()["choices"][0]["message"]["content"]

async def call_deepseek_api(prompt: str) -> str:
    """
    Calls DeepSeek with the given prompt, returns string response.
    """
    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "model": "deepseek-chat",
        "messages": [{"role": "user", "content": prompt}],
        "stream": False
    }
    async with httpx.AsyncClient(timeout=60.0) as client:
        resp = await client.post(DEEPSEEK_API_URL, headers=headers, json=data)
        if resp.status_code != 200:
            raise HTTPException(status_code=resp.status_code, detail=resp.text)
        return resp.json()["choices"][0]["message"]["content"]

async def refine_ensemble(prompt: str, ans_gpt: str, ans_ds: str, strengths_map: dict) -> str:
    """
    Uses GPT as a "Refiner" to merge GPT + DeepSeek answers, referencing the
    category strengths for the domain (e.g., GPT for accuracy, DS for clarity).
    """
    # Turn the strengths_map into a short text we can feed into the prompt
    # e.g. "GPT is better at accuracy, completeness; DeepSeek is better at clarity, conciseness, style."
    categories_text = []
    for cat, winner in strengths_map.items():
        categories_text.append(f"{cat} => {winner}")
    strengths_summary = "; ".join(categories_text)

    refiner_prompt = f"""
The user asked: "{prompt}"

We have two candidate answers:
1) GPT: {ans_gpt}
2) DeepSeek: {ans_ds}

Based on offline evaluations for this domain:
{strengths_summary}

Please produce a single final answer that leverages:
- GPT's strengths where GPT is better,
- DeepSeek's strengths where DeepSeek is better,
- and resolves any conflicts.

Return the best merged answer, ensuring maximum overall quality.
"""

    return await call_openai_api(refiner_prompt)

async def main():
    # 1) Hard-coded test user query:
    user_query = "What are derivatives and how do they work in hedging risk?"
    domain = "finance"  # Let's assume we already determined it's finance

    # 2) Retrieve category strengths for 'finance'
    strengths_for_finance = category_strengths["finance"]
    # e.g. {"accuracy": "GPT", "clarity": "DeepSeek", ...}

    # 3) Call both models in parallel for the actual answer
    ans_gpt_task = call_openai_api(user_query)
    ans_ds_task  = call_deepseek_api(user_query)
    ans_gpt, ans_ds = await asyncio.gather(ans_gpt_task, ans_ds_task)

    # 4) Merge using the refiner approach
    final_answer = await refine_ensemble(user_query, ans_gpt, ans_ds, strengths_for_finance)

    print("----- GPT's Raw Answer -----")
    print(ans_gpt)
    print("\n----- DeepSeek's Raw Answer -----")
    print(ans_ds)
    print("\n----- Refined Ensemble Answer -----")
    print(final_answer)

if __name__ == "__main__":
    asyncio.run(main())

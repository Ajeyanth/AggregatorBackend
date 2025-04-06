import asyncio
import httpx
import json
import os
from fastapi import HTTPException

# Replace with your own or environment variable
OPENAI_API_KEY = "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"
OPENAI_API_URL = "https://api.openai.com/v1/chat/completions"

# Path to the JSON file where we store which model is best per domain/category
CATEGORY_STRENGTHS_FILE = "category_strengths.json"

async def call_openai_api(prompt: str) -> str:
    """
    Calls GPT-3.5-turbo with the given 'prompt' and returns the response text.
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
        response = await client.post(OPENAI_API_URL, headers=headers, json=data)
        if response.status_code != 200:
            raise HTTPException(status_code=response.status_code, detail=response.text)
        return response.json()["choices"][0]["message"]["content"]

async def gpt_classify_domain(query: str) -> str:
    """
    Asks GPT to classify the user query into exactly ONE of these domains:
    [finance, coding, legal, healthcare, marketing, general].
    Returns the domain as a lowercase string.
    """
    classification_prompt = f"""
    You are given a user query.
    Classify it into exactly ONE of the following domains:
    [finance, coding, legal, healthcare, marketing, general].

    The query is: {query}

    Respond ONLY with the domain name,
    and no extra text.
    """
    raw_response = await call_openai_api(classification_prompt)

    domain = raw_response.strip().lower()

    valid_domains = ["finance", "coding", "legal", "healthcare", "marketing", "general"]
    if domain not in valid_domains:
        domain = "general"
    return domain

async def main():
    # 1) Load the category strengths from a JSON file
    #    This file should look like:
    #    {
    #      "finance": {
    #         "accuracy": "GPT",
    #         "clarity": "DeepSeek",
    #         ...
    #      },
    #      "coding": {
    #         ...
    #      },
    #      ...
    #    }
    if not os.path.exists(CATEGORY_STRENGTHS_FILE):
        print(f"ERROR: The file '{CATEGORY_STRENGTHS_FILE}' does not exist.")
        print("Please generate it or place it in the same directory.")
        return

    with open(CATEGORY_STRENGTHS_FILE, "r", encoding="utf-8") as f:
        category_strengths = json.load(f)

    print("GPT-based Domain Classifier Console")
    print("Type a question, I'll classify it into one of [finance, coding, legal, healthcare, marketing, general],")
    print("then show you which model is best at each category for that domain.")
    print("Type 'exit' to quit.\n")

    while True:
        user_input = input("Enter your question: ").strip()
        if user_input.lower() == "exit":
            print("Goodbye!")
            break

        if not user_input:
            # If user just presses enter, skip
            print("No input provided. Try again or type 'exit' to quit.")
            continue

        try:
            # Step 1: Classify domain via GPT
            domain = await gpt_classify_domain(user_input)
            print(f"\nDetected Domain: {domain}")

            # Step 2: Retrieve category strengths
            # If domain not in the file, default to "general"
            strengths_map = category_strengths.get(domain, category_strengths.get("general", {}))

            # If still empty, fallback
            if not strengths_map:
                print("No category strengths data found for this domain; defaulting everything to GPT.\n")
                continue

            # Step 3: Print which model is best for each category
            print("Category strengths for this domain:")
            for category, best_model in strengths_map.items():
                print(f"  - {category}: {best_model}")

            print()  # Extra line break
        except Exception as e:
            print(f"Error occurred: {e}\n")

if __name__ == "__main__":
    asyncio.run(main())

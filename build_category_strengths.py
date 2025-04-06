import json
import os

# Two evaluation files:
#   1) eval_gpt_deepseek.json  (or "model_evaluation.json")
#   2) eval_anthropic_grok.json (or "model_evaluation_anthropic_grok.json")
EVAL_FILE_1 = "model_evaluation.json"  # GPT & DeepSeek results
EVAL_FILE_2 = "model_evaluation_anthropic_grok.json"  # Anthropic & Grok results

# Output for final domain strengths
OUTPUT_FILE = "category_strengths.json"

CATEGORIES = ["accuracy", "clarity", "completeness", "conciseness", "style"]
MODELS = ["GPT", "DeepSeek", "Anthropic", "Grok"]


def load_eval_data(path):
    """
    Loads the JSON from a single evaluation file.
    Example structure:
    {
      "finance": [
         {
            "prompt": "...",
            "gpt_rating": { "accuracy":4.3, ... },
            "deepseek_rating": { ... }
         },
         ...
      ],
      "coding": [...]
    }
    Or for the second file: "anthropic_rating", "grok_rating"
    Returns a dict.
    """
    if not os.path.exists(path):
        print(f"File not found: {path}")
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def merge_evals(eval1, eval2):
    """
    Merges the rating data from eval1 (GPT & DeepSeek) and eval2 (Anthropic & Grok)
    into a single structure keyed by domain -> list of records:
      {
        "prompt": "...",
        "GPT": {...cat scores...},
        "DeepSeek": {...cat scores...},
        "Anthropic": {...cat scores...},
        "Grok": {...cat scores...}
      }

    We'll match records by the "prompt" string. If no matching prompt is found in
    second file, we only keep the first's data, or vice versa.
    """
    merged = {}

    # step 1: read all domains from eval1
    for domain, records in eval1.items():
        merged[domain] = []
        for rec in records:
            prompt = rec.get("prompt", "")
            # build a new struct
            # GPT & DeepSeek
            new_record = {
                "prompt": prompt,
                "GPT": rec.get("gpt_rating", {}),
                "DeepSeek": rec.get("deepseek_rating", {}),
                "Anthropic": {},
                "Grok": {}
            }
            merged[domain].append(new_record)

    # step 2: incorporate data from eval2 (Anthropic & Grok)
    # if domain not in merged, create it
    for domain, records in eval2.items():
        if domain not in merged:
            merged[domain] = []
        for rec in records:
            prompt = rec.get("prompt", "")
            # find matching record by prompt
            # or create new if not found
            existing = None
            for x in merged[domain]:
                if x["prompt"] == prompt:
                    existing = x
                    break
            if existing is None:
                # new record
                existing = {
                    "prompt": prompt,
                    "GPT": {},
                    "DeepSeek": {},
                    "Anthropic": rec.get("anthropic_rating", {}),
                    "Grok": rec.get("grok_rating", {})
                }
                merged[domain].append(existing)
            else:
                # update existing
                existing["Anthropic"] = rec.get("anthropic_rating", {})
                existing["Grok"] = rec.get("grok_rating", {})

    return merged


def compute_best_model_per_domain(merged_data):
    """
    merged_data is domain -> list of records
       each record has "prompt", "GPT", "DeepSeek", "Anthropic", "Grok" dicts (the rating dict).
    We'll accumulate sums for each model, then pick best for each category in that domain.
    Returns a dict:
      {
        "finance": {
           "accuracy": "GPT",
           "clarity": "Anthropic",
           ...
        },
        ...
      }
    """
    domain_strengths = {}

    for domain, records in merged_data.items():
        # We'll track sums for each model in each category
        sums = {model: {cat:0.0 for cat in CATEGORIES} for model in MODELS}
        counts = {model: 0 for model in MODELS}  # how many valid rating records

        for rec in records:
            # rec[model] is a dict with cat => score
            for model in MODELS:
                rating_dict = rec.get(model, {})
                if not isinstance(rating_dict, dict) or not rating_dict:
                    continue
                # increment count
                counts[model] += 1
                # accumulate
                for cat in CATEGORIES:
                    val = rating_dict.get(cat, 0.0)
                    sums[model][cat] += val

        # compute averages
        avgs = {model: {cat:0.0 for cat in CATEGORIES} for model in MODELS}
        for model in MODELS:
            c = counts[model]
            if c > 0:
                for cat in CATEGORIES:
                    avgs[model][cat] = sums[model][cat] / c
            else:
                # no data
                for cat in CATEGORIES:
                    avgs[model][cat] = 0.0

        # pick best model for each category
        cat_map = {}
        for cat in CATEGORIES:
            # find which model has the highest avgs[model][cat]
            best_model = None
            best_score = float("-inf")
            for model in MODELS:
                score = avgs[model][cat]
                if score > best_score:
                    best_score = score
                    best_model = model
            cat_map[cat] = best_model

        domain_strengths[domain] = cat_map

    return domain_strengths


def main():
    # 1) load data from the two eval files
    eval_data_1 = load_eval_data(EVAL_FILE_1)  # GPT & DeepSeek
    eval_data_2 = load_eval_data(EVAL_FILE_2)  # Anthropic & Grok

    # 2) merge
    merged = merge_evals(eval_data_1, eval_data_2)

    # 3) compute best model in each domain/category
    domain_strengths = compute_best_model_per_domain(merged)

    # 4) write out to category_strengths.json
    with open(OUTPUT_FILE, "w", encoding="utf-8") as out:
        json.dump(domain_strengths, out, indent=2, ensure_ascii=False)

    print(f"Wrote category strengths to '{OUTPUT_FILE}'.\nSample content:")
    for domain, catmap in domain_strengths.items():
        print(f"Domain: {domain} => {catmap}")


if __name__ == "__main__":
    main()

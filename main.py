import os
import time
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
    api_key=os.environ["OPENROUTER_API_KEY"],
    base_url="https://openrouter.ai/api/v1",
)

QUESTION = "What is the difference between a process and a thread? Answer in 3 sentences."

MODELS = [
    "meta-llama/llama-3.1-8b-instruct",
    "qwen/qwen-2.5-7b-instruct",
    "qwen/qwen-2.5-72b-instruct",
    "google/gemini-2.5-flash-lite",
]

# price per 1M tokens in USD
PRICES = {
    "meta-llama/llama-3.1-8b-instruct": (0.02, 0.03),
    "qwen/qwen-2.5-7b-instruct":         (0.04, 0.10),
    "qwen/qwen-2.5-72b-instruct":        (0.36, 0.40),
    "google/gemini-2.5-flash-lite":      (0.10, 0.40),
}


TIMEOUT = 30  # seconds per model call


def ask(question, model):
    in_price, out_price = PRICES[model]

    start = time.perf_counter()
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": question}],
        timeout=TIMEOUT,
    )
    latency = time.perf_counter() - start

    answer  = response.choices[0].message.content
    in_tok  = response.usage.prompt_tokens
    out_tok = response.usage.completion_tokens
    cost    = (in_tok * in_price + out_tok * out_price) / 1_000_000

    return answer, latency, in_tok, out_tok, cost


results = {}
for model_id in MODELS:
    try:
        results[model_id] = ask(QUESTION, model_id)
    except Exception as e:
        results[model_id] = e

COL_MODEL   = 36
COL_PREVIEW = 52
COL_LATENCY = 10
COL_COST    = 12

def truncate(text, width):
    return text[:width - 1] + "…" if len(text) > width else text

header = (
    f"{'Model':<{COL_MODEL}}"
    f"{'Preview':<{COL_PREVIEW}}"
    f"{'Latency':>{COL_LATENCY}}"
    f"{'Cost':>{COL_COST}}"
)
divider = "-" * len(header)

print(f"\n{divider}\n{header}\n{divider}")

for model_id in MODELS:
    result = results[model_id]
    if isinstance(result, Exception):
        preview = f"ERROR: {result}"
        lat_str = cost_str = "—"
    else:
        answer, latency, in_tok, out_tok, cost = result
        preview  = answer.replace("\n", " ")
        lat_str  = f"{latency:.2f}s"
        cost_str = f"${cost:.6f}"

    print(
        f"{truncate(model_id, COL_MODEL):<{COL_MODEL}}"
        f"{truncate(preview,  COL_PREVIEW):<{COL_PREVIEW}}"
        f"{lat_str:>{COL_LATENCY}}"
        f"{cost_str:>{COL_COST}}"
    )

print(divider)

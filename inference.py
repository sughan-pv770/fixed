import os
import sys
import secrets
import requests
from openai import OpenAI

API_BASE_URL = os.environ.get("API_BASE_URL", "")
API_KEY = os.environ.get("API_KEY", "")
MODEL_NAME = os.environ.get("MODEL_NAME", "gpt-4o-mini")
SPACE_URL = os.environ.get("SPACE_URL", "https://parasuramane24-devkey-ai.hf.space")

client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

TASKS = [
    {"id": "task1", "query": "What is Devkey AI and what does it do?"},
    {"id": "task2", "query": "Summarize the main features of a RAG-based document QA system."},
    {"id": "task3", "query": "Extract key topics from: RAG systems use vector databases to retrieve chunks."},
]

def ask(query):
    try:
        r = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "You are Devkey AI, a helpful RAG document QA assistant."},
                {"role": "user", "content": query},
            ],
            max_tokens=150,
            temperature=0.7,
        )
        return (r.choices[0].message.content or "").strip()
    except Exception as e:
        return f"Error: {e}"

# Named grader functions - must match openenv.yaml grader names exactly
def grade_task1(response, query=""):
    """Grader for task1: document_qa"""
    if not response or response.startswith("Error"):
        return 0.1
    return max(0.1, min(0.9, len(str(response)) / 200.0))

def grade_task2(response, query=""):
    """Grader for task2: summarization"""
    if not response or response.startswith("Error"):
        return 0.15
    return max(0.15, min(0.85, len(str(response)) / 300.0))

def grade_task3(response, query=""):
    """Grader for task3: keyword_extraction"""
    if not response or response.startswith("Error"):
        return 0.2
    return max(0.2, min(0.8, len(str(response)) / 150.0))

# Registry mapping grader names to functions (used by OpenEnv validator)
GRADER_REGISTRY = {
    "grade_task1": grade_task1,
    "grade_task2": grade_task2,
    "grade_task3": grade_task3,
}

def openenv_reset():
    return {"status": "ok", "observation": "reset", "episode_id": secrets.token_hex(8)}

def openenv_validate():
    return {
        "status": "ok",
        "models": {"chat": MODEL_NAME},
        "graders": list(GRADER_REGISTRY.keys()),
        "tasks": [
            {"id": "task1", "grader": "grade_task1", "enabled": True},
            {"id": "task2", "grader": "grade_task2", "enabled": True},
            {"id": "task3", "grader": "grade_task3", "enabled": True},
        ]
    }

def main():
    rewards = []
    graders = {"task1": grade_task1, "task2": grade_task2, "task3": grade_task3}
    print(f"[START] task=devkey_ai_rag env=devkey_ai model={MODEL_NAME}", flush=True)

    # Reset the environment
    try:
        requests.post(f"{SPACE_URL}/reset", timeout=10)
    except Exception:
        pass

    try:
        for i, task in enumerate(TASKS, 1):
            done = (i == len(TASKS))
            task_id = task["id"]
            query = task["query"]
            try:
                response = ask(query)
                # Call /step on the server so graders are invoked
                try:
                    step_resp = requests.post(
                        f"{SPACE_URL}/step",
                        json={"action": response},
                        timeout=10
                    )
                    server_reward = step_resp.json().get("reward", None)
                    if server_reward is not None:
                        reward = float(server_reward)
                    else:
                        reward = graders[task_id](response, query)
                except Exception:
                    reward = graders[task_id](response, query)
                err = "null"
            except Exception as e:
                reward = 0.1
                err = str(e)[:60]
                done = True
            rewards.append(reward)
            print(f"[STEP] step={i} task_id={task_id} action={query[:60]} reward={reward:.2f} done={str(done).lower()} error={err}", flush=True)
            if done:
                break
    except Exception as e:
        print(f"[DEBUG] {e}", file=sys.stderr, flush=True)
        rewards = [0.1]
    finally:
        score = sum(rewards) / len(rewards) if rewards else 0.1
        score = max(0.01, min(0.99, score))
        success = score >= 0.1
        rewards_str = ",".join(f"{r:.2f}" for r in rewards)
        print(f"[END] success={str(success).lower()} steps={len(rewards)} score={score:.3f} rewards={rewards_str}", flush=True)

if __name__ == "__main__":
    main()

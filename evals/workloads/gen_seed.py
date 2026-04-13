#!/usr/bin/env python3
"""Generate synthetic seed workloads for the evaluation harness.

Each workload has 10 samples — enough to exercise the harness end-to-end.
Real workload captures (200+ samples) will replace these.

Usage:
    uv run python evals/workloads/gen_seed.py
"""

from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).parent

BOILERPLATE_SYSTEM = (
    "You are an expert coding assistant. You help users with software "
    "engineering tasks including writing code, debugging, refactoring, "
    "explaining code, and answering questions about software development. "
    "Always provide clear, concise, and accurate responses. When writing "
    "code, follow best practices and include brief comments where helpful. "
    "If you are unsure about something, say so rather than guessing. "
) * 4  # ~800 chars of repeated boilerplate — realistic for agent system prompts

PYTHON_FILE = '''def merge_sort(arr):
    """Sort a list using merge sort algorithm."""
    if len(arr) <= 1:
        return arr
    mid = len(arr) // 2
    left = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])
    return merge(left, right)

def merge(left, right):
    result = []
    i = j = 0
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1
    result.extend(left[i:])
    result.extend(right[j:])
    return result

def quick_sort(arr):
    """Sort a list using quick sort algorithm."""
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quick_sort(left) + middle + quick_sort(right)
'''

RAG_CONTEXT = (
    "Retrieved context from codebase:\n\n"
    "File: src/auth/middleware.py (lines 1-80)\n"
    "```python\n"
    "class AuthMiddleware:\n"
    "    def __init__(self, app, secret_key):\n"
    "        self.app = app\n"
    "        self.secret_key = secret_key\n"
    "    \n"
    "    async def __call__(self, scope, receive, send):\n"
    "        if scope['type'] == 'http':\n"
    "            headers = dict(scope.get('headers', []))\n"
    "            token = headers.get(b'authorization', b'').decode()\n"
    "            if not self.verify_token(token):\n"
    "                await self.send_401(send)\n"
    "                return\n"
    "        await self.app(scope, receive, send)\n"
    "```\n\n"
    "File: src/auth/tokens.py (lines 1-45)\n"
    "```python\n"
    "import jwt\n"
    "import time\n\n"
    "def create_token(user_id, secret, ttl=3600):\n"
    "    payload = {'sub': user_id, 'exp': time.time() + ttl}\n"
    "    return jwt.encode(payload, secret, algorithm='HS256')\n\n"
    "def verify_token(token, secret):\n"
    "    try:\n"
    "        return jwt.decode(token, secret, algorithms=['HS256'])\n"
    "    except jwt.ExpiredSignatureError:\n"
    "        return None\n"
    "```\n\n"
    "File: docs/AUTH.md (lines 1-30)\n"
    "# Authentication\n"
    "The API uses JWT tokens for authentication. Tokens are issued on login "
    "and must be included in the Authorization header of subsequent requests.\n"
    "Tokens expire after 1 hour by default.\n"
) * 2  # ~2K chars

# ---------------------------------------------------------------------------
# WL1 — edit-heavy
# ---------------------------------------------------------------------------

WL1 = [
    {
        "id": "wl1_001",
        "messages": [
            {"role": "system", "content": BOILERPLATE_SYSTEM},
            {"role": "user", "content": f"Rename the function `merge_sort` to `merge_sort_recursive` in this file:\n```python\n{PYTHON_FILE}\n```"},
        ],
        "labels": {"trivial": True, "edit": True},
    },
    {
        "id": "wl1_002",
        "messages": [
            {"role": "system", "content": BOILERPLATE_SYSTEM},
            {"role": "user", "content": f"Add type hints to all functions in this file:\n```python\n{PYTHON_FILE}\n```"},
        ],
        "labels": {"trivial": False, "edit": True},
    },
    {
        "id": "wl1_003",
        "messages": [
            {"role": "system", "content": BOILERPLATE_SYSTEM},
            {"role": "user", "content": f"Refactor merge_sort and quick_sort to share a common base class SortAlgorithm with a sort() method:\n```python\n{PYTHON_FILE}\n```"},
        ],
        "labels": {"trivial": False, "edit": True},
    },
    {
        "id": "wl1_004",
        "messages": [
            {"role": "system", "content": BOILERPLATE_SYSTEM},
            {"role": "user", "content": f"Fix the bug in the merge function — it doesn't handle empty lists correctly:\n```python\n{PYTHON_FILE}\n```"},
        ],
        "labels": {"trivial": False, "edit": True},
    },
    {
        "id": "wl1_005",
        "messages": [
            {"role": "system", "content": BOILERPLATE_SYSTEM},
            {"role": "user", "content": "Add a docstring to the merge function that says 'Merge two sorted lists'."},
        ],
        "labels": {"trivial": True, "edit": True},
    },
    {
        "id": "wl1_006",
        "messages": [
            {"role": "system", "content": BOILERPLATE_SYSTEM},
            {"role": "user", "content": f"Convert the recursive merge_sort to an iterative bottom-up implementation:\n```python\n{PYTHON_FILE}\n```"},
        ],
        "labels": {"trivial": False, "edit": True},
    },
    {
        "id": "wl1_007",
        "messages": [
            {"role": "system", "content": BOILERPLATE_SYSTEM},
            {"role": "user", "content": "What's the time complexity of merge sort?"},
        ],
        "labels": {"trivial": True, "edit": False},
    },
    {
        "id": "wl1_008",
        "messages": [
            {"role": "system", "content": BOILERPLATE_SYSTEM},
            {"role": "user", "content": f"Add unit tests for merge_sort and quick_sort using pytest:\n```python\n{PYTHON_FILE}\n```"},
        ],
        "labels": {"trivial": False, "edit": True},
    },
    {
        "id": "wl1_009",
        "messages": [
            {"role": "system", "content": BOILERPLATE_SYSTEM},
            {"role": "user", "content": "Remove the quick_sort function, it's no longer needed."},
        ],
        "labels": {"trivial": True, "edit": True},
    },
    {
        "id": "wl1_010",
        "messages": [
            {"role": "system", "content": BOILERPLATE_SYSTEM},
            {"role": "user", "content": f"Optimize this code to use less memory by doing in-place sorting:\n```python\n{PYTHON_FILE}\n```"},
        ],
        "labels": {"trivial": False, "edit": True},
    },
]

# ---------------------------------------------------------------------------
# WL2 �� explanation-heavy
# ---------------------------------------------------------------------------

WL2 = [
    {
        "id": "wl2_001",
        "messages": [
            {"role": "system", "content": BOILERPLATE_SYSTEM},
            {"role": "user", "content": "What does this file do? Explain each function."},
        ],
        "labels": {"trivial": True},
    },
    {
        "id": "wl2_002",
        "messages": [
            {"role": "system", "content": BOILERPLATE_SYSTEM},
            {"role": "user", "content": "How does the merge sort algorithm work? Explain step by step with an example."},
        ],
        "labels": {"trivial": True},
    },
    {
        "id": "wl2_003",
        "messages": [
            {"role": "system", "content": BOILERPLATE_SYSTEM},
            {"role": "user", "content": "What are the trade-offs between merge sort and quick sort in terms of time complexity, space complexity, and stability?"},
        ],
        "labels": {"trivial": False},
    },
    {
        "id": "wl2_004",
        "messages": [
            {"role": "system", "content": BOILERPLATE_SYSTEM},
            {"role": "user", "content": "Explain the difference between stable and unstable sorting algorithms."},
        ],
        "labels": {"trivial": True},
    },
    {
        "id": "wl2_005",
        "messages": [
            {"role": "system", "content": BOILERPLATE_SYSTEM},
            {"role": "user", "content": f"Walk me through how this merge function works line by line:\n```python\n{PYTHON_FILE}\n```"},
        ],
        "labels": {"trivial": False},
    },
    {
        "id": "wl2_006",
        "messages": [
            {"role": "system", "content": BOILERPLATE_SYSTEM},
            {"role": "user", "content": "Why does Python's built-in sort use Timsort instead of merge sort or quick sort?"},
        ],
        "labels": {"trivial": False},
    },
    {
        "id": "wl2_007",
        "messages": [
            {"role": "system", "content": BOILERPLATE_SYSTEM},
            {"role": "user", "content": "What is the worst-case space complexity of merge sort?"},
        ],
        "labels": {"trivial": True},
    },
    {
        "id": "wl2_008",
        "messages": [
            {"role": "system", "content": BOILERPLATE_SYSTEM},
            {"role": "user", "content": "Can you explain recursion to me using merge sort as an example? I'm new to programming."},
        ],
        "labels": {"trivial": True},
    },
    {
        "id": "wl2_009",
        "messages": [
            {"role": "system", "content": BOILERPLATE_SYSTEM},
            {"role": "user", "content": "What would happen if the pivot selection in quick sort always chose the smallest element?"},
        ],
        "labels": {"trivial": False},
    },
    {
        "id": "wl2_010",
        "messages": [
            {"role": "system", "content": BOILERPLATE_SYSTEM},
            {"role": "user", "content": "Compare the practical performance of merge sort vs quick sort for sorting 1 million random integers in Python."},
        ],
        "labels": {"trivial": False},
    },
]

# ---------------------------------------------------------------------------
# WL3 — mixed chat
# ---------------------------------------------------------------------------

WL3 = [
    {"id": "wl3_001", "messages": [{"role": "user", "content": "What is Python?"}], "labels": {"trivial": True}},
    {"id": "wl3_002", "messages": [{"role": "user", "content": "How do I install pip?"}], "labels": {"trivial": True}},
    {"id": "wl3_003", "messages": [{"role": "user", "content": "Explain the differences between REST and GraphQL APIs, with pros and cons for each approach."}], "labels": {"trivial": False}},
    {"id": "wl3_004", "messages": [{"role": "user", "content": "What's a good name for a variable that holds a list of users?"}], "labels": {"trivial": True}},
    {"id": "wl3_005", "messages": [{"role": "user", "content": "Design a database schema for a social media app that supports posts, comments, likes, follows, and direct messages. Include proper indexing strategy."}], "labels": {"trivial": False}},
    {"id": "wl3_006", "messages": [{"role": "user", "content": "What does 'git rebase' do?"}], "labels": {"trivial": True}},
    {"id": "wl3_007", "messages": [{"role": "user", "content": "Write a Python function that checks if a string is a palindrome."}], "labels": {"trivial": True}},
    {"id": "wl3_008", "messages": [{"role": "user", "content": "Explain how garbage collection works in Python, including reference counting and the generational collector."}], "labels": {"trivial": False}},
    {"id": "wl3_009", "messages": [{"role": "user", "content": "What's the difference between == and is in Python?"}], "labels": {"trivial": True}},
    {"id": "wl3_010", "messages": [{"role": "user", "content": "Help me architect a microservices system for an e-commerce platform. Consider service boundaries, data ownership, inter-service communication, eventual consistency, and failure handling."}], "labels": {"trivial": False}},
]

# ---------------------------------------------------------------------------
# WL4 — RAG-heavy
# ---------------------------------------------------------------------------

WL4 = [
    {
        "id": "wl4_001",
        "messages": [
            {"role": "system", "content": BOILERPLATE_SYSTEM + "\n\n" + RAG_CONTEXT},
            {"role": "user", "content": "How does the authentication middleware work?"},
        ],
        "labels": {"trivial": True},
    },
    {
        "id": "wl4_002",
        "messages": [
            {"role": "system", "content": BOILERPLATE_SYSTEM + "\n\n" + RAG_CONTEXT},
            {"role": "user", "content": "Is there a security vulnerability in the token verification?"},
        ],
        "labels": {"trivial": False},
    },
    {
        "id": "wl4_003",
        "messages": [
            {"role": "system", "content": BOILERPLATE_SYSTEM + "\n\n" + RAG_CONTEXT},
            {"role": "user", "content": "What's the default token TTL?"},
        ],
        "labels": {"trivial": True},
    },
    {
        "id": "wl4_004",
        "messages": [
            {"role": "system", "content": BOILERPLATE_SYSTEM + "\n\n" + RAG_CONTEXT},
            {"role": "user", "content": "Suggest improvements to the auth middleware to support API keys and OAuth2 in addition to JWT."},
        ],
        "labels": {"trivial": False},
    },
    {
        "id": "wl4_005",
        "messages": [
            {"role": "system", "content": BOILERPLATE_SYSTEM + "\n\n" + RAG_CONTEXT},
            {"role": "user", "content": "What algorithm is used for JWT signing?"},
        ],
        "labels": {"trivial": True},
    },
    {
        "id": "wl4_006",
        "messages": [
            {"role": "system", "content": BOILERPLATE_SYSTEM + "\n\n" + RAG_CONTEXT},
            {"role": "user", "content": "Design a rate limiting strategy for the auth endpoints that prevents brute force attacks while allowing legitimate traffic."},
        ],
        "labels": {"trivial": False},
    },
    {
        "id": "wl4_007",
        "messages": [
            {"role": "system", "content": BOILERPLATE_SYSTEM + "\n\n" + RAG_CONTEXT},
            {"role": "user", "content": "Does the middleware handle WebSocket connections?"},
        ],
        "labels": {"trivial": True},
    },
    {
        "id": "wl4_008",
        "messages": [
            {"role": "system", "content": BOILERPLATE_SYSTEM + "\n\n" + RAG_CONTEXT},
            {"role": "user", "content": "Audit this auth code for OWASP top 10 vulnerabilities and suggest fixes for each issue found."},
        ],
        "labels": {"trivial": False},
    },
    {
        "id": "wl4_009",
        "messages": [
            {"role": "system", "content": BOILERPLATE_SYSTEM + "\n\n" + RAG_CONTEXT},
            {"role": "user", "content": "What happens when a token expires?"},
        ],
        "labels": {"trivial": True},
    },
    {
        "id": "wl4_010",
        "messages": [
            {"role": "system", "content": BOILERPLATE_SYSTEM + "\n\n" + RAG_CONTEXT},
            {"role": "user", "content": "Implement a refresh token mechanism that uses rotating refresh tokens with token family tracking to detect token theft."},
        ],
        "labels": {"trivial": False},
    },
]

# ---------------------------------------------------------------------------
# Write JSONL files
# ---------------------------------------------------------------------------

def write_jsonl(path: Path, samples: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for s in samples:
            row = {"id": s["id"], "workload": s["id"].split("_")[0], "messages": s["messages"]}
            if "labels" in s:
                row["labels"] = s["labels"]
            f.write(json.dumps(row) + "\n")
    print(f"  {path}: {len(samples)} samples")


if __name__ == "__main__":
    print("Generating seed workloads:")
    write_jsonl(HERE / "wl1_edit.jsonl", WL1)
    write_jsonl(HERE / "wl2_explain.jsonl", WL2)
    write_jsonl(HERE / "wl3_chat.jsonl", WL3)
    write_jsonl(HERE / "wl4_rag.jsonl", WL4)
    print("Done.")

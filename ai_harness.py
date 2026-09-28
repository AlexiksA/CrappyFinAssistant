import requests
import json
from datetime import datetime
from ddgs import DDGS
from playwright.sync_api import sync_playwright
import trafilatura

url = "http://192.168.88.241:11434/api/chat"
old_model = "huihui_ai/qwen2.5-abliterate:7b-instruct"
model = "qwen3.5:4b"

max_steps = 18
max_repeats = 2
"""
In this place I will build schemas for al l the tools I will use
Looks like a normal json that will prepend user input
"""

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "Get the current time and date",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_web",
            "description": "Search the web and return titles, urls, short snippets",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The search Query"}
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "fetch_page",
            "description": "Fetch the visible text content of a web page, given its URL",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "The URL of the page to fetch",
                    }
                },
                "required": ["url"],
            },
        },
    },
]


# tools
def get_current_time():
    return datetime.now().strftime("%A, %d %B %Y, %H:%M:%S")


def search_web(query, max_results=5):
    with DDGS() as ddgs:
        results = ddgs.text(query, max_results=max_results)
    return [
        {"title": r["title"], "url": r["href"], "snippet": r["body"]} for r in results
    ]


def fetch_page(url):
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        page.goto(url, timeout=15000)
        html = page.content()
        browser.close()
    text = trafilatura.extract(html)

    if text is None:
        text = "Could not extract readable content from this page..."
    return text[:8000]


available_tools = {
    "get_current_time": get_current_time,
    "search_web": search_web,
    "fetch_page": fetch_page,
}

user_content = input()

messages = [
    {
        "role": "user",
        "content": user_content
        + " Don't forget to get the current time! Don't forget that if you want more detail you should call the fetch_page function.",
    }
]

last_signature = None
repeat_count = 0
num_ctx = 2**13
for step in range(max_steps):
    payload = {
        "model": model,
        "messages": messages,
        "tools": tools,
        "stream": False,
        "options": {"num_ctx": num_ctx},
    }
    response = requests.post(url, json=payload)
    data = response.json()

    # print(json.dumps(data, indent=2))

    assistant_message = data["message"]
    messages.append(assistant_message)

    tool_calls = assistant_message.get("tool_calls")

    if not tool_calls:
        print(assistant_message["content"])
        break

    stop_now = False

    for call in tool_calls:
        name = call["function"]["name"]
        args = call["function"]["arguments"]
        print(f"calling {name} with args {args}")
        signature = (name, json.dumps(args, indent=2))

        if signature == last_signature:
            repeat_count += 1
        else:
            repeat_count = 1
            last_signature = signature

        if repeat_count > max_repeats:
            print(
                f"Stopping: '{name}' was called with the same args {repeat_count} times in a row"
            )
            stop_now = True
            break

        if name not in available_tools:
            result = f"Error: unknown tool '{name}'"  # model a halucinat
        else:
            try:
                result = available_tools[name](**args)
            except Exception as e:
                result = f"Error running '{name}': {e}"

        content = result if isinstance(result, str) else json.dumps(result)
        messages.append({"role": "tool", "content": content})

    if stop_now:
        break
else:
    print(f"Stopping: hit the {max_steps}-step limit without a final answer.")

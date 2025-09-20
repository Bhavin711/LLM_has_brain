from llama_cpp import Llama
import json
import os

# ------------------ MODEL SETUP ------------------
llm = Llama(
    model_path="/LLAMA3.2_1B/Llama-3.2-1B-Instruct-Q8_0.gguf", # Put the model path 
    n_ctx=2048,     # context size (increase if supported by model)
    n_threads=6     # adjust to your CPU cores
)

# ------------------ MEMORY SETUP ------------------
MEMORY_FILE = "chat_memory.json"

# Load memory from file if exists
if os.path.exists(MEMORY_FILE):
    with open(MEMORY_FILE, "r") as f:
        memory = json.load(f)
else:
    memory = []

def save_memory():
    with open(MEMORY_FILE, "w") as f:
        json.dump(memory, f, indent=2)

# ------------------ PROMPT BUILDER ------------------
MAX_MEMORY_TOKENS = 1600  # keep room for input + response

def build_prompt(memory):
    """Builds a chat-style prompt with proper tokens for LLaMA 3.2"""
    prompt = ""
    for msg in memory:
        role = msg["role"]
        content = msg["content"]

        if role == "system":
            prompt += f"<|start_header_id|>system<|end_header_id|>\n{content}<|eot_id|>"
        elif role == "user":
            prompt += f"<|start_header_id|>user<|end_header_id|>\n{content}<|eot_id|>"
        elif role == "assistant":
            prompt += f"<|start_header_id|>assistant<|end_header_id|>\n{content}<|eot_id|>"

    # Signal assistant to reply next
    prompt += "<|start_header_id|>assistant<|end_header_id|>\n"
    return prompt

def trim_memory():
    """Drop oldest messages if memory grows too big"""
    global memory
    while len(build_prompt(memory).split()) > MAX_MEMORY_TOKENS:
        memory.pop(0)

# ------------------ CHAT FUNCTION ------------------
def chat(user_input):
    memory.append({"role": "user", "content": user_input})
    trim_memory()

    prompt = build_prompt(memory)

    output = llm(
        prompt,
        max_tokens=200,
        stop=["<|eot_id|>"]
    )
    reply = output["choices"][0]["text"].strip()

    memory.append({"role": "assistant", "content": reply})
    save_memory()

    return reply

# ------------------ MAIN LOOP ------------------
print("🤖 LLAMA-1B Chat with Memory (type 'exit' to quit)\n")

while True:
    try:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"]:
            print("Goodbye 👋")
            break

        response = chat(user_input)
        print("Bot:", response)

    except KeyboardInterrupt:
        print("\nExiting...")
        break

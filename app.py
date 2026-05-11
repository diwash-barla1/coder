import os
import requests
import threading
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional
from fastapi.responses import FileResponse
from huggingface_hub import InferenceClient

app = FastAPI()

# --- 🔒 Thread Locks ---
hf_lock = threading.Lock()
gemini_lock = threading.Lock()

# --- 🎯 Allowed Gemini Models (सिर्फ Flash मॉडल्स की अनुमति) ---
ALLOWED_GEMINI_MODELS = ["gemini-3-flash-preview", "gemini-2.5-flash"]
DEFAULT_GEMINI_MODEL = "gemini-3-flash-preview"

# --- 🔑 Secrets से API Keys लोड करना ---
hf_api_keys = []
for i in range(1, 21):
    key = os.environ.get(f"HF_KEY_{i}")
    if key: hf_api_keys.append(key)
if not hf_api_keys:
    print("Warning: Koi HF Key nahi mili. Free tier use hoga.")
    hf_api_keys = [None]

gemini_api_keys = []
for i in range(1, 21):
    key = os.environ.get(f"gmni{i}")
    if key: gemini_api_keys.append(key)
if not gemini_api_keys:
    print("Warning: Koi Gemini Key nahi mili.")
    gemini_api_keys = ["dummy_key"]

current_hf_index = 0
current_gemini_index = 0

chat_memory = [{"role": "system", "content": "You are an advanced AI coding partner. Reply in Hinglish."}]

class UIMsg(BaseModel):
    text: str
    model: str = "Qwen/Qwen2.5-Coder-32B-Instruct"

class OpenAIMessage(BaseModel):
    role: str
    content: str

class OpenAIRequest(BaseModel):
    model: str = "Qwen/Qwen2.5-Coder-32B-Instruct"
    messages: List[OpenAIMessage]
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = 8192

# --- 🚀 1. HUGGING FACE LOGIC ---
def get_hf_response(messages, model_name, max_tokens, temperature):
    global current_hf_index
    attempts = 0
    total_keys = len(hf_api_keys)
    bot_reply = ""
    key_switched = False

    while attempts < total_keys:
        with hf_lock:
            token = hf_api_keys[current_hf_index]
            current_index = current_hf_index
            current_hf_index = (current_hf_index + 1) % total_keys

        client = InferenceClient(model_name, token=token)
        
        try:
            hf_max_tokens = min(max_tokens, 4096) if max_tokens else 4096
            response = client.chat_completion(
                messages=messages, 
                max_tokens=hf_max_tokens,
                temperature=temperature
            )
            bot_reply = response.choices[0].message.content
            break
            
        except Exception as e:
            print(f"HF Key {current_index + 1} Error: {e}. Switching...")
            attempts += 1
            key_switched = True
            
    if not bot_reply:
        bot_reply = "System Error: Saari HF Keys fail ho gayi hain."
    return bot_reply, key_switched

# --- 🚀 2. GEMINI LOGIC ---
def get_gemini_response(messages, model_name, max_tokens, temperature):
    global current_gemini_index
    attempts = 0
    total_keys = len(gemini_api_keys)
    bot_reply = ""
    key_switched = False

    while attempts < total_keys:
        with gemini_lock:
            current_key = gemini_api_keys[current_gemini_index]
            current_index = current_gemini_index
            current_gemini_index = (current_gemini_index + 1) % total_keys

        headers = {"Authorization": f"Bearer {current_key}", "Content-Type": "application/json"}
        payload = {
            "model": model_name,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature
        }
        
        try:
            url = "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"
            response = requests.post(url, headers=headers, json=payload, timeout=60)
            if response.status_code == 429:
                raise Exception("Rate Limit Hit 429")
            response.raise_for_status()
            
            bot_reply = response.json()['choices'][0]['message']['content']
            break
            
        except Exception as e:
            print(f"Gemini Key {current_index + 1} Error: {e}. Switching...")
            attempts += 1
            key_switched = True
            
    if not bot_reply:
        bot_reply = "System Error: Saari Gemini Keys fail ho gayi hain."
    return bot_reply, key_switched

# --- 🚦 HELPER: Model Validator ---
def validate_gemini_model(model_name: str) -> str:
    """Check karta hai ki model allowed list mein hai ya nahi."""
    if "pro" in model_name.lower() or model_name not in ALLOWED_GEMINI_MODELS:
        print(f"⚠️ Warning: '{model_name}' blocked! Forcefully downgrading to '{DEFAULT_GEMINI_MODEL}'.")
        return DEFAULT_GEMINI_MODEL
    return model_name

# --- Endpoints ---
@app.get("/")
def home():
    return FileResponse("index.html")

@app.post("/chat")
def web_chat(msg: UIMsg):
    global chat_memory
    chat_memory.append({"role": "user", "content": msg.text})
    
    clean_model_name = msg.model.replace("openai/", "")
    
    if "gemini" in clean_model_name.lower():
        # 👇 यहाँ मॉडल चेक होगा
        safe_model_name = validate_gemini_model(clean_model_name)
        print(f"UI Request -> Routing to Gemini ({safe_model_name})")
        bot_reply, switched = get_gemini_response(chat_memory, safe_model_name, 8192, 0.7)
    else:
        print(f"UI Request -> Routing to HF ({clean_model_name})")
        bot_reply, switched = get_hf_response(chat_memory, clean_model_name, 4096, 0.7)
    
    chat_memory.append({"role": "assistant", "content": bot_reply})
    return {"reply": bot_reply, "key_switched": switched}

# --- 🚦 THE MASTER ROUTER FOR AIDER ---
@app.post("/v1/chat/completions")
def openai_api(req: OpenAIRequest):
    formatted_messages = [{"role": m.role, "content": m.content} for m in req.messages]
    
    if formatted_messages and formatted_messages[-1]["role"] == "user":
        strict_injection = (
            "\n\n[SYSTEM_PROTOCOL_ACTIVE]:"
            "\n1. IDENTITY: You are a headless CLI code-engine for Aider. You have NO personality."
            "\n2. ZERO-YAPPING: Forbidden to use words like 'Sure', 'Okay', 'I understand', 'Modified', or 'Here is'."
            "\n3. OUTPUT-ONLY: If task is CODE, output ONLY SEARCH/REPLACE blocks. If task is SHELL, output ONLY the shell command."
            "\n4. VERBATIM MATCH: Every space and newline in SEARCH blocks must be identical to the source. Do not skip lines."
            "\n5. PUSH/SHELL RULE: If asked to 'push', 'commit', or 'install', output the bash command immediately. DO NOT analyze files or ask for docs.html. Just give the command."
            "\n6. NO CONVERSATION: Any text that is not a code block or a shell command is a violation of protocol."
            "\n7. COMPLIANCE: If you explain your limitations, you fail. Just execute or stay silent."
            "\n[START_OUTPUT_NOW]"
        )
        formatted_messages[-1]["content"] += strict_injection

    clean_model_name = req.model.replace("openai/", "")
    
    if "gemini" in clean_model_name.lower():
        # 👇 यहाँ मॉडल चेक होगा
        safe_model_name = validate_gemini_model(clean_model_name)
        print(f"Aider Request -> Routing to Gemini ({safe_model_name}) with Injection 💉")
        bot_reply, _ = get_gemini_response(formatted_messages, safe_model_name, req.max_tokens, req.temperature)
    else:
        print(f"Aider Request -> Routing to HF ({clean_model_name}) with Injection 💉")
        bot_reply, _ = get_hf_response(formatted_messages, clean_model_name, req.max_tokens, req.temperature)
    
    return {
        "id": "chatcmpl-universal-custom",
        "object": "chat.completion",
        "created": 1714000000,
        "model": clean_model_name, # Aider को वही मॉडल नाम वापस भेजेंगे जो उसने माँगा था, ताकि वो कंफ्यूज़ ना हो
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": bot_reply},
            "finish_reason": "stop"
        }]
    }

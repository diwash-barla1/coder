import os
import requests
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional
from fastapi.responses import FileResponse
from huggingface_hub import InferenceClient

app = FastAPI()

# --- 🔑 Secrets से API Keys लोड करना ---
# 1. Hugging Face Keys
hf_api_keys = []
for i in range(1, 20):
    key = os.environ.get(f"HF_KEY_{i}")
    if key: hf_api_keys.append(key)
if not hf_api_keys:
    print("Warning: Koi HF Key nahi mili. Free tier use hoga.")
    hf_api_keys = [None]

# 2. Gemini Keys
gemini_api_keys = []
for i in range(1, 20):
    key = os.environ.get(f"gmni{i}")
    if key: gemini_api_keys.append(key)
if not gemini_api_keys:
    print("Warning: Koi Gemini Key nahi mili.")
    gemini_api_keys = ["dummy_key"]

current_hf_index = 0
current_gemini_index = 0

# 🧠 UI के लिए मेमोरी
chat_memory = [{"role": "system", "content": "You are an advanced AI coding partner. Reply in Hinglish."}]

# --- Pydantic Models ---
class UIMsg(BaseModel):
    text: str

class OpenAIMessage(BaseModel):
    role: str
    content: str

class OpenAIRequest(BaseModel):
    model: str = "Qwen/Qwen2.5-Coder-32B-Instruct" # Default Model
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
        token = hf_api_keys[current_hf_index]
        client = InferenceClient(model_name, token=token)
        
        try:
            # HF API में max_tokens limit 8192 से कम हो सकती है, इसलिए सेफ साइड 4096 रखा है
            hf_max_tokens = min(max_tokens, 4096) if max_tokens else 4096
            response = client.chat_completion(
                messages=messages, 
                max_tokens=hf_max_tokens,
                temperature=temperature
            )
            bot_reply = response.choices[0].message.content
            break
        except Exception as e:
            print(f"HF Key {current_hf_index + 1} Error: {e}. Switching...")
            current_hf_index = (current_hf_index + 1) % total_keys
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
        current_key = gemini_api_keys[current_gemini_index]
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
            print(f"Gemini Key {current_gemini_index + 1} Error: {e}. Switching...")
            current_gemini_index = (current_gemini_index + 1) % total_keys
            attempts += 1
            key_switched = True
            
    if not bot_reply:
        bot_reply = "System Error: Saari Gemini Keys fail ho gayi hain."
    return bot_reply, key_switched

# --- Endpoints ---
@app.get("/")
def home():
    return FileResponse("index.html")

@app.post("/chat")
def web_chat(msg: UIMsg):
    global chat_memory
    chat_memory.append({"role": "user", "content": msg.text})
    
    # Web UI के लिए बाय डिफ़ॉल्ट हम HF का Qwen मॉडल यूज़ कर रहे हैं
    bot_reply, switched = get_hf_response(chat_memory, "Qwen/Qwen2.5-Coder-32B-Instruct", 2048, 0.7)
    
    chat_memory.append({"role": "assistant", "content": bot_reply})
    return {"reply": bot_reply, "key_switched": switched}

# --- 🚦 THE MASTER ROUTER FOR AIDER ---
@app.post("/v1/chat/completions")
def openai_api(req: OpenAIRequest):
    formatted_messages = [{"role": m.role, "content": m.content} for m in req.messages]
    
    # 'openai/' प्रिफिक्स हटाना
    clean_model_name = req.model.replace("openai/", "")
    
    # 🧠 स्मार्ट राऊटिंग लॉजिक
    if "gemini" in clean_model_name.lower():
        print(f"Routing to Gemini API -> Model: {clean_model_name}")
        bot_reply, _ = get_gemini_response(formatted_messages, clean_model_name, req.max_tokens, req.temperature)
    else:
        print(f"Routing to Hugging Face API -> Model: {clean_model_name}")
        bot_reply, _ = get_hf_response(formatted_messages, clean_model_name, req.max_tokens, req.temperature)
    
    return {
        "id": "chatcmpl-universal-custom",
        "object": "chat.completion",
        "created": 1714000000,
        "model": clean_model_name,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": bot_reply},
            "finish_reason": "stop"
        }]
    }

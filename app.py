import os
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional
from fastapi.responses import FileResponse
from huggingface_hub import InferenceClient

app = FastAPI()

# 🔑 Secrets से API Keys लोड करना
api_keys = []
for i in range(1, 20):
    key = os.environ.get(f"HF_KEY_{i}")
    if key:
        api_keys.append(key)

if not api_keys:
    print("Warning: Koi API Key nahi mili! Free tier par chal raha hai.")
    api_keys = [None]

current_key_index = 0

# 🧠 UI के लिए मेमोरी
chat_memory = [{"role": "system", "content": "You are an advanced AI coding partner. Reply in Hinglish."}]

# --- Pydantic Models ---
class UIMsg(BaseModel):
    text: str

class OpenAIMessage(BaseModel):
    role: str
    content: str

class OpenAIRequest(BaseModel):
    model: str = "Qwen/Qwen2.5-Coder-32B-Instruct"
    messages: List[OpenAIMessage]
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = 2048

# --- कोर लॉजिक (मॉडल से रिस्पांस लाना और Key रोटेट करना) ---
def get_ai_response(messages, max_tokens=1024, temperature=0.7):
    global current_key_index
    attempts = 0
    total_keys = len(api_keys)
    bot_reply = ""
    key_switched_flag = False

    while attempts < total_keys:
        token = api_keys[current_key_index]
        client = InferenceClient("Qwen/Qwen2.5-Coder-32B-Instruct", token=token)
        
        try:
            response = client.chat_completion(
                messages=messages, 
                max_tokens=max_tokens,
                temperature=temperature
            )
            bot_reply = response.choices[0].message.content
            break
            
        except Exception as e:
            print(f"Key {current_key_index + 1} Error: {e}. Switching...")
            current_key_index = (current_key_index + 1) % total_keys
            attempts += 1
            key_switched_flag = True
            
    if not bot_reply:
        bot_reply = "System Error: Saari API Keys fail ho gayi hain."
        
    return bot_reply, key_switched_flag

# --- Endpoints ---

# 1. UI लोड करने के लिए (अलग की गई HTML फाइल को सर्व करेगा)
@app.get("/")
def home():
    return FileResponse("index.html")

# 2. Web UI के चैट के लिए
@app.post("/chat")
def web_chat(msg: UIMsg):
    global chat_memory
    chat_memory.append({"role": "user", "content": msg.text})
    
    bot_reply, switched = get_ai_response(chat_memory)
    
    chat_memory.append({"role": "assistant", "content": bot_reply})
    return {"reply": bot_reply, "key_switched": switched}

# 3. AIDER / OPENAI COMPATIBLE API
@app.post("/v1/chat/completions")
def openai_api(req: OpenAIRequest):
    hf_messages = [{"role": m.role, "content": m.content} for m in req.messages]
    
    bot_reply, _ = get_ai_response(hf_messages, req.max_tokens, req.temperature)
    
    return {
        "id": "chatcmpl-hf-custom",
        "object": "chat.completion",
        "created": 1714000000,
        "model": req.model,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": bot_reply},
            "finish_reason": "stop"
        }]
    }

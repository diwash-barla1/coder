import os
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.responses import HTMLResponse
from huggingface_hub import InferenceClient
import uvicorn

app = FastAPI()

# 🔑 Secrets से सारी API Keys निकालना (HF_KEY_1, HF_KEY_2...)
api_keys = []
for i in range(1, 20):  # 20 keys तक सपोर्ट करेगा, आप और बढ़ा सकते हैं
    key = os.environ.get(f"HF_KEY_{i}")
    if key:
        api_keys.append(key)

# अगर कोई Key नहीं मिली, तो बिना Key (Free API) के चलाने का बैकअप
if not api_keys:
    print("Warning: Koi API Key nahi mili! Free tier par chal raha hai.")
    api_keys = [None]

current_key_index = 0  # ट्रैक रखने के लिए कि अभी कौन सी Key चल रही है

# 🧠 ग्लोबल मेमोरी
chat_memory = [
    {"role": "system", "content": "You are a highly advanced AI coding partner. Reply in Hinglish."}
]

class Msg(BaseModel):
    text: str

# UI वाला HTML (वही मस्त डार्क थीम)
html_content = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Pro AI Bot</title>
    <style>
        body { background: #0f172a; color: #f8fafc; font-family: -apple-system, sans-serif; margin: 0; display: flex; flex-direction: column; height: 100vh; }
        #chat { flex: 1; overflow-y: auto; padding: 20px; }
        .bubble { margin: 10px 0; padding: 12px; border-radius: 12px; max-width: 80%; line-height: 1.5; }
        .user { background: #334155; align-self: flex-end; margin-left: auto; }
        .bot { background: #1e293b; border: 1px solid #334155; }
        .system { background: #7f1d1d; color: #fca5a5; font-size: 12px; text-align: center; border-radius: 8px; margin: 5px auto; padding: 5px 10px; width: fit-content; }
        .input-box { padding: 20px; background: #1e293b; display: flex; gap: 10px; }
        input { flex: 1; background: #0f172a; border: 1px solid #334155; color: white; padding: 12px; border-radius: 8px; outline: none; }
        button { background: #38bdf8; color: #0f172a; border: none; padding: 12px 20px; border-radius: 8px; font-weight: bold; cursor: pointer; }
    </style>
</head>
<body>
    <div id="chat"></div>
    <div class="input-box">
        <input type="text" id="userInput" placeholder="Partner, command do...">
        <button onclick="send()">Send</button>
    </div>
    <script>
        async function send() {
            const input = document.getElementById('userInput');
            const chat = document.getElementById('chat');
            if(!input.value) return;
            
            const text = input.value;
            chat.innerHTML += `<div class="bubble user">${text}</div>`;
            input.value = '';
            chat.scrollTop = chat.scrollHeight;

            const res = await fetch('/chat', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({text: text})
            });
            const data = await res.json();
            
            // अगर Key स्विच हुई है तो UI में एक छोटा सा अलर्ट दिखाएं
            if(data.key_switched) {
                chat.innerHTML += `<div class="system">⚠️ Rate Limit Hit: Auto-switched to next API Key</div>`;
            }
            
            chat.innerHTML += `<div class="bubble bot">${data.reply}</div>`;
            chat.scrollTop = chat.scrollHeight;
        }
    </script>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
def home():
    return html_content

@app.post("/chat")
def chat(msg: Msg):
    global chat_memory, current_key_index
    
    chat_memory.append({"role": "user", "content": msg.text})
    
    attempts = 0
    total_keys = len(api_keys)
    bot_reply = ""
    key_switched_flag = False

    # 🔄 Auto-Switch Logic
    while attempts < total_keys:
        token = api_keys[current_key_index]
        client = InferenceClient("Qwen/Qwen2.5-Coder-32B-Instruct", token=token)
        
        try:
            response = client.chat_completion(messages=chat_memory, max_tokens=1024)
            bot_reply = response.choices[0].message.content
            break  # अगर सक्सेसफुल रहा, तो लूप से बाहर आ जाओ
            
        except Exception as e:
            # अगर एरर आया, तो अगली Key पर स्विच करो
            print(f"Key {current_key_index + 1} failed with error: {e}. Switching to next...")
            current_key_index = (current_key_index + 1) % total_keys
            attempts += 1
            key_switched_flag = True
    
    if not bot_reply:
        bot_reply = "System Error: Saari API Keys fail ho gayi hain. Kripya baad mein try karein."

    chat_memory.append({"role": "assistant", "content": bot_reply})
    
    return {"reply": bot_reply, "key_switched": key_switched_flag}

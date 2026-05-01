from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.responses import HTMLResponse
from huggingface_hub import InferenceClient
import uvicorn

app = FastAPI()

# मॉडल सेटअप
client = InferenceClient("Qwen/Qwen2.5-Coder-32B-Instruct")

# 🧠 ग्लोबल मेमोरी (नोट: स्पेस रीस्टार्ट होने पर यह खाली हो जाएगी)
chat_memory = [
    {"role": "system", "content": "You are a smart AI partner with long-term memory. Reply in Hinglish."}
]

class Msg(BaseModel):
    text: str

# सिंपल और क्लीन मोबाइल-फ्रेंडली UI
html_content = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI Memory Bot</title>
    <style>
        body { background: #0f172a; color: #f8fafc; font-family: -apple-system, sans-serif; margin: 0; display: flex; flex-direction: column; height: 100vh; }
        #chat { flex: 1; overflow-y: auto; padding: 20px; }
        .bubble { margin: 10px 0; padding: 12px; border-radius: 12px; max-width: 80%; line-height: 1.5; }
        .user { background: #334155; align-self: flex-end; margin-left: auto; }
        .bot { background: #1e293b; border: 1px solid #334155; }
        .input-box { padding: 20px; background: #1e293b; display: flex; gap: 10px; }
        input { flex: 1; background: #0f172a; border: 1px solid #334155; color: white; padding: 12px; border-radius: 8px; outline: none; }
        button { background: #38bdf8; color: #0f172a; border: none; padding: 12px 20px; border-radius: 8px; font-weight: bold; cursor: pointer; }
    </style>
</head>
<body>
    <div id="chat"></div>
    <div class="input-box">
        <input type="text" id="userInput" placeholder="Partner, kuch pucho...">
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
    global chat_memory
    chat_memory.append({"role": "user", "content": msg.text})
    
    # मॉडल से रिस्पांस लेना
    response = client.chat_completion(messages=chat_memory, max_tokens=500)
    reply = response.choices[0].message.content
    
    chat_memory.append({"role": "assistant", "content": reply})
    return {"reply": reply}

if __name__ == '__main__':
    # Hugging Face Spaces default port is 7860
    port = int(os.environ.get("PORT", 7860))
    app.run(host='0.0.0.0', port=port, debug=True)

# पाइथन का हल्का इमेज इस्तेमाल कर रहे हैं
FROM python:3.9

# काम करने की जगह सेट करें
WORKDIR /code

# रिक्वायरमेंट्स कॉपी और इंस्टॉल करें
COPY ./requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# बाकी सारा कोड कॉपी करें
COPY . .

# पोर्ट 7860 को एक्सपोज़ करें (HF का डिफ़ॉल्ट)
EXPOSE 7860

# FastAPI को रन करने की कमांड
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]

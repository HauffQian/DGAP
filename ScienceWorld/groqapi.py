import os

from groq import Groq
internal_proxy = '127.0.0.1:7890'
os.environ['HTTP_PROXY'] = internal_proxy
os.environ['HTTPS_PROXY'] = internal_proxy

client = Groq(
    api_key=os.environ.get("GROQ_API_KEY"),
)

chat_completion = client.chat.completions.create(
    messages=[
        {
            "role": "user",
            "content": "Explain the importance of fast language models",
        }
    ],
    model="llama3-8b-8192",
)

print(chat_completion.choices[0].message.content)
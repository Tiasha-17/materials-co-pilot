from dotenv import load_dotenv
from groq import Groq

load_dotenv()

client = Groq()

response = client.chat.completions.create(
    model="openai/gpt-oss-120b",
    messages=[
        {
            "role": "user",
            "content": "In one sentence, what is materials informatics?"
        }
    ]
)

print(response.choices[0].message.content)
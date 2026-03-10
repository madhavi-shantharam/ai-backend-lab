from openai import OpenAI
from dotenv import load_dotenv
import os

#load .env file
load_dotenv()

print("OPENAI_API_KEY:", os.getenv("OPENAI_API_KEY"))

# create client using environment variable
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
response = client.responses.create(
	model="gpt-3.5-turbo",
	input="Say hello in one sentence"
)

if hasattr(response, "output_text"):
	print(response.output_text)
else:
	print(response)

print(response.output_text)

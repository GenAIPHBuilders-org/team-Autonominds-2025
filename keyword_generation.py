from google import genai
import os
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

client = genai.Client(api_key=api_key)

def generate_keywords(title):
    prompt = f"""Extract the most relevant and high-impact academic keywords from the following research title.
    Focus only on terms that would be useful for retrieving top-tier scholarly papers.
    Limit the output to 5–10 keywords.
    Separate each keyword or phrase using a comma.
    
    Research Title: '{title}'"""
    
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=prompt
    )
    
    return response.text.strip().replace('\n', '').split(", ") #strip() and replace() removes `\n` at the end

if __name__ == "__main__":
    title = input("Enter title: ")
    keywords = generate_keywords(title)
    
    print(keywords)

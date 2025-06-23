import os
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables from .env file in the project root
load_dotenv()

# Configure the genai library with the API key
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise RuntimeError(
        "FATAL: GEMINI_API_KEY not found in environment variables. "
        "Please ensure it is set in your .env file."
    )

genai.configure(api_key=api_key)
print("INFO: Google Generative AI client configured successfully.")

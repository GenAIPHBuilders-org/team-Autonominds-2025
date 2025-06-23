from google import genai
import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=api_key)

def generate_topics(title):
    prompt = f"""You are an expert academic research assistant.
Given a research title, generate a list of 3-5 most relevant and meaningful academic topics associated with it.
Your output should include a mix of core, adjacent, and emerging topics that could appear in academic journals.
Prioritize conceptual breadth, domain relevance, and interdisciplinary connections.

Research Title: {title}

IMPORTANT RULES:
Do not include any explanation, preamble, or commentary.
Output the topics as a comma-separated list, with no bullet points, no numbering, and no newline characters.
Each topic should be short (max 5 words) and represent a real academic field or concept.
Do not include anything like “Sure!” or “Here are the topics:”. Only return the list.

Output Format: Topic 1, Topic 2, Topic 3

Example (for title: “The Effect of Prior Programming Experience on Cognitive Load in First-Year Computer Science Students”)
Cognitive Load Theory, Educational Psychology, Computer Science Education, Programming Experience and Expertise, Human-Computer Interaction (HCI), Novice vs Expert Learning Models"""
    
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=prompt
    )
    
    return response.text.strip().replace('\n', '').split(", ")

def generate_keywords(related_topics):
    keywords_by_topic = {}
    
    for topic in related_topics:
        prompt = f"""You are an expert researcher and domain authority in the field of {topic}. Your task is to generate a list of detailed, search-optimized academic keywords and search phrases that a researcher would use to find scholarly literature specifically related to this topic.

These keywords must vary in scope and specificity — including synonyms, technical terms, and real-world search phrases. Prioritize both precision and coverage, ensuring that the result is suitable for use in academic databases.

Topic: {topic}

IMPORTANT RULES:

Do not include any explanation, preamble, or commentary.

Output the keywords as a comma-separated list, with no bullet points, no numbering, and no newline characters.

Use a mix of short and phrase-based keywords (3–5 words max per phrase).

Do not include any punctuation other than the separating commas.

Return exactly 5 to 10 keywords."""
        
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt
        )
        
        keywords = response.text.strip().replace('\n', '').split(", ")
        keywords_by_topic[topic] = keywords
        
    return keywords_by_topic

if __name__ == "__main__":
    title = input("Enter title: ")
    
    related_topics = generate_topics(title)
    keywords_by_topic = generate_keywords(related_topics)
    
    for topic, keywords in keywords_by_topic.items():
        print(f"\nTopic: {topic}\nKeywords: {keywords}")

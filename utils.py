import json
import os
import google.generativeai as genai
from dotenv import load_dotenv
from datetime import datetime

# Load environment
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
if api_key:
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-1.5-flash")
else:
    model = None

DATA_FILE = os.path.join(os.path.dirname(__file__), "data", "reviews.json")

def load_reviews():
    if not os.path.exists(DATA_FILE):
        return []
    try:
        with open(DATA_FILE, "r") as f:
            return json.load(f)
    except json.JSONDecodeError:
        return []

def save_review(rating, review_text, ai_response):
    reviews = load_reviews()
    new_review = {
        "rating": rating,
        "text": review_text,
        "ai_response": ai_response,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    reviews.append(new_review)
    with open(DATA_FILE, "w") as f:
        json.dump(reviews, f, indent=2)
    return new_review

def generate_user_response(rating, review_text):
    if not model:
        return "AI Service Unavailable (Missing Key)"
    
    prompt = f"""
    You are a helpful customer service AI. A user has left a review.
    Rating: {rating}/5 stars.
    Review: "{review_text}"
    
    Write a short, polite, and personalized response to this user. 
    If the rating is low, apologize and ask for feedback. 
    If high, thank them enthusiastically.
    Keep it under 50 words.
    """
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error generating response: {str(e)}"

def analyze_reviews_for_admin(reviews):
    if not model or not reviews:
        return "No reviews to analyze.", "No actions recommended."
    
    # Prepare text for analysis (limit to last 50 reviews to fit context window if needed)
    reviews_text = "\n".join([f"- {r['rating']} stars: {r['text']}" for r in reviews[-50:]])
    
    prompt = f"""
    Analyze the following customer reviews:
    
    {reviews_text}
    
    Perform two tasks:
    1. Summarize the key themes (positive and negative).
    2. Recommend 3 specific actions the business should take to improve.
    
    Output in strictly valid JSON format:
    {{
      "summary": "...",
      "recommendations": ["Action 1...", "Action 2...", "Action 3..."]
    }}
    """
    try:
        response = model.generate_content(prompt)
        text = response.text.replace("```json", "").replace("```", "").strip()
        data = json.loads(text)
        return data.get("summary", "Analysis failed"), data.get("recommendations", [])
    except Exception as e:
        return f"Error in analysis: {str(e)}", []

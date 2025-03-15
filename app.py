import streamlit as st
import asyncio
import requests
import google.generativeai as genai
from collections import deque
from pymongo import MongoClient, ASCENDING
from datetime import datetime, timedelta
from googleapiclient.discovery import build
import uuid
import hashlib
from io import BytesIO
import random
import PyPDF2
import re

# Hardcoded Environment Variables
GOOGLE_API_KEY = "AIzaSyBaCx9eHQYUjaCH-iJdzmR9LCszYKnWTtc"
SEARCH_ENGINE_ID = "e6da2fcb52c994349"
GEMINI_API_KEY = "AIzaSyCRYqVb1Bu1DTXv7iHuXzz0WP4oJxRAy1w"
MONGO_URI = "mongodb+srv://raghuyanala:Kanna%401249@cluster0.wkvyw.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"

# Configure Gemini AI
if not GEMINI_API_KEY:
    st.error("🚨 Missing Gemini API Key.")
else:
    genai.configure(api_key=GEMINI_API_KEY)

# MongoDB Setup
client = MongoClient(MONGO_URI)
db = client["chatbot_db"]
chat_collection = db["chat_history"]
profile_collection = db["user_profiles"]
MAX_CHAT_HISTORY = 500

# Gemini Model Configuration
generation_config = {
    "temperature": 1,
    "top_p": 0.95,
    "top_k": 64,
    "max_output_tokens": 8192,
    "response_mime_type": "text/plain",
}

model = genai.GenerativeModel(
    model_name="learnlm-1.5-pro-experimental",
    generation_config=generation_config,
)

# Session State Initialization
if "user_id" not in st.session_state:
    st.session_state.user_id = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = deque(maxlen=50)
if "query_processing" not in st.session_state:
    st.session_state.query_processing = False
if "last_query" not in st.session_state:
    st.session_state.last_query = None
if "last_response" not in st.session_state:
    st.session_state.last_response = None
if "user_preferences" not in st.session_state:
    st.session_state.user_preferences = {"tone": "formal", "detail_level": "medium", "language": "en", "format": "paragraph"}
if "notifications" not in st.session_state:
    st.session_state.notifications = []
if "last_proactive_time" not in st.session_state:
    st.session_state.last_proactive_time = 0
if "last_query_time" not in st.session_state:
    st.session_state.last_query_time = 0

# UI Title
st.title("🤖 Ultimate AI Chatbot")

# Heuristic Multi-Scale Attention (No CPU/Training)
def heuristic_multi_scale_attention(query):
    # Simulate multi-scale attention with simple rules
    words = query.lower().split()
    length = len(words)
    
    # Scale 1: Short-term (word count)
    short_scale = min(1.0, length / 5)  # Favor short, concise queries
    
    # Scale 2: Mid-term (specificity via keywords)
    specific_keywords = {"what", "how", "why", "who", "where", "when", "explain", "describe"}
    mid_scale = sum(1 for word in words if word in specific_keywords) / max(1, length)
    
    # Scale 3: Long-term (structure)
    long_scale = 1.0 if "?" in query or len(re.findall(r"\w+", query)) > 3 else 0.5
    
    # Combine scales (weighted average)
    focus_score = (0.3 * short_scale + 0.4 * mid_scale + 0.3 * long_scale)
    return min(max(focus_score, 0.1), 1.0)  # Clamp between 0.1 and 1.0

# DailyDialog-Inspired Focus Score Adjustment
def adjust_focus_score(query, focus_score):
    # Proxy for act/emotion based on query content (no dataset needed)
    words = query.lower().split()
    if any(word in words for word in ["what", "how", "why", "who", "where", "when"]):  # Question-like (act 2)
        act_bonus = 0.2
        if any(word in words for word in ["great", "good", "happy", "cool"]):  # Positive emotion proxy (4)
            emotion_bonus = 0.1
        else:
            emotion_bonus = 0.0
    elif any(word in words for word in ["tell", "give", "show"]):  # Inform-like (act 1)
        act_bonus = 0.1
        emotion_bonus = 0.0
    else:  # Directive/commissive or vague (act 3/4)
        act_bonus = -0.1
        emotion_bonus = 0.0 if "please" in words else -0.1
    
    return min(max(focus_score + act_bonus + emotion_bonus, 0.1), 1.0)

# Profile Management Functions
def create_profile(username, password):
    user_id = str(uuid.uuid4())
    hashed_password = hashlib.sha256(password.encode()).hexdigest()
    profile = {
        "user_id": user_id,
        "username": username,
        "password": hashed_password,
        "preferences": {"tone": "formal", "detail_level": "medium", "language": "en", "format": "paragraph"},
        "created_at": datetime.utcnow().timestamp(),
        "interests": {},
        "query_count": 0,
        "last_query_time": 0
    }
    profile_collection.insert_one(profile)
    return user_id

def authenticate_user(username, password):
    hashed_password = hashlib.sha256(password.encode()).hexdigest()
    user = profile_collection.find_one({"username": username, "password": hashed_password})
    if user:
        prefs = user.get("preferences", {})
        updates = {}
        if "language" not in prefs:
            prefs["language"] = "en"
            updates["preferences"] = prefs
        if "format" not in prefs:
            prefs["format"] = "paragraph"
            updates["preferences"] = prefs
        if "last_query_time" not in user:
            updates["last_query_time"] = 0
        if updates:
            profile_collection.update_one({"user_id": user["user_id"]}, {"$set": updates})
        return user["user_id"]
    return None

def get_user_preferences(user_id):
    user = profile_collection.find_one({"user_id": user_id})
    if user:
        prefs = user.get("preferences", {"tone": "formal", "detail_level": "medium", "language": "en", "format": "paragraph"})
        updates = {}
        if "language" not in prefs:
            prefs["language"] = "en"
            updates["preferences"] = prefs
        if "format" not in prefs:
            prefs["format"] = "paragraph"
            updates["preferences"] = prefs
        if "last_query_time" not in user:
            updates["last_query_time"] = 0
        if updates:
            profile_collection.update_one({"user_id": user_id}, {"$set": updates})
        return prefs
    return {"tone": "formal", "detail_level": "medium", "language": "en", "format": "paragraph"}

def update_user_preferences(user_id, preferences):
    profile_collection.update_one({"user_id": user_id}, {"$set": {"preferences": preferences}})

def update_user_interests(user_id, interests):
    profile_collection.update_one({"user_id": user_id}, {"$set": {"interests": interests}})

def update_query_count(user_id):
    current_time = datetime.utcnow().timestamp()
    profile_collection.update_one({"user_id": user_id}, {"$inc": {"query_count": 1}, "$set": {"last_query_time": current_time}})
    st.session_state.last_query_time = current_time

# Google Search Function
def perform_google_search(query):
    try:
        service = build("customsearch", "v1", developerKey=GOOGLE_API_KEY)
        res = service.cse().list(q=query, cx=SEARCH_ENGINE_ID, num=3).execute()
        search_results = res.get("items", [])
        return "\n".join([f"- [{item['title']}]({item['link']})\n{item['snippet']}" for item in search_results]) or "No results found."
    except Exception as e:
        return f"❌ Google Search Error: {e}"

# Chat History Management
def fetch_chat_history(user_id, limit=5):
    return list(chat_collection.find({"user_id": user_id}, {"_id": 0, "user": 1, "ai": 1, "rating": 1})
                .sort("timestamp", -1).limit(limit))

def store_chat(user_id, query, response, rating=None):
    chat_entry = {
        "user_id": user_id,
        "user": query,
        "ai": response,
        "timestamp": datetime.utcnow().timestamp(),
        "rating": rating
    }
    chat_collection.insert_one(chat_entry)
    if chat_collection.count_documents({"user_id": user_id}) > MAX_CHAT_HISTORY:
        oldest = chat_collection.find_one({"user_id": user_id}, sort=[("timestamp", ASCENDING)])
        chat_collection.delete_one({"_id": oldest["_id"]})

# Conversation Summarization
def summarize_history(user_id):
    history = fetch_chat_history(user_id, 20)
    if not history:
        return "No recent conversation to summarize."
    summary = "Recent chat summary:\n"
    for chat in history:
        summary += f"- You asked: '{chat['user'][:50]}...', I replied: '{chat['ai'][:50]}...'\n"
    return summary.strip()

# Detect User Interests
def detect_user_interests(user_id):
    history = fetch_chat_history(user_id, 20)
    interests = {}
    for chat in history:
        words = chat["user"].lower().split()
        for word in words:
            if len(word) > 3:
                interests[word] = interests.get(word, 0) + 1
    return dict(sorted(interests.items(), key=lambda x: x[1], reverse=True)[:3])

# Proactive Suggestion
def generate_proactive_suggestion(user_id):
    interests = detect_user_interests(user_id)
    if interests and random.random() > 0.3:
        top_interest = max(interests, key=interests.get)
        return f"Hey, noticed you’re into {top_interest}. Want to chat about it?"
    topics = ["latest news", "fun trivia", "math puzzles"]
    return f"How about we discuss {random.choice(topics)}?"

# Multi-Modal Processing (PDF Only)
def process_uploaded_file(file):
    if file.type == "application/pdf":
        try:
            pdf_reader = PyPDF2.PdfReader(file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() or ""
            return f"Extracted text from PDF: {text}" if text else "No text could be extracted from the PDF."
        except Exception as e:
            return f"Error processing PDF: {str(e)}"
    return "Unsupported file type (only PDFs are supported)."

# AI Query Function using Gemini AI
async def query_ai(query, user_id, file_content=None):
    past_context = summarize_history(user_id) if len(fetch_chat_history(user_id)) > 10 else "\n".join([f"Q: {chat['user']}\nA: {chat['ai']}" for chat in fetch_chat_history(user_id)])
    google_results = perform_google_search(query) if not file_content else "N/A"
    preferences = get_user_preferences(user_id)
    focus_score = heuristic_multi_scale_attention(query)
    focus_score = adjust_focus_score(query, focus_score)
    lang = preferences["language"]

    if file_content:
        query = f"{query}\n\nFile Content: {file_content}"

    prompt = f"""
    **User Query**: "{query}"
    **Contextual History (last 3 interactions only)**: 
    {past_context}
    **Google Search Results**: 
    {google_results}
    **User Preferences**: Tone: {preferences['tone']}, Detail Level: {preferences['detail_level']}, Language: {lang}, Format: {preferences['format']}
    **Focus Score**: {focus_score:.2f}
    **Instructions**:
    - Respond in a {preferences['tone']} tone with {preferences['detail_level']} detail in {lang}.
    - Format as {preferences['format']} (e.g., paragraphs or bullet points).
    - Focus strictly on answering the query; do not add unrelated commentary or tangents.
    - Use contextual history only if directly relevant to the query; otherwise, ignore it.
    - Incorporate file content or Google Search results if provided and relevant.
    - Keep responses concise and to the point unless high detail is requested.
    - If unclear, ask for clarification politely without speculation.
    """

    try:
        response = await asyncio.to_thread(
            model.start_chat().send_message, prompt
        )
        return response.text.strip()
    except Exception as e:
        st.error(f"❌ Gemini AI Error: {e}")
        return f"Sorry, something went wrong! How can I assist with '{query}'?"

# Profile UI
def profile_ui():
    if not st.session_state.user_id:
        st.subheader("User Profile")
        action = st.radio("Choose an action:", ["Login", "Sign Up"])
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        
        if action == "Sign Up" and st.button("Sign Up"):
            if profile_collection.find_one({"username": username}):
                st.error("Username exists!")
            else:
                user_id = create_profile(username, password)
                st.session_state.user_id = user_id
                st.session_state.user_preferences = get_user_preferences(user_id)
                st.success("Profile created!")
                st.rerun()
        elif action == "Login" and st.button("Login"):
            user_id = authenticate_user(username, password)
            if user_id:
                st.session_state.user_id = user_id
                st.session_state.user_preferences = get_user_preferences(user_id)
                st.session_state.last_query_time = profile_collection.find_one({"user_id": user_id}).get("last_query_time", 0)
                st.success("Logged in!")
                st.rerun()
            else:
                st.error("Invalid credentials!")
    else:
        st.subheader(f"Welcome, User {st.session_state.user_id[:8]}!")
        with st.expander("Preferences"):
            tone = st.selectbox("Tone", ["formal", "casual"], index=["formal", "casual"].index(st.session_state.user_preferences["tone"]))
            detail = st.selectbox("Detail Level", ["low", "medium", "high"], index=["low", "medium", "high"].index(st.session_state.user_preferences["detail_level"]))
            current_lang = st.session_state.user_preferences.get("language", "en")
            lang = st.selectbox("Language", ["en", "es", "fr", "hi"], index=["en", "es", "fr", "hi"].index(current_lang))
            current_format = st.session_state.user_preferences.get("format", "paragraph")
            format = st.selectbox("Format", ["paragraph", "bullet"], index=["paragraph", "bullet"].index(current_format))
            if st.button("Update Preferences"):
                new_prefs = {"tone": tone, "detail_level": detail, "language": lang, "format": format}
                update_user_preferences(st.session_state.user_id, new_prefs)
                st.session_state.user_preferences = new_prefs
                st.success("Preferences updated!")
        if st.button("Logout"):
            st.session_state.user_id = None
            st.session_state.chat_history.clear()
            st.session_state.last_query_time = 0
            st.rerun()

# Analytics Dashboard
def analytics_ui(user_id):
    with st.expander("Analytics Dashboard"):
        history = fetch_chat_history(user_id, 50)
        if history:
            st.write(f"Total Interactions: {len(history)}")
            interests = detect_user_interests(user_id)
            st.write("Top Interests:", ", ".join([f"{k} ({v})" for k, v in interests.items()]))
            ratings = [chat.get("rating", 3) for chat in history if chat.get("rating")]
            st.write(f"Average Rating: {sum(ratings)/len(ratings):.2f}" if ratings else "No ratings yet.")

# Chatbot UI
def chatbot_ui():
    if not st.session_state.user_id:
        st.warning("Please log in or sign up.")
        return

    with st.sidebar:
        st.subheader("Notifications")
        for notif in st.session_state.notifications[-3:]:
            st.info(notif)
        if st.button("Clear Notifications"):
            st.session_state.notifications.clear()
            st.rerun()

    user = profile_collection.find_one({"user_id": st.session_state.user_id})
    if user:
        current_time = datetime.utcnow().timestamp()
        last_query_time = user.get("last_query_time", 0)
        if current_time - last_query_time < 2:
            st.warning("Please wait a moment before sending another query.")
            return
        if user.get("query_count", 0) > 50:
            st.error("Query limit reached for today.")
            return

    if current_time - st.session_state.last_proactive_time > 30 and not st.session_state.query_processing:
        async def proactive_suggestion():
            suggestion = await query_ai(generate_proactive_suggestion(st.session_state.user_id), st.session_state.user_id)
            st.session_state.notifications.append(f"AI Suggests: {suggestion}")
            st.session_state.last_proactive_time = current_time
            st.rerun()
        asyncio.run(proactive_suggestion())

    uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])
    file_content = process_uploaded_file(uploaded_file) if uploaded_file else None

    query = st.chat_input("💬 Type your message...") or st.session_state.last_query
    
    if query and not st.session_state.query_processing:
        if query != st.session_state.last_query or file_content:
            st.session_state.query_processing = True
            st.session_state.last_query = query
            st.session_state.last_response = None
            update_query_count(st.session_state.user_id)

            def process_query_sync():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                ai_response = loop.run_until_complete(query_ai(query, st.session_state.user_id, file_content))
                loop.close()
                store_chat(st.session_state.user_id, query, ai_response)
                st.session_state.last_response = ai_response
                st.session_state.chat_history.append({"User": query, "AI": ai_response})
                st.session_state.query_processing = False
                st.rerun()
            process_query_sync()

    st.subheader("Chat History")
    for i, chat in enumerate(reversed(fetch_chat_history(st.session_state.user_id, 20))):
        with st.chat_message("user"):
            st.markdown(f"**You**: {chat['user']}")
        with st.chat_message("ai"):
            st.markdown(f"**AI**: {chat['ai']}")
            rating = st.slider(f"Rate this", 1, 5, chat.get("rating", 3), key=f"rating_{i}")
            if st.button("Submit Rating", key=f"submit_{i}"):
                store_chat(st.session_state.user_id, chat["user"], chat["ai"], rating)
                st.success(f"Rating {rating} submitted!")
        st.markdown("---")

    analytics_ui(st.session_state.user_id)

# Main App Layout
profile_ui()
chatbot_ui()

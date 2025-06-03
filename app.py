import streamlit as st
import asyncio
import aiohttp
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
import threading
import logging

# Configure logging
logging.basicConfig(filename="cognichat.log", level=logging.ERROR)

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

# Create indexes for performance
chat_collection.create_index([("user_id", ASCENDING), ("timestamp", ASCENDING)])
profile_collection.create_index([("user_id", ASCENDING)])
profile_collection.create_index([("username", ASCENDING)])

# Gemini Model Configuration
generation_config = {
    "temperature": 1,
    "top_p": 0.95,
    "top_k": 64,
    "max_output_tokens": 8192,
    "response_mime_type": "text/plain",
}

model = genai.GenerativeModel(
    model_name="learnlm-2.0-flash-experimental",
    generation_config=generation_config,
    tools='code_execution',
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
if "last_query_time" not in st.session_state:
    st.session_state.last_query_time = 0

# UI Title
st.title("CogniChat - Your Intelligent Assistant")

# Heuristic Multi-Scale Attention
def heuristic_multi_scale_attention(query):
    words = query.lower().split()
    length = len(words)
    short_scale = min(1.0, length / 5)
    specific_keywords = {"what", "how", "why", "who", "where", "when", "explain", "describe"}
    mid_scale = sum(1 for word in words if word in specific_keywords) / max(1, length)
    long_scale = 1.0 if "?" in query or len(re.findall(r"\w+", query)) > 3 else 0.5
    focus_score = (0.3 * short_scale + 0.4 * mid_scale + 0.3 * long_scale)
    return min(max(focus_score, 0.1), 1.0)

# DailyDialog-Inspired Focus Score Adjustment
def adjust_focus_score(query, focus_score):
    words = query.lower().split()
    if any(word in words for word in ["what", "how", "why", "who", "where", "when"]):
        act_bonus = 0.2
        if any(word in words for word in ["great", "good", "happy", "cool"]):
            emotion_bonus = 0.1
        else:
            emotion_bonus = 0.0
    elif any(word in words for word in ["tell", "give", "show"]):
        act_bonus = 0.1
        emotion_bonus = 0.0
    else:
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
        logging.error(f"Google Search Error: {str(e)}, Query: {query}")
        return f"❌ Google Search Error: {e}. Try simplifying your query."

# Chat History Management
@st.cache_data(ttl=60)
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

def clear_chat_history(user_id):
    chat_collection.delete_many({"user_id": user_id})
    st.session_state.chat_history.clear()

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

# Query Suggestion
def generate_query_suggestion(user_id):
    interests = detect_user_interests(user_id)
    if interests:
        top_interest = max(interests, key=interests.get)
        return f"Explore more about {top_interest}?"
    topics = ["latest news", "fun trivia", "math puzzles"]
    return f"Try asking about {random.choice(topics)}?"

# Ascending
# Multi-Modal Processing (PDF Only)
def process_uploaded_file(file):
    if file and file.type == "application/pdf":
        try:
            pdf_reader = PyPDF2.PdfReader(file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() or ""
            return f"Extracted text from PDF: {text}" if text else "No text could be extracted from the PDF."
        except Exception as e:
            logging.error(f"PDF Processing Error: {str(e)}")
            return f"Error processing PDF: {str(e)}. Please upload a valid PDF."
    return "Unsupported file type (only PDFs are supported)."

# AI Query Function
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
    **Contextual History**: 
    {past_context}
    **Google Search Results**: 
    {google_results}
    **User Preferences**: Tone: {preferences['tone']}, Detail Level: {preferences['detail_level']}, Language: {lang}, Format: {preferences['format']}
    **Focus Score**: {focus_score:.2f}
    **Instructions**:
    - Respond in a {preferences['tone']} tone with {preferences['detail_level']} detail in {lang}.
    - Format as {preferences['format']} (e.g., paragraphs or bullet points).
    - Use contextual history for personalization; summarize if long.
    - Incorporate file content or external API data if relevant.
    - Keep greetings engaging; ensure questions are answered accurately.
    - If unclear, ask for clarification politely.
    """

    async def try_api_call(attempts=3, timeout=30):
        for attempt in range(attempts):
            try:
                async with aiohttp.ClientSession() as session:
                    response = await asyncio.wait_for(
                        asyncio.to_thread(model.start_chat().send_message, prompt),
                        timeout=timeout
                    )
                    return response.text.strip()
            except asyncio.TimeoutError:
                if attempt < attempts - 1:
                    await asyncio.sleep(2 ** attempt)
                    continue
                return "Sorry, the AI took too long to respond. Please try again or simplify your query."
            except Exception as e:
                logging.error(f"query_ai error: {str(e)}, Query: {query}, User: {user_id}")
                if attempt < attempts - 1:
                    await asyncio.sleep(2 ** attempt)
                    continue
                return f"Sorry, something went wrong! How can I assist with '{query}'? Error: {str(e)}"
    
    return await try_api_call()

# Profile UI
def profile_ui():
    with st.expander("User Profile", expanded=not st.session_state.user_id):
        if not st.session_state.user_id:
            st.subheader("Login or Sign Up")
            action = st.radio("Choose an action:", ["Login", "Sign Up"])
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            
            if action == "Sign Up" and st.button("Sign Up"):
                if profile_collection.find_one({"username": username}):
                    st.error("Username already exists! Try a different one.")
                else:
                    user_id = create_profile(username, password)
                    st.session_state.user_id = user_id
                    st.session_state.user_preferences = get_user_preferences(user_id)
                    st.success("Profile created successfully!")
                    st.rerun()
            elif action == "Login" and st.button("Login"):
                user_id = authenticate_user(username, password)
                if user_id:
                    st.session_state.user_id = user_id
                    st.session_state.user_preferences = get_user_preferences(user_id)
                    st.session_state.last_query_time = profile_collection.find_one({"user_id": user_id}).get("last_query_time", 0)
                    st.success("Logged in successfully!")
                    st.rerun()
                else:
                    st.error("Invalid username or password. Please try again.")
        else:
            st.subheader(f"Welcome, {profile_collection.find_one({'user_id': st.session_state.user_id})['username']}!")
            with st.form("preferences_form"):
                tone = st.selectbox("Response Tone", ["formal", "casual"], index=["formal", "casual"].index(st.session_state.user_preferences["tone"]))
                detail = st.selectbox("Detail Level", ["low", "medium", "high"], index=["low", "medium", "high"].index(st.session_state.user_preferences["detail_level"]))
                lang = st.selectbox("Language", ["en", "es", "fr", "hi"], index=["en", "es", "fr", "hi"].index(st.session_state.user_preferences.get("language", "en")))
                format = st.selectbox("Response Format", ["paragraph", "bullet"], index=["paragraph", "bullet"].index(st.session_state.user_preferences.get("format", "paragraph")))
                if st.form_submit_button("Update Preferences"):
                    new_prefs = {"tone": tone, "detail_level": detail, "language": lang, "format": format}
                    update_user_preferences(st.session_state.user_id, new_prefs)
                    st.session_state.user_preferences = new_prefs
                    st.success("Preferences updated successfully!")
            if st.button("Logout"):
                st.session_state.user_id = None
                st.session_state.chat_history.clear()
                st.session_state.last_query_time = 0
                st.success("Logged out successfully!")
                st.rerun()

# Analytics Dashboard
def analytics_ui(user_id):
    with st.expander("Analytics Dashboard"):
        history = fetch_chat_history(user_id, 50)
        if history:
            st.metric("Total Interactions", len(history))
            interests = detect_user_interests(user_id)
            st.write("**Top Interests**: " + ", ".join([f"{k} ({v})" for k, v in interests.items()]))
            ratings = [chat.get("rating", 3) for chat in history if chat.get("rating")]
            st.metric("Average Rating", f"{sum(ratings)/len(ratings):.2f}" if ratings else "No ratings yet")
        else:
            st.info("No interactions yet. Start chatting to see analytics!")

# Chatbot UI
def chatbot_ui():
    if not st.session_state.user_id:
        st.warning("Please log in or sign up to start chatting.")
        return

    user = profile_collection.find_one({"user_id": st.session_state.user_id})
    if user:
        current_time = datetime.utcnow().timestamp()
        last_query_time = user.get("last_query_time", 0)
        if current_time - last_query_time < 5:
            st.warning("Please wait a few seconds before sending another query.")
            return
        if user.get("query_count", 0) > 50:
            st.error("Daily query limit reached. Try again tomorrow.")
            return

    with st.expander("Query Suggestions"):
        st.info(generate_query_suggestion(st.session_state.user_id))

    uploaded_file = st.file_uploader("Upload a PDF (Optional)", type=["pdf"])
    file_content = process_uploaded_file(uploaded_file) if uploaded_file else None

    query = st.chat_input("💬 Type your message...") or st.session_state.last_query
    
    if query and not st.session_state.query_processing:
        if query != st.session_state.last_query or file_content:
            st.session_state.query_processing = True
            st.session_state.last_query = query
            st.session_state.last_response = None
            update_query_count(st.session_state.user_id)

            with st.spinner("Processing your query..."):
                def process_query_sync():
                    loop = asyncio.get_event_loop() if asyncio.get_event_loop().is_running() else asyncio.new_event_loop()
                    try:
                        ai_response = loop.run_until_complete(query_ai(query, st.session_state.user_id, file_content))
                        store_chat(st.session_state.user_id, query, ai_response)
                        st.session_state.last_response = ai_response
                        st.session_state.chat_history.append({"User": query, "AI": ai_response})
                    except Exception as e:
                        logging.error(f"process_query_sync error: {str(e)}, Query: {query}, User: {st.session_state.user_id}")
                        st.session_state.last_response = f"Error processing query: {str(e)}. Please try again."
                    finally:
                        st.session_state.query_processing = False
                        if not asyncio.get_event_loop().is_running():
                            loop.close()
                        st.rerun()

                threading.Thread(target=process_query_sync).start()

    with st.expander("Chat History", expanded=True):
        col1, col2 = st.columns([3, 1])
        with col1:
            history_limit = st.slider("Messages to display", 5, 20, 5)
        with col2:
            if st.button("Clear History"):
                clear_chat_history(st.session_state.user_id)
                st.success("Chat history cleared!")
                st.rerun()
        for i, chat in enumerate(reversed(fetch_chat_history(st.session_state.user_id, history_limit))):
            with st.chat_message("user"):
                st.markdown(f"**You**: {chat['user']}")
            with st.chat_message("ai"):
                st.markdown(f"**AI**: {chat['ai']}")
                rating = st.slider(f"Rate this response", 1, 5, chat.get("rating", 3), key=f"rating_{i}")
                if st.button("Submit Rating", key=f"submit_{i}"):
                    store_chat(st.session_state.user_id, chat["user"], chat["ai"], rating)
                    st.success(f"Rating {rating} submitted!")
            st.markdown("---")

    analytics_ui(st.session_state.user_id)

# Main App Layout
profile_ui()
chatbot_ui()

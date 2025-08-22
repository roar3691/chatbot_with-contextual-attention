"""
Agent-based architecture for the chatbot system.
This module implements specialized agents that handle different aspects of the chatbot functionality.
"""

import asyncio
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
import google.generativeai as genai
from googleapiclient.discovery import build
from datetime import datetime
import random
import re


class BaseAgent(ABC):
    """Base class for all agents in the system."""
    
    def __init__(self, name: str):
        self.name = name
    
    @abstractmethod
    async def execute(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the agent's primary function."""
        pass
    
    def get_capabilities(self) -> List[str]:
        """Return list of capabilities this agent provides."""
        return []


class SearchAgent(BaseAgent):
    """Agent responsible for external search operations."""
    
    def __init__(self, google_api_key: str, search_engine_id: str):
        super().__init__("SearchAgent")
        self.google_api_key = google_api_key
        self.search_engine_id = search_engine_id
    
    async def execute(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Perform Google search and return results."""
        query = request.get("query", "")
        if not query:
            return {"error": "No search query provided"}
        
        try:
            service = build("customsearch", "v1", developerKey=self.google_api_key)
            res = service.cse().list(q=query, cx=self.search_engine_id, num=3).execute()
            search_results = res.get("items", [])
            
            formatted_results = "\n".join([
                f"- [{item['title']}]({item['link']})\n{item['snippet']}" 
                for item in search_results
            ]) if search_results else "No results found."
            
            return {"results": formatted_results, "raw_results": search_results}
        except Exception as e:
            return {"error": f"Google Search Error: {e}", "results": "N/A"}
    
    def get_capabilities(self) -> List[str]:
        return ["web_search", "information_retrieval"]


class ContextAgent(BaseAgent):
    """Agent responsible for managing chat history and user context."""
    
    def __init__(self, chat_collection, profile_collection):
        super().__init__("ContextAgent")
        self.chat_collection = chat_collection
        self.profile_collection = profile_collection
    
    async def execute(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Retrieve and process user context."""
        user_id = request.get("user_id")
        action = request.get("action", "get_context")
        
        if action == "get_context":
            return await self._get_user_context(user_id)
        elif action == "summarize_history":
            return await self._summarize_history(user_id)
        elif action == "get_preferences":
            return await self._get_preferences(user_id)
        else:
            return {"error": f"Unknown action: {action}"}
    
    async def _get_user_context(self, user_id: str) -> Dict[str, Any]:
        """Get comprehensive user context."""
        # Get chat history
        history = list(self.chat_collection.find(
            {"user_id": user_id}
        ).sort("timestamp", -1).limit(10))
        
        # Get user preferences
        user_profile = self.profile_collection.find_one({"user_id": user_id})
        preferences = user_profile.get("preferences", {
            "tone": "formal", 
            "detail_level": "medium", 
            "language": "en", 
            "format": "paragraph"
        }) if user_profile else {
            "tone": "formal", 
            "detail_level": "medium", 
            "language": "en", 
            "format": "paragraph"
        }
        
        # Format history
        formatted_history = "\n".join([
            f"Q: {chat['user']}\nA: {chat['ai']}" 
            for chat in reversed(history)
        ])
        
        return {
            "history": formatted_history,
            "preferences": preferences,
            "user_profile": user_profile
        }
    
    async def _summarize_history(self, user_id: str) -> Dict[str, Any]:
        """Summarize user's chat history."""
        history = list(self.chat_collection.find(
            {"user_id": user_id}
        ).sort("timestamp", -1).limit(20))
        
        if len(history) <= 10:
            formatted_history = "\n".join([
                f"Q: {chat['user']}\nA: {chat['ai']}" 
                for chat in reversed(history)
            ])
            return {"summary": formatted_history}
        
        # For longer histories, create a summary
        recent_topics = []
        for chat in history[:10]:
            recent_topics.append(chat['user'])
        
        summary = f"Recent conversation topics: {', '.join(recent_topics[:5])}"
        return {"summary": summary}
    
    async def _get_preferences(self, user_id: str) -> Dict[str, Any]:
        """Get user preferences."""
        user_profile = self.profile_collection.find_one({"user_id": user_id})
        preferences = user_profile.get("preferences", {
            "tone": "formal", 
            "detail_level": "medium", 
            "language": "en", 
            "format": "paragraph"
        }) if user_profile else {
            "tone": "formal", 
            "detail_level": "medium", 
            "language": "en", 
            "format": "paragraph"
        }
        
        return {"preferences": preferences}
    
    def get_capabilities(self) -> List[str]:
        return ["context_management", "history_summarization", "user_preferences"]


class AttentionAgent(BaseAgent):
    """Agent responsible for focus scoring and attention mechanisms."""
    
    def __init__(self):
        super().__init__("AttentionAgent")
    
    async def execute(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate focus and attention scores."""
        query = request.get("query", "")
        action = request.get("action", "calculate_focus")
        
        if action == "calculate_focus":
            focus_score = self._heuristic_multi_scale_attention(query)
            adjusted_score = self._adjust_focus_score(query, focus_score)
            
            return {
                "focus_score": focus_score,
                "adjusted_score": adjusted_score,
                "final_score": adjusted_score
            }
        else:
            return {"error": f"Unknown action: {action}"}
    
    def _heuristic_multi_scale_attention(self, query: str) -> float:
        """Calculate base focus score using heuristic multi-scale attention."""
        if not query:
            return 0.1
        
        # Length-based attention
        length_score = min(len(query.split()) / 20.0, 1.0)
        
        # Question-based attention
        question_words = ["what", "how", "why", "when", "where", "who", "which"]
        question_score = 0.3 if any(word in query.lower() for word in question_words) else 0.1
        
        # Complexity-based attention
        complexity_indicators = ["explain", "analyze", "compare", "detailed", "comprehensive"]
        complexity_score = 0.4 if any(word in query.lower() for word in complexity_indicators) else 0.2
        
        # Technical terms attention
        technical_terms = ["algorithm", "code", "programming", "technical", "implementation"]
        technical_score = 0.3 if any(word in query.lower() for word in technical_terms) else 0.1
        
        focus_score = (length_score + question_score + complexity_score + technical_score) / 4.0
        return min(max(focus_score, 0.1), 1.0)
    
    def _adjust_focus_score(self, query: str, focus_score: float) -> float:
        """Adjust focus score based on emotional and action indicators."""
        # Action words bonus
        action_words = ["help", "solve", "fix", "create", "build", "implement"]
        act_bonus = 0.1 if any(word in query.lower() for word in action_words) else 0.0
        
        # Emotion words bonus  
        emotion_words = ["urgent", "important", "please", "need", "struggling"]
        emotion_bonus = 0.15 if any(word in query.lower() for word in emotion_words) else 0.0
        
        return min(max(focus_score + act_bonus + emotion_bonus, 0.1), 1.0)
    
    def get_capabilities(self) -> List[str]:
        return ["attention_scoring", "focus_calculation", "query_analysis"]


class ResponseAgent(BaseAgent):
    """Agent responsible for generating AI responses."""
    
    def __init__(self, model: genai.GenerativeModel):
        super().__init__("ResponseAgent")
        self.model = model
    
    async def execute(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Generate AI response using provided context."""
        query = request.get("query", "")
        context = request.get("context", {})
        search_results = request.get("search_results", "N/A")
        focus_score = request.get("focus_score", 0.5)
        file_content = request.get("file_content")
        
        # Extract context components
        history = context.get("history", "")
        preferences = context.get("preferences", {})
        
        # Prepare query with file content if provided
        if file_content:
            query = f"{query}\n\nFile Content: {file_content}"
        
        # Build comprehensive prompt
        prompt = self._build_prompt(query, history, search_results, preferences, focus_score)
        
        try:
            response = await asyncio.to_thread(
                self.model.start_chat().send_message, prompt
            )
            return {"response": response.text.strip(), "success": True}
        except Exception as e:
            return {
                "response": f"Sorry, something went wrong! How can I assist with '{query}'?",
                "success": False,
                "error": str(e)
            }
    
    def _build_prompt(self, query: str, history: str, search_results: str, 
                     preferences: Dict[str, str], focus_score: float) -> str:
        """Build comprehensive prompt for AI response generation."""
        lang = preferences.get("language", "en")
        
        prompt = f"""
        **User Query**: "{query}"
        **Contextual History**: 
        {history}
        **Search Results**: 
        {search_results}
        **User Preferences**: Tone: {preferences.get('tone', 'formal')}, Detail Level: {preferences.get('detail_level', 'medium')}, Language: {lang}, Format: {preferences.get('format', 'paragraph')}
        **Focus Score**: {focus_score:.2f}
        **Instructions**:
        - Respond in a {preferences.get('tone', 'formal')} tone with {preferences.get('detail_level', 'medium')} detail in {lang}.
        - Format as {preferences.get('format', 'paragraph')} (e.g., paragraphs or bullet points).
        - Use contextual history for personalization; summarize if long.
        - Incorporate file content or external search data if relevant.
        - Keep greetings engaging; ensure questions are answered accurately.
        - If unclear, ask for clarification politely.
        """
        
        return prompt
    
    def get_capabilities(self) -> List[str]:
        return ["response_generation", "prompt_building", "ai_interaction"]


class ProactiveAgent(BaseAgent):
    """Agent responsible for generating proactive suggestions."""
    
    def __init__(self, chat_collection, profile_collection):
        super().__init__("ProactiveAgent")
        self.chat_collection = chat_collection
        self.profile_collection = profile_collection
    
    async def execute(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Generate proactive suggestions for the user."""
        user_id = request.get("user_id")
        
        # Detect user interests
        interests = self._detect_user_interests(user_id)
        
        if interests and random.random() > 0.3:
            top_interest = max(interests, key=interests.get)
            suggestion = f"Hey, noticed you're into {top_interest}. Want to chat about it?"
        else:
            topics = ["latest news", "fun trivia", "math puzzles", "creative writing", "technology trends"]
            suggestion = f"How about we discuss {random.choice(topics)}?"
        
        return {"suggestion": suggestion, "interests": interests}
    
    def _detect_user_interests(self, user_id: str) -> Dict[str, int]:
        """Detect user interests from chat history."""
        history = list(self.chat_collection.find(
            {"user_id": user_id}
        ).sort("timestamp", -1).limit(20))
        
        interest_keywords = {
            "technology": ["code", "programming", "tech", "software", "AI", "machine learning"],
            "science": ["science", "research", "experiment", "theory", "physics", "chemistry"],
            "business": ["business", "marketing", "finance", "startup", "investment"],
            "health": ["health", "fitness", "exercise", "nutrition", "wellness"],
            "education": ["learn", "study", "education", "course", "tutorial"],
            "entertainment": ["movie", "music", "game", "book", "entertainment"]
        }
        
        interests = {}
        for chat in history:
            user_query = chat.get("user", "").lower()
            for topic, keywords in interest_keywords.items():
                count = sum(1 for keyword in keywords if keyword in user_query)
                interests[topic] = interests.get(topic, 0) + count
        
        # Filter out interests with count 0
        return {k: v for k, v in interests.items() if v > 0}
    
    def get_capabilities(self) -> List[str]:
        return ["proactive_suggestions", "interest_detection", "user_engagement"]


class AgentCoordinator:
    """Coordinates multiple agents to handle complex requests."""
    
    def __init__(self):
        self.agents = {}
        self.execution_log = []
    
    def register_agent(self, agent: BaseAgent):
        """Register an agent with the coordinator."""
        self.agents[agent.name] = agent
    
    async def process_request(self, request_type: str, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process a request by coordinating appropriate agents."""
        self.execution_log.append(f"Processing {request_type} request")
        
        if request_type == "chat_query":
            return await self._handle_chat_query(request_data)
        elif request_type == "proactive_suggestion":
            return await self._handle_proactive_suggestion(request_data)
        else:
            return {"error": f"Unknown request type: {request_type}"}
    
    async def _handle_chat_query(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle a chat query by coordinating multiple agents."""
        user_id = request_data.get("user_id")
        query = request_data.get("query")
        file_content = request_data.get("file_content")
        
        try:
            # Step 1: Get user context
            context_result = await self.agents["ContextAgent"].execute({
                "user_id": user_id,
                "action": "get_context"
            })
            
            # Step 2: Calculate attention scores
            attention_result = await self.agents["AttentionAgent"].execute({
                "query": query,
                "action": "calculate_focus"
            })
            
            # Step 3: Perform search (if no file content)
            search_result = {"results": "N/A"}
            if not file_content and "SearchAgent" in self.agents:
                search_result = await self.agents["SearchAgent"].execute({
                    "query": query
                })
            
            # Step 4: Generate response
            response_result = await self.agents["ResponseAgent"].execute({
                "query": query,
                "context": context_result,
                "search_results": search_result.get("results", "N/A"),
                "focus_score": attention_result.get("final_score", 0.5),
                "file_content": file_content
            })
            
            return {
                "response": response_result.get("response", ""),
                "success": response_result.get("success", False),
                "context": context_result,
                "attention": attention_result,
                "search": search_result,
                "execution_log": self.execution_log.copy()
            }
            
        except Exception as e:
            return {"error": f"Agent coordination error: {e}", "success": False}
    
    async def _handle_proactive_suggestion(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle proactive suggestion generation."""
        if "ProactiveAgent" not in self.agents:
            return {"error": "ProactiveAgent not available"}
        
        return await self.agents["ProactiveAgent"].execute(request_data)
    
    def get_agent_status(self) -> Dict[str, Any]:
        """Get status of all registered agents."""
        status = {}
        for name, agent in self.agents.items():
            status[name] = {
                "capabilities": agent.get_capabilities(),
                "status": "active"
            }
        return status
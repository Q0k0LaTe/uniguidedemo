import asyncio
import json
import logging
import os
import re
import urllib.parse
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from enum import Enum

from openai import OpenAI
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import uvicorn
import httpx

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

if not os.getenv("OPENAI_API_KEY"):
    logger.warning("OPENAI_API_KEY environment variable not set!")

app = FastAPI(title="UniGuide AI Chatbot")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Data Models
class IntentType(Enum):
    COLLEGE_MATCH = "college_match"
    ESSAY_REVISE = "essay_revise"
    SCHEDULE_PLAN = "schedule_plan"
    GENERAL_QA = "general_qa"

@dataclass
class UserProfile:
    gpa: Optional[float] = None
    sat_score: Optional[int] = None
    act_score: Optional[int] = None
    interests: List[str] = None
    budget: Optional[Dict[str, Any]] = None
    location_preference: List[str] = None
    major_preference: List[str] = None
    
    def __post_init__(self):
        if self.interests is None:
            self.interests = []
        if self.location_preference is None:
            self.location_preference = []
        if self.major_preference is None:
            self.major_preference = []

@dataclass
class College:
    name: str
    location: str
    ranking: Optional[int] = None
    tuition: Optional[int] = None
    acceptance_rate: Optional[float] = None
    avg_sat: Optional[int] = None
    avg_gpa: Optional[float] = None
    majors: List[str] = None
    description: str = ""
    fit_score: float = 0.0
    
    def __post_init__(self):
        if self.majors is None:
            self.majors = []

# Web Search Tool for University Data
class UniversitySearchTool:
    def __init__(self):
        self.search_api_url = "https://api.search.brave.com/res/v1/web/search"
        self.headers = {
            "Accept": "application/json",
            "Accept-Encoding": "gzip",
            "X-Subscription-Token": os.getenv("BRAVE_API_KEY", "")  # Optional: use Brave Search API
        }
    
    async def search_universities(self, query: str) -> List[Dict]:
        """Search for universities using web search"""
        try:
            # Use DuckDuckGo search if no Brave API key
            if not os.getenv("BRAVE_API_KEY"):
                return await self._duckduckgo_search(query)
            else:
                return await self._brave_search(query)
        except Exception as e:
            logger.error(f"Error in university search: {e}")
            return []
    
    async def _duckduckgo_search(self, query: str) -> List[Dict]:
        """Fallback search using DuckDuckGo instant answer API"""
        try:
            async with httpx.AsyncClient() as client:
                # Search for university information
                encoded_query = urllib.parse.quote_plus(query)
                search_url = f"https://api.duckduckgo.com/?q={encoded_query}&format=json&no_html=1&skip_disambig=1"
                response = await client.get(search_url, timeout=10)
                
                if response.status_code == 200:
                    data = response.json()
                    results = []
                    
                    # Extract from abstract
                    if data.get("Abstract"):
                        results.append({
                            "title": data.get("Heading", query),
                            "snippet": data.get("Abstract", ""),
                            "url": data.get("AbstractURL", "")
                        })
                    
                    # Extract from related topics
                    for topic in data.get("RelatedTopics", [])[:5]:
                        if isinstance(topic, dict) and topic.get("Text"):
                            results.append({
                                "title": topic.get("FirstURL", "").split("/")[-1].replace("_", " "),
                                "snippet": topic.get("Text", ""),
                                "url": topic.get("FirstURL", "")
                            })
                    
                    return results
                
                return []
        except Exception as e:
            logger.error(f"DuckDuckGo search error: {e}")
            return []
    
    async def _brave_search(self, query: str) -> List[Dict]:
        """Search using Brave Search API (if API key available)"""
        try:
            async with httpx.AsyncClient() as client:
                params = {
                    "q": query,
                    "count": 10,
                    "search_lang": "en",
                    "country": "US",
                    "safesearch": "moderate"
                }
                
                response = await client.get(
                    self.search_api_url,
                    headers=self.headers,
                    params=params,
                    timeout=10
                )
                
                if response.status_code == 200:
                    data = response.json()
                    results = []
                    
                    for result in data.get("web", {}).get("results", []):
                        results.append({
                            "title": result.get("title", ""),
                            "snippet": result.get("description", ""),
                            "url": result.get("url", "")
                        })
                    
                    return results
                
                return []
        except Exception as e:
            logger.error(f"Brave search error: {e}")
            return []

class CollegeSearchTool:
    def __init__(self):
        self.university_search = UniversitySearchTool()
    
    async def search(self, query_params: Dict) -> List[College]:
        """Search colleges using web search and AI processing"""
        try:
            # Build search query
            search_query = self._build_search_query(query_params)
            
            # Search the web for university information
            search_results = await self.university_search.search_universities(search_query)
            
            # Process search results with AI to extract structured data
            colleges = await self._process_search_results(search_results, query_params)
            
            return colleges[:10]  # Return top 10 results
            
        except Exception as e:
            logger.error(f"Error in college search: {e}")
            return []
    
    def _build_search_query(self, query_params: Dict) -> str:
        """Build search query from parameters"""
        query_parts = ["universities colleges"]
        
        if "majors" in query_params:
            majors_str = " ".join(query_params["majors"])
            query_parts.append(f"{majors_str} programs")
        
        if "location" in query_params:
            location_str = " ".join(query_params["location"])
            query_parts.append(f"in {location_str}")
        
        query_parts.append("admission requirements tuition ranking")
        
        return " ".join(query_parts)
    
    async def _process_search_results(self, search_results: List[Dict], query_params: Dict) -> List[College]:
        """Use AI to extract structured college data from search results"""
        if not search_results:
            return []
        
        # Prepare search results text for AI processing
        search_text = ""
        for result in search_results[:8]:  # Limit to first 8 results
            search_text += f"Title: {result.get('title', '')}\n"
            search_text += f"Description: {result.get('snippet', '')}\n"
            search_text += f"URL: {result.get('url', '')}\n\n"
        
        system_prompt = """
        You are a university data extraction expert. Extract structured information about universities/colleges from the provided search results.
        
        For each university mentioned, extract:
        - name: Full university name
        - location: City, State/Country
        - ranking: National ranking (if mentioned)
        - tuition: Annual tuition cost in USD (if mentioned)
        - acceptance_rate: Acceptance rate as decimal (e.g. 0.15 for 15%)
        - avg_sat: Average SAT score (if mentioned)
        - avg_gpa: Average GPA (if mentioned)
        - majors: List of strong/notable programs
        - description: Brief description highlighting key strengths
        
        Return a JSON array of universities. Only include universities with sufficient information.
        If specific numbers aren't mentioned, omit those fields.
        
        Example format:
        [
          {
            "name": "Stanford University",
            "location": "Stanford, CA",
            "ranking": 5,
            "tuition": 56000,
            "acceptance_rate": 0.04,
            "avg_sat": 1520,
            "avg_gpa": 3.9,
            "majors": ["Computer Science", "Engineering", "Business"],
            "description": "Elite private research university known for innovation and technology programs"
          }
        ]
        """
        
        try:
            response = await client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Extract university data from these search results:\n\n{search_text}"}
                ],
                temperature=0.1,
                max_tokens=2000
            )
            
            # Parse AI response
            ai_response = response.choices[0].message.content.strip()
            
            colleges_data = None
            # Try to extract JSON from markdown or direct array
            # Regex tries to find ```json [...] ``` or just [...]
            json_markdown_match = re.search(r"```json\\s*(\\[.*?\\])```|(\\[.*?\\])", ai_response, re.DOTALL)
            
            if json_markdown_match:
                # Prioritize the capture group from ```json [...] ``` (group 1)
                # Otherwise, use the standalone array (group 2)
                json_str = json_markdown_match.group(1) or json_markdown_match.group(2)
                if json_str:
                    try:
                        colleges_data = json.loads(json_str)
                    except json.JSONDecodeError as e:
                        logger.error(f"Error decoding JSON from regex match: {e}. Content: {json_str}")
                        # Fallback to trying the whole response if regex-extracted part fails
                        colleges_data = None 

            if colleges_data is None: # If regex didn't match or parsing the match failed
                try:
                    colleges_data = json.loads(ai_response)
                except json.JSONDecodeError as e:
                    logger.error(f"Error decoding JSON from full AI response: {e}. Content: {ai_response}")
                    return [] # Cannot proceed if JSON is invalid

            if colleges_data is None: # Should not happen if the above logic is correct, but as a safeguard
                logger.error(f"Could not parse JSON from AI response. Content: {ai_response}")
                return []

            # Convert to College objects
            colleges = []
            for college_data in colleges_data:
                college = College(
                    name=college_data.get("name", ""),
                    location=college_data.get("location", ""),
                    ranking=college_data.get("ranking"),
                    tuition=college_data.get("tuition"),
                    acceptance_rate=college_data.get("acceptance_rate"),
                    avg_sat=college_data.get("avg_sat"),
                    avg_gpa=college_data.get("avg_gpa"),
                    majors=college_data.get("majors", []),
                    description=college_data.get("description", "")
                )
                colleges.append(college)
            
            return colleges
            
        except Exception as e:
            logger.error(f"Error processing search results with AI: {e}")
            return []

class FitScoringTool:
    def score_college(self, college: College, user_profile: UserProfile) -> float:
        """Calculate fit score between college and user profile"""
        score = 0.0
        factors = 0
        
        # Academic fit (40% weight)
        if user_profile.gpa and user_profile.sat_score and college.avg_gpa and college.avg_sat:
            gpa_diff = abs(college.avg_gpa - user_profile.gpa)
            sat_diff = abs(college.avg_sat - user_profile.sat_score)
            
            # Normalize scores (closer = better fit)
            gpa_score = max(0, 1 - (gpa_diff / 4.0))  # GPA scale 0-4
            sat_score = max(0, 1 - (sat_diff / 800))   # SAT scale ~800-1600
            
            score += (gpa_score + sat_score) * 0.4
            factors += 1
        
        # Major interest fit (30% weight)
        if user_profile.major_preference and college.majors:
            major_matches = sum(1 for major in user_profile.major_preference 
                              if any(major.lower() in college_major.lower() for college_major in college.majors))
            if len(user_profile.major_preference) > 0:
                major_score = major_matches / len(user_profile.major_preference)
                score += major_score * 0.3
                factors += 1
        
        # Location preference fit (20% weight)
        if user_profile.location_preference and college.location:
            location_matches = sum(1 for loc in user_profile.location_preference 
                                 if loc.lower() in college.location.lower())
            if location_matches > 0:
                score += 0.2
            factors += 1
        
        # Acceptance rate consideration (10% weight)
        if college.acceptance_rate:
            # Higher acceptance rate = better chance = higher score for safety
            acceptance_score = min(1.0, college.acceptance_rate * 2)  # Cap at 1.0
            score += acceptance_score * 0.1
            factors += 1
        
        return score if factors > 0 else 0.5  # Default score if no factors

class UniGuideAgent:
    def __init__(self):
        self.college_search = CollegeSearchTool()
        self.fit_scoring = FitScoringTool()
        self.conversation_history = []
        self.user_profile = UserProfile()
    
    def extract_user_info(self, message: str) -> Dict[str, Any]:
        """Extract user information from natural language using OpenAI"""
        system_prompt = """
        You are an information extraction assistant for college admissions. 
        Extract relevant information from the user's message and return it as JSON.
        
        Look for:
        - GPA (as float, e.g., 3.7)
        - SAT score (as integer, e.g., 1450)
        - ACT score (as integer, e.g., 32)
        - Interests/majors (as list of strings)
        - Budget preferences (as JSON object, e.g., {"max_annual_tuition": 50000, "type": "under"} or {"min_annual_tuition": 30000, "max_annual_tuition": 45000, "type": "range"})
        - Location preferences (as list of strings)
        
        Example response:
        {
            "gpa": 3.8,
            "sat_score": 1450,
            "interests": ["computer science", "artificial intelligence"],
            "location_preference": ["California", "Massachusetts"],
            "budget": {"max_annual_tuition": 50000, "type": "under"}
        }
        
        Only include fields that are mentioned. Return empty object {} if no relevant info found.
        If budget is mentioned as a single number, assume it's max_annual_tuition and type "under".
        """
        
        try:
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": message}
                ],
                temperature=0.1
            )
            
            extracted_info = json.loads(response.choices[0].message.content)
            return extracted_info
        except Exception as e:
            logger.error(f"Error extracting user info: {e}")
            return {}
    
    def detect_intent(self, message: str) -> IntentType:
        """Detect user intent using OpenAI"""
        system_prompt = """
        You are an intent classifier for a college guidance chatbot.
        Classify the user's message into one of these categories:
        
        1. college_match - User wants college recommendations
        2. essay_revise - User wants help with essays
        3. schedule_plan - User wants help with deadlines/planning
        4. general_qa - General questions about college admissions
        
        Respond with only the category name.
        """
        
        try:
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": message}
                ],
                temperature=0.1
            )
            
            intent_str = response.choices[0].message.content.strip()
            return IntentType(intent_str)
        except Exception as e:
            logger.error(f"Error detecting intent: {e}")
            return IntentType.GENERAL_QA
    
    def update_profile(self, extracted_info: Dict[str, Any]):
        """Update user profile with extracted information"""
        if "gpa" in extracted_info:
            self.user_profile.gpa = extracted_info["gpa"]
        if "sat_score" in extracted_info:
            self.user_profile.sat_score = extracted_info["sat_score"]
        if "act_score" in extracted_info:
            self.user_profile.act_score = extracted_info["act_score"]
        if "interests" in extracted_info:
            self.user_profile.major_preference.extend(extracted_info["interests"])
        if "location_preference" in extracted_info:
            self.user_profile.location_preference.extend(extracted_info["location_preference"])
        if "budget" in extracted_info:
            # Ensure budget is stored as a dictionary if provided
            if isinstance(extracted_info["budget"], dict):
                self.user_profile.budget = extracted_info["budget"]
            elif isinstance(extracted_info["budget"], str): # Basic fallback if LLM returns string
                try:
                    # Attempt to parse if it's a JSON string by chance
                    parsed_budget = json.loads(extracted_info["budget"])
                    if isinstance(parsed_budget, dict):
                        self.user_profile.budget = parsed_budget
                    else: # If it's a plain string, try to make a structure
                        value = int("".join(filter(str.isdigit, extracted_info["budget"])))
                        if value > 0:
                             self.user_profile.budget = {"max_annual_tuition": value, "type": "under_fallback"}
                except (json.JSONDecodeError, ValueError):
                     logger.warning(f"Could not parse budget string into dict: {extracted_info['budget']}")
                     self.user_profile.budget = {"description": extracted_info["budget"], "type": "string_fallback"}
    
    async def process_college_match(self, message: str) -> str:
        """Process college recommendation request"""
        steps = []
        steps.append("🔍 **Step 1**: Analyzing your academic profile and preferences...")
        
        # Extract user information
        extracted_info = self.extract_user_info(message)
        self.update_profile(extracted_info)
        
        steps.append("🌐 **Step 2**: Searching live university databases and websites...")
        
        # Search colleges using web search
        search_params = {}
        if self.user_profile.major_preference:
            search_params["majors"] = self.user_profile.major_preference
        if self.user_profile.location_preference:
            search_params["location"] = self.user_profile.location_preference
        
        # Updated budget handling
        if self.user_profile.budget and isinstance(self.user_profile.budget, dict):
            if "max_annual_tuition" in self.user_profile.budget:
                try:
                    search_params["max_tuition"] = int(self.user_profile.budget["max_annual_tuition"])
                except ValueError:
                    logger.warning(f"Could not parse max_annual_tuition: {self.user_profile.budget['max_annual_tuition']}")
            # Could add handling for "min_annual_tuition" or "type": "range" here if needed for search_params

        colleges = await self.college_search.search(search_params)
        
        steps.append("🎯 **Step 3**: Calculating fit scores using real-time data...")
        
        # Calculate fit scores
        for college in colleges:
            college.fit_score = self.fit_scoring.score_college(college, self.user_profile)
        
        # Sort by fit score
        colleges.sort(key=lambda x: x.fit_score, reverse=True)
        
        steps.append("✅ **Step 4**: Generating personalized recommendations with current data...")
        
        # Generate response
        response = "\n\n".join(steps) + "\n\n"
        response += "## 🎓 **Real-Time University Recommendations**\n\n"
        
        # Profile summary
        profile_summary = f"**Your Profile**: "
        profile_parts = []
        if self.user_profile.gpa:
            profile_parts.append(f"GPA {self.user_profile.gpa}")
        if self.user_profile.sat_score:
            profile_parts.append(f"SAT {self.user_profile.sat_score}")
        if self.user_profile.major_preference:
            profile_parts.append(f"Interests: {', '.join(self.user_profile.major_preference[:3])}")
        
        response += profile_summary + ", ".join(profile_parts) + "\n\n"
        
        if not colleges:
            response += "**Note**: I searched current university data but found limited results. Let me provide some general guidance based on your profile instead.\n\n"
            # Fallback to general advice
            response += await self.process_general_qa(f"I'm looking for college recommendations with GPA {self.user_profile.gpa}, SAT {self.user_profile.sat_score}, interested in {', '.join(self.user_profile.major_preference)}")
        else:
            # Top recommendations with real data
            for i, college in enumerate(colleges[:5], 1):
                fit_percentage = int(college.fit_score * 100)
                response += f"**{i}. {college.name}** ({fit_percentage}% match)\n"
                response += f"📍 {college.location}"
                
                if college.tuition:
                    response += f" | 💰 ${college.tuition:,}/year"
                if college.acceptance_rate:
                    response += f" | 📈 {college.acceptance_rate:.1%} acceptance rate"
                
                response += "\n"
                
                if college.avg_sat and college.avg_gpa:
                    response += f"📊 Avg SAT: {college.avg_sat} | Avg GPA: {college.avg_gpa}\n"
                
                if college.majors:
                    response += f"🎯 Strong in: {', '.join(college.majors[:3])}\n"
                
                if college.description:
                    response += f"💡 {college.description}\n"
                
                response += "\n"
        
        response += "💡 **Note**: This data is sourced from live web searches for the most current information available."
        
        return response
    
    async def process_general_qa(self, message: str) -> str:
        """Process general Q&A using OpenAI"""
        system_prompt = """
        You are UniGuide AI, a helpful college admissions assistant. 
        Provide accurate, helpful information about college admissions, applications, 
        essays, deadlines, and related topics. Be encouraging and supportive.
        
        Keep responses conversational but informative. If you don't know something specific,
        suggest they consult official sources or a guidance counselor.
        """
        
        try:
            # Include recent conversation context
            messages = [{"role": "system", "content": system_prompt}]
            
            # Add recent conversation history (last 4 exchanges)
            for msg in self.conversation_history[-4:]:
                messages.append(msg)
            
            messages.append({"role": "user", "content": message})
            
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=messages,
                temperature=0.7,
                max_tokens=500
            )
            
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Error in general QA: {e}")
            return "I apologize, but I'm having trouble processing your request right now. Please try again or ask a different question."
    
    async def process_message(self, message: str) -> str:
        """Main message processing function"""
        # Add to conversation history
        self.conversation_history.append({"role": "user", "content": message})
        
        # Detect intent
        intent = self.detect_intent(message)
        
        # Process based on intent
        if intent == IntentType.COLLEGE_MATCH:
            response = await self.process_college_match(message)
        elif intent == IntentType.ESSAY_REVISE:
            response = "📝 **Essay assistance is coming soon!** For now, I recommend focusing on your personal story and unique experiences. Would you like college recommendations instead?"
        elif intent == IntentType.SCHEDULE_PLAN:
            response = "📅 **Schedule planning is coming soon!** I can help you understand application deadlines though. Most early decision deadlines are November 1st, and regular decision deadlines are typically January 1st."
        else:
            response = await self.process_general_qa(message)
        
        # Add response to conversation history
        self.conversation_history.append({"role": "assistant", "content": response})
        
        # Keep conversation history manageable
        if len(self.conversation_history) > 10:
            self.conversation_history = self.conversation_history[-10:]
        
        return response

# WebSocket connection manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[WebSocket, UniGuideAgent] = {} # Store agent per websocket

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections[websocket] = UniGuideAgent() # Create agent for this connection

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            del self.active_connections[websocket]

    async def send_personal_message(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)

    def get_agent(self, websocket: WebSocket) -> Optional[UniGuideAgent]: # Helper to get agent
        return self.active_connections.get(websocket)

manager = ConnectionManager()

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    agent_instance = manager.get_agent(websocket)
    if not agent_instance: # Should not happen if connect was successful
        logger.error(f"Agent not found for websocket: {websocket}")
        await websocket.close(code=1011) # Internal error
        return

    try:
        while True:
            # Receive message from client
            data = await websocket.receive_text()
            
            try:
                message_data = json.loads(data)
            except json.JSONDecodeError:
                logger.error(f"Invalid JSON received: {data}")
                await manager.send_personal_message(
                    json.dumps({
                        "type": "error",
                        "message": "Invalid message format. Please send JSON.",
                        "timestamp": datetime.now().isoformat()
                    }),
                    websocket
                )
                continue # Skip processing this message

            user_message = message_data.get("message", "")
            
            if user_message.strip():
                # Process message with agent_instance
                response = await agent_instance.process_message(user_message)
                
                # Send response back to client
                await manager.send_personal_message(
                    json.dumps({
                        "type": "ai_response",
                        "message": response,
                        "timestamp": datetime.now().isoformat()
                    }),
                    websocket
                )
    
    except WebSocketDisconnect:
        manager.disconnect(websocket)

@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

# Serve the main HTML page (same as before)
@app.get("/", response_class=HTMLResponse)
async def get_index():
    html_content = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>UniGuide AI | AI对话</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
    <link href="https://fonts.googleapis.com/css2?family=Noto+Sans+SC:wght@300;400;500;700&display=swap" rel="stylesheet">
    <style>
        body { font-family: 'Noto Sans SC', sans-serif; background-color: #f8f9ff; }
        .chat-container { height: calc(100vh - 160px); }
        .message { max-width: 80%; animation: fadeIn 0.3s ease-in-out; }
        .user-message { background-color: #4f46e5; color: white; border-radius: 18px 18px 0 18px; }
        .ai-message { background-color: #f3f4f6; color: #1f2937; border-radius: 18px 18px 18px 0; line-height: 1.6; }
        .ai-message h1, .ai-message h2, .ai-message h3 { font-weight: bold; margin: 12px 0 8px 0; }
        .ai-message h2 { font-size: 1.1em; color: #4f46e5; }
        .ai-message ul, .ai-message ol { margin: 8px 0; padding-left: 20px; }
        .ai-message li { margin: 4px 0; }
        .ai-message strong { font-weight: 600; color: #1f2937; }
        .ai-message p { margin: 8px 0; }
        .typing-indicator span { animation: blink 1.4s infinite both; }
        .typing-indicator span:nth-child(2) { animation-delay: 0.2s; }
        .typing-indicator span:nth-child(3) { animation-delay: 0.4s; }
        .message-input { border-radius: 24px; transition: all 0.3s ease; }
        .message-input:focus { box-shadow: 0 0 0 2px rgba(79, 70, 229, 0.3); }
        .send-button { transition: all 0.2s ease; }
        .send-button:hover { transform: scale(1.05); }
        .send-button:active { transform: scale(0.95); }
        .send-button:disabled { opacity: 0.5; cursor: not-allowed; transform: none; }
        .suggestions button { transition: all 0.2s ease; }
        .suggestions button:hover { transform: translateY(-2px); }
        .welcome-container { background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%); border-radius: 16px; box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.1), 0 10px 10px -5px rgba(0, 0, 0, 0.04); }
        .connection-status { position: fixed; top: 20px; right: 20px; padding: 8px 16px; border-radius: 20px; font-size: 14px; font-weight: 500; z-index: 1000; }
        .connected { background-color: #10b981; color: white; }
        .disconnected { background-color: #ef4444; color: white; }
        .connecting { background-color: #f59e0b; color: white; }
        @keyframes fadeIn { from { opacity: 0; transform: translateY(10px); } to { opacity: 1; transform: translateY(0); } }
        @keyframes blink { 0% { opacity: 0.2; } 20% { opacity: 1; } 100% { opacity: 0.2; } }
        .custom-scrollbar::-webkit-scrollbar { width: 6px; }
        .custom-scrollbar::-webkit-scrollbar-track { background: transparent; }
        .custom-scrollbar::-webkit-scrollbar-thumb { background-color: #d1d5db; border-radius: 3px; }
        .custom-scrollbar::-webkit-scrollbar-thumb:hover { background-color: #9ca3af; }
    </style>
</head>
<body class="h-screen flex flex-col">
    <div id="connection-status" class="connection-status connecting">🔄 连接中...</div>
    <nav class="bg-white shadow-sm py-3 px-4 flex items-center justify-between">
        <div class="flex items-center">
            <svg class="h-8 w-8 text-blue-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 6.253v13m0-13C10.832 5.477 9.246 5 7.5 5S4.168 5.477 3 6.253v13C4.168 18.477 5.754 18 7.5 18s3.332.477 4.5 1.253m0-13C13.168 5.477 14.754 5 16.5 5c1.747 0 3.332.477 4.5 1.253v13C19.832 18.477 18.247 18 16.5 18c-1.746 0-3.332.477-4.5 1.253"></path>
            </svg>
            <span class="ml-2 text-blue-600 text-xl font-bold">UniGuide AI</span>
        </div>
        <div class="flex items-center">
            <div class="text-xs text-green-600 mr-4">🌐 Live Data</div>
            <button id="clear-chat" class="text-gray-600 hover:text-gray-900 mr-4">
                <svg class="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16"></path>
                </svg>
            </button>
            <div class="h-8 w-8 rounded-full bg-blue-100 flex items-center justify-center">
                <span class="text-blue-600 font-bold text-sm">用户</span>
            </div>
        </div>
    </nav>
    <div class="flex-1 flex flex-col overflow-hidden">
        <div id="chat-container" class="chat-container flex-1 overflow-y-auto p-4 custom-scrollbar">
            <div class="welcome-container text-white p-6 mb-8">
                <h2 class="text-2xl font-bold mb-3">🌐 UniGuide AI - 实时大学数据助手</h2>
                <p class="mb-4">我是您的AI大学推荐顾问，使用<strong>实时网络搜索</strong>为您提供最新的大学信息和个性化申请建议。我会搜索当前的大学排名、录取要求、学费信息等，确保为您提供最准确的数据。</p>
                <div class="bg-white bg-opacity-10 p-4 rounded-lg">
                    <h3 class="font-medium mb-2">🔍 实时搜索能力：</h3>
                    <ul class="space-y-1">
                        <li class="flex items-start"><svg class="h-5 w-5 mr-2 flex-shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 13l4 4L19 7"></path></svg><span>实时搜索大学官网和权威教育网站</span></li>
                        <li class="flex items-start"><svg class="h-5 w-5 mr-2 flex-shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 13l4 4L19 7"></path></svg><span>获取最新录取要求和学费信息</span></li>
                        <li class="flex items-start"><svg class="h-5 w-5 mr-2 flex-shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 13l4 4L19 7"></path></svg><span>AI智能分析和结构化数据提取</span></li>
                        <li class="flex items-start"><svg class="h-5 w-5 mr-2 flex-shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 13l4 4L19 7"></path></svg><span>基于当前数据的个性化匹配算法</span></li>
                    </ul>
                </div>
            </div>
            <div class="suggestions mb-8">
                <h3 class="text-sm font-medium text-gray-500 mb-3">🚀 试试这些搜索：</h3>
                <div class="grid grid-cols-1 md:grid-cols-2 gap-3">
                    <button class="suggestion-btn bg-white border border-gray-200 hover:border-blue-300 p-3 rounded-lg text-left shadow-sm"><p class="font-medium text-gray-800">我是理科生，GPA 3.8，SAT 1450，对计算机科学感兴趣，帮我搜索适合的大学</p></button>
                    <button class="suggestion-btn bg-white border border-gray-200 hover:border-blue-300 p-3 rounded-lg text-left shadow-sm"><p class="font-medium text-gray-800">搜索加州地区商科专业排名靠前的大学，预算5万美元以内</p></button>
                    <button class="suggestion-btn bg-white border border-gray-200 hover:border-blue-300 p-3 rounded-lg text-left shadow-sm"><p class="font-medium text-gray-800">查找工程专业强的公立大学，我的SAT成绩1350</p></button>
                    <button class="suggestion-btn bg-white border border-gray-200 hover:border-blue-300 p-3 rounded-lg text-left shadow-sm"><p class="font-medium text-gray-800">MIT和斯坦福大学最新的录取要求和申请deadline是什么？</p></button>
                </div>
            </div>
            <div id="messages" class="space-y-4">
                <div id="typing-indicator" class="flex justify-start hidden">
                    <div class="message ai-message p-4">
                        <div class="typing-indicator flex space-x-1">
                            <span class="h-2 w-2 bg-gray-500 rounded-full"></span>
                            <span class="h-2 w-2 bg-gray-500 rounded-full"></span>
                            <span class="h-2 w-2 bg-gray-500 rounded-full"></span>
                        </div>
                    </div>
                </div>
            </div>
        </div>
        <div class="p-4 border-t border-gray-200 bg-white">
            <div class="relative">
                <textarea id="message-input" class="message-input w-full border border-gray-300 rounded-lg py-3 px-4 pr-12 focus:outline-none focus:border-blue-500 resize-none" rows="1" placeholder="描述您的情况，我会实时搜索最适合的大学信息..." style="min-height: 50px; max-height: 150px;"></textarea>
                <button id="send-button" class="send-button absolute right-3 bottom-3 bg-blue-600 text-white rounded-full p-2 hover:bg-blue-700 focus:outline-none">
                    <svg class="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 19l9 2-9-18-9 18 9-2zm0 0v-8"></path>
                    </svg>
                </button>
            </div>
            <div class="text-xs text-gray-500 mt-2 flex items-center justify-between">
                <span>按 Enter 发送，Shift + Enter 换行 | 🌐 实时搜索大学数据中...</span>
                <span id="message-count" class="text-gray-400">0 条消息</span>
            </div>
        </div>
    </div>
    <script>
        let ws = null, messageCount = 0;
        const messageInput = document.getElementById('message-input');
        const sendButton = document.getElementById('send-button');
        const messagesContainer = document.getElementById('messages');
        const typingIndicator = document.getElementById('typing-indicator');
        const chatContainer = document.getElementById('chat-container');
        const suggestionBtns = document.querySelectorAll('.suggestion-btn');
        const connectionStatus = document.getElementById('connection-status');
        const messageCountEl = document.getElementById('message-count');
        const clearChatBtn = document.getElementById('clear-chat');
        
        function initWebSocket() {
            const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
            const wsUrl = `${protocol}//${window.location.host}/ws`;
            ws = new WebSocket(wsUrl);
            
            ws.onopen = function(event) {
                console.log('WebSocket连接已建立');
                updateConnectionStatus('connected');
            };
            
            ws.onmessage = function(event) {
                const data = JSON.parse(event.data);
                if (data.type === 'ai_response') {
                    handleAIResponse(data.message);
                }
            };
            
            ws.onclose = function(event) {
                console.log('WebSocket连接已关闭');
                updateConnectionStatus('disconnected');
                setTimeout(initWebSocket, 3000);
            };
            
            ws.onerror = function(error) {
                console.error('WebSocket错误:', error);
                updateConnectionStatus('disconnected');
            };
        }
        
        function updateConnectionStatus(status) {
            const statusEl = connectionStatus;
            statusEl.className = `connection-status ${status}`;
            
            switch(status) {
                case 'connected':
                    statusEl.textContent = '🟢 已连接';
                    sendButton.disabled = false;
                    break;
                case 'disconnected':
                    statusEl.textContent = '🔴 连接断开';
                    sendButton.disabled = true;
                    break;
                case 'connecting':
                    statusEl.textContent = '🔄 连接中...';
                    sendButton.disabled = true;
                    break;
            }
        }
        
        function handleAIResponse(message) {
            typingIndicator.classList.add('hidden');
            const aiMessageDiv = document.createElement('div');
            aiMessageDiv.className = 'flex justify-start';
            const htmlContent = marked.parse(message);
            aiMessageDiv.innerHTML = `<div class="message ai-message p-4">${htmlContent}</div>`;
            messagesContainer.appendChild(aiMessageDiv);
            scrollToBottom();
            sendButton.disabled = false;
            messageInput.disabled = false;
        }
        
        messageInput.addEventListener('input', function() {
            this.style.height = 'auto';
            this.style.height = (this.scrollHeight) + 'px';
        });
        
        function sendMessage() {
            const message = messageInput.value.trim();
            if (!message || !ws || ws.readyState !== WebSocket.OPEN) return;
            
            addUserMessage(message);
            ws.send(JSON.stringify({message: message}));
            messageInput.value = '';
            messageInput.style.height = 'auto';
            typingIndicator.classList.remove('hidden');
            scrollToBottom();
            sendButton.disabled = true;
            messageInput.disabled = true;
            messageCount += 2;
            updateMessageCount();
        }
        
        function addUserMessage(message) {
            const userMessageDiv = document.createElement('div');
            userMessageDiv.className = 'flex justify-end';
            userMessageDiv.innerHTML = `<div class="message user-message p-4"><p>${escapeHtml(message)}</p></div>`;
            messagesContainer.appendChild(userMessageDiv);
            scrollToBottom();
        }
        
        function escapeHtml(text) {
            const div = document.createElement('div');
            div.textContent = text;
            return div.innerHTML;
        }
        
        function scrollToBottom() {
            chatContainer.scrollTop = chatContainer.scrollHeight;
        }
        
        function updateMessageCount() {
            messageCountEl.textContent = `${messageCount} 条消息`;
        }
        
        sendButton.addEventListener('click', sendMessage);
        
        messageInput.addEventListener('keydown', function(e) {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                sendMessage();
            }
        });
        
        suggestionBtns.forEach(btn => {
            btn.addEventListener('click', function() {
                const question = this.querySelector('p').textContent;
                messageInput.value = question;
                messageInput.style.height = 'auto';
                messageInput.style.height = (messageInput.scrollHeight) + 'px';
                sendMessage();
            });
        });
        
        clearChatBtn.addEventListener('click', function() {
            if (confirm('确定要清空所有对话吗？')) {
                const welcomeContainer = document.querySelector('.welcome-container').parentElement;
                const suggestionsContainer = document.querySelector('.suggestions');
                messagesContainer.innerHTML = '';
                messagesContainer.appendChild(welcomeContainer);
                messagesContainer.appendChild(suggestionsContainer);
                messageCount = 0;
                updateMessageCount();
                typingIndicator.classList.add('hidden');
            }
        });
        
        document.addEventListener('DOMContentLoaded', function() {
            updateConnectionStatus('connecting');
            initWebSocket();
        });
        
        document.addEventListener('visibilitychange', function() {
            if (!document.hidden && (!ws || ws.readyState === WebSocket.CLOSED)) {
                initWebSocket();
            }
        });
    </script>
</body>
</html>"""
    return html_content

# Add httpx to requirements
# Update requirements.txt to include:
"""
fastapi>=0.104.1
uvicorn[standard]>=0.24.0
websockets>=12.0
openai>=1.68.2
python-multipart>=0.0.6
pydantic>=2.5.0
httpx>=0.25.0
"""

# CRITICAL: This is the fix for Render port binding
if __name__ == "__main__":
    # Get port from environment variable (Render sets this automatically)
    port = int(os.environ.get("PORT", 8000))
    
    # Log the port being used
    logger.info(f"Starting server on port {port}")
    
    # Start the server with explicit host and port
    uvicorn.run(
        app, 
        host="0.0.0.0",  # Bind to all interfaces
        port=port,       # Use Render's PORT
        log_level="info"
    )
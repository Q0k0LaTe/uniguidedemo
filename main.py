import asyncio
import json
import logging
import os
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
    budget: Optional[str] = None
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
    ranking: int
    location: str
    tuition: int
    acceptance_rate: float
    avg_sat: int
    avg_gpa: float
    majors: List[str]
    fit_score: float = 0.0

# Expanded Mock College Database
MOCK_COLLEGES = [
    College("Harvard University", 1, "Cambridge, MA", 54000, 0.04, 1520, 4.0, ["Computer Science", "Business", "Medicine", "Law"]),
    College("Stanford University", 2, "Stanford, CA", 56000, 0.04, 1510, 3.95, ["Computer Science", "Engineering", "Business"]),
    College("MIT", 3, "Cambridge, MA", 53000, 0.07, 1540, 4.0, ["Computer Science", "Engineering", "Physics"]),
    College("Yale University", 4, "New Haven, CT", 59000, 0.06, 1515, 3.95, ["Liberal Arts", "Law", "Medicine", "Business"]),
    College("Princeton University", 5, "Princeton, NJ", 57000, 0.05, 1525, 3.95, ["Engineering", "Liberal Arts", "Economics"]),
    College("Carnegie Mellon", 15, "Pittsburgh, PA", 58000, 0.15, 1480, 3.8, ["Computer Science", "Engineering", "Business"]),
    College("UC Berkeley", 20, "Berkeley, CA", 45000, 0.16, 1450, 3.9, ["Computer Science", "Engineering", "Business"]),
    College("University of Michigan", 25, "Ann Arbor, MI", 50000, 0.23, 1430, 3.85, ["Computer Science", "Engineering", "Business"]),
    College("Georgia Tech", 30, "Atlanta, GA", 35000, 0.21, 1460, 3.8, ["Computer Science", "Engineering"]),
    College("NYU", 35, "New York, NY", 54000, 0.16, 1420, 3.75, ["Business", "Arts", "Computer Science"]),
    College("Purdue University", 40, "West Lafayette, IN", 30000, 0.58, 1350, 3.7, ["Computer Science", "Engineering"]),
    College("University of Washington", 45, "Seattle, WA", 38000, 0.48, 1400, 3.75, ["Computer Science", "Engineering", "Medicine"]),
    College("Boston University", 50, "Boston, MA", 58000, 0.18, 1440, 3.8, ["Business", "Engineering", "Medicine"]),
    College("Virginia Tech", 55, "Blacksburg, VA", 32000, 0.65, 1320, 3.6, ["Computer Science", "Engineering"]),
    College("Penn State", 60, "University Park, PA", 35000, 0.55, 1310, 3.6, ["Engineering", "Business"]),
]

# AI Agent Tools
class CollegeSearchTool:
    def __init__(self):
        self.colleges = MOCK_COLLEGES
    
    def search(self, query_params: Dict) -> List[College]:
        """Search colleges based on query parameters"""
        results = self.colleges.copy()
        
        # Filter by major
        if "majors" in query_params:
            target_majors = query_params["majors"]
            results = [c for c in results if any(major.lower() in [m.lower() for m in c.majors] for major in target_majors)]
        
        # Filter by location
        if "location" in query_params:
            location_prefs = query_params["location"]
            if location_prefs:
                results = [c for c in results if any(loc.lower() in c.location.lower() for loc in location_prefs)]
        
        # Filter by budget (tuition)
        if "max_tuition" in query_params:
            max_tuition = query_params["max_tuition"]
            results = [c for c in results if c.tuition <= max_tuition]
        
        return results[:10]  # Return top 10 results

class FitScoringTool:
    def score_college(self, college: College, user_profile: UserProfile) -> float:
        """Calculate fit score between college and user profile"""
        score = 0.0
        factors = 0
        
        # Academic fit (40% weight)
        if user_profile.gpa and user_profile.sat_score:
            gpa_diff = abs(college.avg_gpa - user_profile.gpa)
            sat_diff = abs(college.avg_sat - user_profile.sat_score)
            
            # Normalize scores (closer = better fit)
            gpa_score = max(0, 1 - (gpa_diff / 4.0))  # GPA scale 0-4
            sat_score = max(0, 1 - (sat_diff / 800))   # SAT scale ~800-1600
            
            score += (gpa_score + sat_score) * 0.4
            factors += 1
        
        # Major interest fit (30% weight)
        if user_profile.major_preference:
            major_matches = sum(1 for major in user_profile.major_preference 
                              if any(major.lower() in college_major.lower() for college_major in college.majors))
            major_score = major_matches / len(user_profile.major_preference)
            score += major_score * 0.3
            factors += 1
        
        # Location preference fit (20% weight)
        if user_profile.location_preference:
            location_matches = sum(1 for loc in user_profile.location_preference 
                                 if loc.lower() in college.location.lower())
            if location_matches > 0:
                score += 0.2
            factors += 1
        
        # Acceptance rate consideration (10% weight)
        # Higher acceptance rate = better chance = higher score for safety
        acceptance_score = min(1.0, college.acceptance_rate * 2)  # Cap at 1.0
        score += acceptance_score * 0.1
        factors += 1
        
        return score if factors > 0 else 0.0

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
        - Budget preferences (as string description)
        - Location preferences (as list of strings)
        
        Example response:
        {
            "gpa": 3.8,
            "sat_score": 1450,
            "interests": ["computer science", "artificial intelligence"],
            "location_preference": ["California", "Massachusetts"],
            "budget": "under 50000"
        }
        
        Only include fields that are mentioned. Return empty object {} if no relevant info found.
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
            self.user_profile.budget = extracted_info["budget"]
    
    async def process_college_match(self, message: str) -> str:
        """Process college recommendation request"""
        steps = []
        steps.append("🔍 **Step 1**: Analyzing your academic profile and preferences...")
        
        # Extract user information
        extracted_info = self.extract_user_info(message)
        self.update_profile(extracted_info)
        
        steps.append("📊 **Step 2**: Searching college database...")
        
        # Search colleges
        search_params = {}
        if self.user_profile.major_preference:
            search_params["majors"] = self.user_profile.major_preference
        if self.user_profile.location_preference:
            search_params["location"] = self.user_profile.location_preference
        if self.user_profile.budget and "under" in self.user_profile.budget.lower():
            try:
                budget_amount = int(''.join(filter(str.isdigit, self.user_profile.budget)))
                search_params["max_tuition"] = budget_amount
            except:
                pass
        
        colleges = self.college_search.search(search_params)
        
        steps.append("🎯 **Step 3**: Calculating fit scores based on your profile...")
        
        # Calculate fit scores
        for college in colleges:
            college.fit_score = self.fit_scoring.score_college(college, self.user_profile)
        
        # Sort by fit score
        colleges.sort(key=lambda x: x.fit_score, reverse=True)
        
        steps.append("✅ **Step 4**: Generating personalized recommendations...")
        
        # Generate response
        response = "\n\n".join(steps) + "\n\n"
        response += "## 🎓 **Personalized College Recommendations**\n\n"
        
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
        
        # Top recommendations
        for i, college in enumerate(colleges[:5], 1):
            fit_percentage = int(college.fit_score * 100)
            response += f"**{i}. {college.name}** ({fit_percentage}% match)\n"
            response += f"📍 {college.location} | 💰 ${college.tuition:,}/year | 📈 {college.acceptance_rate:.1%} acceptance rate\n"
            response += f"📊 Avg SAT: {college.avg_sat} | Avg GPA: {college.avg_gpa}\n"
            response += f"🎯 Strong in: {', '.join(college.majors[:3])}\n\n"
        
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

# Global agent instance
agent = UniGuideAgent()

# WebSocket connection manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def send_personal_message(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)

manager = ConnectionManager()

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            # Receive message from client
            data = await websocket.receive_text()
            message_data = json.loads(data)
            user_message = message_data.get("message", "")
            
            if user_message.strip():
                # Process message with agent
                response = await agent.process_message(user_message)
                
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
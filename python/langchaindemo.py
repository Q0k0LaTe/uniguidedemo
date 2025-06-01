#!/usr/bin/env python3
"""
LangChain AI Agent Implementation
A comprehensive example showing how to build an AI agent with tools, memory, and reasoning capabilities.
"""

import os
from typing import List, Dict, Any, Optional
from datetime import datetime
import json

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("Note: Install python-dotenv to use .env file: pip install python-dotenv")

# LangChain imports
from langchain.agents import create_openai_tools_agent, AgentExecutor
from langchain_core.tools import Tool
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.memory import ConversationBufferWindowMemory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langchain_core.callbacks import StreamingStdOutCallbackHandler

# Custom tool imports
import requests
import sqlite3
from pathlib import Path

class WeatherTool:
    """Custom tool for getting weather information"""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv('WEATHER_API_KEY')
    
    def get_weather(self, location: str) -> str:
        """Get current weather for a location"""
        if not self.api_key:
            return f"Weather data unavailable - API key not configured for {location}"
        
        try:
            url = f"http://api.openweathermap.org/data/2.5/weather"
            params = {
                'q': location,
                'appid': self.api_key,
                'units': 'metric'
            }
            response = requests.get(url, params=params, timeout=10)
            data = response.json()
            
            if response.status_code == 200:
                temp = data['main']['temp']
                desc = data['weather'][0]['description']
                humidity = data['main']['humidity']
                return f"Weather in {location}: {temp}°C, {desc}, humidity: {humidity}%"
            else:
                return f"Could not get weather for {location}: {data.get('message', 'Unknown error')}"
        except Exception as e:
            return f"Weather lookup failed: {str(e)}"

class NoteTool:
    """Custom tool for managing notes with SQLite"""
    
    def __init__(self, db_path: str = "agent_notes.db"):
        self.db_path = db_path
        self._init_db()
    
    def _init_db(self):
        """Initialize the notes database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS notes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                tags TEXT
            )
        ''')
        conn.commit()
        conn.close()
    
    def save_note(self, title: str, content: str, tags: str = "") -> str:
        """Save a note to the database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO notes (title, content, tags) VALUES (?, ?, ?)",
                (title, content, tags)
            )
            conn.commit()
            note_id = cursor.lastrowid
            conn.close()
            return f"Note '{title}' saved successfully with ID {note_id}"
        except Exception as e:
            return f"Failed to save note: {str(e)}"
    
    def search_notes(self, query: str) -> str:
        """Search notes by title or content"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute('''
                SELECT id, title, content, created_at FROM notes 
                WHERE title LIKE ? OR content LIKE ? 
                ORDER BY created_at DESC LIMIT 10
            ''', (f'%{query}%', f'%{query}%'))
            
            results = cursor.fetchall()
            conn.close()
            
            if not results:
                return f"No notes found matching '{query}'"
            
            formatted_results = []
            for note_id, title, content, created_at in results:
                preview = content[:100] + "..." if len(content) > 100 else content
                formatted_results.append(f"ID {note_id}: {title}\n  {preview}\n  Created: {created_at}")
            
            return "Found notes:\n" + "\n\n".join(formatted_results)
        except Exception as e:
            return f"Failed to search notes: {str(e)}"

class CalculatorTool:
    """Safe calculator tool"""
    
    def calculate(self, expression: str) -> str:
        """Safely evaluate mathematical expressions"""
        try:
            # Remove any potentially dangerous characters
            allowed_chars = set('0123456789+-*/.() ')
            if not all(c in allowed_chars for c in expression):
                return "Error: Only basic mathematical operations are allowed"
            
            # Evaluate the expression safely
            result = eval(expression, {"__builtins__": {}}, {})
            return f"Result: {result}"
        except Exception as e:
            return f"Calculation error: {str(e)}"

class LangChainAgent:
    """Main AI Agent class using LangChain"""
    
    def __init__(self, deepseek_api_key: str, weather_api_key: Optional[str] = None):
        self.llm = ChatOpenAI(
            model="deepseek-chat",
            temperature=0.1,
            openai_api_key=deepseek_api_key,
            openai_api_base="https://api.deepseek.com",
            streaming=True,
            callbacks=[StreamingStdOutCallbackHandler()]
        )
        
        # Initialize custom tools
        self.weather_tool = WeatherTool(weather_api_key)
        self.note_tool = NoteTool()
        self.calc_tool = CalculatorTool()
        
        # Set up memory
        self.memory = ConversationBufferWindowMemory(
            k=10,  # Keep last 10 exchanges
            memory_key="chat_history",
            return_messages=True
        )
        
        # Create tools
        self.tools = self._create_tools()
        
        # Create agent
        self.agent = self._create_agent()
        
        # Create agent executor
        self.agent_executor = AgentExecutor(
            agent=self.agent,
            tools=self.tools,
            memory=self.memory,
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=5
        )
    
    def _create_tools(self) -> List[Tool]:
        """Create and return list of available tools"""
        return [
            Tool(
                name="web_search",
                description="Search the internet for current information. Use this when you need up-to-date information or answers to questions you're not sure about.",
                func=DuckDuckGoSearchRun().run
            ),
            Tool(
                name="weather",
                description="Get current weather information for any city or location. Input should be a city name or location.",
                func=self.weather_tool.get_weather
            ),
            Tool(
                name="calculator",
                description="Perform mathematical calculations. Input should be a mathematical expression like '2 + 2' or '10 * 5 / 2'.",
                func=self.calc_tool.calculate
            ),
            Tool(
                name="save_note",
                description="Save a note with title and content. Input should be in format 'title|content|tags' where tags are optional.",
                func=lambda x: self._parse_and_save_note(x)
            ),
            Tool(
                name="search_notes",
                description="Search through saved notes by title or content. Input should be the search query.",
                func=self.note_tool.search_notes
            )
        ]
    
    def _parse_and_save_note(self, input_str: str) -> str:
        """Parse note input and save note"""
        parts = input_str.split('|')
        if len(parts) < 2:
            return "Error: Please provide input in format 'title|content|tags'"
        
        title = parts[0].strip()
        content = parts[1].strip()
        tags = parts[2].strip() if len(parts) > 2 else ""
        
        return self.note_tool.save_note(title, content, tags)
    
    def _create_agent(self):
        """Create the LangChain agent with custom prompt"""
        
        system_prompt = """You are a helpful AI assistant with access to various tools. 
        
        Your capabilities include:
        - Searching the web for current information
        - Getting weather information for any location
        - Performing mathematical calculations
        - Saving and searching through notes
        
        Guidelines:
        - Always be helpful, accurate, and concise
        - Use tools when needed to provide the most up-to-date and accurate information
        - If you're unsure about something, use the web search tool
        - When saving notes, use a clear title and organize information well
        - Show your reasoning process when solving complex problems
        - Be conversational and friendly
        
        Current date and time: {datetime}
        """.format(datetime=datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad")
        ])
        
        return create_openai_tools_agent(
            llm=self.llm,
            tools=self.tools,
            prompt=prompt
        )
    
    def chat(self, message: str) -> str:
        """Main chat interface"""
        try:
            print(f"Available tools: {[tool.name for tool in self.tools]}")
            response = self.agent_executor.invoke({"input": message})
            return response["output"]
        except Exception as e:
            return f"I encountered an error: {str(e)}. Please try rephrasing your question."
    
    def get_conversation_history(self) -> List[Dict[str, Any]]:
        """Get the conversation history"""
        messages = self.memory.chat_memory.messages
        history = []
        for msg in messages:
            if isinstance(msg, HumanMessage):
                history.append({"role": "human", "content": msg.content})
            elif isinstance(msg, AIMessage):
                history.append({"role": "assistant", "content": msg.content})
        return history
    
    def clear_memory(self):
        """Clear the conversation memory"""
        self.memory.clear()
        print("Conversation memory cleared.")

def main():
    """Main function to run the agent"""
    # Set up API keys (you'll need to provide these)
    deepseek_api_key = os.getenv('DEEPSEEK_API_KEY')
    weather_api_key = os.getenv('WEATHER_API_KEY')  # Optional
    
    if not deepseek_api_key:
        print("Error: Please set your DEEPSEEK_API_KEY environment variable")
        print("You can get an API key from: https://platform.deepseek.com/")
        return
    
    # Initialize the agent
    print("Initializing AI Agent with DeepSeek...")
    agent = LangChainAgent(deepseek_api_key, weather_api_key)
    print("Agent ready! Type 'quit' to exit, 'clear' to clear memory, or 'history' to see conversation history.\n")
    
    # Main chat loop
    while True:
        try:
            user_input = input("\nYou: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'bye']:
                print("Goodbye!")
                break
            elif user_input.lower() == 'clear':
                agent.clear_memory()
                continue
            elif user_input.lower() == 'history':
                history = agent.get_conversation_history()
                print("\n--- Conversation History ---")
                for msg in history[-10:]:  # Show last 10 messages
                    role = msg['role'].title()
                    content = msg['content'][:200] + "..." if len(msg['content']) > 200 else msg['content']
                    print(f"{role}: {content}")
                print("--- End History ---")
                continue
            elif not user_input:
                continue
            
            print("\nAgent: ", end="")
            response = agent.chat(user_input)
            print(f"\n{response}")
            
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"\nError: {str(e)}")

# Example usage and testing
if __name__ == "__main__":
    # Example of how to use the agent programmatically
    example_usage = """
    # Example usage:
    
    # 1. Install required packages:
    # pip install langchain langchain-openai langchain-community langchain-core requests python-dotenv
    
    # 2. Set environment variables (Windows PowerShell):
    # $env:DEEPSEEK_API_KEY="your-deepseek-api-key"
    # OR create a .env file with:
    # DEEPSEEK_API_KEY=your-deepseek-api-key
    # WEATHER_API_KEY=your-weather-api-key
    
    # 3. Run the agent
    # python langchain_agent.py
    
    # 3. Try these example queries:
    # - "What's the weather like in New York?"
    # - "Calculate 15% tip on a $87.50 bill"
    # - "Search for the latest news about AI"
    # - "Save a note: Meeting Notes|Discussed project timeline and budget allocation|work,meeting"
    # - "Search my notes for meeting"
    """
    
    print("LangChain AI Agent")
    print("=" * 50)
    print(example_usage)
    print("=" * 50)
    
    main()
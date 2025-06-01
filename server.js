require('dotenv').config();
const express = require('express');
const http = require('http');
const WebSocket = require('ws');
const path = require('path');
const { OpenAI } = require('openai');
const { URLSearchParams } = require('url'); // For DuckDuckGo query encoding
const fetch = require('node-fetch'); // For making HTTP requests in search tools

const app = express();
const server = http.createServer(app);
const wss = new WebSocket.Server({ server });

const PORT = process.env.PORT || 8000;

// Initialize OpenAI client
let openai;
if (process.env.OPENAI_API_KEY) {
    openai = new OpenAI({
        apiKey: process.env.OPENAI_API_KEY,
    });
} else {
    console.warn('OPENAI_API_KEY environment variable not set! The application may not function correctly.');
}

// --- Data Models (mirroring Python, can be simple objects or classes) ---
class UserProfile {
    constructor() {
        this.gpa = null;
        this.sat_score = null;
        this.act_score = null;
        this.interests = [];
        this.budget = null; // Will be { max_annual_tuition: number, type: string }
        this.location_preference = [];
        this.major_preference = [];
    }
}

class College {
    constructor(name, location, ranking = null, tuition = null, acceptance_rate = null, avg_sat = null, avg_gpa = null, majors = [], description = "", fit_score = 0.0) {
        this.name = name;
        this.location = location;
        this.ranking = ranking;
        this.tuition = tuition;
        this.acceptance_rate = acceptance_rate;
        this.avg_sat = avg_sat;
        this.avg_gpa = avg_gpa;
        this.majors = majors;
        this.description = description;
        this.fit_score = fit_score;
    }
}

const IntentType = {
    COLLEGE_MATCH: "college_match",
    ESSAY_REVISE: "essay_revise",
    SCHEDULE_PLAN: "schedule_plan",
    GENERAL_QA: "general_qa"
};

// --- Connection Manager ---
const connectionManager = {
    activeConnections: new Map(), // Map<WebSocket, UniGuideAgent>

    addConnection(ws, agent) {
        this.activeConnections.set(ws, agent);
        console.log(`Connection added. Total connections: ${this.activeConnections.size}`);
    },

    removeConnection(ws) {
        this.activeConnections.delete(ws);
        console.log(`Connection removed. Total connections: ${this.activeConnections.size}`);
    },

    getAgent(ws) {
        return this.activeConnections.get(ws);
    }
};


// --- UniGuideAgent Class ---
class UniGuideAgent {
    constructor(wsInstance, openaiClient) {
        this.ws = wsInstance;
        this.openai = openaiClient;
        this.userProfile = new UserProfile();
        this.conversationHistory = [];
        // Placeholder for tools, to be implemented
        // this.collegeSearchTool = new CollegeSearchTool(this);
        // this.fitScoringTool = new FitScoringTool();
        console.log("UniGuideAgent initialized for a new connection.");
    }

    async sendStatusUpdate(message) {
        if (this.ws.readyState === WebSocket.OPEN) {
            this.ws.send(JSON.stringify({
                type: 'status_update',
                message: message,
                timestamp: new Date().toISOString()
            }));
        }
    }

    async extractUserInfo(message) {
        await this.sendStatusUpdate("正在提取您的信息...");
        if (!this.openai) {
            console.warn("OpenAI client not available for extractUserInfo.");
            return {};
        }
        const system_prompt = `
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
        `;
        try {
            const response = await this.openai.chat.completions.create({
                model: "gpt-3.5-turbo",
                messages: [
                    { "role": "system", "content": system_prompt },
                    { "role": "user", "content": message }
                ],
                temperature: 0.1,
                response_format: { type: "json_object" },
            });
            const extracted_info = JSON.parse(response.choices[0].message.content);
            console.log("Extracted user info:", extracted_info);
            return extracted_info;
        } catch (error) {
            console.error("Error extracting user info:", error);
            await this.sendStatusUpdate("提取用户信息时出错。");
            return {};
        }
    }

    updateProfile(extractedInfo) {
        if (extractedInfo.gpa) this.userProfile.gpa = parseFloat(extractedInfo.gpa);
        if (extractedInfo.sat_score) this.userProfile.sat_score = parseInt(extractedInfo.sat_score);
        if (extractedInfo.act_score) this.userProfile.act_score = parseInt(extractedInfo.act_score);
        if (extractedInfo.interests && Array.isArray(extractedInfo.interests)) {
            this.userProfile.major_preference = [...new Set([...this.userProfile.major_preference, ...extractedInfo.interests])];
        }
        if (extractedInfo.location_preference && Array.isArray(extractedInfo.location_preference)) {
            this.userProfile.location_preference = [...new Set([...this.userProfile.location_preference, ...extractedInfo.location_preference])];
        }
        if (extractedInfo.budget && typeof extractedInfo.budget === 'object') {
            this.userProfile.budget = extractedInfo.budget;
        } else if (typeof extractedInfo.budget === 'number') { // Handle simple budget number
            this.userProfile.budget = { max_annual_tuition: extractedInfo.budget, type: "under" };
        }
        console.log("Updated user profile:", this.userProfile);
    }

    async detectIntent(message) {
        await this.sendStatusUpdate("正在识别您的意图...");
        if (!this.openai) {
            console.warn("OpenAI client not available for detectIntent.");
            return IntentType.GENERAL_QA; // Fallback
        }
        const system_prompt = `
        You are an intent classifier for a college guidance chatbot.
        Classify the user's message into one of these categories: ${Object.values(IntentType).join(", ")}.
        Respond with only the category name.
        `;
        try {
            const response = await this.openai.chat.completions.create({
                model: "gpt-3.5-turbo",
                messages: [
                    { "role": "system", "content": system_prompt },
                    { "role": "user", "content": message }
                ],
                temperature: 0.1
            });
            const intentStr = response.choices[0].message.content.trim();
            if (Object.values(IntentType).includes(intentStr)) {
                console.log("Detected intent:", intentStr);
                return intentStr;
            }
            console.warn("Unknown intent detected:", intentStr);
            return IntentType.GENERAL_QA;
        } catch (error) {
            console.error("Error detecting intent:", error);
            await this.sendStatusUpdate("识别用户意图时出错。");
            return IntentType.GENERAL_QA;
        }
    }

    async processCollegeMatch(message) {
        await this.sendStatusUpdate("正在分析您的学术背景与偏好以匹配大学...");
        const extractedInfo = await this.extractUserInfo(message);
        this.updateProfile(extractedInfo);

        // Placeholder for actual college search and scoring logic
        await this.sendStatusUpdate("正在搜索大学数据库（占位符）...");
        await new Promise(resolve => setTimeout(resolve, 1000)); // Simulate search
        
        await this.sendStatusUpdate("正在计算匹配得分（占位符）...");
        await new Promise(resolve => setTimeout(resolve, 500));  // Simulate scoring

        let response = `## 🎓 **实时大学推荐 (Node.js 占位符)**

`;
        response += `**您的档案**: GPA ${this.userProfile.gpa || 'N/A'}, SAT ${this.userProfile.sat_score || 'N/A'}, 专业 ${this.userProfile.major_preference.join(', ') || 'N/A'}

`;
        response += `1. **Node.js 虚拟大学** (90% 匹配)
`;
        response += `📍 虚拟城市, VS | 💰 $30,000/年 | 📈 50% 录取率
`;
        response += `📊 平均 SAT: 1300 | 平均 GPA: 3.7
`;
        response += `🎯 强势专业: 计算机科学, 软件工程
`;
        response += `💡 一所优秀的虚拟大学，专注于 Node.js 开发课程。

`;
        response += `💡 **注意**: 此数据为占位符，实际功能待实现。
`;
        
        await this.sendStatusUpdate("个性化推荐报告生成完毕！");
        return response;
    }

    async processGeneralQA(message) {
        await this.sendStatusUpdate("正在思考您的问题...");
        if (!this.openai) {
            return "抱歉，我现在无法回答通用问题，因为 OpenAI 服务未配置。";
        }
        const system_prompt = `
        You are UniGuide AI, a helpful college admissions assistant.
        Provide accurate, helpful information about college admissions, applications,
        essays, deadlines, and related topics. Be encouraging and supportive.

        Keep responses conversational but informative. If you don't know something specific,
        suggest they consult official sources or a guidance counselor.
        Answer in Mandarin Chinese.
        `;
        try {
            const messages = [{ "role": "system", "content": system_prompt }];
            this.conversationHistory.slice(-4).forEach(msg => messages.push(msg)); // Add recent history
            messages.push({ "role": "user", "content": message });

            const response = await this.openai.chat.completions.create({
                model: "gpt-3.5-turbo",
                messages: messages,
                temperature: 0.7,
                max_tokens: 500
            });
            return response.choices[0].message.content;
        } catch (error) {
            console.error("Error in general QA:", error);
            return "抱歉，处理您的问题时发生了错误。";
        }
    }

    async processMessage(userMessageContent) {
        console.log(`Agent processing message: ${userMessageContent}`);
        this.conversationHistory.push({ "role": "user", "content": userMessageContent });

        const intent = await this.detectIntent(userMessageContent);
        let responseContent;

        switch (intent) {
            case IntentType.COLLEGE_MATCH:
                responseContent = await this.processCollegeMatch(userMessageContent);
                break;
            case IntentType.ESSAY_REVISE:
                responseContent = "📝 **论文修改助手即将上线！** 目前，我建议您专注于展现您的个人故事和独特经历。需要大学推荐吗？";
                await this.sendStatusUpdate(responseContent);
                break;
            case IntentType.SCHEDULE_PLAN:
                responseContent = "📅 **规划日程助手即将上线！** 我可以帮您了解申请截止日期。多数早申截止日期在11月1日，常规申请通常在1月1日。";
                await this.sendStatusUpdate(responseContent);
                break;
            default: // GENERAL_QA
                responseContent = await this.processGeneralQA(userMessageContent);
                break;
        }

        this.conversationHistory.push({ "role": "assistant", "content": responseContent });
        if (this.conversationHistory.length > 10) {
            this.conversationHistory = this.conversationHistory.slice(-10);
        }

        if (this.ws.readyState === WebSocket.OPEN) {
            this.ws.send(JSON.stringify({
                type: 'ai_response',
                message: responseContent,
                timestamp: new Date().toISOString()
            }));
        }
        return responseContent; // For logging or other server-side use if needed
    }

    cleanup() {
        // Placeholder for any cleanup logic when a connection is closed
        console.log("Agent cleanup for a closed connection.");
    }
}

// Serve static files (HTML, CSS, client-side JS)
app.use(express.static(path.join(__dirname, 'public')));

// Serve the main HTML page
app.get('/', (req, res) => {
    res.sendFile(path.join(__dirname, 'public', 'index.html'));
});

// WebSocket connection handling
wss.on('connection', (ws) => {
    console.log('Client connected via WebSocket.');
    const agent = new UniGuideAgent(ws, openai); // Pass openai client
    connectionManager.addConnection(ws, agent);

    ws.on('message', async (message) => {
        console.log('Raw message from client:', message.toString());
        try {
            const parsedMessage = JSON.parse(message.toString());
            const userMessage = parsedMessage.message;

            if (userMessage && userMessage.trim() !== "") {
                const currentAgent = connectionManager.getAgent(ws);
                if (currentAgent) {
                    await currentAgent.processMessage(userMessage);
                } else {
                    console.error("Agent not found for this WebSocket connection.");
                    ws.send(JSON.stringify({ type: 'error', message: 'Internal server error: Agent not found.' }));
                }
            } else {
                ws.send(JSON.stringify({ type: 'error', message: 'Empty message received.' , timestamp: new Date().toISOString()}));
            }
        } catch (error) {
            console.error('Error processing message or parsing JSON on server:', error);
            ws.send(JSON.stringify({ type: 'error', message: 'Error processing your request on the server.', timestamp: new Date().toISOString() }));
        }
    });

    ws.on('close', () => {
        console.log('Client disconnected.');
        const agentToCleanup = connectionManager.getAgent(ws);
        if (agentToCleanup) {
            agentToCleanup.cleanup();
        }
        connectionManager.removeConnection(ws);
    });

    ws.on('error', (error) => {
        console.error('WebSocket error:', error);
        // Optionally try to clean up agent if ws instance is available and error is connection-fatal
        const agentToCleanup = connectionManager.getAgent(ws);
        if (agentToCleanup) {
            agentToCleanup.cleanup(); // Perform cleanup
            connectionManager.removeConnection(ws); // Remove from manager
        }
    });
});

// Health check endpoint
app.get('/health', (req, res) => {
    res.json({ status: 'healthy', timestamp: new Date().toISOString() });
});

server.listen(PORT, () => {
    console.log(`Server is listening on port ${PORT}`);
}); 
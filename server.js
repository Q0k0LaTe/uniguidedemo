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

const BRAVE_API_KEY = process.env.BRAVE_API_KEY;
if (!BRAVE_API_KEY) {
    console.warn('BRAVE_API_KEY not set. Web search functionality will be limited or fall back to DuckDuckGo if implemented.');
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
        this.majors = majors || [];
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

// --- University Search Tool (Raw Web Search) ---
class UniversitySearchTool {
    constructor() {
        this.braveSearchApiUrl = "https://api.search.brave.com/res/v1/web/search";
        this.braveHeaders = {
            "Accept": "application/json",
            "Accept-Encoding": "gzip",
            "X-Subscription-Token": BRAVE_API_KEY
        };
    }

    async _brave_search(query) {
        if (!BRAVE_API_KEY) {
            console.warn("Brave API key not available for _brave_search. Skipping.");
            return [];
        }
        try {
            const params = new URLSearchParams({
                q: query,
                count: '10',
                search_lang: 'en',
                country: 'US',
                safesearch: 'moderate'
            });
            const response = await fetch(`${this.braveSearchApiUrl}?${params.toString()}`, {
                headers: this.braveHeaders,
                timeout: 10000 // 10 seconds timeout
            });

            if (response.ok) {
                const data = await response.json();
                return (data.web?.results || []).map(result => ({
                    title: result.title || "",
                    snippet: result.description || "",
                    url: result.url || ""
                }));
            }
            console.error(`Brave search failed with status: ${response.status}`);
            return [];
        } catch (error) {
            console.error("Error in Brave search:", error);
            return [];
        }
    }

    async _duckduckgo_search(query) {
        try {
            const encodedQuery = encodeURIComponent(query);
            const searchUrl = `https://api.duckduckgo.com/?q=${encodedQuery}&format=json&no_html=1&skip_disambig=1`;
            const response = await fetch(searchUrl, { timeout: 10000 });

            if (response.ok) {
                const data = await response.json();
                const results = [];
                if (data.Abstract) {
                    results.push({
                        title: data.Heading || query,
                        snippet: data.Abstract || "",
                        url: data.AbstractURL || ""
                    });
                }
                (data.RelatedTopics || []).slice(0, 5).forEach(topic => {
                    if (topic.Text) {
                        results.push({
                            title: topic.FirstURL ? topic.FirstURL.split("/").pop().replace(/_/g, " ") : "",
                            snippet: topic.Text || "",
                            url: topic.FirstURL || ""
                        });
                    }
                });
                return results;
            }
            console.error(`DuckDuckGo search failed with status: ${response.status}`);
            return [];
        } catch (error) {
            console.error("DuckDuckGo search error:", error);
            return [];
        }
    }

    async search_universities(query) {
        if (BRAVE_API_KEY) {
            return await this._brave_search(query);
        }
        console.log("BRAVE_API_KEY not found, falling back to DuckDuckGo.");
        return await this._duckduckgo_search(query);
    }
}

// --- College Search Tool (Processes Raw Search with AI) ---
class CollegeSearchTool {
    constructor(openaiClient, agentInstance) {
        this.universitySearch = new UniversitySearchTool();
        this.openai = openaiClient;
        this.agent = agentInstance; // To send status updates
    }

    _build_search_query(queryParams) {
        let queryParts = ["universities colleges"];
        if (queryParams.majors && queryParams.majors.length > 0) {
            queryParts.push(`${queryParams.majors.join(" ")} programs`);
        }
        if (queryParams.location && queryParams.location.length > 0) {
            queryParts.push(`in ${queryParams.location.join(" ")}`);
        }
        if (queryParams.max_tuition) {
            queryParts.push(`tuition under $${queryParams.max_tuition}`);
        }
        queryParts.push("admission requirements tuition ranking");
        return queryParts.join(" ");
    }

    async _process_search_results(searchResults, queryParams) {
        if (!searchResults || searchResults.length === 0) return [];
        if (!this.openai) {
            console.warn("OpenAI client not available for _process_search_results.");
            await this.agent.sendStatusUpdate("AI处理模块不可用，无法解析搜索结果。");
            return [];
        }

        let searchText = "";
        searchResults.slice(0, 8).forEach(result => {
            searchText += `Title: ${result.title}\nDescription: ${result.snippet}\nURL: ${result.url}\n\n`;
        });

        const system_prompt = `
        You are a university data extraction expert. Extract structured information about universities/colleges from the provided search results.
        For each university mentioned, extract:
        - name: Full university name (String)
        - location: City, State/Country (String)
        - ranking: National ranking (Integer, if mentioned)
        - tuition: Annual tuition cost in USD (Integer, if mentioned, remove symbols like $ ,)
        - acceptance_rate: Acceptance rate as decimal (Float, e.g. 0.15 for 15%)
        - avg_sat: Average SAT score (Integer, if mentioned)
        - avg_gpa: Average GPA (Float, if mentioned)
        - majors: List of strong/notable programs (Array of Strings)
        - description: Brief description highlighting key strengths (String)
        Return a JSON array of universities. Only include universities with sufficient information.
        If specific numbers aren't mentioned, omit those fields or set to null.
        Ensure tuition, ranking, avg_sat, avg_gpa, acceptance_rate are numbers if present, otherwise null.
        Majors should be an array of strings.
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
        `;

        try {
            const response = await this.openai.chat.completions.create({
                model: "gpt-3.5-turbo",
                messages: [
                    { "role": "system", "content": system_prompt },
                    { "role": "user", "content": `Extract university data from these search results:\n\n${searchText}` }
                ],
                temperature: 0.1,
                max_tokens: 2000,
                response_format: { type: "json_object" },
            });

            const aiResponse = response.choices[0].message.content.trim();
            let collegesData = JSON.parse(aiResponse); // Expecting an object with a key (e.g., "universities") that holds the array
            
            // Check if the response is an object with a key containing the array, or the array itself
            if (typeof collegesData === 'object' && !Array.isArray(collegesData)) {
                const key = Object.keys(collegesData)[0]; // Get the first key
                collegesData = collegesData[key]; // Assume the array is under this key
            }

            if (!Array.isArray(collegesData)) {
                 console.error("AI did not return a valid JSON array for colleges. Received:", collegesData);
                 await this.agent.sendStatusUpdate("AI未能正确解析大学数据格式。");
                 return [];
            }

            return collegesData.map(cd => new College(
                cd.name, cd.location, 
                cd.ranking ? parseInt(cd.ranking) : null, 
                cd.tuition ? parseInt(String(cd.tuition).replace(/[^\d]/g, '')) : null, 
                cd.acceptance_rate ? parseFloat(cd.acceptance_rate) : null, 
                cd.avg_sat ? parseInt(cd.avg_sat) : null, 
                cd.avg_gpa ? parseFloat(cd.avg_gpa) : null, 
                cd.majors || [], 
                cd.description
            ));
        } catch (error) {
            console.error("Error processing search results with AI:", error, "AI Response:", aiResponse);
            await this.agent.sendStatusUpdate("使用AI处理搜索结果时发生错误。");
            return [];
        }
    }

    async search(queryParams) {
        try {
            await this.agent.sendStatusUpdate("正在构建搜索查询...");
            const searchQuery = this._build_search_query(queryParams);

            await this.agent.sendStatusUpdate(`正在使用查询"${searchQuery.substring(0,100)}..."搜索大学数据库与网站...`);
            const searchResults = await this.universitySearch.search_universities(searchQuery);

            if (searchResults && searchResults.length > 0) {
                await this.agent.sendStatusUpdate("正在使用AI处理搜索结果，提取结构化大学信息...");
            } else {
                await this.agent.sendStatusUpdate("未找到相关网络搜索结果，尝试调整搜索条件。");
                return [];
            }
            
            const colleges = await this._process_search_results(searchResults, queryParams);
            return colleges.slice(0, 10);
        } catch (error) {
            console.error("Error in college search tool:", error);
            await this.agent.sendStatusUpdate(`大学搜索过程中发生错误: ${error.message}`);
            return [];
        }
    }
}

// --- Fit Scoring Tool ---
class FitScoringTool {
    score_college(college, userProfile) {
        let score = 0.0;
        let factors = 0;

        if (userProfile.gpa && userProfile.sat_score && college.avg_gpa && college.avg_sat) {
            const gpaDiff = Math.abs(college.avg_gpa - userProfile.gpa);
            const satDiff = Math.abs(college.avg_sat - userProfile.sat_score);
            const gpaScore = Math.max(0, 1 - (gpaDiff / 4.0));
            const satScore = Math.max(0, 1 - (satDiff / 800));
            score += (gpaScore + satScore) * 0.4;
            factors += 1;
        }

        if (userProfile.major_preference && userProfile.major_preference.length > 0 && college.majors && college.majors.length > 0) {
            const majorMatches = userProfile.major_preference.reduce((acc, major) => 
                acc + (college.majors.some(cm => cm.toLowerCase().includes(major.toLowerCase())) ? 1 : 0), 0);
            const majorScore = majorMatches / userProfile.major_preference.length;
            score += majorScore * 0.3;
            factors += 1;
        }

        if (userProfile.location_preference && userProfile.location_preference.length > 0 && college.location) {
            const locationMatches = userProfile.location_preference.some(loc => college.location.toLowerCase().includes(loc.toLowerCase()));
            if (locationMatches) {
                score += 0.2;
            }
            factors += 1; 
        }
        
        if (college.acceptance_rate) {
            const acceptanceScore = Math.min(1.0, college.acceptance_rate * 2);
            score += acceptanceScore * 0.1;
            factors += 1;
        }
        
        // Budget consideration (simple filter, not direct scoring factor here, applied in processCollegeMatch)

        return factors > 0 ? score : 0.5; // Default if no factors match
    }
}

// --- Connection Manager ---
const connectionManager = {
    activeConnections: new Map(),
    addConnection(ws, agent) { this.activeConnections.set(ws, agent); console.log(`Connection added. Total: ${this.activeConnections.size}`); },
    removeConnection(ws) { this.activeConnections.delete(ws); console.log(`Connection removed. Total: ${this.activeConnections.size}`); },
    getAgent(ws) { return this.activeConnections.get(ws); }
};

// --- UniGuideAgent Class ---
class UniGuideAgent {
    constructor(wsInstance, openaiClient) {
        this.ws = wsInstance;
        this.openai = openaiClient;
        this.userProfile = new UserProfile();
        this.conversationHistory = [];
        this.collegeSearchTool = new CollegeSearchTool(openaiClient, this); 
        this.fitScoringTool = new FitScoringTool();
        console.log("UniGuideAgent initialized with tools.");
    }

    async sendStatusUpdate(message) {
        if (this.ws.readyState === WebSocket.OPEN) {
            this.ws.send(JSON.stringify({ type: 'status_update', message, timestamp: new Date().toISOString() }));
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
        } else if (typeof extractedInfo.budget === 'number') { 
            this.userProfile.budget = { max_annual_tuition: extractedInfo.budget, type: "under" };
        }
        console.log("Updated user profile:", this.userProfile);
    }
    
    async detectIntent(message) {
        await this.sendStatusUpdate("正在识别您的意图...");
        if (!this.openai) {
            console.warn("OpenAI client not available for detectIntent.");
            return IntentType.GENERAL_QA; 
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
            console.warn("Unknown intent detected:", intentStr, "Falling back to GENERAL_QA");
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

        let searchParams = {};
        if (this.userProfile.major_preference && this.userProfile.major_preference.length > 0) {
            searchParams.majors = this.userProfile.major_preference;
        }
        if (this.userProfile.location_preference && this.userProfile.location_preference.length > 0) {
            searchParams.location = this.userProfile.location_preference;
        }
        if (this.userProfile.budget && this.userProfile.budget.max_annual_tuition) {
            searchParams.max_tuition = parseInt(this.userProfile.budget.max_annual_tuition);
        }

        let colleges = await this.collegeSearchTool.search(searchParams);

        if (!colleges || colleges.length === 0) {
            await this.sendStatusUpdate("根据您的详细要求，目前未能找到匹配的大学。尝试提供更多信息或调整搜索条件。");
            return await this.processGeneralQA("I couldn't find specific colleges for my detailed request. Can you give some general advice based on my profile?");
        }

        await this.sendStatusUpdate(`找到了 ${colleges.length} 所潜在匹配的大学，正在计算匹配得分...`);
        
        colleges.forEach(college => {
            college.fit_score = this.fitScoringTool.score_college(college, this.userProfile);
        });

        if (searchParams.max_tuition) {
            colleges = colleges.filter(college => college.tuition === null || college.tuition <= searchParams.max_tuition);
             if (colleges.length === 0) {
                await this.sendStatusUpdate("根据您的预算，没有找到符合条件的大学。显示未经过预算筛选的结果。");
            }
        }

        colleges.sort((a, b) => b.fit_score - a.fit_score);

        await this.sendStatusUpdate("个性化推荐报告生成完毕！");

        let response = "## 🎓 **实时大学推荐**\n\n";
        let profileParts = [];
        if (this.userProfile.gpa) profileParts.push(`GPA ${this.userProfile.gpa}`);
        if (this.userProfile.sat_score) profileParts.push(`SAT ${this.userProfile.sat_score}`);
        if (this.userProfile.major_preference.length > 0) profileParts.push(`专业: ${this.userProfile.major_preference.join(', ')}`);
        if (this.userProfile.budget && this.userProfile.budget.max_annual_tuition) {
            profileParts.push(`预算上限: $${this.userProfile.budget.max_annual_tuition}/年`);
        }
        response += `**您的档案**: ${profileParts.join(' | ') || '未提供完整档案'}\n\n`;

        if (colleges.length === 0) {
            response += `**抱歉，根据您的具体条件，未能找到完全匹配的大学。** 您可以尝试放宽一些搜索条件，例如预算或地点，或者提供更多关于您的偏好的信息。\n\n`;
             response += await this.processGeneralQA("I couldn't find specific colleges for my detailed request. Can you give some general advice based on my profile?");
        } else {
            colleges.slice(0, 5).forEach((college, i) => {
                const fitPercentage = Math.round(college.fit_score * 100);
                response += `**${i + 1}. ${college.name}** (${fitPercentage}% 匹配)\n`;
                response += `📍 ${college.location}`;
                if (college.tuition) response += ` | 💰 $${college.tuition.toLocaleString()}/年`;
                if (college.acceptance_rate) response += ` | 📈 ${(college.acceptance_rate * 100).toFixed(1)}% 录取率`;
                response += "\n";
                if (college.avg_sat && college.avg_gpa) response += `📊 平均 SAT: ${college.avg_sat} | 平均 GPA: ${college.avg_gpa}\n`;
                if (college.majors && college.majors.length > 0) response += `🎯 强势专业: ${college.majors.slice(0, 3).join(', ')}\n`;
                if (college.description) response += `💡 ${college.description}\n`;
                response += "\n";
            });
        }
        response += "💡 **注意**: 此数据通过实时网络搜索和AI分析生成，仅供参考。请务必访问大学官网获取最准确信息。";
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
            this.conversationHistory.slice(-4).forEach(msg => messages.push(msg));
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

        if (!this.openai) {
            const errorMsg = "抱歉，AI服务当前未配置，我暂时无法处理您的请求。请稍后再试或联系管理员。";
            await this.sendStatusUpdate(errorMsg);
            if (this.ws.readyState === WebSocket.OPEN) {
                this.ws.send(JSON.stringify({ type: 'ai_response', message: errorMsg, timestamp: new Date().toISOString() }));
            }
            return errorMsg;
        }

        const intent = await this.detectIntent(userMessageContent);
        let responseContent;

        switch (intent) {
            case IntentType.COLLEGE_MATCH:
                responseContent = await this.processCollegeMatch(userMessageContent);
                break;
            case IntentType.ESSAY_REVISE:
                responseContent = "📝 **论文修改助手即将上线！** 目前，我建议您专注于展现您的个人故事和独特经历。需要大学推荐吗？";
                break;
            case IntentType.SCHEDULE_PLAN:
                responseContent = "📅 **规划日程助手即将上线！** 我可以帮您了解申请截止日期。多数早申截止日期在11月1日，常规申请通常在1月1日。";
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
        return responseContent; 
    }

    cleanup() {
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
    const agent = new UniGuideAgent(ws, openai); 
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
                    ws.send(JSON.stringify({ type: 'error', message: 'Internal server error: Agent not found.', timestamp: new Date().toISOString() }));
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
        const agentToCleanup = connectionManager.getAgent(ws);
        if (agentToCleanup) {
            agentToCleanup.cleanup(); 
            connectionManager.removeConnection(ws); 
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
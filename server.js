require('dotenv').config();
const express = require('express');
const http = require('http');
const WebSocket = require('ws');
const path = require('path');
const { OpenAI } = require('openai');
const fetch = require('node-fetch');
const marked = require('marked');

const app = express();
const server = http.createServer(app);
const wss = new WebSocket.Server({ server });

const PORT = process.env.PORT || 8000;

// ============ CONFIGURATION VALIDATION ============
class ConfigValidator {
    static validateEnvironment() {
        const required = ['OPENAI_API_KEY'];
        const optional = ['BRAVE_API_KEY'];
        
        const missing = required.filter(key => !process.env[key]);
        if (missing.length > 0) {
            throw new Error(`❌ Missing required environment variables: ${missing.join(', ')}`);
        }
        
        const warnings = optional.filter(key => !process.env[key]);
        if (warnings.length > 0) {
            console.warn(`⚠️ Optional environment variables not set: ${warnings.join(', ')} - Will use fallback methods`);
        }
        
        console.log('✅ Environment configuration validated');
    }
}

// Validate environment on startup
try {
    ConfigValidator.validateEnvironment();
} catch (error) {
    console.error(error.message);
    process.exit(1);
}

// Initialize OpenAI client
const openai = new OpenAI({
    apiKey: process.env.OPENAI_API_KEY,
    timeout: 30000, // 30 second timeout
    maxRetries: 3,
    fetch: (url, init) => {
        return fetch(url, {
            ...init,
            agent: process.env.HTTPS_PROXY ? 
                new (require('https-proxy-agent'))(process.env.HTTPS_PROXY) : 
                undefined,
            timeout: 30000
        });
    }
});

// ============ DATA MODELS ============
class UserProfile {
    constructor() {
        this.gpa = null;
        this.sat_score = null;
        this.act_score = null;
        this.interests = [];
        this.budget = null;
        this.location_preference = [];
        this.major_preference = [];
    }

    toString() {
        const parts = [];
        if (this.gpa) parts.push(`GPA: ${this.gpa}`);
        if (this.sat_score) parts.push(`SAT: ${this.sat_score}`);
        if (this.act_score) parts.push(`ACT: ${this.act_score}`);
        if (this.major_preference.length > 0) parts.push(`Majors: ${this.major_preference.join(', ')}`);
        if (this.location_preference.length > 0) parts.push(`Locations: ${this.location_preference.join(', ')}`);
        if (this.budget) parts.push(`Budget: $${this.budget.max_annual_tuition || 'N/A'}`);
        return parts.join(' | ') || 'No profile data';
    }
}

class College {
    constructor(data) {
        this.name = data.name || 'Unknown University';
        this.location = data.location || 'Unknown Location';
        this.ranking = this.parseNumber(data.ranking);
        this.tuition = this.parseNumber(data.tuition);
        this.acceptance_rate = this.parseFloat(data.acceptance_rate);
        this.avg_sat = this.parseNumber(data.avg_sat);
        this.avg_gpa = this.parseFloat(data.avg_gpa);
        this.majors = Array.isArray(data.majors) ? data.majors : [];
        this.description = data.description || '';
        this.fit_score = 0.0;
        this.source_url = data.source_url || '';
    }

    parseNumber(value) {
        if (!value) return null;
        const num = parseInt(String(value).replace(/[^\d]/g, ''));
        return isNaN(num) ? null : num;
    }

    parseFloat(value) {
        if (!value) return null;
        const num = parseFloat(value);
        return isNaN(num) ? null : num;
    }

    isValid() {
        return this.name && this.name !== 'Unknown University' && 
               this.location && this.location !== 'Unknown Location';
    }

    toDisplayString() {
        let str = `**${this.name}**\n`;
        str += `📍 ${this.location}`;
        if (this.tuition) str += ` | 💰 $${this.tuition.toLocaleString()}/year`;
        if (this.acceptance_rate) str += ` | 📈 ${(this.acceptance_rate * 100).toFixed(1)}% acceptance`;
        if (this.ranking) str += ` | 🏆 #${this.ranking} ranked`;
        str += '\n';
        if (this.avg_sat && this.avg_gpa) str += `📊 Avg SAT: ${this.avg_sat} | Avg GPA: ${this.avg_gpa}\n`;
        if (this.majors.length > 0) str += `🎯 Strong in: ${this.majors.slice(0, 3).join(', ')}\n`;
        if (this.description) str += `💡 ${this.description}\n`;
        return str;
    }
}

const IntentType = {
    COLLEGE_MATCH: "college_match",
    ESSAY_REVISE: "essay_revise", 
    SCHEDULE_PLAN: "schedule_plan",
    GENERAL_QA: "general_qa"
};

// ============ IMPROVED SEARCH SYSTEM ============
class RobustUniversitySearch {
    constructor() {
        this.cache = new Map();
        this.cacheTimeout = 10 * 60 * 1000; // 10 minutes
        this.staticCollegeData = this.loadStaticCollegeData();
    }

    loadStaticCollegeData() {
        // Comprehensive fallback data for when APIs fail
        return [
            {
                name: "Massachusetts Institute of Technology",
                location: "Cambridge, MA",
                ranking: 2,
                tuition: 57590,
                acceptance_rate: 0.04,
                avg_sat: 1520,
                avg_gpa: 4.17,
                majors: ["Computer Science", "Engineering", "Physics", "Mathematics"],
                description: "Leading technology and engineering research university"
            },
            {
                name: "Stanford University", 
                location: "Stanford, CA",
                ranking: 6,
                tuition: 56169,
                acceptance_rate: 0.04,
                avg_sat: 1505,
                avg_gpa: 4.18,
                majors: ["Computer Science", "Engineering", "Business", "Medicine"],
                description: "Premier research university in Silicon Valley"
            },
            {
                name: "Harvard University",
                location: "Cambridge, MA", 
                ranking: 3,
                tuition: 54269,
                acceptance_rate: 0.03,
                avg_sat: 1515,
                avg_gpa: 4.18,
                majors: ["Liberal Arts", "Medicine", "Law", "Business"],
                description: "Prestigious Ivy League institution"
            },
            {
                name: "California Institute of Technology",
                location: "Pasadena, CA",
                ranking: 9,
                tuition: 58680,
                acceptance_rate: 0.06,
                avg_sat: 1545,
                avg_gpa: 4.19,
                majors: ["Engineering", "Physics", "Computer Science", "Chemistry"],
                description: "Elite science and engineering focused institute"
            },
            {
                name: "University of California, Berkeley",
                location: "Berkeley, CA",
                ranking: 22,
                tuition: 14312,
                acceptance_rate: 0.11,
                avg_sat: 1405,
                avg_gpa: 3.89,
                majors: ["Computer Science", "Engineering", "Business", "Liberal Arts"],
                description: "Top public research university"
            },
            {
                name: "Carnegie Mellon University",
                location: "Pittsburgh, PA",
                ranking: 25,
                tuition: 59864,
                acceptance_rate: 0.11,
                avg_sat: 1480,
                avg_gpa: 3.87,
                majors: ["Computer Science", "Engineering", "Robotics", "Drama"],
                description: "Premier computer science and engineering programs"
            },
            {
                name: "University of Michigan",
                location: "Ann Arbor, MI",
                ranking: 21,
                tuition: 15948,
                acceptance_rate: 0.18,
                avg_sat: 1435,
                avg_gpa: 3.88,
                majors: ["Engineering", "Business", "Medicine", "Liberal Arts"],
                description: "Large public research university with strong programs"
            },
            {
                name: "Georgia Institute of Technology",
                location: "Atlanta, GA",
                ranking: 38,
                tuition: 12852,
                acceptance_rate: 0.16,
                avg_sat: 1465,
                avg_gpa: 4.04,
                majors: ["Engineering", "Computer Science", "Business"],
                description: "Top engineering and technology programs"
            },
            {
                name: "University of Texas at Austin",
                location: "Austin, TX",
                ranking: 38,
                tuition: 11448,
                acceptance_rate: 0.29,
                avg_sat: 1355,
                avg_gpa: 3.71,
                majors: ["Business", "Engineering", "Liberal Arts", "Computer Science"],
                description: "Large public university with diverse strong programs"
            },
            {
                name: "University of Washington",
                location: "Seattle, WA",
                ranking: 59,
                tuition: 12076,
                acceptance_rate: 0.48,
                avg_sat: 1315,
                avg_gpa: 3.75,
                majors: ["Computer Science", "Medicine", "Engineering", "Business"],
                description: "Strong research programs especially in tech and medicine"
            }
        ];
    }

    async searchUniversities(query) {
        console.log(`🔍 Searching universities with query: "${query}"`);
        
        // Check cache first
        const cacheKey = query.toLowerCase();
        const cached = this.getFromCache(cacheKey);
        if (cached) {
            console.log('📋 Using cached search results');
            return cached;
        }

        const searchStrategies = [
            () => this.braveSearch(query),
            () => this.duckDuckGoSearch(query),
            () => this.staticDataSearch(query)
        ];

        for (let i = 0; i < searchStrategies.length; i++) {
            try {
                console.log(`📡 Trying search strategy ${i + 1}...`);
                const results = await searchStrategies[i]();
                
                if (results && results.length > 0) {
                    console.log(`✅ Search strategy ${i + 1} successful - found ${results.length} results`);
                    this.setCache(cacheKey, results);
                    return results;
                }
            } catch (error) {
                console.warn(`⚠️ Search strategy ${i + 1} failed:`, error.message);
                continue;
            }
        }

        console.error('❌ All search strategies failed');
        return [];
    }

    async braveSearch(query) {
        if (!process.env.BRAVE_API_KEY) {
            throw new Error('Brave API key not available');
        }

        const params = new URLSearchParams({
            q: `${query} university college admission requirements tuition`,
            count: '10',
            search_lang: 'en',
            country: 'US'
        });

        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), 30000);

        try {
            const response = await fetch(`https://api.search.brave.com/res/v1/web/search?${params}`, {
                headers: {
                    'Accept': 'application/json',
                    'X-Subscription-Token': process.env.BRAVE_API_KEY
                },
                agent: process.env.HTTPS_PROXY ? 
                    new (require('https-proxy-agent'))(process.env.HTTPS_PROXY) : 
                    undefined,
                signal: controller.signal,
                timeout: 30000
            });

            clearTimeout(timeoutId);

            if (!response.ok) {
                throw new Error(`Brave search failed: ${response.status}`);
            }

            const data = await response.json();
            return (data.web?.results || []).map(result => ({
                title: result.title || '',
                snippet: result.description || '',
                url: result.url || ''
            }));
        } catch (error) {
            clearTimeout(timeoutId);
            if (error.name === 'AbortError') {
                throw new Error('Brave search timed out');
            }
            throw error;
        }
    }

    async duckDuckGoSearch(query) {
        const encodedQuery = encodeURIComponent(`${query} university college`);
        const searchUrl = `https://api.duckduckgo.com/?q=${encodedQuery}&format=json&no_html=1&skip_disambig=1`;
        
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), 30000);

        try {
            const response = await fetch(searchUrl, {
                agent: process.env.HTTPS_PROXY ? 
                    new (require('https-proxy-agent'))(process.env.HTTPS_PROXY) : 
                    undefined,
                signal: controller.signal,
                timeout: 30000
            });
            
            clearTimeout(timeoutId);

            if (!response.ok) {
                throw new Error(`DuckDuckGo search failed: ${response.status}`);
            }

            const data = await response.json();
            const results = [];

            if (data.Abstract) {
                results.push({
                    title: data.Heading || query,
                    snippet: data.Abstract,
                    url: data.AbstractURL || ''
                });
            }

            (data.RelatedTopics || []).slice(0, 8).forEach(topic => {
                if (topic.Text) {
                    results.push({
                        title: topic.FirstURL ? topic.FirstURL.split('/').pop().replace(/_/g, ' ') : '',
                        snippet: topic.Text,
                        url: topic.FirstURL || ''
                    });
                }
            });

            return results;
        } catch (error) {
            clearTimeout(timeoutId);
            if (error.name === 'AbortError') {
                throw new Error('DuckDuckGo search timed out');
            }
            throw error;
        }
    }

    staticDataSearch(query) {
        console.log('📚 Using static college database as fallback');
        const keywords = query.toLowerCase().split(' ');
        
        const matchedColleges = this.staticCollegeData.filter(college => {
            const searchText = `${college.name} ${college.location} ${college.majors.join(' ')} ${college.description}`.toLowerCase();
            return keywords.some(keyword => searchText.includes(keyword));
        });

        return matchedColleges.map(college => ({
            title: college.name,
            snippet: `${college.description} Located in ${college.location}. Tuition: $${college.tuition}`,
            url: `https://www.${college.name.toLowerCase().replace(/\s+/g, '')}.edu`,
            college_data: college // Include structured data directly
        }));
    }

    getFromCache(key) {
        const cached = this.cache.get(key);
        if (cached && Date.now() - cached.timestamp < this.cacheTimeout) {
            return cached.data;
        }
        this.cache.delete(key);
        return null;
    }

    setCache(key, data) {
        this.cache.set(key, {
            data,
            timestamp: Date.now()
        });
    }
}

// ============ IMPROVED AI PROCESSING ============
class SafeJSONParser {
    static parseCollegeData(aiResponse) {
        console.log('🧠 Parsing AI response for college data...');
        
        try {
            // Multiple parsing strategies
            const strategies = [
                () => this.parseJSONMarkdown(aiResponse),
                () => this.parseDirectJSON(aiResponse),
                () => this.parseJSONArray(aiResponse)
            ];

            for (const strategy of strategies) {
                try {
                    const result = strategy();
                    if (result && Array.isArray(result) && result.length > 0) {
                        console.log(`✅ Successfully parsed ${result.length} colleges`);
                        return result.filter(college => this.validateCollegeData(college));
                    }
                } catch (e) {
                    continue;
                }
            }

            throw new Error('All parsing strategies failed');

        } catch (error) {
            console.error('❌ Failed to parse college data:', error.message);
            return [];
        }
    }

    static parseJSONMarkdown(text) {
        const jsonMatch = text.match(/```json\s*([\s\S]*?)\s*```/);
        if (jsonMatch) {
            return JSON.parse(jsonMatch[1]);
        }
        throw new Error('No JSON markdown found');
    }

    static parseDirectJSON(text) {
        return JSON.parse(text);
    }

    static parseJSONArray(text) {
        const arrayMatch = text.match(/\[[\s\S]*\]/);
        if (arrayMatch) {
            return JSON.parse(arrayMatch[0]);
        }
        throw new Error('No JSON array found');
    }

    static validateCollegeData(college) {
        if (!college || typeof college !== 'object') return false;
        if (!college.name || !college.location) return false;
        
        // Clean and validate data
        if (college.tuition && typeof college.tuition === 'string') {
            college.tuition = parseInt(college.tuition.replace(/[^\d]/g, '')) || null;
        }
        if (college.ranking && typeof college.ranking === 'string') {
            college.ranking = parseInt(college.ranking) || null;
        }
        if (college.acceptance_rate && typeof college.acceptance_rate === 'string') {
            college.acceptance_rate = parseFloat(college.acceptance_rate) || null;
        }

        return true;
    }
}

// ============ COLLEGE SEARCH AND PROCESSING ============
class CollegeSearchTool {
    constructor(openaiClient, agent) {
        this.universitySearch = new RobustUniversitySearch();
        this.openai = openaiClient;
        this.agent = agent;
    }

    buildSearchQuery(queryParams) {
        const parts = ['universities colleges'];
        
        if (queryParams.majors?.length > 0) {
            parts.push(`${queryParams.majors.join(' ')} programs`);
        }
        if (queryParams.location?.length > 0) {
            parts.push(`in ${queryParams.location.join(' ')}`);
        }
        if (queryParams.max_tuition) {
            parts.push(`tuition under $${queryParams.max_tuition}`);
        }
        
        parts.push('admission requirements ranking');
        return parts.join(' ');
    }

    async search(queryParams) {
        try {
            await this.agent.sendStatusUpdate('🔍 构建搜索查询...');
            const searchQuery = this.buildSearchQuery(queryParams);

            await this.agent.sendStatusUpdate('📡 搜索大学数据库和网站...');
            const searchResults = await this.universitySearch.searchUniversities(searchQuery);

            if (!searchResults?.length) {
                await this.agent.sendStatusUpdate('⚠️ 未找到搜索结果，使用静态数据库...');
                return this.fallbackToStaticData(queryParams);
            }

            // Check if we got direct college data from static search
            const directData = searchResults.filter(r => r.college_data);
            if (directData.length > 0) {
                console.log('📋 Using direct college data from static database');
                return directData.map(r => new College(r.college_data)).filter(c => c.isValid());
            }

            await this.agent.sendStatusUpdate('🧠 使用AI解析搜索结果...');
            return await this.processSearchResultsWithAI(searchResults, queryParams);

        } catch (error) {
            console.error('❌ College search error:', error);
            await this.agent.sendStatusUpdate(`❌ 搜索出错: ${error.message}`);
            return this.fallbackToStaticData(queryParams);
        }
    }

    async processSearchResultsWithAI(searchResults, queryParams) {
        if (!searchResults?.length) return [];

        const searchText = searchResults.slice(0, 8)
            .map(result => `Title: ${result.title}\nDescription: ${result.snippet}\nURL: ${result.url}`)
            .join('\n\n');

        const systemPrompt = `You are a university data extraction expert. Extract structured information about universities/colleges from search results.

For each university mentioned, extract:
- name: Full university name
- location: City, State/Country  
- ranking: National ranking (number only)
- tuition: Annual tuition in USD (number only, no symbols)
- acceptance_rate: As decimal (0.15 for 15%)
- avg_sat: Average SAT score (number)
- avg_gpa: Average GPA (decimal)
- majors: Array of strong programs
- description: Brief description of strengths

Return ONLY a JSON array. Example:
[
  {
    "name": "Stanford University",
    "location": "Stanford, CA", 
    "ranking": 6,
    "tuition": 56169,
    "acceptance_rate": 0.04,
    "avg_sat": 1505,
    "avg_gpa": 4.18,
    "majors": ["Computer Science", "Engineering", "Business"],
    "description": "Premier research university in Silicon Valley"
  }
]`;

        try {
            const response = await this.openai.chat.completions.create({
                model: "gpt-3.5-turbo",
                messages: [
                    { role: "system", content: systemPrompt },
                    { role: "user", content: `Extract university data from:\n\n${searchText}` }
                ],
                temperature: 0.1,
                max_tokens: 2000
            });

            const collegesData = SafeJSONParser.parseCollegeData(response.choices[0].message.content);
            return collegesData.map(data => new College(data)).filter(college => college.isValid());

        } catch (error) {
            console.error('❌ AI processing failed:', error);
            await this.agent.sendStatusUpdate('⚠️ AI解析失败，使用备用数据...');
            return this.fallbackToStaticData(queryParams);
        }
    }

    fallbackToStaticData(queryParams) {
        console.log('📚 Using fallback static college data');
        
        let colleges = this.universitySearch.staticCollegeData.map(data => new College(data));
        
        // Apply filters
        if (queryParams.majors?.length > 0) {
            colleges = colleges.filter(college => 
                queryParams.majors.some(major => 
                    college.majors.some(collegeMajor => 
                        collegeMajor.toLowerCase().includes(major.toLowerCase())
                    )
                )
            );
        }
        
        if (queryParams.location?.length > 0) {
            colleges = colleges.filter(college =>
                queryParams.location.some(loc =>
                    college.location.toLowerCase().includes(loc.toLowerCase())
                )
            );
        }
        
        if (queryParams.max_tuition) {
            colleges = colleges.filter(college => 
                !college.tuition || college.tuition <= queryParams.max_tuition
            );
        }
        
        return colleges.slice(0, 10);
    }
}

// ============ FIT SCORING SYSTEM ============
class FitScoringTool {
    scoreCollege(college, userProfile) {
        let totalScore = 0;
        let weightSum = 0;

        // Academic fit (40% weight)
        const academicScore = this.calculateAcademicFit(college, userProfile);
        if (academicScore !== null) {
            totalScore += academicScore * 0.4;
            weightSum += 0.4;
        }

        // Major fit (30% weight)
        const majorScore = this.calculateMajorFit(college, userProfile);
        if (majorScore !== null) {
            totalScore += majorScore * 0.3;
            weightSum += 0.3;
        }

        // Location fit (20% weight)
        const locationScore = this.calculateLocationFit(college, userProfile);
        if (locationScore !== null) {
            totalScore += locationScore * 0.2;
            weightSum += 0.2;
        }

        // Affordability bonus (10% weight)
        const affordabilityScore = this.calculateAffordabilityScore(college, userProfile);
        if (affordabilityScore !== null) {
            totalScore += affordabilityScore * 0.1;
            weightSum += 0.1;
        }

        return weightSum > 0 ? totalScore / weightSum : 0.5;
    }

    calculateAcademicFit(college, userProfile) {
        if (!userProfile.gpa && !userProfile.sat_score) return null;
        
        let academicScore = 0;
        let factors = 0;

        if (userProfile.gpa && college.avg_gpa) {
            const gpaDiff = Math.abs(college.avg_gpa - userProfile.gpa);
            const gpaScore = Math.max(0, 1 - (gpaDiff / 2)); // Scale 0-4 GPA range
            academicScore += gpaScore;
            factors++;
        }

        if (userProfile.sat_score && college.avg_sat) {
            const satDiff = Math.abs(college.avg_sat - userProfile.sat_score);
            const satScore = Math.max(0, 1 - (satDiff / 400)); // Scale SAT differences
            academicScore += satScore;
            factors++;
        }

        return factors > 0 ? academicScore / factors : null;
    }

    calculateMajorFit(college, userProfile) {
        if (!userProfile.major_preference?.length || !college.majors?.length) return null;

        let matches = 0;
        for (const userMajor of userProfile.major_preference) {
            const hasMatch = college.majors.some(collegeMajor =>
                collegeMajor.toLowerCase().includes(userMajor.toLowerCase()) ||
                userMajor.toLowerCase().includes(collegeMajor.toLowerCase())
            );
            if (hasMatch) matches++;
        }

        return matches / userProfile.major_preference.length;
    }

    calculateLocationFit(college, userProfile) {
        if (!userProfile.location_preference?.length) return null;

        const hasLocationMatch = userProfile.location_preference.some(location =>
            college.location.toLowerCase().includes(location.toLowerCase())
        );

        return hasLocationMatch ? 1.0 : 0.0;
    }

    calculateAffordabilityScore(college, userProfile) {
        if (!userProfile.budget?.max_annual_tuition || !college.tuition) return null;

        const budget = userProfile.budget.max_annual_tuition;
        if (college.tuition <= budget) {
            // Better score for colleges well within budget
            return Math.min(1.0, budget / college.tuition - 0.5);
        } else {
            // Penalty for over-budget colleges
            return Math.max(0, 1 - (college.tuition - budget) / budget);
        }
    }
}

// ============ IMPROVED CONNECTION MANAGEMENT ============
class ConnectionManager {
    constructor() {
        this.activeConnections = new Map();
        this.connectionTimeouts = new Map();
        this.maxConnections = 50;
        this.connectionTimeout = 30 * 60 * 1000; // 30 minutes
    }

    addConnection(ws, agent) {
        if (this.activeConnections.size >= this.maxConnections) {
            ws.close(1008, 'Server at capacity');
            return false;
        }

        this.activeConnections.set(ws, agent);
        
        // Set connection timeout
        const timeout = setTimeout(() => {
            console.log('⏰ Connection timeout - closing connection');
            ws.close(1000, 'Connection timeout');
        }, this.connectionTimeout);
        
        this.connectionTimeouts.set(ws, timeout);
        console.log(`➕ Connection added. Total: ${this.activeConnections.size}`);
        return true;
    }

    removeConnection(ws) {
        const agent = this.activeConnections.get(ws);
        if (agent) {
            agent.cleanup();
        }

        this.activeConnections.delete(ws);

        const timeout = this.connectionTimeouts.get(ws);
        if (timeout) {
            clearTimeout(timeout);
            this.connectionTimeouts.delete(ws);
        }

        console.log(`➖ Connection removed. Total: ${this.activeConnections.size}`);
    }

    getAgent(ws) {
        return this.activeConnections.get(ws);
    }
}

// ============ MAIN AGENT CLASS ============
class UniGuideAgent {
    constructor(ws, openaiClient) {
        this.ws = ws;
        this.openai = openaiClient;
        this.userProfile = new UserProfile();
        this.conversationHistory = [];
        this.collegeSearchTool = new CollegeSearchTool(openaiClient, this);
        this.fitScoringTool = new FitScoringTool();
        
        console.log('🤖 UniGuideAgent initialized');
    }

    async sendStatusUpdate(message) {
        if (this.ws.readyState === WebSocket.OPEN) {
            this.ws.send(JSON.stringify({
                type: 'status_update',
                message,
                timestamp: new Date().toISOString()
            }));
        }
    }

    async sendResponse(message) {
        if (this.ws.readyState === WebSocket.OPEN) {
            this.ws.send(JSON.stringify({
                type: 'ai_response',
                message,
                timestamp: new Date().toISOString()
            }));
        }
    }

    async extractUserInfo(message) {
        const systemPrompt = `Extract college admissions information from user messages. Return JSON only.

Extract these fields when mentioned:
- gpa: Float (0.0-4.0)
- sat_score: Integer (400-1600)  
- act_score: Integer (1-36)
- interests: Array of majors/subjects
- location_preference: Array of states/cities
- budget: Object with max_annual_tuition (number)

Example output:
{
  "gpa": 3.8,
  "sat_score": 1450,
  "interests": ["computer science", "engineering"],
  "location_preference": ["California", "New York"],
  "budget": {"max_annual_tuition": 50000}
}

Return {} if no relevant information found.`;

        try {
            const response = await this.openai.chat.completions.create({
                model: "gpt-3.5-turbo",
                messages: [
                    { role: "system", content: systemPrompt },
                    { role: "user", content: message }
                ],
                temperature: 0.1,
                max_tokens: 500
            });

            return JSON.parse(response.choices[0].message.content);
        } catch (error) {
            console.error('❌ Failed to extract user info:', error);
            return {};
        }
    }

    updateProfile(extractedInfo) {
        let updated = false;

        if (extractedInfo.gpa && extractedInfo.gpa >= 0 && extractedInfo.gpa <= 4) {
            this.userProfile.gpa = parseFloat(extractedInfo.gpa);
            updated = true;
        }
        if (extractedInfo.sat_score && extractedInfo.sat_score >= 400 && extractedInfo.sat_score <= 1600) {
            this.userProfile.sat_score = parseInt(extractedInfo.sat_score);
            updated = true;
        }
        if (extractedInfo.act_score && extractedInfo.act_score >= 1 && extractedInfo.act_score <= 36) {
            this.userProfile.act_score = parseInt(extractedInfo.act_score);
            updated = true;
        }
        if (extractedInfo.interests && Array.isArray(extractedInfo.interests)) {
            // Merge new interests with existing ones
            const newInterests = extractedInfo.interests.filter(interest => 
                !this.userProfile.major_preference.some(existing => 
                    existing.toLowerCase() === interest.toLowerCase()
                )
            );
            this.userProfile.major_preference.push(...newInterests);
            updated = true;
        }
        if (extractedInfo.location_preference && Array.isArray(extractedInfo.location_preference)) {
            const newLocations = extractedInfo.location_preference.filter(location => 
                !this.userProfile.location_preference.some(existing => 
                    existing.toLowerCase() === location.toLowerCase()
                )
            );
            this.userProfile.location_preference.push(...newLocations);
            updated = true;
        }
        if (extractedInfo.budget && extractedInfo.budget.max_annual_tuition) {
            this.userProfile.budget = {
                max_annual_tuition: parseInt(extractedInfo.budget.max_annual_tuition),
                type: "under"
            };
            updated = true;
        }

        if (updated) {
            console.log('📝 Updated user profile:', this.userProfile.toString());
        }
        return updated;
    }

    async detectIntent(message) {
        const keywords = {
            [IntentType.COLLEGE_MATCH]: ['recommend', 'university', 'college', 'match', 'find', 'search', 'suggest', '推荐', '大学', '学校', '匹配', '寻找'],
            [IntentType.ESSAY_REVISE]: ['essay', 'personal statement', 'application', 'writing', '论文', '文书', '申请'],
            [IntentType.SCHEDULE_PLAN]: ['deadline', 'timeline', 'schedule', 'plan', '截止', '时间', '计划', '规划'],
            [IntentType.GENERAL_QA]: ['help', 'advice', 'question', '帮助', '建议', '问题']
        };

        const messageLower = message.toLowerCase();
        
        // Score each intent based on keyword matches
        const scores = {};
        for (const [intent, intentKeywords] of Object.entries(keywords)) {
            scores[intent] = intentKeywords.filter(keyword => 
                messageLower.includes(keyword.toLowerCase())
            ).length;
        }

        // Find intent with highest score
        const topIntent = Object.entries(scores).reduce((a, b) => 
            scores[a[0]] > scores[b[0]] ? a : b
        )[0];

        // If college match has any matches or no clear intent, default to college match
        if (scores[IntentType.COLLEGE_MATCH] > 0 || Math.max(...Object.values(scores)) === 0) {
            return IntentType.COLLEGE_MATCH;
        }

        return topIntent;
    }

    async processCollegeMatch(message) {
        try {
            await this.sendStatusUpdate('🔍 分析您的学术背景和偏好...');
            
            // Extract and update user information
            const extractedInfo = await this.extractUserInfo(message);
            const profileUpdated = this.updateProfile(extractedInfo);
            
            if (profileUpdated) {
                await this.sendStatusUpdate('📝 已更新您的档案信息');
            }

            // Build search parameters
            const searchParams = {};
            if (this.userProfile.major_preference.length > 0) {
                searchParams.majors = this.userProfile.major_preference;
            }
            if (this.userProfile.location_preference.length > 0) {
                searchParams.location = this.userProfile.location_preference;
            }
            if (this.userProfile.budget?.max_annual_tuition) {
                searchParams.max_tuition = this.userProfile.budget.max_annual_tuition;
            }

            // Search for colleges
            await this.sendStatusUpdate('🎯 搜索匹配的大学...');
            const colleges = await this.collegeSearchTool.search(searchParams);

            if (!colleges || colleges.length === 0) {
                await this.sendStatusUpdate('⚠️ 未找到匹配结果，提供通用建议...');
                return await this.generateFallbackAdvice();
            }

            // Calculate fit scores
            await this.sendStatusUpdate(`📊 为 ${colleges.length} 所大学计算匹配度...`);
            colleges.forEach(college => {
                college.fit_score = this.fitScoringTool.scoreCollege(college, this.userProfile);
            });

            // Apply budget filter if specified
            let filteredColleges = colleges;
            if (this.userProfile.budget?.max_annual_tuition) {
                const budgetFiltered = colleges.filter(college => 
                    !college.tuition || college.tuition <= this.userProfile.budget.max_annual_tuition
                );
                if (budgetFiltered.length > 0) {
                    filteredColleges = budgetFiltered;
                } else {
                    await this.sendStatusUpdate('💰 预算范围内无匹配大学，显示所有结果');
                }
            }

            // Sort by fit score
            filteredColleges.sort((a, b) => b.fit_score - a.fit_score);

            await this.sendStatusUpdate('✅ 生成个性化推荐报告...');
            return this.generateCollegeReport(filteredColleges.slice(0, 8));

        } catch (error) {
            console.error('❌ Error in college matching:', error);
            await this.sendStatusUpdate('❌ 处理过程中出现错误');
            return await this.generateFallbackAdvice();
        }
    }

    generateCollegeReport(colleges) {
        let report = "## 🎓 **个性化大学推荐报告**\n\n";
        
        // User profile summary
        report += `**您的档案:** ${this.userProfile.toString()}\n\n`;
        
        if (colleges.length === 0) {
            report += "😔 **抱歉，未找到完全匹配的大学。**\n\n";
            report += "💡 **建议:**\n";
            report += "- 尝试扩大地理位置范围\n";
            report += "- 考虑调整预算限制\n";
            report += "- 提供更多专业偏好信息\n\n";
            return report;
        }

        // Categorize colleges by fit score
        const excellent = colleges.filter(c => c.fit_score >= 0.8);
        const good = colleges.filter(c => c.fit_score >= 0.6 && c.fit_score < 0.8);
        const decent = colleges.filter(c => c.fit_score < 0.6);

        if (excellent.length > 0) {
            report += "## 🌟 **高度匹配 (80%+)**\n\n";
            excellent.forEach((college, i) => {
                report += this.formatCollegeEntry(college, i + 1);
            });
        }

        if (good.length > 0) {
            report += "## ✅ **良好匹配 (60-79%)**\n\n";
            good.forEach((college, i) => {
                report += this.formatCollegeEntry(college, excellent.length + i + 1);
            });
        }

        if (decent.length > 0) {
            report += "## 📋 **其他选项 (60%以下)**\n\n";
            decent.forEach((college, i) => {
                report += this.formatCollegeEntry(college, excellent.length + good.length + i + 1);
            });
        }

        // Add recommendations
        report += "\n## 💡 **申请建议**\n\n";
        report += this.generateApplicationAdvice(colleges);
        
        report += "\n---\n";
        report += "*💻 此报告基于实时搜索数据和AI分析生成。请访问大学官网获取最新信息。*";
        
        return report;
    }

    formatCollegeEntry(college, index) {
        const fitPercentage = Math.round(college.fit_score * 100);
        let entry = `**${index}. ${college.name}** (${fitPercentage}% 匹配)\n`;
        entry += college.toDisplayString();
        entry += "\n";
        return entry;
    }

    generateApplicationAdvice(colleges) {
        let advice = "";
        
        const avgFitScore = colleges.reduce((sum, c) => sum + c.fit_score, 0) / colleges.length;
        
        if (avgFitScore >= 0.75) {
            advice += "🎯 **您的档案与这些大学高度匹配！**\n";
            advice += "- 重点准备高匹配度大学的申请\n";
            advice += "- 确保申请材料突出您的优势\n";
        } else if (avgFitScore >= 0.5) {
            advice += "📈 **您有很好的申请机会**\n";
            advice += "- 考虑提升标准化考试成绩\n";
            advice += "- 加强相关专业的课外活动\n";
        } else {
            advice += "🚀 **需要更多努力来提升竞争力**\n";
            advice += "- 重点提升GPA和标准化考试成绩\n";
            advice += "- 寻找安全学校作为保底选择\n";
        }

        // Budget advice
        const expensiveColleges = colleges.filter(c => c.tuition && c.tuition > 50000);
        if (expensiveColleges.length > 0) {
            advice += "- 💰 考虑申请奖学金和助学金\n";
        }

        return advice;
    }

    async generateFallbackAdvice() {
        const advice = `## 🎓 **大学申请指导**

根据您提供的信息，我来为您提供一些通用的大学申请建议：

### 📊 **学术准备**
- **GPA:** 保持3.5+的成绩，顶尖大学通常要求3.8+
- **标准化考试:** SAT 1400+ 或 ACT 32+ 对大多数好大学很重要
- **课程选择:** 选择具有挑战性的AP或IB课程

### 🎯 **专业选择建议**
- **STEM领域:** 计算机科学、工程、数学需求量大
- **商科:** 金融、市场营销、管理等就业前景好
- **医学预科:** 竞争激烈但回报丰厚

### 🌍 **地理位置考虑**
- **加州:** 斯坦福、UC系统、加州理工
- **东海岸:** 常春藤联盟、MIT、约翰霍普金斯
- **德州:** UT奥斯汀、Rice大学
- **其他:** 密歇根大学、威斯康辛大学等公立强校

### 💰 **费用规划**
- **公立大学:** 州内学费通常$10,000-15,000
- **私立大学:** 学费通常$50,000-70,000
- **奖学金:** 积极申请merit-based和need-based奖学金

请提供更多具体信息，我可以给出更精准的推荐！`;

        return advice;
    }

    async processGeneralQA(message) {
        const systemPrompt = `You are UniGuide AI, a helpful college admissions consultant. 
        
        Provide practical, accurate advice about:
        - College applications and admissions
        - Standardized tests (SAT/ACT) 
        - Essays and personal statements
        - Financial aid and scholarships
        - Academic planning and course selection
        
        Be encouraging but realistic. Respond in Chinese when the user writes in Chinese, English otherwise.
        Keep responses informative but concise (under 300 words).`;

        try {
            const messages = [{ role: "system", content: systemPrompt }];
            
            // Add recent conversation context
            this.conversationHistory.slice(-4).forEach(msg => messages.push(msg));
            messages.push({ role: "user", content: message });

            const response = await this.openai.chat.completions.create({
                model: "gpt-3.5-turbo",
                messages: messages,
                temperature: 0.7,
                max_tokens: 400
            });

            return response.choices[0].message.content;

        } catch (error) {
            console.error('❌ General QA error:', error);
            return "抱歉，我现在无法回答您的问题。请稍后再试，或者询问具体的大学推荐需求。";
        }
    }

    async processMessage(userMessage) {
        console.log(`🗣️ Processing message: ${userMessage.substring(0, 100)}...`);
        
        // Add to conversation history
        this.conversationHistory.push({ role: "user", content: userMessage });

        try {
            // Detect intent
            const intent = await this.detectIntent(userMessage);
            console.log(`🎯 Detected intent: ${intent}`);
            
            let response;

            switch (intent) {
                case IntentType.COLLEGE_MATCH:
                    response = await this.processCollegeMatch(userMessage);
                    break;
                    
                case IntentType.ESSAY_REVISE:
                    response = "📝 **Essay Review Service Coming Soon!**\n\n" +
                              "目前essay修改功能正在开发中。建议您：\n" +
                              "- 突出个人独特经历和成长\n" +
                              "- 展现您的激情和目标\n" +
                              "- 保持真实性和原创性\n\n" +
                              "需要大学推荐服务吗？";
                    break;
                    
                case IntentType.SCHEDULE_PLAN:
                    response = "📅 **Application Timeline Planner**\n\n" +
                              "重要申请截止日期：\n" +
                              "- **Early Decision/Action:** 11月1日\n" +
                              "- **Regular Decision:** 1月1日-1月15日\n" +
                              "- **奖学金申请:** 通常与入学申请同时\n" +
                              "- **FAFSA:** 10月1日开始\n\n" +
                              "需要帮您制定详细的申请时间表吗？";
                    break;
                    
                default: // GENERAL_QA
                    response = await this.processGeneralQA(userMessage);
                    break;
            }

            // Add response to conversation history
            this.conversationHistory.push({ role: "assistant", content: response });
            
            // Keep conversation history manageable
            if (this.conversationHistory.length > 12) {
                this.conversationHistory = this.conversationHistory.slice(-12);
            }

            await this.sendResponse(response);
            return response;

        } catch (error) {
            console.error('❌ Error processing message:', error);
            const errorResponse = "抱歉，处理您的请求时出现了问题。请稍后再试或重新描述您的需求。";
            await this.sendResponse(errorResponse);
            return errorResponse;
        }
    }

    cleanup() {
        console.log('🧹 Cleaning up agent resources');
        this.conversationHistory = [];
        // Clear any other resources if needed
    }
}

// ============ SERVER SETUP ============
const connectionManager = new ConnectionManager();

// Serve static files
app.use(express.static('public'));

// Health check endpoint
app.get('/health', (req, res) => {
    res.json({ 
        status: 'healthy', 
        timestamp: new Date().toISOString(),
        connections: connectionManager.activeConnections.size
    });
});

// Main page route
app.get('/', (req, res) => {
    res.send(`<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>UniGuide AI - 智能大学推荐助手</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
    <style>
        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'PingFang SC', 'Hiragino Sans GB', 'Microsoft YaHei', sans-serif; }
        .chat-container { height: calc(100vh - 180px); }
        .message { animation: slideIn 0.3s ease-out; }
        .user-message { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); }
        .ai-message { background: #f8fafc; border: 1px solid #e2e8f0; }
        .typing-indicator span { animation: bounce 1.4s infinite; }
        .typing-indicator span:nth-child(2) { animation-delay: 0.2s; }
        .typing-indicator span:nth-child(3) { animation-delay: 0.4s; }
        @keyframes slideIn { from { opacity: 0; transform: translateY(20px); } to { opacity: 1; transform: translateY(0); } }
        @keyframes bounce { 0%, 80%, 100% { transform: scale(0); } 40% { transform: scale(1); } }
        .status-message { background: linear-gradient(90deg, #4f46e5, #7c3aed); }
    </style>
</head>
<body class="bg-gray-50">
    <div class="min-h-screen flex flex-col">
        <!-- Header -->
        <header class="bg-white shadow-sm border-b">
            <div class="max-w-4xl mx-auto px-4 py-4 flex items-center justify-between">
                <div class="flex items-center space-x-3">
                    <div class="w-10 h-10 bg-gradient-to-r from-blue-500 to-purple-600 rounded-lg flex items-center justify-center">
                        <span class="text-white font-bold text-lg">🎓</span>
                    </div>
                    <div>
                        <h1 class="text-xl font-bold text-gray-900">UniGuide AI</h1>
                        <p class="text-sm text-gray-500">智能大学推荐助手</p>
                    </div>
                </div>
                <div class="flex items-center space-x-4">
                    <div id="connection-status" class="px-3 py-1 rounded-full text-sm font-medium bg-yellow-100 text-yellow-800">
                        🔄 连接中...
                    </div>
                    <button id="clear-chat" class="p-2 text-gray-400 hover:text-gray-600 transition-colors">
                        <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16"></path>
                        </svg>
                    </button>
                </div>
            </div>
        </header>

        <!-- Chat Container -->
        <main class="flex-1 overflow-hidden">
            <div class="max-w-4xl mx-auto h-full flex flex-col">
                <div id="chat-container" class="chat-container flex-1 overflow-y-auto p-4">
                    <div id="messages" class="space-y-4">
                        <!-- Welcome Message -->
                        <div class="bg-gradient-to-r from-blue-500 to-purple-600 text-white p-6 rounded-xl">
                            <h2 class="text-2xl font-bold mb-3">🌟 欢迎使用 UniGuide AI</h2>
                            <p class="mb-4">我是您的AI大学申请顾问，使用实时搜索为您提供个性化的大学推荐。我会分析您的学术背景，搜索最新的大学信息，并计算最佳匹配度。</p>
                            <div class="grid grid-cols-2 gap-4 mt-4">
                                <div class="bg-white bg-opacity-20 p-3 rounded-lg">
                                    <div class="font-semibold">🔍 实时搜索</div>
                                    <div class="text-sm opacity-90">最新大学数据</div>
                                </div>
                                <div class="bg-white bg-opacity-20 p-3 rounded-lg">
                                    <div class="font-semibold">🤖 AI分析</div>
                                    <div class="text-sm opacity-90">智能匹配算法</div>
                                </div>
                            </div>
                        </div>

                        <!-- Quick Start Options -->
                        <div class="bg-white p-6 rounded-xl border border-gray-200">
                            <h3 class="font-semibold text-gray-900 mb-4">🚀 快速开始</h3>
                            <div class="grid gap-3">
                                <button class="suggestion-btn text-left p-4 bg-gray-50 hover:bg-blue-50 rounded-lg border border-gray-200 hover:border-blue-300 transition-all">
                                    <div class="font-medium text-gray-900">我想学计算机科学</div>
                                    <div class="text-sm text-gray-600 mt-1">GPA 3.8, SAT 1450, 预算5万美元</div>
                                </button>
                                <button class="suggestion-btn text-left p-4 bg-gray-50 hover:bg-blue-50 rounded-lg border border-gray-200 hover:border-blue-300 transition-all">
                                    <div class="font-medium text-gray-900">推荐加州的商科大学</div>
                                    <div class="text-sm text-gray-600 mt-1">找性价比高的商学院</div>
                                </button>
                                <button class="suggestion-btn text-left p-4 bg-gray-50 hover:bg-blue-50 rounded-lg border border-gray-200 hover:border-blue-300 transition-all">
                                    <div class="font-medium text-gray-900">工程专业的公立大学</div>
                                    <div class="text-sm text-gray-600 mt-1">SAT 1350, 寻找性价比高的选择</div>
                                </button>
                            </div>
                        </div>

                        <!-- Typing Indicator -->
                        <div id="typing-indicator" class="hidden">
                            <div class="flex justify-start">
                                <div class="ai-message p-4 rounded-xl max-w-md">
                                    <div class="flex items-center space-x-2">
                                        <div class="typing-indicator flex space-x-1">
                                            <span class="w-2 h-2 bg-blue-500 rounded-full"></span>
                                            <span class="w-2 h-2 bg-blue-500 rounded-full"></span>
                                            <span class="w-2 h-2 bg-blue-500 rounded-full"></span>
                                        </div>
                                        <span class="text-sm text-gray-600">正在思考...</span>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>

                <!-- Input Area -->
                <div class="p-4 bg-white border-t">
                    <div class="flex space-x-4">
                        <div class="flex-1">
                            <textarea
                                id="message-input"
                                placeholder="描述您的情况，比如：我的GPA是3.7，SAT 1400，想学计算机科学，预算4万美元，推荐一些大学..."
                                class="w-full p-3 border border-gray-300 rounded-lg resize-none focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                                rows="2"
                            ></textarea>
                        </div>
                        <button
                            id="send-button"
                            class="px-6 py-3 bg-gradient-to-r from-blue-500 to-purple-600 text-white rounded-lg hover:shadow-lg transform hover:scale-105 transition-all duration-200 disabled:opacity-50 disabled:transform-none"
                        >
                            <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 19l9 2-9-18-9 18 9-2zm0 0v-8"></path>
                            </svg>
                        </button>
                    </div>
                    <div class="flex justify-between items-center mt-2 text-sm text-gray-500">
                        <span>按 Enter 发送，Shift + Enter 换行</span>
                        <span id="message-count">0 条对话</span>
                    </div>
                </div>
            </div>
        </main>
    </div>

    <script>
        let ws = null;
        let messageCount = 0;
        
        const elements = {
            messageInput: document.getElementById('message-input'),
            sendButton: document.getElementById('send-button'),
            messagesContainer: document.getElementById('messages'),
            typingIndicator: document.getElementById('typing-indicator'),
            chatContainer: document.getElementById('chat-container'),
            connectionStatus: document.getElementById('connection-status'),
            messageCount: document.getElementById('message-count'),
            clearChat: document.getElementById('clear-chat')
        };

        function initWebSocket() {
            const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
            const wsUrl = protocol + '//' + window.location.host + '/ws';
            
            ws = new WebSocket(wsUrl);
            updateConnectionStatus('connecting');

            ws.onopen = () => {
                console.log('✅ WebSocket connected');
                updateConnectionStatus('connected');
            };

            ws.onmessage = (event) => {
                const data = JSON.parse(event.data);
                
                if (data.type === 'ai_response') {
                    handleAIResponse(data.message);
                } else if (data.type === 'status_update') {
                    showStatusMessage(data.message);
                }
            };

            ws.onclose = () => {
                console.log('❌ WebSocket disconnected');
                updateConnectionStatus('disconnected');
                setTimeout(initWebSocket, 3000);
            };

            ws.onerror = (error) => {
                console.error('WebSocket error:', error);
                updateConnectionStatus('error');
            };
        }

        function updateConnectionStatus(status) {
            const statusEl = elements.connectionStatus;
            
            const statusConfig = {
                connecting: { class: 'bg-yellow-100 text-yellow-800', text: '🔄 连接中...' },
                connected: { class: 'bg-green-100 text-green-800', text: '🟢 已连接' },
                disconnected: { class: 'bg-red-100 text-red-800', text: '🔴 已断开' },
                error: { class: 'bg-red-100 text-red-800', text: '❌ 连接错误' }
            };

            const config = statusConfig[status];
            statusEl.className = 'px-3 py-1 rounded-full text-sm font-medium ' + config.class;
            statusEl.textContent = config.text;
            elements.sendButton.disabled = status !== 'connected';
            elements.messageInput.disabled = status !== 'connected';
        }

        function addUserMessage(message) {
            const messageDiv = document.createElement('div');
            messageDiv.className = 'flex justify-end';
            messageDiv.innerHTML = '<div class="user-message text-white p-4 rounded-xl max-w-md"><p>' + 
                escapeHtml(message) + '</p></div>';
            elements.messagesContainer.appendChild(messageDiv);
            scrollToBottom();
        }

        function sendMessage() {
            const message = elements.messageInput.value.trim();
            if (!message || !ws || ws.readyState !== WebSocket.OPEN) return;

            addUserMessage(message);
            ws.send(JSON.stringify({ message }));
            
            elements.messageInput.value = '';
            elements.messageInput.style.height = 'auto';
            elements.typingIndicator.classList.remove('hidden');
            elements.sendButton.disabled = true;
            elements.messageInput.disabled = true;
            
            messageCount += 2;
            updateMessageCount();
            scrollToBottom();
        }

        function escapeHtml(text) {
            const div = document.createElement('div');
            div.textContent = text;
            return div.innerHTML;
        }

        function scrollToBottom() {
            elements.chatContainer.scrollTop = elements.chatContainer.scrollHeight;
        }

        function updateMessageCount() {
            elements.messageCount.textContent = messageCount + ' 条对话';
        }

        function showStatusMessage(message) {
            const statusDiv = document.createElement('div');
            statusDiv.className = 'flex justify-center my-2';
            statusDiv.innerHTML = '<div class="status-message text-white px-4 py-2 rounded-full text-sm">' + 
                message + '</div>';
            elements.messagesContainer.appendChild(statusDiv);
            scrollToBottom();
        }

        function handleAIResponse(message) {
            elements.typingIndicator.classList.add('hidden');
            
            const messageDiv = document.createElement('div');
            messageDiv.className = 'flex justify-start';
            
            const htmlContent = marked.parse(message);
            messageDiv.innerHTML = '<div class="ai-message p-4 rounded-xl max-w-4xl border border-gray-200">' + 
                htmlContent + '</div>';
            
            elements.messagesContainer.appendChild(messageDiv);
            scrollToBottom();
            
            elements.sendButton.disabled = false;
            elements.messageInput.disabled = false;
        }

        // Event Listeners
        elements.sendButton.addEventListener('click', sendMessage);

        elements.messageInput.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                sendMessage();
            }
        });

        elements.messageInput.addEventListener('input', function() {
            this.style.height = 'auto';
            this.style.height = this.scrollHeight + 'px';
        });

        // Suggestion buttons
        document.querySelectorAll('.suggestion-btn').forEach(btn => {
            btn.addEventListener('click', function() {
                const question = this.querySelector('.font-medium').textContent;
                elements.messageInput.value = question;
                elements.messageInput.style.height = 'auto';
                elements.messageInput.style.height = elements.messageInput.scrollHeight + 'px';
                sendMessage();
            });
        });

        // Clear chat
        elements.clearChat.addEventListener('click', () => {
            if (confirm('确定要清空所有对话吗？')) {
                const welcomeElements = elements.messagesContainer.querySelectorAll('.bg-gradient-to-r, .bg-white');
                elements.messagesContainer.innerHTML = '';
                welcomeElements.forEach(el => elements.messagesContainer.appendChild(el));
                messageCount = 0;
                updateMessageCount();
                elements.typingIndicator.classList.add('hidden');
            }
        });

        // Initialize
        document.addEventListener('DOMContentLoaded', () => {
            initWebSocket();
            elements.messageInput.focus();
        });

        // Reconnect on visibility change
        document.addEventListener('visibilitychange', () => {
            if (!document.hidden && (!ws || ws.readyState === WebSocket.CLOSED)) {
                initWebSocket();
            }
        });
    </script>
</body>
</html>`);
});

// ============ WEBSOCKET HANDLING ============
wss.on('connection', (ws) => {
    console.log('🔌 New WebSocket connection');
    
    const agent = new UniGuideAgent(ws, openai);
    
    if (!connectionManager.addConnection(ws, agent)) {
        console.log('❌ Connection rejected - server at capacity');
        return;
    }

    ws.on('message', async (message) => {
        try {
            const data = JSON.parse(message.toString());
            const userMessage = data.message?.trim();

            if (!userMessage) {
                ws.send(JSON.stringify({
                    type: 'error',
                    message: '请输入有效的消息内容',
                    timestamp: new Date().toISOString()
                }));
                return;
            }

            const currentAgent = connectionManager.getAgent(ws);
            if (currentAgent) {
                await currentAgent.processMessage(userMessage);
            } else {
                console.error('❌ Agent not found for WebSocket');
                ws.send(JSON.stringify({
                    type: 'error',
                    message: '服务器内部错误：找不到对应的AI助手',
                    timestamp: new Date().toISOString()
                }));
            }

        } catch (error) {
            console.error('❌ Error processing message:', error);
            ws.send(JSON.stringify({
                type: 'error',
                message: '处理消息时发生错误，请重试',
                timestamp: new Date().toISOString()
            }));
        }
    });

    ws.on('close', (code, reason) => {
        console.log(`🔌 WebSocket closed: ${code} - ${reason}`);
        connectionManager.removeConnection(ws);
    });

    ws.on('error', (error) => {
        console.error('❌ WebSocket error:', error);
        connectionManager.removeConnection(ws);
    });
});

// ============ ERROR HANDLING ============
process.on('uncaughtException', (error) => {
    console.error('❌ Uncaught Exception:', error);
    // Don't exit in production, just log the error
});

process.on('unhandledRejection', (reason, promise) => {
    console.error('❌ Unhandled Rejection at:', promise, 'reason:', reason);
});

// ============ GRACEFUL SHUTDOWN ============
process.on('SIGTERM', () => {
    console.log('📴 Received SIGTERM, shutting down gracefully...');
    server.close(() => {
        console.log('✅ Server closed successfully');
        process.exit(0);
    });
});

process.on('SIGINT', () => {
    console.log('📴 Received SIGINT, shutting down gracefully...');
    server.close(() => {
        console.log('✅ Server closed successfully');
        process.exit(0);
    });
});

// ============ START SERVER ============
server.listen(PORT, '0.0.0.0', () => {
    console.log(`
🚀 ===============================================
   UniGuide AI Server Started Successfully!
   
   🌐 Server: http://localhost:${PORT}
   📊 Health: http://localhost:${PORT}/health
   🔌 WebSocket: ws://localhost:${PORT}/ws
   
   📝 Environment:
   ✅ OpenAI API: ${process.env.OPENAI_API_KEY ? 'Configured' : 'Missing'}
   ${process.env.BRAVE_API_KEY ? '✅' : '⚠️'} Brave Search: ${process.env.BRAVE_API_KEY ? 'Configured' : 'Using fallback'}
   
   🎯 Features:
   ✅ Real-time college search
   ✅ AI-powered matching algorithm  
   ✅ Comprehensive user profiling
   ✅ Robust error handling
   ✅ Connection management
   
===============================================`);
});

// Export for testing
module.exports = { 
    app, 
    server, 
    UniGuideAgent, 
    CollegeSearchTool, 
    FitScoringTool,
    ConnectionManager,
    RobustUniversitySearch
};
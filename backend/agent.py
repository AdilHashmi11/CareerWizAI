# backend/agent.py
import os
import google.generativeai as genai
from dotenv import load_dotenv
from .tools import search_tool
from google.api_core import exceptions
import re
import json
import logging

# Load environment variables
load_dotenv()

# Configure logging
logger = logging.getLogger(__name__)

# --- ENHANCED CONVERSATIONAL SYSTEM PROMPT ---
SYSTEM_PROMPT = """
# IDENTITY AND CORE MISSION
You are CareerWiz, a seasoned career strategist with 15+ years of experience across diverse industries. Your expertise spans career transitions, skill development, salary negotiations, and market intelligence. You engage in natural, flowing conversations that build upon previous interactions to provide increasingly personalized guidance.

# CRITICAL LINK FORMATTING REQUIREMENTS
- **ALWAYS include working URLs** when you receive search results
- **Format ALL links as**: **[Source Title](actual_url)**
- **Never create fake or placeholder links** - only use actual URLs from search results
- **Include at least 3-5 relevant links** in each response when search results are available
- **Mix different types of sources**: job sites, courses, YouTube videos, articles, etc.
- **YouTube videos should be marked with 🎥** for easy identification

# MEMORY AND CONTEXT AWARENESS
- **Conversation Memory**: Actively reference and build upon the last 10+ messages to create seamless dialogue continuity
- **Progressive Learning**: Each interaction should deepen your understanding of the user's goals, challenges, and preferences
- **Context Integration**: Weave previous discussions naturally into new responses without explicitly stating "you mentioned earlier"
- **Adaptive Guidance**: Adjust your communication style and depth based on the user's demonstrated expertise level

# SEARCH-FIRST INTELLIGENCE PROTOCOL
- **Immediate Search**: For ANY career topic (jobs, skills, courses, salaries, trends), use the search tool FIRST
- **Current Market Data**: Prioritize recent information over general knowledge to ensure relevance
- **Validation**: Use search results to validate and enhance your expert knowledge
- **Link Integration**: Extract and format ALL working URLs as: **[Descriptive Title](actual_url)**
- **No Placeholders**: Never create fake links or generic references

# CONVERSATIONAL EXCELLENCE STANDARDS

## Natural Dialogue Flow
- Respond conversationally without rigid section templates
- Use strategic emojis from the approved context list to enhance readability and engagement
- Vary response structure based on context - sometimes bullet points, sometimes flowing paragraphs
- Ask follow-up questions naturally when clarification would be genuinely helpful

## Smart Emoji Usage Context
Use these emojis strategically to enhance communication effectiveness:

**Career Development & Growth:**
- 🚀 for career advancement, launches, and new opportunities
- 📈 for growth, progress, salary increases, market trends
- 🎯 for goals, targeting specific outcomes, focus areas
- 💡 for insights, ideas, strategic thinking, lightbulb moments
- 🔑 for key skills, essential requirements, unlock potential

**Professional Skills & Learning:**
- 🧠 for knowledge, learning, skill development, intelligence
- 💻 for tech roles, programming, digital skills, remote work
- 📚 for education, courses, certifications, training programs
- ⚡ for quick wins, immediate actions, energy, momentum
- 🎓 for education, degrees, academic achievements

**Industry & Market Intelligence:**
- 📊 for data, analytics, market research, compensation info
- 🌐 for global opportunities, remote work, international markets
- 🏢 for corporate roles, companies, business environments
- 💼 for professional contexts, business opportunities, networking
- 🔍 for research, job searching, investigation, discovery

**Success & Achievement:**
- ✅ for completion, success, positive outcomes, validation
- 🏆 for achievements, winning, excellence, top performance
- 💪 for strength, capability, confidence, overcoming challenges
- ⭐ for standout opportunities, premium options, excellence

**Communication & Relationships:**
- 🤝 for networking, partnerships, collaboration, agreements
- 💬 for communication skills, interviews, presentations
- 🎤 for public speaking, leadership, voice, influence

**Warning & Caution (use sparingly):**
- ⚠️ for important warnings, market cautions, realistic expectations
- 🚨 for urgent action needed, critical timing, important alerts

**Timing & Planning:**
- ⏰ for timing, deadlines, time-sensitive opportunities
- 📅 for planning, scheduling, timelines, milestones

**Money & Compensation:**
- 💰 for salary, compensation, financial benefits, ROI
- 💵 for specific money amounts, pay ranges, cost discussions

**EMOJI USAGE RULES:**
- Maximum 4-5 emojis per response to maintain professionalism
- Use emojis to highlight key points, not decorate every sentence
- Choose emojis that directly relate to the career context being discussed
- Place emojis strategically at the beginning of important points or sections
- Never use emojis in place of words - they should enhance, not replace communication

## Proactive Value Delivery
- **Range-Based Recommendations**: Instead of asking "what's your skill level?", provide options for beginners through advanced professionals
- **Multi-Path Guidance**: Present 2-3 viable approaches with clear trade-offs
- **Immediate Actionability**: Include specific next steps the user can take today
- **Resource Variety**: Offer mix of free and premium options, online and offline resources

## Communication Style
- **Confident Authority**: Speak as the expert you are - no apologetic language about limitations
- **Practical Focus**: Balance strategic thinking with tactical execution
- **Personal Touch**: Use "you" and "your" to maintain engagement
- **Encouraging Realism**: Be honest about challenges while maintaining optimism about outcomes

# RESPONSE QUALITY GUIDELINES

## Content Standards
- **Depth Over Breadth**: Better to cover fewer topics thoroughly than many topics superficially
- **Market Relevance**: Ground all advice in current industry realities
- **Personalized Insights**: Connect general advice to the user's specific situation
- **Forward-Looking**: Consider industry evolution and future skill demands

## Format Flexibility
- **Short Queries**: 2-3 paragraphs with key insights and 1-2 actionable recommendations
- **Complex Questions**: More detailed responses with clear structure but natural flow
- **Follow-ups**: Build directly on previous conversation without restating context
- **Resource Sharing**: Integrate links naturally within explanatory text

# LINK INTEGRATION EXAMPLES

When you get search results, integrate them naturally like this:

"For Python certification, I recommend starting with **[Python Institute PCAP Certification](https://pythoninstitute.org/pcap)** which is industry-recognized. You can also explore **[Python for Everybody Specialization on Coursera](https://coursera.org/specializations/python)** for comprehensive learning. 

For practical tutorials, check out **[🎥 Python Tutorial for Beginners - Programming with Mosh](https://youtube.com/watch?v=_uQrJ0TkZlc)** which covers fundamentals excellently."

# PROHIBITED BEHAVIORS
- Never apologize for search limitations or technical constraints
- Don't mention APIs, tools, or system processes
- Avoid rigid section headers unless genuinely helpful for organization
- No generic platitudes or filler content
- Don't ask for information you could reasonably provide ranges for
- **NEVER create placeholder links or fake URLs**
- **NEVER ignore search results when available**

# SPECIALIZED EXPERTISE AREAS
- **Career Transitions**: Industry switching, role pivoting, skill bridging
- **Compensation Strategy**: Salary research, negotiation tactics, total compensation analysis  
- **Professional Development**: Skill gap analysis, learning path optimization, certification ROI
- **Market Intelligence**: Industry trends, job market conditions, remote work insights
- **Personal Branding**: Resume optimization, LinkedIn strategy, portfolio development

# EXAMPLE INTERACTION PATTERNS

**For Skill Development Queries:**
"Based on current market demand, here are three learning paths that align with your background... [specific recommendations with timelines and resources including actual links]"

**For Career Change Questions:** "Your experience in [previous role] actually translates well to [target field]. Here's how to position yourself... [tactical advice with real examples and relevant job board links]"

**For Salary Inquiries:**
"Let me pull the latest compensation data for your role and location... [current market rates with context and negotiation insights, including links to salary research tools]"

# SUCCESS METRICS
Every response should:
1. Advance the user's career objectives meaningfully
2. Provide at least one immediately actionable insight
3. Include current market data when relevant
4. Build naturally on previous conversation elements
5. Feel like advice from a trusted career mentor
6. **Include working links to relevant resources when search results are available**

Remember: You're not just an information provider - you're a strategic career partner helping users navigate complex professional decisions with confidence and clarity. Always include real, working links to help users take immediate action on your recommendations.
"""

def clean_response_text(text: str) -> str:
    """Clean and format the response text for optimal presentation."""
    # Remove any XML-style tags that might slip through
    text = re.sub(r'<[^>]+>', '', text)
    
    # Ensure proper markdown link formatting - keep the bold formatting
    # Convert any malformed links to proper format with bold
    text = re.sub(r'(?<!\*)\[([^\]]+)\]\(([^)]+)\)(?!\*)', r'**[\1](\2)**', text)
    
    # Remove any placeholder link patterns
    text = re.sub(r'\[RESOURCE_LINK[^\]]*\]', '', text)
    text = re.sub(r'\[[^\]]*PLACEHOLDER[^\]]*\]', '', text)
    
    # Clean up excessive whitespace
    text = re.sub(r'\n\s*\n\s*\n', '\n\n', text)
    
    return text.strip()

def validate_links_in_response(text: str) -> str:
    """Validate that all links in the response are properly formatted."""
    # Find all markdown links (both bold and regular)
    link_pattern = r'(?:\*\*)?\[([^\]]+)\]\(([^)]+)\)(?:\*\*)?'
    links = re.findall(link_pattern, text)
    
    # Remove any links that don't have valid URLs
    def is_valid_url(url):
        return url.startswith(('http://', 'https://')) and '.' in url and len(url) > 10
    
    valid_links_count = 0
    for title, url in links:
        if not is_valid_url(url):
            # Remove the invalid link
            invalid_patterns = [
                f'**[{title}]({url})**',
                f'[{title}]({url})',
            ]
            for pattern in invalid_patterns:
                if pattern in text:
                    text = text.replace(pattern, f'_{title}_')
                    logger.warning(f"Removed invalid link: {pattern}")
        else:
            valid_links_count += 1
    
    logger.info(f"✅ Response contains {valid_links_count} valid links")
    return text

def parse_agent_json(text: str) -> dict:
    """
    Parses agent output to extract JSON content if present.
    If no JSON is found, it formats the entire text as a full_response.
    """
    json_pattern = r'```json(.*?)```'
    json_match = re.search(json_pattern, text, re.DOTALL)
    
    if json_match:
        try:
            json_content = json_match.group(1).strip()
            data = json.loads(json_content)
            
            # The system prompt asks for a 'reply' object
            if 'reply' in data:
                reply_data = data['reply']
                
                # Ensure text is present, fallback to summary or an empty string
                full_response_text = reply_data.get('text', '')
                
                # Extract tasks and resources
                tasks = reply_data.get('tasks', [])
                resources = reply_data.get('resources', [])
                
                return {
                    "full_response": full_response_text,
                    "tasks": tasks,
                    "resources": resources,
                    "source": "AI Plan"
                }
            else:
                logger.warning("Agent JSON is missing 'reply' key")
        except json.JSONDecodeError:
            logger.error("Failed to parse JSON from agent response.")
            
    # Fallback if no valid JSON or structured data is found
    logger.info("No valid structured JSON found. Returning raw text.")
    return {
        "full_response": text,
        "tasks": [],
        "resources": [],
        "source": "AI Text"
    }

class CareerCounselorAgent:
    def __init__(self):
        """Initialize the CareerWiz agent with enhanced error handling."""
        try:
            # Validate API key
            api_key = os.getenv("GOOGLE_API_KEY")
            if not api_key:
                raise ValueError("GOOGLE_API_KEY not found in environment variables")
            
            genai.configure(api_key=api_key)
            
            # Initialize with a supported, stable model
            self.model = genai.GenerativeModel(
                model_name=os.getenv("GEMINI_MODEL", 'gemini-2.5-flash-lite'), # Using env variable
                system_instruction=SYSTEM_PROMPT,
                tools=[search_tool.search_for_gemini]
            )
            
            # Start chat session with automatic function calling and memory
            self.chat = self.model.start_chat(
                enable_automatic_function_calling=True,
                history=[]  # Will maintain conversation history
            )
            
            # Track conversation context for memory management
            self.conversation_turns = 0
            self.max_history_turns = 20  # Keep last 20 turns for context
            
            logger.info("✅ CareerWiz conversational agent initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize CareerWiz agent: {str(e)}")
            raise RuntimeError(f"Agent initialization failed: {str(e)}")

    def reset_conversation(self):
        """Reset the conversation history."""
        try:
            self.chat = self.model.start_chat(
                enable_automatic_function_calling=True,
                history=[]
            )
            self.conversation_turns = 0
            logger.info("🔄 Conversation history reset")
        except Exception as e:
            logger.error(f"❌ Failed to reset conversation: {str(e)}")

    def is_career_related(self, query: str) -> bool:
        """Determine if a query is career-related with expanded detection."""
        career_keywords = {
            'job', 'career', 'salary', 'work', 'employment', 'skill', 'course',
            'training', 'certification', 'interview', 'resume', 'cv', 'hire',
            'promotion', 'profession', 'industry', 'company', 'business',
            'freelance', 'remote', 'internship', 'apprenticeship', 'bootcamp',
            'developer', 'engineer', 'manager', 'analyst', 'consultant',
            'marketing', 'sales', 'finance', 'hr', 'design', 'coding',
            'programming', 'data', 'tech', 'startup', 'corporate',
            'networking', 'portfolio', 'linkedin', 'negotiate', 'transition'
        }
        
        query_lower = query.lower()
        return any(keyword in query_lower for keyword in career_keywords)

    def generate_conversational_response(self, query: str, username: str = None) -> str:
        """Generate a brief conversational response for non-career or casual queries."""
        user_ref = f"{username}, " if username else ""
        
        if len(query.strip()) < 15:  # Very brief query
            return f"Hi {user_ref}I'm here to help with your career journey! 🚀 What specific career challenge or opportunity are you thinking about? Whether it's exploring new roles, developing skills, or planning your next move, I can provide targeted guidance based on current market insights."
        
        # Try to find a career angle in the query
        return f"Hello {user_ref}I specialize in career strategy and professional development. 💼 While that's an interesting topic, I'd love to help you tackle a career-related challenge. Are you looking to advance in your current field, explore new opportunities, negotiate compensation, or develop new skills? Let's focus on accelerating your professional growth! 📈"

    def manage_conversation_memory(self):
        """Manage conversation history to maintain context while preventing token overflow."""
        if self.conversation_turns > self.max_history_turns:
            # Keep only recent conversation history
            try:
                # Get current history
                current_history = self.chat.history
                
                # Keep last 16 messages (8 exchanges) for context
                if len(current_history) > 16:
                    recent_history = current_history[-16:]
                    
                    # Create new chat with recent history
                    self.chat = self.model.start_chat(
                        enable_automatic_function_calling=True,
                        history=recent_history
                    )
                    
                    logger.info("🧠 Conversation memory trimmed to maintain context")
                
                # Reset counter
                self.conversation_turns = len(self.chat.history) // 2
                
            except Exception as e:
                logger.warning(f"Memory management warning: {str(e)}")

    def enhance_response_with_links(self, response_text: str) -> str:
        """Ensure response includes proper link formatting and validation."""
        # Check if response already has properly formatted links
        link_pattern = r'\*\*\[([^\]]+)\]\(([^)]+)\)\*\*'
        existing_links = re.findall(link_pattern, response_text)
        
        if existing_links:
            logger.info(f"🔗 Response contains {len(existing_links)} formatted links")
            # Validate existing links
            for title, url in existing_links:
                if not url.startswith(('http://', 'https://')):
                    logger.warning(f"⚠️ Invalid URL detected: {url}")
        else:
            logger.info("ℹ️ Response does not contain formatted links")
        
        return response_text

    def process_query(self, query: str, username: str = None) -> dict:
        """Process user query with enhanced conversational flow and memory management."""
        logger.info(f"💬 Processing conversational query from {username or 'anonymous'}: {query[:100]}...")
        
        try:
            # Increment conversation counter
            self.conversation_turns += 1
            
            # Manage memory if needed
            if self.conversation_turns % 5 == 0:  # Check every 5 turns
                self.manage_conversation_memory()
            
            # Handle very short or non-career queries conversationally
            if len(query.strip()) < 8:
                conversational_response = self.generate_conversational_response(query, username)
                return {
                    "full_response": conversational_response,
                    "source": "Conversational AI",
                    "search_performed": False
                }
            
            # For non-career queries, still try to provide some career-relevant angle
            if not self.is_career_related(query):
                # Give it one chance to find a career connection
                career_angle_query = f"How does this relate to career development or professional growth: {query}. Provide career-relevant guidance with resources and links."
                try:
                    response = self.chat.send_message(career_angle_query)
                    if response and response.text:
                        # Use the new parser to get structured data
                        parsed_response = parse_agent_json(response.text.strip())
                        cleaned_response = clean_response_text(parsed_response["full_response"])
                        enhanced_response = self.enhance_response_with_links(cleaned_response)
                        final_response = validate_links_in_response(enhanced_response)
                        
                        return {
                            "full_response": final_response,
                            "source": "Career Strategy",
                            "search_performed": self.check_if_search_performed(response),
                            "tasks": parsed_response.get("tasks", []),
                            "resources": parsed_response.get("resources", [])
                        }
                except:
                    # Fall back to conversational response
                    conversational_response = self.generate_conversational_response(query, username)
                    return {
                        "full_response": conversational_response,
                        "source": "Conversational AI", 
                        "search_performed": False,
                        "tasks": [],
                        "resources": []
                    }
            
            # Prepare the query with conversational context and search instruction
            enhanced_query = f"{query}\n\nIMPORTANT: Please search for current information and include actual working links in your response formatted as **[Source Title](URL)**."
            
            if username:
                contextual_query = f"{username} asks: {enhanced_query}"
            else:
                contextual_query = enhanced_query
            
            # Send query to the model with conversation history
            response = self.chat.send_message(contextual_query)

            # Check if the response is a tool call before trying to access `.text`
            if not response.parts or not hasattr(response.parts[0], 'text'):
                logger.warning("Empty response from conversational model or response is a function call")
                # Handle the case where the response might be a tool call with no text
                # The library should handle this automatically. If it doesn't, we return a fallback.
                return {
                    "error": "I couldn't generate a response right now. It seems I need to perform an action first. Please try again or rephrase your question about a specific career topic."
                }
            
            # Clean and validate the response
            raw_text = response.text.strip()
            parsed_response = parse_agent_json(raw_text)

            cleaned_response = clean_response_text(parsed_response["full_response"])
            enhanced_response = self.enhance_response_with_links(cleaned_response)
            final_response = validate_links_in_response(enhanced_response)
            
            # Determine if search was performed by checking function calls
            search_performed = self.check_if_search_performed(response)
            source = "Market Intelligence + Career Strategy" if search_performed else "Career Expertise"
            
            logger.info(f"✅ Conversational response generated successfully. Search: {search_performed}, Turn: {self.conversation_turns}")
            
            return {
                "full_response": final_response,
                "source": source,
                "search_performed": search_performed,
                "tasks": parsed_response.get("tasks", []),
                "resources": parsed_response.get("resources", [])
            }
            
        except exceptions.ResourceExhausted:
            logger.error("❌ API quota exceeded")
            return {
                "error": "I'm experiencing high demand right now. Please try again in a few minutes, and I'll be ready to help with your career questions."
            }
            
        except exceptions.InvalidArgument as e:
            logger.error(f"❌ Invalid request: {str(e)}")
            return {
                "error": "There was an issue processing your request. Could you try rephrasing your career question?"
            }
            
        except Exception as e:
            logger.error(f"❌ Unexpected error processing conversational query: {str(e)}")
            return {
                "error": "I encountered an unexpected issue. Please try again with a different career-related question."
            }

    def check_if_search_performed(self, response) -> bool:
        """Check if search was performed by analyzing the response."""
        search_performed = False
        
        if response.candidates and response.candidates[0].content.parts:
            for part in response.candidates[0].content.parts:
                if hasattr(part, 'function_call') and part.function_call:
                    if part.function_call.name == 'search':
                        search_performed = True
                        logger.info("🔍 Search function was called")
                        break
        
        return search_performed


# Create and export the career agent instance
try:
    career_agent = CareerCounselorAgent()
    logger.info("✅ Global career_agent instance created successfully")
except Exception as e:
    logger.error(f"❌ Failed to create career_agent instance: {str(e)}")
    career_agent = None

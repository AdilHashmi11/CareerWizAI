# backend/tools.py
import os
import json
import requests
from typing import List, Dict, Any, Optional
import google.generativeai as genai
from dotenv import load_dotenv
import logging
from urllib.parse import urlparse

# Load environment variables
load_dotenv()

# Configure logging
logger = logging.getLogger(__name__)

class SearchTool:
    """Enhanced search tool with robust error handling and result validation."""
    
    def __init__(self):
        """Initialize search tool with API configuration."""
        self.serper_api_key = os.getenv("SERPER_API_KEY")
        
        # Log initialization status
        if self.serper_api_key:
            logger.info("🔧 SearchTool initialized with Serper API")
        else:
            logger.warning("⚠️ SearchTool initialized without Serper API key")
    
    def is_valid_url(self, url: str) -> bool:
        """Validate if a URL is properly formatted."""
        try:
            result = urlparse(url)
            return all([result.scheme, result.netloc]) and result.scheme in ['http', 'https']
        except Exception:
            return False
    
    def search_serper(self, query: str, num: int = 8) -> List[Dict[str, Any]]:
        """Enhanced Serper API search with better error handling."""
        if not self.serper_api_key:
            logger.warning("Serper API key not available")
            return []
        
        try:
            url = "https://google.serper.dev/search"
            payload = {
                "q": query,
                "num": min(num, 10),  # Limit to prevent quota issues
                "gl": "us",
                "hl": "en"
            }
            
            headers = {
                'X-API-KEY': self.serper_api_key,
                'Content-Type': 'application/json'
            }
            
            logger.debug(f"🔍 Searching Serper for: '{query}'")
            
            response = requests.post(
                url, 
                headers=headers, 
                json=payload, 
                timeout=15
            )
            response.raise_for_status()
            
            data = response.json()
            results = []
            
            # Process organic results
            if "organic" in data and data["organic"]:
                for item in data["organic"]:
                    title = item.get("title", "").strip()
                    link = item.get("link", "").strip()
                    snippet = item.get("snippet", "").strip()
                    
                    # Validate required fields
                    if title and link and self.is_valid_url(link):
                        results.append({
                            "title": title,
                            "url": link,
                            "description": snippet,
                            "source": "Google Search"
                        })
            
            # Process knowledge graph if available
            if "knowledgeGraph" in data:
                kg = data["knowledgeGraph"]
                if kg.get("title") and kg.get("website") and self.is_valid_url(kg.get("website", "")):
                    results.insert(0, {
                        "title": f"{kg.get('title')} - Official Website",
                        "url": kg.get("website"),
                        "description": kg.get("description", "Official website"),
                        "source": "Knowledge Graph"
                    })
            
            # Look for YouTube results specifically
            youtube_results = self.extract_youtube_results(data)
            results.extend(youtube_results)
            
            logger.info(f"🟢 Serper: Found {len(results)} valid results for '{query}'")
            return results
            
        except requests.exceptions.Timeout:
            logger.error("🔴 Serper search timeout")
            return []
        except requests.exceptions.RequestException as e:
            logger.error(f"🔴 Serper request failed: {str(e)}")
            return []
        except Exception as e:
            logger.error(f"🔴 Serper unexpected error: {str(e)}")
            return []
    
    def extract_youtube_results(self, search_data: dict) -> List[Dict[str, Any]]:
        """Extract YouTube video results from search data."""
        youtube_results = []
        
        # Check for videos section in search results
        if "videos" in search_data:
            for video in search_data["videos"][:3]:  # Limit to top 3 videos
                title = video.get("title", "").strip()
                link = video.get("link", "").strip()
                duration = video.get("duration", "")
                
                if title and link and "youtube.com" in link:
                    description = f"YouTube Video"
                    if duration:
                        description += f" ({duration})"
                    
                    youtube_results.append({
                        "title": f"🎥 {title}",
                        "url": link,
                        "description": description,
                        "source": "YouTube"
                    })
        
        return youtube_results
    
    def prioritize_career_results(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Enhanced result prioritization for career-focused content."""
        
        # High-priority career domains
        priority_domains = [
            'linkedin.com', 'indeed.com', 'glassdoor.com', 'coursera.org',
            'udemy.com', 'edx.org', 'pluralsight.com', 'codecademy.com',
            'freecodecamp.org', 'datacamp.com', 'udacity.com',
            'monster.com', 'careerbuilder.com', 'ziprecruiter.com',
            'stackoverflow.com', 'github.com', 'medium.com',
            'harvard.edu', 'mit.edu', 'stanford.edu', 'youtube.com'
        ]
        
        # Regional job boards
        regional_domains = [
            'rozee.pk', 'bayt.com', 'naukri.com', 'jobstreet.com',
            'seek.com', 'reed.co.uk', 'xing.com'
        ]
        
        # Career-relevant keywords
        career_keywords = [
            'salary', 'job', 'career', 'course', 'training', 'certification',
            'skill', 'learning', 'bootcamp', 'degree', 'program', 'interview',
            'resume', 'portfolio', 'experience', 'internship', 'remote'
        ]
        
        scored_results = []
        
        for result in results:
            score = 0
            url_lower = result.get('url', '').lower()
            title_lower = result.get('title', '').lower()
            desc_lower = result.get('description', '').lower()
            
            # Domain scoring
            for domain in priority_domains:
                if domain in url_lower:
                    score += 15
                    # Extra boost for YouTube educational content
                    if domain == 'youtube.com' and any(keyword in title_lower for keyword in ['tutorial', 'course', 'learn', 'training']):
                        score += 10
                    break
            
            for domain in regional_domains:
                if domain in url_lower:
                    score += 10
                    break
            
            # Keyword scoring
            for keyword in career_keywords:
                if keyword in title_lower:
                    score += 5
                if keyword in desc_lower:
                    score += 2
            
            # Boost educational and certification content
            edu_indicators = ['course', 'certification', 'training', 'learn', 'tutorial']
            if any(indicator in title_lower for indicator in edu_indicators):
                score += 8
            
            # Penalize spam and irrelevant content
            spam_indicators = [
                'download', 'crack', 'free download', 'torrent', 'hack',
                'cheat', 'generator', 'bot', 'automation'
            ]
            if any(spam in title_lower or spam in desc_lower for spam in spam_indicators):
                score -= 20
            
            # Only include results with positive scores
            if score > 0:
                scored_results.append((score, result))
        
        # Sort by score and return top results
        scored_results.sort(key=lambda x: x[0], reverse=True)
        final_results = [result for score, result in scored_results[:10]]  # Increased to 10
        
        logger.debug(f"📊 Prioritized {len(final_results)} career-relevant results")
        return final_results
    
    def format_results_for_agent(self, results: List[Dict[str, Any]]) -> str:
        """Format search results specifically for the AI agent to use in responses."""
        if not results:
            return "No relevant results found."
        
        formatted_output = []
        formatted_output.append(f"SEARCH RESULTS ({len(results)} found):")
        formatted_output.append("=" * 50)
        
        for i, result in enumerate(results, 1):
            title = result.get('title', 'Untitled')
            url = result.get('url', '')
            description = result.get('description', 'No description available')
            source = result.get('source', 'Web')
            
            # Format each result with clear structure
            formatted_output.append(f"\n{i}. **{title}**")
            formatted_output.append(f"   Source: {source}")
            formatted_output.append(f"   URL: {url}")
            formatted_output.append(f"   Description: {description}")
        
        formatted_output.append("\n" + "=" * 50)
        formatted_output.append("IMPORTANT: When referencing these results in your response, format links as:")
        formatted_output.append("**[Source Title](actual_url)**")
        formatted_output.append("Always include the actual working URLs from the search results above.")
        
        return "\n".join(formatted_output)
    
    def search(self, queries: List[str]) -> str:
        """
        Main search function for the Gemini tool interface.
        Returns formatted string with search results for the agent to use.
        """
        if not queries:
            return "ERROR: No search queries provided"
        
        logger.info(f"🔍 Starting search with {len(queries)} queries: {queries[:3]}")
        
        all_results = []
        successful_searches = 0
        
        # Search with each query (limit to 3 to prevent quota issues)
        for query in queries[:3]:
            if not query.strip():
                continue
                
            results = self.search_serper(query.strip())
            if results:
                all_results.extend(results)
                successful_searches += 1
        
        # Remove duplicates based on URL
        seen_urls = set()
        unique_results = []
        
        for result in all_results:
            url = result.get('url', '')
            if url and url not in seen_urls:
                seen_urls.add(url)
                unique_results.append(result)
        
        # Prioritize career-relevant results
        final_results = self.prioritize_career_results(unique_results)
        
        # Format results for the agent
        if final_results:
            formatted_results = self.format_results_for_agent(final_results)
            logger.info(f"✅ Search completed: {len(final_results)} results from {successful_searches} successful searches")
            return formatted_results
        else:
            logger.warning(f"⚠️ No results found from {successful_searches} searches")
            return "No relevant career results found. Try different search terms or check your internet connection."

# Create search tool instance
search_tool = SearchTool()

# Gemini function declaration for the search tool
search_function_declaration = genai.protos.FunctionDeclaration(
    name="search",
    description="Search the web for current career information including jobs, courses, training, certifications, salary data, industry trends, and YouTube tutorials. This tool provides up-to-date links and resources that you MUST include in your responses with proper formatting.",
    parameters=genai.protos.Schema(
        type=genai.protos.Type.OBJECT,
        properties={
            "queries": genai.protos.Schema(
                type=genai.protos.Type.ARRAY,
                items=genai.protos.Schema(type=genai.protos.Type.STRING),
                description="List of search queries to find relevant career information. Include variations like 'software developer salary 2024', 'python certification courses', 'remote data science jobs', 'career transition tutorials YouTube'"
            )
        },
        required=["queries"]
    )
)

# Tool configuration for Gemini
search_tool.search_for_gemini = genai.protos.Tool(
    function_declarations=[search_function_declaration]
)

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
    
    def prioritize_career_results(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Enhanced result prioritization for career-focused content."""
        
        # High-priority career domains
        priority_domains = [
            'linkedin.com', 'indeed.com', 'glassdoor.com', 'coursera.org',
            'udemy.com', 'edx.org', 'pluralsight.com', 'codecademy.com',
            'freecodecamp.org', 'datacamp.com', 'udacity.com',
            'monster.com', 'careerbuilder.com', 'ziprecruiter.com',
            'stackoverflow.com', 'github.com', 'medium.com',
            'harvard.edu', 'mit.edu', 'stanford.edu'
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
        final_results = [result for score, result in scored_results[:8]]
        
        logger.debug(f"📊 Prioritized {len(final_results)} career-relevant results")
        return final_results
    
    def search(self, queries: List[str]) -> str:
        """
        Main search function for the Gemini tool interface.
        Returns JSON string with search results.
        """
        if not queries:
            return json.dumps({
                "success": False,
                "message": "No search queries provided",
                "results": []
            })
        
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
        
        # Prepare response
        if final_results:
            response_data = {
                "success": True,
                "total_results": len(final_results),
                "results": final_results,
                "search_queries": queries[:3],
                "searches_performed": successful_searches
            }
            logger.info(f"✅ Search completed: {len(final_results)} results from {successful_searches} successful searches")
        else:
            response_data = {
                "success": False,
                "total_results": 0,
                "results": [],
                "search_queries": queries[:3],
                "searches_performed": successful_searches,
                "message": "No relevant career results found. Try different search terms."
            }
            logger.warning(f"⚠️ No results found from {successful_searches} searches")
        
        return json.dumps(response_data, indent=2)

# Create search tool instance
search_tool = SearchTool()

# Gemini function declaration for the search tool
search_function_declaration = genai.protos.FunctionDeclaration(
    name="search",
    description="Search the web for current career information including jobs, courses, training, certifications, salary data, and industry trends. Use this tool for any career-related query to get up-to-date market information.",
    parameters=genai.protos.Schema(
        type=genai.protos.Type.OBJECT,
        properties={
            "queries": genai.protos.Schema(
                type=genai.protos.Type.ARRAY,
                items=genai.protos.Schema(type=genai.protos.Type.STRING),
                description="List of search queries to find relevant career information. Include variations like 'software developer salary 2024', 'python certification courses', 'remote data science jobs'"
            )
        },
        required=["queries"]
    )
)

# Tool configuration for Gemini
search_tool.search_for_gemini = genai.protos.Tool(
    function_declarations=[search_function_declaration]
)

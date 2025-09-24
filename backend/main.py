# backend/main.py

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, validator, Field
import os
import sys
import time
import logging
from typing import Optional, Dict, Any
from contextlib import asynccontextmanager
import traceback
import re

# Path configuration
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
FRONTEND_DIR = os.path.join(PROJECT_ROOT, "frontend")
sys.path.append(PROJECT_ROOT)

# Enhanced logging configuration with Unicode support
class UnicodeFormatter(logging.Formatter):
    """Custom formatter that handles Unicode characters safely."""
    def format(self, record):
        # Replace problematic Unicode characters with safe alternatives
        if hasattr(record, 'msg') and isinstance(record.msg, str):
            # Replace common emojis with text alternatives
            emoji_replacements = {
                '🚀': '[ROCKET]',
                '✅': '[SUCCESS]', 
                '❌': '[ERROR]',
                '⚠️': '[WARNING]',
                '💬': '[CHAT]',
                '📊': '[STATS]',
                '🔧': '[TOOL]',
                '🧙': '[WIZARD]',
                '📈': '[GROWTH]',
                '💡': '[IDEA]',
                '🎯': '[TARGET]',
                '💻': '[TECH]',
                '📚': '[EDUCATION]',
                '🏆': '[ACHIEVEMENT]',
                '🔍': '[SEARCH]',
                '🎥': '[VIDEO]',
                '💼': '[BUSINESS]',
                '🛑': '[STOP]'
            }
            
            msg = str(record.msg)
            for emoji, replacement in emoji_replacements.items():
                msg = msg.replace(emoji, replacement)
            record.msg = msg
            
        return super().format(record)

# Configure logging with Unicode-safe formatter
unicode_formatter = UnicodeFormatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Setup handlers
file_handler = logging.FileHandler('career_wiz.log', encoding='utf-8')
console_handler = logging.StreamHandler(sys.stdout)

file_handler.setFormatter(unicode_formatter)
console_handler.setFormatter(unicode_formatter)

# Configure root logger
logging.basicConfig(
    level=logging.INFO,
    handlers=[file_handler, console_handler]
)

logger = logging.getLogger(__name__)

# Global application state
career_agent_instance = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Enhanced lifespan management with better error handling."""
    global career_agent_instance
    
    # Startup
    logger.info("[ROCKET] Initializing CareerWiz application...")
    
    try:
        # Import and initialize the career agent
        from backend.agent import career_agent
        
        if career_agent is None:
            raise RuntimeError("Career agent failed to initialize during import")
        
        career_agent_instance = career_agent
        logger.info("[SUCCESS] CareerWiz agent initialized successfully")
        
        # Test the agent with a simple query
        test_result = career_agent_instance.process_query("test")
        if "error" in test_result:
            logger.warning(f"[WARNING] Agent test returned error: {test_result['error']}")
        else:
            logger.info("[SUCCESS] Agent test completed successfully")
        
        yield
        
    except Exception as e:
        logger.error(f"[ERROR] Critical startup failure: {str(e)}")
        logger.error(traceback.format_exc())
        raise RuntimeError(f"Failed to initialize CareerWiz: {str(e)}")
    
    finally:
        # Shutdown
        logger.info("[STOP] Shutting down CareerWiz application...")

# FastAPI application setup
app = FastAPI(
    title="CareerWiz API",
    description="AI-powered career counseling and guidance platform with intelligent search capabilities",
    version="2.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Rate limiting implementation
class RateLimiter:
    """Enhanced rate limiter with different limits for different endpoints."""
    
    def __init__(self):
        self.requests = {}
        self.limits = {
            "chat": {"max_requests": 20, "window": 60},  # 20 chat requests per minute
            "default": {"max_requests": 50, "window": 60}  # 50 other requests per minute
        }
    
    def is_allowed(self, client_ip: str, endpoint_type: str = "default") -> bool:
        current_time = time.time()
        limit_config = self.limits.get(endpoint_type, self.limits["default"])
        
        key = f"{client_ip}:{endpoint_type}"
        
        if key not in self.requests:
            self.requests[key] = []
        
        # Clean old requests
        self.requests[key] = [
            req_time for req_time in self.requests[key]
            if current_time - req_time < limit_config["window"]
        ]
        
        if len(self.requests[key]) < limit_config["max_requests"]:
            self.requests[key].append(current_time)
            return True
        
        return False

rate_limiter = RateLimiter()

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# Enhanced middleware
@app.middleware("http")
async def comprehensive_middleware(request: Request, call_next):
    """Combined rate limiting and logging middleware."""
    start_time = time.time()
    client_ip = request.client.host
    
    # Skip rate limiting for health checks and static files
    skip_rate_limit = (
        request.url.path in ["/health", "/api/status", "/"] or 
        request.url.path.startswith("/static") or
        request.url.path.startswith("/docs") or
        request.url.path.startswith("/redoc")
    )
    
    if not skip_rate_limit:
        endpoint_type = "chat" if request.url.path == "/chat" else "default"
        
        if not rate_limiter.is_allowed(client_ip, endpoint_type):
            logger.warning(f"[WARNING] Rate limit exceeded for {client_ip} on {endpoint_type}")
            return JSONResponse(
                status_code=429,
                content={
                    "error": "Rate limit exceeded. Please wait before making another request.",
                    "retry_after": 60
                }
            )
    
    # Process request
    try:
        response = await call_next(request)
        process_time = time.time() - start_time
        
        # Log request
        logger.info(
            f"[STATS] {request.method} {request.url.path} - "
            f"{response.status_code} - {process_time:.3f}s - {client_ip}"
        )
        
        return response
        
    except Exception as e:
        process_time = time.time() - start_time
        logger.error(f"[ERROR] Middleware error: {request.url.path} - {str(e)} - {process_time:.3f}s")
        
        return JSONResponse(
            status_code=500,
            content={"error": "Internal server error occurred"}
        )

# Pydantic models with enhanced validation
class ChatRequest(BaseModel):
    """Enhanced chat request model with better validation."""
    message: str = Field(..., min_length=1, max_length=2000)
    username: Optional[str] = Field(None, max_length=100)
    
    @validator('message')
    def validate_message(cls, v):
        """Validate and clean the message."""
        if not v or not v.strip():
            raise ValueError('Message cannot be empty')
        
        # Remove excessive whitespace
        cleaned = ' '.join(v.strip().split())
        
        if len(cleaned) < 1:
            raise ValueError('Message cannot be empty after cleaning')
        
        return cleaned
    
    @validator('username')
    def validate_username(cls, v):
        """Validate and clean the username."""
        if v is not None:
            cleaned = v.strip()
            if not cleaned:
                return None
            
            # Basic sanitization
            cleaned = ''.join(char for char in cleaned if char.isalnum() or char in ' -_.')
            
            return cleaned[:100]  # Ensure length limit
        return v

class ChatResponse(BaseModel):
    """Standardized chat response model with link tracking."""
    reply: Dict[str, Any]
    timestamp: str
    processing_time: float
    status: str
    agent_version: str = "2.1.0"
    links_included: Optional[int] = None

class HealthResponse(BaseModel):
    """Health check response model."""
    status: str
    timestamp: str
    version: str = "2.1.0"
    agent_status: str
    uptime_seconds: Optional[float] = None

# Utility functions
def get_client_info(request: Request) -> Dict[str, str]:
    """Extract client information for logging."""
    return {
        "ip": request.client.host,
        "user_agent": request.headers.get("user-agent", "unknown")[:200],  # Limit length
        "timestamp": time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())
    }

def count_links_in_response(response_text: str) -> int:
    """Count the number of valid links in the response."""
    if not response_text:
        return 0
    
    # Find all markdown links (both bold and regular)
    link_pattern = r'(?:\*\*)?\[([^\]]+)\]\(([^)]+)\)(?:\*\*)?'
    links = re.findall(link_pattern, response_text)
    
    valid_links = 0
    for title, url in links:
        if url.startswith(('http://', 'https://')) and '.' in url and len(url) > 10:
            valid_links += 1
    
    return valid_links

# Frontend serving
@app.get("/", response_class=FileResponse, tags=["Frontend"])
async def serve_frontend():
    """Serve the main frontend application."""
    index_path = os.path.join(FRONTEND_DIR, "index.html")
    
    if not os.path.exists(index_path):
        logger.error(f"[ERROR] Frontend not found at: {index_path}")
        raise HTTPException(
            status_code=500,
            detail="Frontend application not found. Please ensure index.html exists in the frontend directory."
        )
    
    return FileResponse(index_path)

# Mount static files with error handling
try:
    if os.path.exists(FRONTEND_DIR):
        app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")
        logger.info(f"[SUCCESS] Static files mounted from: {FRONTEND_DIR}")
    else:
        logger.warning(f"[WARNING] Frontend directory not found: {FRONTEND_DIR}")
except Exception as e:
    logger.error(f"[ERROR] Failed to mount static files: {str(e)}")

# Application startup time tracking
app_start_time = time.time()

# Main API endpoints
@app.post("/chat", response_model=ChatResponse, tags=["AI Chat"])
async def chat_with_counselor(request: ChatRequest, http_request: Request):
    """
    Enhanced chat endpoint with comprehensive error handling and link validation.
    """
    start_time = time.time()
    client_info = get_client_info(http_request)
    
    logger.info(
        f"[CHAT] Chat request from {client_info['ip']}: "
        f"'{request.message[:100]}{'...' if len(request.message) > 100 else ''}' "
        f"(User: {request.username or 'Anonymous'})"
    )
    
    try:
        # Validate agent availability
        if career_agent_instance is None:
            logger.error("[ERROR] Career agent not available")
            raise HTTPException(
                status_code=503,
                detail="Career counseling service is temporarily unavailable. Please try again later."
            )
        
        # Process the query
        agent_response = career_agent_instance.process_query(
            query=request.message,
            username=request.username
        )
        
        # Calculate processing time
        processing_time = time.time() - start_time
        timestamp = time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())
        
        # Determine response status
        status = "error" if "error" in agent_response else "success"
        
        # Count links in the response
        links_count = 0
        if "full_response" in agent_response and agent_response["full_response"]:
            links_count = count_links_in_response(agent_response["full_response"])
        
        # Log the result
        if status == "error":
            logger.warning(
                f"[WARNING] Agent error for {client_info['ip']}: "
                f"{agent_response.get('error', 'Unknown error')}"
            )
        else:
            source = agent_response.get('source', 'AI Analysis')
            search_performed = agent_response.get('search_performed', False)
            logger.info(
                f"[SUCCESS] Chat response generated in {processing_time:.3f}s - "
                f"Source: {source} - Search: {search_performed} - Links: {links_count}"
            )
        
        # Return standardized response
        return ChatResponse(
            reply=agent_response,
            timestamp=timestamp,
            processing_time=round(processing_time, 3),
            status=status,
            links_included=links_count if links_count > 0 else None
        )
        
    except HTTPException:
        # Re-raise HTTP exceptions for proper FastAPI handling
        raise
        
    except Exception as e:
        # Handle unexpected server errors
        processing_time = time.time() - start_time
        error_msg = f"Unexpected server error: {str(e)}"
        
        logger.error(
            f"[ERROR] Critical chat error for {client_info['ip']}: {error_msg}"
        )
        logger.error(traceback.format_exc())
        
        # Return error in standard format
        return ChatResponse(
            reply={
                "error": "I encountered an unexpected issue. Please try again with a different question.",
                "full_response": None,
                "source": "Error Handler",
                "search_performed": False
            },
            timestamp=time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime()),
            processing_time=round(processing_time, 3),
            status="error"
        )

@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """
    Comprehensive health check endpoint.
    """
    timestamp = time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())
    uptime = time.time() - app_start_time
    
    # Check agent status
    agent_status = "healthy"
    overall_status = "healthy"
    
    try:
        if career_agent_instance is None:
            agent_status = "unavailable"
            overall_status = "degraded"
        else:
            # Quick agent test
            test_result = career_agent_instance.process_query("health check")
            if "error" in test_result:
                agent_status = "degraded"
                overall_status = "degraded"
    except Exception as e:
        logger.warning(f"[WARNING] Health check agent test failed: {str(e)}")
        agent_status = "error"
        overall_status = "degraded"
    
    return HealthResponse(
        status=overall_status,
        timestamp=timestamp,
        agent_status=agent_status,
        uptime_seconds=round(uptime, 2)
    )

@app.get("/api/status", tags=["System"])
async def detailed_status():
    """
    Detailed system status information.
    """
    uptime = time.time() - app_start_time
    
    # Check components
    components = {
        "career_agent": "healthy" if career_agent_instance is not None else "unavailable",
        "search_tool": "integrated",
        "frontend": "mounted" if os.path.exists(FRONTEND_DIR) else "not_found",
        "rate_limiter": "active",
        "logging": "active",
        "link_validation": "enabled"
    }
    
    # Environment check
    env_status = {
        "google_api_key": "configured" if os.getenv("GOOGLE_API_KEY") else "missing",
        "serper_api_key": "configured" if os.getenv("SERPER_API_KEY") else "missing"
    }
    
    return {
        "api_version": "2.1.0",
        "status": "operational" if components["career_agent"] == "healthy" else "degraded",
        "timestamp": time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime()),
        "uptime_seconds": round(uptime, 2),
        "components": components,
        "environment": env_status,
        "endpoints": {
            "chat": "/chat",
            "health": "/health", 
            "status": "/api/status",
            "docs": "/docs",
            "frontend": "/"
        },
        "features": {
            "ai_agent": "Gemini 1.5 Flash",
            "search_engine": "Serper API",
            "rate_limiting": "Active",
            "conversation_memory": "Enabled",
            "error_recovery": "Enhanced",
            "link_validation": "Enabled",
            "youtube_integration": "Enabled",
            "global_locations": "Pakistan, USA, UK, Germany, China, Australia, Singapore"
        }
    }

@app.post("/api/reset-conversation", tags=["AI Chat"])
async def reset_conversation(http_request: Request):
    """
    Reset the conversation context for the current user.
    """
    client_info = get_client_info(http_request)
    
    try:
        if career_agent_instance is None:
            raise HTTPException(status_code=503, detail="Agent not available")
        
        career_agent_instance.reset_conversation()
        
        logger.info(f"[SUCCESS] Conversation reset for {client_info['ip']}")
        
        return {
            "success": True,
            "message": "Conversation context has been reset",
            "timestamp": time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())
        }
        
    except Exception as e:
        logger.error(f"[ERROR] Failed to reset conversation: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": "Failed to reset conversation context"
            }
        )

# Enhanced error handlers
@app.exception_handler(404)
async def not_found_handler(request: Request, exc):
    """Enhanced 404 handler with logging."""
    client_ip = request.client.host
    logger.warning(f"[WARNING] 404 Not Found: {request.url.path} from {client_ip}")
    
    return JSONResponse(
        status_code=404,
        content={
            "error": "The requested resource was not found",
            "path": request.url.path,
            "suggestion": "Check the API documentation at /docs"
        }
    )

@app.exception_handler(500)
async def internal_error_handler(request: Request, exc):
    """Enhanced 500 handler with detailed logging."""
    client_ip = request.client.host
    error_details = str(exc)
    
    logger.error(f"[ERROR] 500 Internal Server Error: {request.url.path} - {error_details}")
    logger.error(traceback.format_exc())
    
    return JSONResponse(
        status_code=500,
        content={
            "error": "An internal server error occurred",
            "message": "Please try again later. If the problem persists, contact support.",
            "timestamp": time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())
        }
    )

@app.exception_handler(422)
async def validation_error_handler(request: Request, exc):
    """Handle Pydantic validation errors."""
    logger.warning(f"[WARNING] Validation error: {request.url.path} - {str(exc)}")
    
    return JSONResponse(
        status_code=422,
        content={
            "error": "Request validation failed",
            "details": "Please check your request format and try again",
            "timestamp": time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())
        }
    )

# Development vs Production configuration
if __name__ == "__main__":
    import uvicorn
    
    logger.info("[WIZARD] Starting CareerWiz in development mode...")
    logger.info(f"[BUSINESS] Frontend directory: {FRONTEND_DIR}")
    logger.info(f"[SUCCESS] Environment: Google API configured: {bool(os.getenv('GOOGLE_API_KEY'))}")
    logger.info(f"[SEARCH] Search API configured: {bool(os.getenv('SERPER_API_KEY'))}")
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
        access_log=True
    )
else:
    logger.info("[WIZARD] CareerWiz loaded for production deployment")
    logger.info("[STATS] Application version: 2.1.0")

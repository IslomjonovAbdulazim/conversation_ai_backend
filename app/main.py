from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import logging

from app.config import settings
from app.database import create_tables
from app.routers import auth, users, folders, words, quiz, voice

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Vocabulary Learning App",
    description="English-Uzbek vocabulary learning platform with gamification",
    version="1.0.0",
    debug=settings.debug
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure properly for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(auth.router, prefix="/auth", tags=["Authentication"])
app.include_router(users.router, prefix="/user", tags=["Users"])
app.include_router(folders.router, prefix="/folders", tags=["Folders"])
app.include_router(words.router, prefix="/words", tags=["Words"])
app.include_router(quiz.router, prefix="/quiz", tags=["Quiz"])
app.include_router(voice.router, prefix="/voice", tags=["Voice"])

# Startup event
@app.on_event("startup")
async def startup_event():
    """Initialize database and create tables"""
    logger.info("Starting Vocabulary Learning App...")
    create_tables()
    logger.info("Database tables created successfully")

# Root endpoint
@app.get("/")
async def root():
    return {
        "message": "Vocabulary Learning App API",
        "version": "1.0.0",
        "status": "active"
    }

# Health check endpoint
@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": "2025-01-15T12:00:00Z"
    }

# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.error(f"Global exception: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "internal_error",
            "message": "Something went wrong on our end",
            "request_id": f"req_{hash(str(request.url))}"
        }
    )

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=settings.port,
        reload=settings.debug
    )
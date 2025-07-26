from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import logging

from app.config import settings
from app.database import create_tables
from app.routers import auth, folders, words, voice

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Vocabulary Learning App",
    description="English-Uzbek vocabulary learning platform",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(auth.router, prefix="/auth", tags=["Authentication"])
app.include_router(folders.router, prefix="/folders", tags=["Folders"])
app.include_router(words.router, prefix="/words", tags=["Words"])
app.include_router(voice.router, prefix="/voice", tags=["Voice"])

@app.on_event("startup")
async def startup_event():
    """Initialize database"""
    logger.info("Starting Vocabulary Learning App...")
    create_tables()
    logger.info("Database tables created successfully")

@app.get("/")
async def root():
    return {"message": "Vocabulary Learning App API", "status": "active"}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=settings.port, reload=settings.debug)
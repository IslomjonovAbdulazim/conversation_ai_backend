from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import requests
import logging

from app.routers.auth import get_current_user
from app.models import User
from app.config import settings

logger = logging.getLogger(__name__)
router = APIRouter()


# Pydantic models
class VoiceAgent(BaseModel):
    id: str
    topic: str
    title: str
    description: str
    image_url: str


class StartVoiceSessionRequest(BaseModel):
    agent_id: str


class VoiceSessionResponse(BaseModel):
    session_id: str
    websocket_url: str
    agent_id: str


class VoiceAgentsResponse(BaseModel):
    agents: List[VoiceAgent]


# Hardcoded voice agents (simplified)
VOICE_AGENTS = [
    VoiceAgent(
        id="agent_cars",
        topic="cars",
        title="Car Expert",
        description="Talk about cars, engines, and automotive topics",
        image_url="https://images.unsplash.com/photo-1550355291-bbee04a92027?w=400"
    ),
    VoiceAgent(
        id="agent_travel",
        topic="travel",
        title="Travel Guide",
        description="Discuss travel destinations and experiences",
        image_url="https://images.unsplash.com/photo-1488646953014-85cb44e25828?w=400"
    ),
    VoiceAgent(
        id="agent_food",
        topic="food",
        title="Food Expert",
        description="Talk about cooking, recipes, and cuisine",
        image_url="https://images.unsplash.com/photo-1567620905732-2d1ec7ab7445?w=400"
    ),
    VoiceAgent(
        id="agent_sports",
        topic="sports",
        title="Sports Coach",
        description="Discuss sports, fitness, and athletics",
        image_url="https://images.unsplash.com/photo-1571019613454-1cb2f99b2d8b?w=400"
    )
]


@router.get("/agents", response_model=VoiceAgentsResponse)
async def get_voice_agents(current_user: User = Depends(get_current_user)):
    """Get all available voice conversation agents"""
    return VoiceAgentsResponse(agents=VOICE_AGENTS)


@router.post("/start", response_model=VoiceSessionResponse)
async def start_voice_session(
        request: StartVoiceSessionRequest,
        current_user: User = Depends(get_current_user)
):
    """Start a voice conversation session"""
    try:
        # Validate agent exists
        agent = next((a for a in VOICE_AGENTS if a.id == request.agent_id), None)
        if not agent:
            raise HTTPException(status_code=404, detail="Voice agent not found")

        # For now, return a mock websocket URL
        # In production, you'd integrate with ElevenLabs API
        session_id = f"session_{current_user.id}_{request.agent_id}"
        websocket_url = f"wss://api.elevenlabs.io/v1/convai/conversation?agent_id={request.agent_id}"

        return VoiceSessionResponse(
            session_id=session_id,
            websocket_url=websocket_url,
            agent_id=request.agent_id
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error starting voice session: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to start voice session")


@router.post("/stop")
async def stop_voice_session(
        session_id: str,
        current_user: User = Depends(get_current_user)
):
    """Stop a voice conversation session"""
    try:
        # In production, you'd cleanup the session with ElevenLabs
        return {
            "success": True,
            "message": "Voice session stopped",
            "session_id": session_id
        }

    except Exception as e:
        logger.error(f"Error stopping voice session: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to stop voice session")


@router.delete("/cleanup")
async def cleanup_voice_sessions(current_user: User = Depends(get_current_user)):
    """Cleanup all voice sessions for current user"""
    try:
        # In production, you'd cleanup all sessions for this user
        return {
            "success": True,
            "message": "All voice sessions cleaned up"
        }

    except Exception as e:
        logger.error(f"Error cleaning up voice sessions: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to cleanup voice sessions")
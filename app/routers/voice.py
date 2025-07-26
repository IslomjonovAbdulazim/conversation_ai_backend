from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import List, Dict
import logging
import uuid
from datetime import datetime

from app.database import get_db
from app.models import User, VoiceAgent
from app.routers.auth import get_current_user
from app.services.elevenlabs import get_conversation_url

logger = logging.getLogger(__name__)
router = APIRouter()


# Pydantic models
class VoiceAgentResponse(BaseModel):
    id: int
    topic: str
    title: str
    description: str
    image_url: str
    agent_id: str
    is_active: bool


class VoiceAgentsResponse(BaseModel):
    agents: List[VoiceAgentResponse]


class StartTopicRequest(BaseModel):
    agent_id: int


class StartTopicResponse(BaseModel):
    session_id: str
    agent: Dict
    websocket_url: str
    connection_expires_at: str


class StopTopicRequest(BaseModel):
    session_id: str


class StopTopicResponse(BaseModel):
    success: bool
    message: str
    session_stats: Dict


# In-memory session management
active_voice_sessions = {}


@router.get("/agents", response_model=VoiceAgentsResponse)
async def get_voice_agents(
        current_user: User = Depends(get_current_user),
        db: Session = Depends(get_db)
):
    """
    Get available voice agents for topics
    """
    try:
        # Get all active voice agents
        agents = db.query(VoiceAgent).filter(
            VoiceAgent.is_active == True
        ).order_by(VoiceAgent.id).all()

        agent_list = []
        for agent in agents:
            agent_list.append(VoiceAgentResponse(
                id=agent.id,
                topic=agent.topic,
                title=agent.title,
                description=agent.description,
                image_url=agent.image_url,
                agent_id=agent.agent_id,
                is_active=agent.is_active
            ))

        logger.info(f"Retrieved {len(agent_list)} voice agents for user {current_user.id}")

        return VoiceAgentsResponse(agents=agent_list)

    except Exception as e:
        logger.error(f"Error getting voice agents: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to fetch voice agents"
        )


@router.post("/topic/start", response_model=StartTopicResponse)
async def start_topic_conversation(
        request: StartTopicRequest,
        current_user: User = Depends(get_current_user),
        db: Session = Depends(get_db)
):
    """
    Start voice conversation for specific topic
    """
    try:
        # Get voice agent
        voice_agent = db.query(VoiceAgent).filter(
            VoiceAgent.id == request.agent_id,
            VoiceAgent.is_active == True
        ).first()

        if not voice_agent:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Voice agent not found"
            )

        # Generate unique session ID
        session_id = f"voice_session_{uuid.uuid4().hex[:12]}"

        # Get WebSocket URL from ElevenLabs
        websocket_url = await get_conversation_url(voice_agent.agent_id)

        if not websocket_url:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to create voice conversation"
            )

        # Store session data
        session_data = {
            "session_id": session_id,
            "user_id": current_user.id,
            "agent_id": voice_agent.id,
            "elevenlabs_agent_id": voice_agent.agent_id,
            "topic": voice_agent.topic,
            "started_at": datetime.utcnow(),
            "websocket_url": websocket_url
        }

        active_voice_sessions[session_id] = session_data

        # Set expiry (30 minutes from now)
        from datetime import timedelta
        expires_at = datetime.utcnow() + timedelta(minutes=30)

        logger.info(f"Started voice session {session_id} for topic '{voice_agent.topic}' by user {current_user.id}")

        return StartTopicResponse(
            session_id=session_id,
            agent={
                "id": voice_agent.id,
                "topic": voice_agent.topic,
                "title": voice_agent.title,
                "agent_id": voice_agent.agent_id
            },
            websocket_url=websocket_url,
            connection_expires_at=expires_at.isoformat()
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error starting voice conversation: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to start voice conversation"
        )


@router.post("/topic/stop", response_model=StopTopicResponse)
async def stop_topic_conversation(
        request: StopTopicRequest,
        current_user: User = Depends(get_current_user)
):
    """
    Stop voice conversation session
    """
    try:
        # Get session data
        session_data = active_voice_sessions.get(request.session_id)

        if not session_data:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Voice session not found"
            )

        # Verify session belongs to current user
        if session_data["user_id"] != current_user.id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied to this voice session"
            )

        # Calculate session duration
        ended_at = datetime.utcnow()
        duration = int((ended_at - session_data["started_at"]).total_seconds())

        # Create session stats
        session_stats = {
            "session_id": request.session_id,
            "duration": duration,
            "topic": session_data["topic"],
            "started_at": session_data["started_at"].isoformat(),
            "ended_at": ended_at.isoformat()
        }

        # Remove session from memory
        del active_voice_sessions[request.session_id]

        logger.info(f"Stopped voice session {request.session_id} after {duration} seconds")

        return StopTopicResponse(
            success=True,
            message="Voice session ended successfully",
            session_stats=session_stats
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error stopping voice conversation: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to stop voice conversation"
        )


# Optional: Cleanup expired sessions periodically
@router.get("/sessions/cleanup")
async def cleanup_expired_sessions():
    """
    Clean up expired voice sessions (for maintenance)
    """
    try:
        current_time = datetime.utcnow()
        expired_sessions = []

        for session_id, session_data in list(active_voice_sessions.items()):
            # Sessions expire after 30 minutes
            from datetime import timedelta
            if current_time - session_data["started_at"] > timedelta(minutes=30):
                expired_sessions.append(session_id)
                del active_voice_sessions[session_id]

        logger.info(f"Cleaned up {len(expired_sessions)} expired voice sessions")

        return {
            "success": True,
            "cleaned_sessions": len(expired_sessions),
            "active_sessions": len(active_voice_sessions)
        }

    except Exception as e:
        logger.error(f"Error cleaning up sessions: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to cleanup sessions"
        )
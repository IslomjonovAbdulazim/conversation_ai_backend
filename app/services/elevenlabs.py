import requests
import logging
from typing import Optional, Dict
import json

from app.config import settings

logger = logging.getLogger(__name__)


class ElevenLabsService:

    def __init__(self):
        self.base_url = "https://api.elevenlabs.io/v1"
        self.headers = {
            "Accept": "application/json",
            "xi-api-key": settings.elevenlabs_api_key
        }

    async def get_conversation_url(self, agent_id: str) -> Optional[str]:
        """
        Get signed WebSocket URL for conversational AI
        """
        try:
            # For public agents, we can directly connect
            if agent_id:
                websocket_url = f"wss://api.elevenlabs.io/v1/convai/conversation?agent_id={agent_id}"
                logger.info(f"Generated WebSocket URL for agent {agent_id}")
                return websocket_url

            return None

        except Exception as e:
            logger.error(f"Error getting conversation URL: {str(e)}")
            return None

    async def get_signed_url(self, agent_id: str) -> Optional[str]:
        """
        Get signed URL for private agents (if needed)
        """
        try:
            url = f"{self.base_url}/convai/conversations"

            payload = {
                "agent_id": agent_id
            }

            response = requests.post(
                url,
                headers=self.headers,
                json=payload,
                timeout=10
            )

            if response.status_code == 200:
                data = response.json()
                signed_url = data.get("signed_url")

                if signed_url:
                    logger.info(f"Generated signed URL for agent {agent_id}")
                    return signed_url

            logger.error(f"Failed to get signed URL: {response.status_code} - {response.text}")
            return None

        except Exception as e:
            logger.error(f"Error getting signed URL: {str(e)}")
            return None

    async def list_voices(self) -> Optional[list]:
        """
        Get available voices from ElevenLabs
        """
        try:
            url = f"{self.base_url}/voices"

            response = requests.get(
                url,
                headers=self.headers,
                timeout=10
            )

            if response.status_code == 200:
                data = response.json()
                voices = data.get("voices", [])
                logger.info(f"Retrieved {len(voices)} voices from ElevenLabs")
                return voices

            logger.error(f"Failed to get voices: {response.status_code}")
            return None

        except Exception as e:
            logger.error(f"Error getting voices: {str(e)}")
            return None

    async def create_agent(self, agent_config: dict) -> Optional[str]:
        """
        Create new conversational AI agent (if needed for future)
        """
        try:
            url = f"{self.base_url}/convai/agents"

            response = requests.post(
                url,
                headers={**self.headers, "Content-Type": "application/json"},
                json=agent_config,
                timeout=30
            )

            if response.status_code == 200:
                data = response.json()
                agent_id = data.get("agent_id")

                if agent_id:
                    logger.info(f"Created new agent: {agent_id}")
                    return agent_id

            logger.error(f"Failed to create agent: {response.status_code} - {response.text}")
            return None

        except Exception as e:
            logger.error(f"Error creating agent: {str(e)}")
            return None

    async def get_agent_details(self, agent_id: str) -> Optional[dict]:
        """
        Get agent details and configuration
        """
        try:
            url = f"{self.base_url}/convai/agents/{agent_id}"

            response = requests.get(
                url,
                headers=self.headers,
                timeout=10
            )

            if response.status_code == 200:
                agent_data = response.json()
                logger.info(f"Retrieved agent details for {agent_id}")
                return agent_data

            logger.error(f"Failed to get agent details: {response.status_code}")
            return None

        except Exception as e:
            logger.error(f"Error getting agent details: {str(e)}")
            return None

    async def text_to_speech(self, text: str, voice_id: str) -> Optional[bytes]:
        """
        Convert text to speech (for testing)
        """
        try:
            url = f"{self.base_url}/text-to-speech/{voice_id}"

            payload = {
                "text": text,
                "voice_settings": {
                    "stability": 0.5,
                    "similarity_boost": 0.5
                }
            }

            response = requests.post(
                url,
                headers={**self.headers, "Content-Type": "application/json"},
                json=payload,
                timeout=30
            )

            if response.status_code == 200:
                logger.info(f"Generated TTS audio for text: {text[:50]}...")
                return response.content

            logger.error(f"TTS failed: {response.status_code}")
            return None

        except Exception as e:
            logger.error(f"Error in text-to-speech: {str(e)}")
            return None


# Create service instance
elevenlabs_service = ElevenLabsService()


async def get_conversation_url(agent_id: str) -> Optional[str]:
    """
    Main function to get conversation WebSocket URL
    """
    return await elevenlabs_service.get_conversation_url(agent_id)


async def get_signed_url(agent_id: str) -> Optional[str]:
    """
    Main function to get signed URL for private agents
    """
    return await elevenlabs_service.get_signed_url(agent_id)


async def list_voices() -> Optional[list]:
    """
    Main function to list available voices
    """
    return await elevenlabs_service.list_voices()


async def create_agent(agent_config: dict) -> Optional[str]:
    """
    Main function to create new agent
    """
    return await elevenlabs_service.create_agent(agent_config)


async def get_agent_details(agent_id: str) -> Optional[dict]:
    """
    Main function to get agent details
    """
    return await elevenlabs_service.get_agent_details(agent_id)
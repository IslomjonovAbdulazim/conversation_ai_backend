from google.cloud import vision
import logging
import os
from app.config import settings

logger = logging.getLogger(__name__)


class VisionService:
    def __init__(self):
        self.client = None
        self._initialize_client()

    def _initialize_client(self):
        """Initialize Google Vision client"""
        try:
            # Set the API key as environment variable
            os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = settings.google_vision_api_key
            self.client = vision.ImageAnnotatorClient()
            logger.info("Google Vision client initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Google Vision client: {str(e)}")
            self.client = None

    def extract_text(self, image_content: bytes) -> list:
        """Extract text from image using Google Vision API"""
        try:
            if not self.client:
                raise Exception("Google Vision client not initialized")

            image = vision.Image(content=image_content)
            response = self.client.text_detection(image=image)

            if response.error.message:
                raise Exception(f"Google Vision API error: {response.error.message}")

            extracted_words = []

            if response.text_annotations:
                # Skip the first annotation (full text) and process individual words
                for annotation in response.text_annotations[1:]:
                    text = annotation.description.strip()

                    # Filter out non-alphabetic text and very short words
                    if text.isalpha() and len(text) > 2:
                        extracted_words.append({
                            "text": text.lower(),
                            "confidence": 0.9  # Google Vision doesn't provide word-level confidence
                        })

            return extracted_words[:20]  # Return max 20 words

        except Exception as e:
            logger.error(f"Error extracting text from image: {str(e)}")
            return []


# Create service instance
vision_service = VisionService()
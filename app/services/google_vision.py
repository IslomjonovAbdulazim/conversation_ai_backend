from google.cloud import vision
from fastapi import UploadFile
import io
import re
import logging
from typing import List

from app.config import settings

logger = logging.getLogger(__name__)


class GoogleVisionService:

    def __init__(self):
        self.client = None

    def get_client(self):
        """Initialize Vision client with API key"""
        if self.client is None:
            # Use requests to call Google Vision API directly with API key
            import requests
            self.use_api_key = True
            return None

        return self.client

    async def call_vision_api_with_key(self, image_content: bytes):
        """Call Google Vision API directly using API key"""
        import requests
        import base64

        url = f"https://vision.googleapis.com/v1/images:annotate?key={settings.google_vision_api_key}"

        # Encode image to base64
        image_base64 = base64.b64encode(image_content).decode('utf-8')

        payload = {
            "requests": [
                {
                    "image": {
                        "content": image_base64
                    },
                    "features": [
                        {
                            "type": "TEXT_DETECTION",
                            "maxResults": 50
                        }
                    ]
                }
            ]
        }

        response = requests.post(url, json=payload, timeout=30)
        response.raise_for_status()

        return response.json()

    async def extract_text_from_image(self, image_file: UploadFile) -> List[str]:
        """
        Extract English text from image and return list of words
        """
        try:
            # Read image content
            content = await image_file.read()

            # Call Google Vision API with API key
            response_data = await self.call_vision_api_with_key(content)

            # Check for errors
            if "error" in response_data:
                logger.error(f"Google Vision API error: {response_data['error']}")
                return []

            responses = response_data.get("responses", [])
            if not responses:
                logger.info("No response from Google Vision API")
                return []

            text_annotations = responses[0].get("textAnnotations", [])

            if not text_annotations:
                logger.info("No text detected in image")
                return []

            # First annotation contains all detected text
            full_text = text_annotations[0].get("description", "")

            # Extract English words
            english_words = self.extract_english_words(full_text)

            logger.info(f"Extracted {len(english_words)} English words from image")
            return english_words

        except Exception as e:
            logger.error(f"Error extracting text from image: {str(e)}")
            return []

    def extract_english_words(self, text: str) -> List[str]:
        """
        Extract English words from text, filter out common words
        """
        try:
            # Extract words using regex (letters only, 3+ characters)
            words = re.findall(r'\b[a-zA-Z]{3,}\b', text)

            # Convert to lowercase and remove duplicates
            words = [word.lower() for word in words]
            unique_words = list(set(words))

            # Common words to exclude (stop words)
            common_words = {
                'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'any', 'can',
                'had', 'her', 'was', 'one', 'our', 'out', 'day', 'get', 'has', 'him',
                'his', 'how', 'man', 'new', 'now', 'old', 'see', 'two', 'way', 'who',
                'boy', 'did', 'its', 'let', 'put', 'say', 'she', 'too', 'use', 'may',
                'will', 'with', 'have', 'this', 'that', 'they', 'from', 'been', 'into',
                'what', 'were', 'said', 'each', 'which', 'their', 'time', 'would', 'there',
                'could', 'other', 'after', 'first', 'well', 'water', 'long', 'little',
                'very', 'when', 'much', 'before', 'here', 'through', 'just', 'form',
                'sentence', 'great', 'think', 'where', 'help', 'too', 'line', 'right',
                'tell', 'does', 'even', 'back', 'good', 'also', 'around', 'came'
            }

            # Filter out common words and very short words
            filtered_words = [
                word for word in unique_words
                if word not in common_words and len(word) >= 4
            ]

            # Limit to 20 words to avoid overwhelming
            return filtered_words[:20]

        except Exception as e:
            logger.error(f"Error extracting English words: {str(e)}")
            return []


# Create service instance
google_vision_service = GoogleVisionService()


async def extract_text_from_image(image_file: UploadFile) -> List[str]:
    """
    Main function to extract text from image
    """
    return await google_vision_service.extract_text_from_image(image_file)
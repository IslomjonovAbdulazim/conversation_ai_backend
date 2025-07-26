import requests
import base64
import logging
from app.config import settings

logger = logging.getLogger(__name__)


class VisionService:
    def __init__(self):
        self.api_key = settings.google_vision_api_key
        self.base_url = "https://vision.googleapis.com/v1/images:annotate"

    def extract_text(self, image_content: bytes) -> list:
        """Extract text from image using Google Vision API with API key"""
        try:
            if not self.api_key:
                raise Exception("Google Vision API key not configured")

            # Encode image to base64
            image_base64 = base64.b64encode(image_content).decode('utf-8')

            # Prepare request payload
            payload = {
                "requests": [
                    {
                        "image": {
                            "content": image_base64
                        },
                        "features": [
                            {
                                "type": "TEXT_DETECTION",
                                "maxResults": 20
                            }
                        ]
                    }
                ]
            }

            # Make API request
            url = f"{self.base_url}?key={self.api_key}"
            response = requests.post(url, json=payload, timeout=30)
            response.raise_for_status()

            result = response.json()

            if 'responses' not in result or not result['responses']:
                return []

            response_data = result['responses'][0]

            if 'error' in response_data:
                raise Exception(f"Google Vision API error: {response_data['error']['message']}")

            extracted_words = []

            if 'textAnnotations' in response_data:
                # Skip the first annotation (full text) and process individual words
                for annotation in response_data['textAnnotations'][1:]:
                    text = annotation['description'].strip()

                    # Filter out non-alphabetic text and very short words
                    if text.isalpha() and len(text) > 2:
                        extracted_words.append({
                            "text": text.lower(),
                            "confidence": 0.9  # Google Vision doesn't provide word-level confidence in this format
                        })

            return extracted_words[:20]  # Return max 20 words

        except requests.exceptions.RequestException as e:
            logger.error(f"HTTP error in Google Vision API: {str(e)}")
            return []
        except Exception as e:
            logger.error(f"Error extracting text from image: {str(e)}")
            return []


# Create service instance
vision_service = VisionService()
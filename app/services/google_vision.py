# app/services/google_vision.py - OPTIMIZED WITH AI WORD FILTERING
import re
import base64
import logging
from typing import List
from PIL import Image
import io
import requests
from fastapi import UploadFile, HTTPException, status
from app.config import settings

logger = logging.getLogger(__name__)


class GoogleVisionService:
    def __init__(self):
        self.client = None
        # Minimal stop words - let AI do the smart filtering
        self.basic_stop_words = {
            'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can', 'had', 'her', 'was', 'one', 'our',
            'out', 'day', 'get', 'has', 'him', 'his', 'how', 'man', 'new', 'now', 'old', 'see', 'two', 'way',
            'who', 'boy', 'did', 'its', 'let', 'put', 'say', 'she', 'too', 'use', 'will', 'with', 'have',
            'this', 'that', 'they', 'from', 'been', 'into', 'what', 'were', 'said', 'each', 'which', 'their',
            'time', 'would', 'there', 'could', 'other', 'after', 'first', 'well', 'very', 'when', 'much',
            'before', 'here', 'through', 'just', 'think', 'where', 'also', 'good', 'come', 'work', 'take',
            'make', 'know', 'back', 'help', 'give', 'tell', 'does', 'even', 'right', 'same'
        }

    def convert_heic_to_jpeg(self, image_content: bytes) -> bytes:
        """Convert HEIC/HEIF to JPEG if needed"""
        try:
            image = Image.open(io.BytesIO(image_content))

            # Convert to RGB if necessary
            if image.mode != 'RGB':
                image = image.convert('RGB')

            # Save as JPEG with good quality
            output = io.BytesIO()
            image.save(output, format='JPEG', quality=85, optimize=True)
            return output.getvalue()

        except Exception as e:
            logger.warning(f"Could not convert image format: {e}")
            return image_content

    def resize_image_if_needed(self, image_content: bytes, max_size: int = 4 * 1024 * 1024) -> bytes:
        """Resize image if it's too large for Vision API"""
        if len(image_content) <= max_size:
            return image_content

        try:
            image = Image.open(io.BytesIO(image_content))

            # Calculate new size to fit within max_size
            width, height = image.size
            ratio = min(1920 / width, 1080 / height, 1.0)  # Max 1920x1080

            new_width = int(width * ratio)
            new_height = int(height * ratio)

            resized_image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)

            output = io.BytesIO()
            resized_image.save(output, format='JPEG', quality=80, optimize=True)
            return output.getvalue()

        except Exception as e:
            logger.error(f"Failed to resize image: {e}")
            return image_content

    async def call_vision_api_with_key(self, image_content: bytes):
        """Call Google Vision API directly using API key with improved error handling"""
        try:
            # Convert and resize image if needed
            processed_image = self.convert_heic_to_jpeg(image_content)
            processed_image = self.resize_image_if_needed(processed_image)

            url = f"https://vision.googleapis.com/v1/images:annotate?key={settings.google_vision_api_key}"

            # Encode to base64
            image_base64 = base64.b64encode(processed_image).decode('utf-8')

            payload = {
                "requests": [{
                    "image": {"content": image_base64},
                    "features": [{"type": "TEXT_DETECTION", "maxResults": 50}]
                }]
            }

            response = requests.post(url, json=payload, timeout=30)

            if response.status_code != 200:
                error_text = response.text
                logger.error(f"Vision API error {response.status_code}: {error_text}")
                return {"error": f"Vision API error: {response.status_code}"}

            return response.json()

        except requests.exceptions.Timeout:
            logger.error("Vision API timeout")
            return {"error": "Vision API timeout"}
        except Exception as e:
            logger.error(f"Vision API call failed: {e}")
            return {"error": str(e)}

    def extract_all_words(self, text: str) -> List[str]:
        """Extract ALL English words from text - minimal filtering"""
        if not text:
            return []

        try:
            # Extract words using regex (letters only, 3+ characters)
            all_words = re.findall(r'\b[a-zA-Z]{3,}\b', text)

            # Convert to lowercase and remove duplicates while preserving order
            seen = set()
            unique_words = []
            for word in all_words:
                word_lower = word.lower()
                if word_lower not in seen:
                    # Only skip very basic stop words and very short/long words
                    if (word_lower not in self.basic_stop_words and
                            3 <= len(word_lower) <= 15):
                        unique_words.append(word_lower)
                        seen.add(word_lower)

            logger.info(f"Extracted {len(unique_words)} raw words from vision")
            return unique_words

        except Exception as e:
            logger.error(f"Error extracting words: {str(e)}")
            return []

    async def extract_text_from_image(self, image_file: UploadFile) -> List[str]:
        """
        Extract text from image and return ALL meaningful words (for AI filtering)
        """
        try:
            # Read image content
            content = await image_file.read()

            # Call Google Vision API
            response_data = await self.call_vision_api_with_key(content)

            # Check for errors
            if "error" in response_data:
                logger.error(f"Google Vision API error: {response_data['error']}")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Vision API error: {response_data['error']}"
                )

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

            # Extract ALL meaningful words - let AI do the smart filtering
            all_words = self.extract_all_words(full_text)

            logger.info(f"Extracted {len(all_words)} words for AI filtering")
            return all_words

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error extracting text from image: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Image processing failed: {str(e)}"
            )


# Create service instance
google_vision_service = GoogleVisionService()


async def extract_text_from_image(image_file: UploadFile) -> List[str]:
    """Main function to extract text from image"""
    return await google_vision_service.extract_text_from_image(image_file)
# app/services/openai_service.py - Simplified and optimized
import openai
import logging
import asyncio
from typing import Optional, List, Dict

from app.config import settings

logger = logging.getLogger(__name__)


class OpenAIService:
    def __init__(self):
        # Initialize OpenAI client
        self.client = openai.OpenAI(api_key=settings.openai_api_key)

    async def translate_to_uzbek(self, english_word: str) -> str:
        """
        Translate English word to Uzbek using OpenAI
        """
        try:
            prompt = f"""
            Translate this English word to Uzbek. Return only the Uzbek translation, nothing else.

            English word: {english_word}

            Uzbek translation:
            """

            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a professional English-Uzbek translator. Provide only the most accurate and commonly used Uzbek translation for the given English word."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                max_tokens=50,
                temperature=0.1
            )

            translation = response.choices[0].message.content.strip()

            # Clean up the response (remove any extra text)
            if ":" in translation:
                translation = translation.split(":")[-1].strip()

            # Remove quotes if present
            translation = translation.strip('"\'')

            logger.info(f"Translated '{english_word}' to '{translation}'")
            return translation

        except Exception as e:
            logger.error(f"Error translating word '{english_word}': {str(e)}")
            # Fallback: return the original word if translation fails
            return english_word

    async def batch_translate_to_uzbek(self, english_words: List[str]) -> Dict[str, str]:
        """
        Translate multiple English words to Uzbek in batches
        """
        try:
            if not english_words:
                return {}

            # Process in chunks of 20 to avoid token limits and rate limiting
            all_translations = {}

            for i in range(0, len(english_words), 20):
                chunk = english_words[i:i + 20]
                chunk_translations = await self._translate_chunk(chunk)
                all_translations.update(chunk_translations)

                # Small delay between chunks to avoid rate limiting
                if i + 20 < len(english_words):
                    await asyncio.sleep(0.5)

            return all_translations

        except Exception as e:
            logger.error(f"Error in batch translation: {str(e)}")
            return {}

    async def _translate_chunk(self, words_chunk: List[str]) -> Dict[str, str]:
        """
        Translate a chunk of words (max 20) in one API call
        """
        try:
            # Format words as numbered list for better parsing
            words_list = "\n".join([f"{i + 1}. {word}" for i, word in enumerate(words_chunk)])

            prompt = f"""
            Translate these English words to Uzbek. Return as JSON format only.

            Words to translate:
            {words_list}

            Return in this EXACT JSON format (no extra text):
            {{
                "{words_chunk[0]}": "translation1",
                "{words_chunk[1] if len(words_chunk) > 1 else 'example'}": "translation2"
            }}
            """

            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": "Professional English-Uzbek translator. Return only valid JSON with translations."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                max_tokens=600,
                temperature=0.1
            )

            # Parse JSON response
            import json
            response_text = response.choices[0].message.content.strip()

            # Clean response (remove markdown formatting if present)
            if response_text.startswith("```json"):
                response_text = response_text.replace("```json", "").replace("```", "").strip()
            elif response_text.startswith("```"):
                response_text = response_text.replace("```", "").strip()

            translations = json.loads(response_text)

            logger.info(f"Batch translated {len(translations)} words in one call")
            return translations

        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing error in batch translation: {str(e)}")
            return {}
        except Exception as e:
            logger.error(f"Error in translation chunk: {str(e)}")
            return {}

    async def generate_example_sentence(self, english_word: str, uzbek_translation: str) -> str:
        """
        Generate example sentence for English word
        """
        try:
            prompt = f"""
            Create a simple, clear example sentence using the English word "{english_word}" (which means "{uzbek_translation}" in Uzbek).

            Requirements:
            - Use everyday, simple English
            - Make the sentence practical and useful for language learners
            - Keep it under 15 words
            - Make sure the meaning of "{english_word}" is clear from context

            Example sentence:
            """

            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": "You are an English teacher creating example sentences for vocabulary learning. Make sentences simple, clear, and practical for everyday use."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                max_tokens=100,
                temperature=0.3
            )

            example_sentence = response.choices[0].message.content.strip()

            # Clean up the response
            if example_sentence.startswith('"'):
                example_sentence = example_sentence.strip('"')
            if example_sentence.startswith("'"):
                example_sentence = example_sentence.strip("'")

            # Remove any prefixes
            prefixes = ["example sentence:", "example:", "sentence:", "here's an example:"]
            for prefix in prefixes:
                if example_sentence.lower().startswith(prefix):
                    example_sentence = example_sentence[len(prefix):].strip()

            # Ensure sentence ends with proper punctuation
            if not example_sentence.endswith(('.', '!', '?')):
                example_sentence += '.'

            logger.info(f"Generated example for '{english_word}': {example_sentence}")
            return example_sentence

        except Exception as e:
            logger.error(f"Error generating example for '{english_word}': {str(e)}")
            # Fallback: create simple sentence
            return f"I use {english_word} every day."


# Create service instance
openai_service = OpenAIService()


# Export functions for backwards compatibility
async def translate_to_uzbek(english_word: str) -> str:
    """Main function to translate English word to Uzbek"""
    return await openai_service.translate_to_uzbek(english_word)


async def generate_example_sentence(english_word: str, uzbek_translation: str) -> str:
    """Main function to generate example sentence"""
    return await openai_service.generate_example_sentence(english_word, uzbek_translation)
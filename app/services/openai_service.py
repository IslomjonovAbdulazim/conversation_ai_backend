# app/services/openai_service.py - Enhanced with multiple translation options
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

    async def get_multiple_translation_options(self, english_word: str) -> List[dict]:
        """
        Get multiple translation options considering different parts of speech and meanings
        """
        try:
            prompt = f"""
            I need MULTIPLE Uzbek translations for the English word "{english_word}". Be creative and think of ALL possible ways this word can be used!

            Consider these examples of how words can have multiple meanings:
            - "book": kitob (noun), band qilmoq (reserve), buyurtma bermoq (order), ro'yxatga olmoq (register)
            - "run": yugurmoq (jog), boshqarmoq (manage), ishlash (operate), qochmoq (flee)
            - "light": yorug'lik (illumination), yengil (not heavy), yoqmoq (turn on), och (pale color)
            - "play": o'ynamoq (game), ijro etmoq (music), spektakl (theater), rol o'ynamoq (act)

            For "{english_word}", think about:
            ✓ Different parts of speech (noun, verb, adjective, adverb)
            ✓ Multiple meanings within the same part of speech
            ✓ Common phrases and expressions
            ✓ Formal vs informal usage
            ✓ Technical vs everyday meanings
            ✓ Regional variations if any

            IMPORTANT: I want AT LEAST 3-5 different options. Don't be conservative - explore ALL possible meanings!

            Format as JSON array:
            [
                {{"translation": "option1", "part_of_speech": "noun", "meaning": "explanation", "confidence": 0.95}},
                {{"translation": "option2", "part_of_speech": "verb", "meaning": "explanation", "confidence": 0.90}},
                {{"translation": "option3", "part_of_speech": "verb", "meaning": "different context", "confidence": 0.85}},
                {{"translation": "option4", "part_of_speech": "adjective", "meaning": "another meaning", "confidence": 0.80}}
            ]

            Give me as many relevant options as possible! Return only the JSON array.
            """

            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a creative English-Uzbek translator who loves exploring ALL possible meanings and translations of words. Your goal is to provide comprehensive, diverse translation options. Never give just one option - always explore multiple meanings, parts of speech, and contexts. Be generous with options!"
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                max_tokens=800,  # Increased for more options
                temperature=0.7  # Higher temperature for more creativity
            )

            # Parse JSON response
            import json
            response_text = response.choices[0].message.content.strip()

            # Clean response (remove markdown formatting if present)
            if response_text.startswith("```json"):
                response_text = response_text.replace("```json", "").replace("```", "").strip()
            elif response_text.startswith("```"):
                response_text = response_text.replace("```", "").strip()

            translation_options = json.loads(response_text)

            # Validate and clean the options
            cleaned_options = []
            for option in translation_options:
                if isinstance(option, dict) and "translation" in option:
                    cleaned_option = {
                        "translation": option.get("translation", "").strip(),
                        "confidence": float(option.get("confidence", 0.8)),
                        "part_of_speech": option.get("part_of_speech", "").strip(),
                        "meaning": option.get("meaning", "").strip()
                    }
                    if cleaned_option["translation"]:
                        cleaned_options.append(cleaned_option)

            # If we got fewer than 2 options, try to get more with a follow-up
            if len(cleaned_options) < 2:
                logger.warning(f"Only got {len(cleaned_options)} options for '{english_word}', trying to get more")
                additional_options = await self._get_additional_translations(english_word, cleaned_options)
                cleaned_options.extend(additional_options)

            logger.info(f"Generated {len(cleaned_options)} translation options for '{english_word}'")
            return cleaned_options

        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing error for translation options: {str(e)}")
            # Try a simpler approach
            return await self._get_fallback_translations(english_word)
        except Exception as e:
            logger.error(f"Error getting translation options for '{english_word}': {str(e)}")
            return await self._get_fallback_translations(english_word)

    async def _get_additional_translations(self, english_word: str, existing_options: List[dict]) -> List[dict]:
        """Get additional translations if the first attempt didn't provide enough"""
        try:
            existing_translations = [opt["translation"] for opt in existing_options]
            existing_text = ", ".join(existing_translations) if existing_translations else "none"

            prompt = f"""
            I already have these translations for "{english_word}": {existing_text}

            Give me MORE different Uzbek translations for "{english_word}". Think of:
            - Synonyms and alternative words
            - Different contexts and situations
            - Formal vs colloquial usage
            - Technical or specialized meanings
            - Related expressions or phrases

            Return 2-4 additional options as JSON array:
            [{{"translation": "new_option1", "part_of_speech": "type", "meaning": "context", "confidence": 0.8}}]
            """

            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "Creative translator providing diverse translation alternatives."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=400,
                temperature=0.8
            )

            import json
            response_text = response.choices[0].message.content.strip()
            if response_text.startswith("```json"):
                response_text = response_text.replace("```json", "").replace("```", "").strip()
            elif response_text.startswith("```"):
                response_text = response_text.replace("```", "").strip()

            additional_options = json.loads(response_text)

            cleaned_additional = []
            for option in additional_options:
                if isinstance(option, dict) and "translation" in option:
                    translation = option.get("translation", "").strip()
                    if translation and translation not in existing_translations:
                        cleaned_additional.append({
                            "translation": translation,
                            "confidence": float(option.get("confidence", 0.75)),
                            "part_of_speech": option.get("part_of_speech", "").strip(),
                            "meaning": option.get("meaning", "").strip()
                        })

            return cleaned_additional

        except Exception as e:
            logger.error(f"Error getting additional translations: {str(e)}")
            return []

    async def _get_fallback_translations(self, english_word: str) -> List[dict]:
        """Fallback method using simpler prompts"""
        try:
            # Get basic translation first
            basic_translation = await self.translate_to_uzbek(english_word)

            options = [{
                "translation": basic_translation,
                "confidence": 0.8,
                "part_of_speech": "unknown",
                "meaning": "primary translation"
            }]

            # Try to get one more alternative
            try:
                prompt = f"Give me an alternative Uzbek translation for '{english_word}' that's different from '{basic_translation}'. Just return the single word or phrase."

                response = self.client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": "Translator providing alternative translations."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=50,
                    temperature=0.5
                )

                alternative = response.choices[0].message.content.strip()
                if alternative and alternative != basic_translation:
                    options.append({
                        "translation": alternative,
                        "confidence": 0.7,
                        "part_of_speech": "alternative",
                        "meaning": "alternative translation"
                    })

            except Exception:
                pass  # If alternative fails, just return the basic one

            return options

        except Exception as e:
            logger.error(f"Fallback translation failed: {str(e)}")
            return [{
                "translation": english_word,
                "confidence": 0.5,
                "part_of_speech": "unknown",
                "meaning": "translation failed"
            }]

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

    async def generate_quiz_question(self, words: list, quiz_type: str):
        """
        Generate quiz questions for different quiz types
        (Used by quiz router)
        """
        try:
            if quiz_type == "reading":
                # Generate reading comprehension passage
                prompt = f"""
                Create a short reading passage (3-4 sentences) using these words: {', '.join([w['word'] for w in words[:8]])}.

                Replace some of the words with blanks like this: _____.
                Make the passage natural and interesting.
                Return the passage and the correct answers.

                Format:
                PASSAGE: [passage with blanks]
                ANSWERS: [list of correct words for each blank]
                """

                response = self.client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {
                            "role": "system",
                            "content": "You are creating reading comprehension exercises for English language learners."
                        },
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    max_tokens=300,
                    temperature=0.5
                )

                return response.choices[0].message.content.strip()

            return None

        except Exception as e:
            logger.error(f"Error generating quiz question: {str(e)}")
            return None


# Create service instance
openai_service = OpenAIService()


# Export functions for backwards compatibility
async def translate_to_uzbek(english_word: str) -> str:
    """Main function to translate English word to Uzbek"""
    return await openai_service.translate_to_uzbek(english_word)


async def generate_example_sentence(english_word: str, uzbek_translation: str) -> str:
    """Main function to generate example sentence"""
    return await openai_service.generate_example_sentence(english_word, uzbek_translation)


async def generate_quiz_question(words: list, quiz_type: str):
    """Main function to generate quiz questions"""
    return await openai_service.generate_quiz_question(words, quiz_type)


async def get_multiple_translation_options(english_word: str) -> List[dict]:
    """Main function to get multiple translation options"""
    return await openai_service.get_multiple_translation_options(english_word)
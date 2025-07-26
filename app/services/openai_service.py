import openai
import logging
from typing import Optional, List, Dict  # Add Dict here

from app.config import settings
from app.config import settings

logger = logging.getLogger(__name__)


class OpenAIService:

    def __init__(self):
        # Initialize OpenAI client
        openai.api_key = settings.openai_api_key
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

            logger.info(f"Translated '{english_word}' to '{translation}'")
            return translation

        except Exception as e:
            logger.error(f"Error translating word '{english_word}': {str(e)}")
            # Fallback: return the original word if translation fails
            return english_word

    async def batch_translate_to_uzbek(self, english_words: List[str]) -> Dict[str, str]:
        """
        Translate multiple English words to Uzbek in a single API call
        """
        try:
            if not english_words:
                return {}

            # Limit batch size to prevent token limits
            if len(english_words) > 30:
                # Process in chunks of 30
                all_translations = {}
                for i in range(0, len(english_words), 30):
                    chunk = english_words[i:i + 30]
                    chunk_translations = await self._translate_chunk(chunk)
                    all_translations.update(chunk_translations)
                return all_translations
            else:
                return await self._translate_chunk(english_words)

        except Exception as e:
            logger.error(f"Error in batch translation: {str(e)}")
            return {}

    async def _translate_chunk(self, words_chunk: List[str]) -> Dict[str, str]:
        """
        Translate a chunk of words (max 30) in one API call
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
                "word1": "translation1",
                "word2": "translation2",
                "word3": "translation3"
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
                max_tokens=800,
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
            # Fallback to individual translations
            return {}
        except Exception as e:
            logger.error(f"Error in translation chunk: {str(e)}")
            return {}
    # Add this function to app/services/openai_service.py

    async def filter_best_vocabulary_words(self, word_list: List[str]) -> List[str]:
        """
        Adaptive word filtering based on text amount - more words for small text, selective for large text
        """
        try:
            if not word_list:
                return []

            word_count = len(word_list)
            unique_words = list(
                set([word.strip().lower() for word in word_list if word.strip() and len(word.strip()) >= 2]))

            # Adaptive strategy based on word count
            if word_count <= 15:
                # Small text: Take almost all words with minimal filtering
                return await self._minimal_filter(unique_words)
            elif word_count <= 40:
                # Medium text: Take most words with light filtering
                return await self._light_filter(unique_words)
            else:
                # Large text: Apply selective filtering for best vocabulary words
                return await self._selective_filter(unique_words)

        except Exception as e:
            logger.error(f"Error filtering vocabulary words: {str(e)}")
            # Fallback: return most words with basic cleanup
            return [word for word in word_list if word and len(word) >= 3][:50]

    async def _minimal_filter(self, word_list: List[str]) -> List[str]:
        """
        Minimal filtering for small text - take almost everything
        """
        try:
            words_text = ", ".join(word_list)

            prompt = f"""
            Clean up this word list by removing only obvious non-words, but keep most vocabulary including common words.

            Word list: {words_text}

            Remove only:
            - Single letters (a, b, c)
            - Numbers and abbreviations (123, etc, vs, etc.)
            - Non-English words
            - Obvious typos or fragments

            Keep everything else including common words like "the", "and", "with", etc.
            Users might want to learn these too.

            Return words comma-separated, no explanations:
            """

            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": "Clean word lists minimally, preserve most vocabulary for learning."
                    },
                    {"role": "user", "content": prompt}
                ],
                max_tokens=300,
                temperature=0.1
            )

            filtered_response = response.choices[0].message.content.strip()
            if filtered_response:
                filtered_words = [word.strip().lower() for word in filtered_response.split(",")]
                filtered_words = [word for word in filtered_words if word and len(word) >= 2]
                logger.info(f"Minimal filter: {len(word_list)} → {len(filtered_words)} words")
                return filtered_words

            return word_list  # Return original if AI fails

        except Exception as e:
            logger.error(f"Error in minimal filter: {str(e)}")
            return word_list

    async def _light_filter(self, word_list: List[str]) -> List[str]:
        """
        Light filtering for medium text - keep most useful words
        """
        try:
            words_text = ", ".join(word_list)

            prompt = f"""
            From this word list, select the most useful words for English learning.

            Word list: {words_text}

            Selection criteria:
            - Keep useful vocabulary words
            - Include some common words that learners need
            - Remove obvious junk (single letters, numbers, fragments)
            - Prefer complete words over word fragments
            - Return 20-35 words maximum

            Return words comma-separated, no explanations:
            """

            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": "Select useful vocabulary while being inclusive of common words learners need."
                    },
                    {"role": "user", "content": prompt}
                ],
                max_tokens=400,
                temperature=0.2
            )

            filtered_response = response.choices[0].message.content.strip()
            if filtered_response:
                filtered_words = [word.strip().lower() for word in filtered_response.split(",")]
                filtered_words = [word for word in filtered_words if word and len(word) >= 2]
                logger.info(f"Light filter: {len(word_list)} → {len(filtered_words)} words")
                return filtered_words[:35]  # Cap at 35

            return word_list[:30]  # Fallback

        except Exception as e:
            logger.error(f"Error in light filter: {str(e)}")
            return word_list[:30]

    async def _selective_filter(self, word_list: List[str]) -> List[str]:
        """
        Selective filtering for large text - focus on best vocabulary words
        """
        try:
            words_text = ", ".join(word_list)

            prompt = f"""
            From this large word list, select the BEST vocabulary words for English learning.

            Word list: {words_text}

            Selection criteria:
            - Prioritize useful vocabulary words
            - Include some common words (not just advanced words)
            - Focus on words learners will actually use
            - Skip obvious junk, fragments, and repeated words
            - Include nouns, verbs, adjectives that are practical
            - Return 25-45 words maximum

            Return words comma-separated, no explanations:
            """

            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": "Expert vocabulary teacher selecting the most valuable words from large text."
                    },
                    {"role": "user", "content": prompt}
                ],
                max_tokens=500,
                temperature=0.3
            )

            filtered_response = response.choices[0].message.content.strip()
            if filtered_response:
                filtered_words = [word.strip().lower() for word in filtered_response.split(",")]
                filtered_words = [word for word in filtered_words if word and len(word) >= 2]
                logger.info(f"Selective filter: {len(word_list)} → {len(filtered_words)} words")
                return filtered_words[:45]  # Cap at 45

            return word_list[:35]  # Fallback

        except Exception as e:
            logger.error(f"Error in selective filter: {str(e)}")
            return word_list[:35]

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
        (This will be used in quiz router)
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


async def translate_to_uzbek(english_word: str) -> str:
    """
    Main function to translate English word to Uzbek
    """
    return await openai_service.translate_to_uzbek(english_word)


async def generate_example_sentence(english_word: str, uzbek_translation: str) -> str:
    """
    Main function to generate example sentence
    """
    return await openai_service.generate_example_sentence(english_word, uzbek_translation)


async def generate_quiz_question(words: list, quiz_type: str):
    """
    Main function to generate quiz questions
    """
    return await openai_service.generate_quiz_question(words, quiz_type)

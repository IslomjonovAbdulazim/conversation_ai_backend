import openai
import logging
from typing import Optional, List  # Add List here

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

    # Add this function to app/services/openai_service.py

    async def filter_best_vocabulary_words(self, word_list: List[str]) -> List[str]:
        """
        Use AI to select the best vocabulary words from extracted text
        """
        try:
            if not word_list:
                return []

            # Join words for prompt
            words_text = ", ".join(word_list)

            prompt = f"""
            From this list of words extracted from an image, select the 10-30 BEST vocabulary words for English learning.

            Word list: {words_text}

            Selection criteria:
            - Choose words that are USEFUL for vocabulary building
            - Skip very common words (a, the, and, etc.)
            - Skip proper nouns, numbers, and abbreviations  
            - Prefer words that are:
              * Concrete nouns (computer, kitchen, business)
              * Useful verbs (develop, manage, create)
              * Important adjectives (efficient, professional, modern)
              * Academic/professional terms
            - Skip: articles, prepositions, pronouns, very basic words
            - Return 10-30 words maximum, prioritize quality over quantity

            Return ONLY the selected words, comma-separated, no explanations:
            """

            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert English vocabulary teacher. Select only the most valuable words for language learning from any given list."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                max_tokens=200,
                temperature=0.3
            )

            filtered_response = response.choices[0].message.content.strip()

            # Parse response into list
            if filtered_response:
                filtered_words = [word.strip().lower() for word in filtered_response.split(",")]
                # Remove any empty strings and ensure uniqueness
                filtered_words = list(set([word for word in filtered_words if word and len(word) >= 3]))

                logger.info(
                    f"AI filtered {len(word_list)} words down to {len(filtered_words)} quality vocabulary words")
                return filtered_words[:30]  # Hard limit of 30

            return []

        except Exception as e:
            logger.error(f"Error filtering vocabulary words: {str(e)}")
            # Fallback: return first 20 words if AI filtering fails
            return word_list[:20]

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

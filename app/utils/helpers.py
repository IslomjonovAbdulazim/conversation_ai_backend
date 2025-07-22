import re
import random
import string
from typing import List, Optional
from datetime import datetime, timezone
import logging

logger = logging.getLogger(__name__)


def validate_email(email: str) -> bool:
    """
    Validate email format
    """
    email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return re.match(email_pattern, email) is not None


def validate_english_word(word: str) -> bool:
    """
    Validate if word contains only English letters
    """
    return bool(re.match(r'^[a-zA-Z\s\-\']+$', word.strip()))


def clean_text(text: str) -> str:
    """
    Clean and normalize text input
    """
    if not text:
        return ""

    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text.strip())

    # Remove special characters except basic punctuation
    text = re.sub(r'[^\w\s\.\,\!\?\-\']', '', text)

    return text


def generate_session_id(prefix: str = "session") -> str:
    """
    Generate unique session ID
    """
    timestamp = int(datetime.now(timezone.utc).timestamp())
    random_part = ''.join(random.choices(string.ascii_lowercase + string.digits, k=8))
    return f"{prefix}_{timestamp}_{random_part}"


def format_duration(seconds: int) -> str:
    """
    Format duration in seconds to human readable format
    """
    if seconds < 60:
        return f"{seconds}s"
    elif seconds < 3600:
        minutes = seconds // 60
        remaining_seconds = seconds % 60
        return f"{minutes}m {remaining_seconds}s"
    else:
        hours = seconds // 3600
        remaining_minutes = (seconds % 3600) // 60
        return f"{hours}h {remaining_minutes}m"


def calculate_accuracy(correct: int, total: int) -> int:
    """
    Calculate accuracy percentage
    """
    if total == 0:
        return 0
    return round((correct / total) * 100)


def extract_quiz_words(words: List[dict], max_count: int = 10) -> List[dict]:
    """
    Extract and shuffle words for quiz, ensuring they have example sentences
    """
    # Filter words that have example sentences
    complete_words = [word for word in words if word.get('example_sentence')]

    # Shuffle and limit
    random.shuffle(complete_words)
    return complete_words[:max_count]


def generate_wrong_options(correct_answer: str, all_options: List[str], count: int = 2) -> List[str]:
    """
    Generate wrong options for multiple choice questions
    """
    # Remove the correct answer from options
    wrong_options = [opt for opt in all_options if opt.lower() != correct_answer.lower()]

    # Shuffle and return requested count
    random.shuffle(wrong_options)
    return wrong_options[:count]


def is_similar_word(word1: str, word2: str, threshold: float = 0.8) -> bool:
    """
    Check if two words are similar (to avoid confusing quiz options)
    """
    word1, word2 = word1.lower(), word2.lower()

    # If words are too similar in length and content, avoid using together
    if abs(len(word1) - len(word2)) <= 1:
        # Simple character overlap check
        common_chars = set(word1) & set(word2)
        overlap_ratio = len(common_chars) / max(len(set(word1)), len(set(word2)))
        return overlap_ratio > threshold

    return False


def filter_quiz_options(correct_word: str, all_words: List[str], count: int = 2) -> List[str]:
    """
    Filter quiz options to avoid too similar words
    """
    wrong_options = []

    for word in all_words:
        if word.lower() != correct_word.lower():
            # Check if word is not too similar to correct answer
            if not is_similar_word(correct_word, word):
                wrong_options.append(word)

            if len(wrong_options) >= count:
                break

    # If we don't have enough different words, just use random ones
    if len(wrong_options) < count:
        remaining = [w for w in all_words if w.lower() != correct_word.lower()]
        random.shuffle(remaining)
        wrong_options.extend(remaining[:count - len(wrong_options)])

    return wrong_options[:count]


def normalize_answer(answer: str) -> str:
    """
    Normalize user answer for comparison
    """
    if not answer:
        return ""

    # Convert to lowercase and strip whitespace
    normalized = answer.lower().strip()

    # Remove extra spaces
    normalized = re.sub(r'\s+', ' ', normalized)

    # Remove common punctuation
    normalized = re.sub(r'[.,!?;:]', '', normalized)

    return normalized


def get_word_difficulty(word: str) -> str:
    """
    Estimate word difficulty based on length and complexity
    """
    length = len(word)

    if length <= 4:
        return "easy"
    elif length <= 7:
        return "medium"
    else:
        return "hard"


def shuffle_letters(word: str) -> str:
    """
    Shuffle letters of a word for anagram game
    """
    letters = list(word.lower())

    # Ensure the shuffled version is different from original
    shuffled = letters.copy()
    attempts = 0
    while ''.join(shuffled) == word.lower() and attempts < 10:
        random.shuffle(shuffled)
        attempts += 1

    return ''.join(shuffled)


def format_quiz_time(seconds: int) -> str:
    """
    Format quiz time for display
    """
    if seconds < 10:
        return f"{seconds}s (Fast!)"
    elif seconds < 30:
        return f"{seconds}s (Good)"
    else:
        return f"{seconds}s"


def log_user_action(user_id: int, action: str, details: Optional[dict] = None):
    """
    Log user actions for analytics
    """
    log_message = f"User {user_id}: {action}"
    if details:
        log_message += f" - {details}"

    logger.info(log_message)


def safe_get(dictionary: dict, key: str, default=None):
    """
    Safely get value from dictionary
    """
    try:
        return dictionary.get(key, default)
    except (AttributeError, TypeError):
        return default


def truncate_text(text: str, max_length: int = 100) -> str:
    """
    Truncate text to specified length
    """
    if not text or len(text) <= max_length:
        return text

    return text[:max_length - 3] + "..."


def is_uzbek_text(text: str) -> bool:
    """
    Simple check if text contains Uzbek characters
    """
    # Uzbek uses Latin script with some special characters
    uzbek_chars = set("ōo'ğgʻqQG'")
    return any(char in uzbek_chars for char in text)


def format_percentage(value: float) -> str:
    """
    Format percentage for display
    """
    return f"{value:.1f}%"
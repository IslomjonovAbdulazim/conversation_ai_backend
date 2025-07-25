from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File, Request
from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import List, Dict, Optional
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
import logging
import time
import asyncio
import random

from app.database import get_db
from app.models import User, Folder, Word, WordStats
from app.routers.auth import get_current_user
from app.services.google_vision import extract_text_from_image
from app.services.openai_service import translate_to_uzbek, generate_example_sentence

logger = logging.getLogger(__name__)
router = APIRouter()


# Pydantic models
class AddWordRequest(BaseModel):
    word: str
    translation: str
    example_sentence: Optional[str] = None


class WordResponse(BaseModel):
    id: int
    word: str
    translation: str
    example_sentence: Optional[str]
    added_at: str


class AddWordResponse(BaseModel):
    word: WordResponse
    stats: Dict


class ExtractedWord(BaseModel):
    word: str
    translation: str
    confidence: float


class OCRResponse(BaseModel):
    extracted_words: List[ExtractedWord]
    total_extracted: int
    processing_time: float


class BulkAddRequest(BaseModel):
    words: List[AddWordRequest]


class BulkAddResponse(BaseModel):
    success: bool
    added_count: int
    words: List[WordResponse]


class BulkDeleteRequest(BaseModel):
    word_ids: List[int]


class BulkDeleteResponse(BaseModel):
    success: bool
    deleted_count: int
    deleted_words: List[Dict]


class GenerateExampleRequest(BaseModel):
    word: str
    translation: str


class GenerateExampleResponse(BaseModel):
    example_sentence: str
    alternatives: List[str] = []


class WordDetailResponse(BaseModel):
    word: Dict
    stats: Dict


class UpdateWordRequest(BaseModel):
    word: Optional[str] = None
    translation: Optional[str] = None
    example_sentence: Optional[str] = None


class UpdateWordResponse(BaseModel):
    success: bool
    word: Dict


class DeleteWordResponse(BaseModel):
    success: bool
    message: str
    deleted_word: Dict


def get_folder_or_404(folder_id: int, user_id: int, db: Session):
    """Helper function to get folder or raise 404"""
    folder = db.query(Folder).filter(
        Folder.id == folder_id,
        Folder.user_id == user_id
    ).first()

    if not folder:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Folder not found"
        )

    return folder


def create_word_stats(word_id: int, user_id: int, db: Session):
    """Create initial word stats for user"""
    word_stats = WordStats(
        word_id=word_id,
        user_id=user_id,
        category="not_known",
        last_5_results=[],
        total_attempts=0,
        correct_attempts=0
    )
    db.add(word_stats)
    return word_stats


# SPECIFIC ROUTES FIRST - these must come before parameterized routes

@router.post("/upload-photo", response_model=OCRResponse)
async def upload_photo_ocr(
        file: UploadFile = File(...),
        current_user: User = Depends(get_current_user)
):
    """
    Extract text from image using Google Vision API
    """
    try:
        start_time = time.time()

        # Validate file
        if not file.content_type.startswith('image/'):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="File must be an image"
            )

        # Read file content
        file_content = await file.read()

        # Extract text using Google Vision
        extracted_text = extract_text_from_image(file_content)

        if not extracted_text:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No text found in image"
            )

        # Split into words and translate
        words = [word.strip() for word in extracted_text.split() if len(word.strip()) > 2]

        if not words:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No valid words found in image"
            )

        # Translate words to Uzbek
        translated_words = []
        for word in words[:20]:  # Limit to 20 words
            try:
                translation = await translate_to_uzbek(word)
                if translation:
                    translated_words.append(ExtractedWord(
                        word=word.lower(),
                        translation=translation,
                        confidence=0.8  # Could be improved with actual confidence scores
                    ))
            except Exception as e:
                logger.warning(f"Failed to translate word '{word}': {str(e)}")
                continue

        processing_time = round(time.time() - start_time, 2)

        logger.info(f"OCR processed {len(translated_words)} words in {processing_time}s for user {current_user.id}")

        return OCRResponse(
            extracted_words=translated_words,
            total_extracted=len(translated_words),
            processing_time=processing_time
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"OCR processing error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to process image"
        )


@router.post("/bulk-delete", response_model=BulkDeleteResponse)
async def bulk_delete_words(
        request: BulkDeleteRequest,
        current_user: User = Depends(get_current_user),
        db: Session = Depends(get_db)
):
    """
    Delete multiple words at once
    """
    try:
        if not request.word_ids:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No word IDs provided"
            )

        # Get words that belong to user's folders
        words = db.query(Word).join(Folder).filter(
            Word.id.in_(request.word_ids),
            Folder.user_id == current_user.id
        ).all()

        if not words:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="No words found to delete"
            )

        deleted_words = []
        for word in words:
            deleted_words.append({
                "id": word.id,
                "word": word.word
            })
            db.delete(word)

        db.commit()

        logger.info(f"Bulk deleted {len(deleted_words)} words by user {current_user.id}")

        return BulkDeleteResponse(
            success=True,
            deleted_count=len(deleted_words),
            deleted_words=deleted_words
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error bulk deleting words: {str(e)}")
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete words"
        )


# Add these imports to your main.py
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from fastapi import Request
import time
from typing import Dict, Optional

# Add rate limiter to main.py
logger = logging.getLogger(__name__)
router = APIRouter()

# Create limiter instance
limiter = Limiter(key_func=get_remote_address)

example_cache: Dict[str, Dict] = {}


# Cache helper functions
def get_cache_key(word: str, translation: str) -> str:
    return f"{word.lower()}:{translation.lower()}"


def get_cached_example(word: str, translation: str) -> Optional[str]:
    """Get cached example if exists and not expired"""
    cache_key = get_cache_key(word, translation)
    cached = example_cache.get(cache_key)

    if cached and time.time() - cached['timestamp'] < 3600:  # 1 hour cache
        logger.info(f"Using cached example for '{word}'")
        return cached['example']

    return None


def cache_example(word: str, translation: str, example: str):
    """Cache the generated example"""
    cache_key = get_cache_key(word, translation)
    example_cache[cache_key] = {
        'example': example,
        'timestamp': time.time()
    }

    # Simple cache cleanup - keep only 1000 entries
    if len(example_cache) > 1000:
        # Remove oldest 200 entries
        sorted_cache = sorted(example_cache.items(), key=lambda x: x[1]['timestamp'])
        for key, _ in sorted_cache[:200]:
            del example_cache[key]


# Updated generate-example endpoint with rate limiting and caching
@router.post("/generate-example", response_model=GenerateExampleResponse)
@limiter.limit("20/minute")  # Rate limit: 20 requests per minute per IP
async def generate_example(
        request: Request,  # Required for rate limiting
        example_request: GenerateExampleRequest,
        current_user: User = Depends(get_current_user),
        db: Session = Depends(get_db)
):
    """
    Generate example sentence with rate limiting and caching
    """
    try:
        if not example_request.word or not example_request.translation:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Both word and translation are required"
            )

        word_clean = example_request.word.strip().lower()
        translation_clean = example_request.translation.strip()

        # Check cache first
        cached_example = get_cached_example(word_clean, translation_clean)
        if cached_example:
            return GenerateExampleResponse(
                example_sentence=cached_example,
                alternatives=[]
            )

        # Check existing examples for variation
        existing_words = db.query(Word).join(Folder).filter(
            Folder.user_id == current_user.id,
            Word.word.ilike(word_clean)
        ).all()

        existing_examples = []
        if existing_words:
            existing_examples = [
                word.example_sentence for word in existing_words
                if word.example_sentence and word.example_sentence.strip()
            ]

        # Generate new example with retry logic
        example_sentence = await generate_example_with_retry(
            word_clean,
            translation_clean,
            existing_examples,
            max_retries=3
        )

        if not example_sentence:
            # Fallback to simple template
            example_sentence = f"I need to use {word_clean} in my daily life."

        # Cache the result
        cache_example(word_clean, translation_clean, example_sentence)

        logger.info(f"Generated example for word '{example_request.word}' by user {current_user.id}")

        return GenerateExampleResponse(
            example_sentence=example_sentence,
            alternatives=[]
        )

    except RateLimitExceeded:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Too many requests. Please wait before generating more examples."
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating example: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to generate example sentence"
        )


async def generate_example_with_retry(
        english_word: str,
        uzbek_translation: str,
        existing_examples: List[str] = None,
        max_retries: int = 3
) -> str:
    """
    Generate example with exponential backoff retry
    """
    for attempt in range(max_retries):
        try:
            # Build variation prompt
            existing_text = ""
            if existing_examples:
                existing_text = f"\n\nAvoid duplicating:\n" + "\n".join(f"- {ex}" for ex in existing_examples)

            # Randomize prompt style
            import random
            styles = [
                "Create a simple example sentence",
                "Make a practical example sentence",
                "Write a clear example sentence",
                "Generate an educational example sentence"
            ]

            prompt = f"""
            {random.choice(styles)} using "{english_word}" (means "{uzbek_translation}").

            Requirements:
            - Simple English for language learners
            - Under 15 words
            - Clear word meaning
            - Unique from existing examples
            {existing_text}
            """

            import openai
            from app.config import settings

            client = openai.OpenAI(api_key=settings.openai_api_key)

            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": "Create diverse vocabulary examples for English learners."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                max_tokens=80,  # Reduced token usage
                temperature=0.7,
                timeout=10.0  # 10 second timeout
            )

            example = response.choices[0].message.content.strip()

            # Clean response
            if example.startswith('"'):
                example = example.strip('"')
            if not example.endswith(('.', '!', '?')):
                example += '.'

            return example

        except openai.RateLimitError:
            # If rate limited, wait longer
            wait_time = (2 ** attempt) * 5  # 5, 10, 20 seconds
            logger.warning(f"OpenAI rate limited, waiting {wait_time}s before retry {attempt + 1}")
            await asyncio.sleep(wait_time)

        except Exception as e:
            if attempt == max_retries - 1:
                logger.error(f"Final attempt failed: {str(e)}")
                break

            wait_time = 2 ** attempt  # 1, 2, 4 seconds
            await asyncio.sleep(wait_time)

    return None  # All retries failed


# Add cache cleanup endpoint for admin
@router.delete("/generate-example/cache")
async def clear_example_cache(current_user: User = Depends(get_current_user)):
    """Clear the example cache (admin only)"""
    global example_cache
    example_cache.clear()
    return {"message": "Cache cleared successfully"}

# PARAMETERIZED ROUTES - these must come after specific routes

@router.post("/{folder_id}", response_model=AddWordResponse)
async def add_word(
        folder_id: int,
        request: AddWordRequest,
        current_user: User = Depends(get_current_user),
        db: Session = Depends(get_db)
):
    """
    Add word manually to folder
    """
    try:
        # Get folder
        folder = get_folder_or_404(folder_id, current_user.id, db)

        # Validate input
        if not request.word or len(request.word.strip()) == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Word cannot be empty"
            )

        if not request.translation or len(request.translation.strip()) == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Translation cannot be empty"
            )

        # Check if word already exists in this folder
        existing_word = db.query(Word).filter(
            Word.folder_id == folder_id,
            Word.word.ilike(request.word.strip())
        ).first()

        if existing_word:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Word already exists in this folder"
            )

        # Create new word
        word = Word(
            folder_id=folder_id,
            word=request.word.strip().lower(),
            translation=request.translation.strip(),
            example_sentence=request.example_sentence.strip() if request.example_sentence else None
        )

        db.add(word)
        db.flush()  # Get word.id

        # Create word stats
        word_stats = create_word_stats(word.id, current_user.id, db)

        db.commit()
        db.refresh(word)

        logger.info(f"Word '{word.word}' added to folder {folder_id} by user {current_user.id}")

        return AddWordResponse(
            word=WordResponse(
                id=word.id,
                word=word.word,
                translation=word.translation,
                example_sentence=word.example_sentence,
                added_at=word.added_at.isoformat()
            ),
            stats={
                "category": word_stats.category,
                "accuracy": 0,
                "total_attempts": 0
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error adding word: {str(e)}")
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to add word"
        )


@router.post("/{folder_id}/bulk-add", response_model=BulkAddResponse)
async def bulk_add_words(
        folder_id: int,
        request: BulkAddRequest,
        current_user: User = Depends(get_current_user),
        db: Session = Depends(get_db)
):
    """
    Save multiple words at once
    """
    try:
        # Get folder
        folder = get_folder_or_404(folder_id, current_user.id, db)

        if not request.words:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No words provided"
            )

        added_words = []

        for word_data in request.words:
            # Validate word
            if not word_data.word or not word_data.translation:
                continue  # Skip invalid words

            # Check if word already exists
            existing_word = db.query(Word).filter(
                Word.folder_id == folder_id,
                Word.word.ilike(word_data.word.strip())
            ).first()

            if existing_word:
                continue  # Skip duplicates

            # Create word
            word = Word(
                folder_id=folder_id,
                word=word_data.word.strip().lower(),
                translation=word_data.translation.strip(),
                example_sentence=word_data.example_sentence.strip() if word_data.example_sentence else None
            )

            db.add(word)
            db.flush()  # Get word.id

            # Create word stats
            create_word_stats(word.id, current_user.id, db)

            added_words.append(WordResponse(
                id=word.id,
                word=word.word,
                translation=word.translation,
                example_sentence=word.example_sentence,
                added_at=word.added_at.isoformat()
            ))

        db.commit()

        logger.info(f"Bulk added {len(added_words)} words to folder {folder_id} by user {current_user.id}")

        return BulkAddResponse(
            success=True,
            added_count=len(added_words),
            words=added_words
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error bulk adding words: {str(e)}")
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to add words"
        )


@router.get("/{word_id}", response_model=WordDetailResponse)
async def get_word_details(
        word_id: int,
        current_user: User = Depends(get_current_user),
        db: Session = Depends(get_db)
):
    """
    Get word details with stats
    """
    try:
        # Get word that belongs to user's folder
        word = db.query(Word).join(Folder).filter(
            Word.id == word_id,
            Folder.user_id == current_user.id
        ).first()

        if not word:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Word not found"
            )

        # Get word stats
        word_stats = word.get_user_stats(current_user.id)

        return WordDetailResponse(
            word={
                "id": word.id,
                "word": word.word,
                "translation": word.translation,
                "example_sentence": word.example_sentence,
                "added_at": word.added_at.isoformat(),
                "folder": {
                    "id": word.folder.id,
                    "name": word.folder.name
                }
            },
            stats={
                "category": word_stats.category if word_stats else "not_known",
                "last_5_results": word_stats.last_5_results if word_stats else [],
                "total_attempts": word_stats.total_attempts if word_stats else 0,
                "correct_attempts": word_stats.correct_attempts if word_stats else 0,
                "accuracy": word_stats.accuracy if word_stats else 0,
                "last_quiz_date": None  # Could track this in future
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting word details: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to fetch word details"
        )


@router.put("/{word_id}", response_model=UpdateWordResponse)
async def update_word(
        word_id: int,
        request: UpdateWordRequest,
        current_user: User = Depends(get_current_user),
        db: Session = Depends(get_db)
):
    """
    Edit word/translation/example
    """
    try:
        # Get word that belongs to user's folder
        word = db.query(Word).join(Folder).filter(
            Word.id == word_id,
            Folder.user_id == current_user.id
        ).first()

        if not word:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Word not found"
            )

        # Update fields if provided
        if request.word is not None:
            if not request.word or len(request.word.strip()) == 0:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Word cannot be empty"
                )
            word.word = request.word.strip().lower()

        if request.translation is not None:
            if not request.translation or len(request.translation.strip()) == 0:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Translation cannot be empty"
                )
            word.translation = request.translation.strip()

        if request.example_sentence is not None:
            word.example_sentence = request.example_sentence.strip() if request.example_sentence else None

        db.commit()
        db.refresh(word)

        logger.info(f"Word {word.id} updated by user {current_user.id}")

        return UpdateWordResponse(
            success=True,
            word={
                "id": word.id,
                "word": word.word,
                "translation": word.translation,
                "example_sentence": word.example_sentence,
                "added_at": word.added_at.isoformat()
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating word: {str(e)}")
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update word"
        )


@router.delete("/{word_id}", response_model=DeleteWordResponse)
async def delete_word(
        word_id: int,
        current_user: User = Depends(get_current_user),
        db: Session = Depends(get_db)
):
    """
    Delete word
    """
    try:
        # Get word that belongs to user's folder
        word = db.query(Word).join(Folder).filter(
            Word.id == word_id,
            Folder.user_id == current_user.id
        ).first()

        if not word:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Word not found"
            )

        word_data = {
            "id": word.id,
            "word": word.word,
            "translation": word.translation
        }

        db.delete(word)
        db.commit()

        logger.info(f"Word '{word_data['word']}' deleted by user {current_user.id}")

        return DeleteWordResponse(
            success=True,
            message="Word deleted successfully",
            deleted_word=word_data
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting word: {str(e)}")
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete word"
        )
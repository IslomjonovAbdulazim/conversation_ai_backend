# app/routers/words.py - Optimized and simplified
from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File
from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import List, Optional
import logging
import time

from app.database import get_db
from app.models import User, Folder, Word, WordStats
from app.services.auth import get_current_user
from app.services.google_vision import google_vision_service
from app.services.openai_service import openai_service

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
    stats: dict


class ExtractedWord(BaseModel):
    word: str
    translation: str
    confidence: float


class OCRResponse(BaseModel):
    extracted_words: List[ExtractedWord]
    total_extracted: int
    processing_time: float


class GenerateExampleRequest(BaseModel):
    word: str
    translation: str


class GenerateExampleResponse(BaseModel):
    example_sentence: str


class WordDetailResponse(BaseModel):
    word: dict
    stats: dict


class UpdateWordRequest(BaseModel):
    word: Optional[str] = None
    translation: Optional[str] = None
    example_sentence: Optional[str] = None


class UpdateWordResponse(BaseModel):
    success: bool
    word: dict


class DeleteWordResponse(BaseModel):
    success: bool
    message: str
    deleted_word: dict

class TranslateWordRequest(BaseModel):
    word: str

# Debug endpoint - remove after testing
@router.post("/translate-debug")
async def translate_debug(
        request: TranslateWordRequest,
        current_user: User = Depends(get_current_user),
        db: Session = Depends(get_db)
):
    """Debug endpoint to test enhanced translation"""
    try:
        word_clean = request.word.strip().lower()

        # Test the enhanced function directly
        logger.info(f"DEBUG: Testing enhanced translation for '{word_clean}'")

        try:
            translation_options = await openai_service.get_multiple_translation_options(word_clean)
            return {
                "status": "success",
                "word": word_clean,
                "raw_options": translation_options,
                "options_count": len(translation_options) if translation_options else 0
            }
        except Exception as e:
            return {
                "status": "error",
                "word": word_clean,
                "error": str(e),
                "error_type": type(e).__name__
            }

    except Exception as e:
        return {
            "status": "endpoint_error",
            "error": str(e)
        }


class TranslateWordRequest(BaseModel):
    word: str


class TranslationOption(BaseModel):
    translation: str
    confidence: float
    part_of_speech: Optional[str] = None
    meaning: Optional[str] = None


class TranslateWordRequest(BaseModel):
    word: str


class TranslateWordResponse(BaseModel):
    word: str
    options: List[TranslationOption]
    total_options: int


# Helper functions
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


def create_word_stats(word_id: int, user_id: int, db: Session) -> WordStats:
    """Create initial word stats for a new word"""
    word_stats = WordStats(
        word_id=word_id,
        user_id=user_id,
        category="not_known",
        last_5_results="",
        total_attempts=0,
        correct_attempts=0
    )
    db.add(word_stats)
    db.flush()
    return word_stats


# Main endpoints
@router.post("/translate", response_model=TranslateWordResponse)
async def translate_word(
        request: TranslateWordRequest,
        current_user: User = Depends(get_current_user),
        db: Session = Depends(get_db)
):
    """Translate a single English word to Uzbek with multiple options"""
    try:
        if not request.word or len(request.word.strip()) == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Word cannot be empty"
            )

        word_clean = request.word.strip().lower()

        # Try to get multiple translation options first
        try:
            logger.info(f"Attempting enhanced translation for '{word_clean}'")
            translation_options = await openai_service.get_multiple_translation_options(word_clean)

            if translation_options and len(translation_options) > 0:
                # Convert to response format
                options = []
                for option in translation_options:
                    options.append(TranslationOption(
                        translation=option.get("translation", ""),
                        confidence=option.get("confidence", 0.8),
                        part_of_speech=option.get("part_of_speech", ""),
                        meaning=option.get("meaning", "")
                    ))

                logger.info(f"Enhanced translation successful: {len(options)} options for '{word_clean}'")

                return TranslateWordResponse(
                    word=word_clean,
                    options=options,
                    total_options=len(options)
                )
            else:
                logger.warning(f"Enhanced translation returned empty for '{word_clean}', falling back")

        except Exception as e:
            logger.error(f"Enhanced translation failed for '{word_clean}': {str(e)}, falling back to simple")

        # Fallback to simple translation
        logger.info(f"Using simple translation fallback for '{word_clean}'")
        simple_translation = await openai_service.translate_to_uzbek(word_clean)

        options = [TranslationOption(
            translation=simple_translation,
            confidence=0.9,
            part_of_speech="unknown",
            meaning="standard translation"
        )]

        logger.info(f"Simple translation completed for '{word_clean}': {simple_translation}")

        return TranslateWordResponse(
            word=word_clean,
            options=options,
            total_options=len(options)
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Translation endpoint failed for '{request.word}': {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to translate word"
        )


@router.post("/generate-example", response_model=GenerateExampleResponse)
async def generate_example(
        request: GenerateExampleRequest,
        current_user: User = Depends(get_current_user),
        db: Session = Depends(get_db)
):
    """Generate example sentence for English word + Uzbek translation"""
    try:
        if not request.word or not request.translation:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Both word and translation are required"
            )

        word_clean = request.word.strip().lower()
        translation_clean = request.translation.strip()

        example_sentence = await openai_service.generate_example_sentence(word_clean, translation_clean)

        if not example_sentence:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to generate example sentence"
            )

        logger.info(f"Generated example for word '{request.word}' by user {current_user.id}")

        return GenerateExampleResponse(
            example_sentence=example_sentence
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating example: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to generate example sentence"
        )


@router.post("/upload-photo", response_model=OCRResponse)
async def upload_photo_ocr(
        file: UploadFile = File(...),
        current_user: User = Depends(get_current_user)
):
    """
    SIMPLIFIED OCR: Extract text from image and translate directly - no filtering
    """
    try:
        start_time = time.time()

        # Validate file type
        allowed_types = ['image/jpeg', 'image/jpg', 'image/png', 'image/heic', 'image/heif']
        if file.content_type not in allowed_types:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unsupported file type: {file.content_type}. Allowed: {', '.join(allowed_types)}"
            )

        # Check file size (max 10MB)
        file_content = await file.read()
        file_size = len(file_content)

        if file_size > 10 * 1024 * 1024:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="File too large. Maximum size is 10MB"
            )

        # Step 1: Extract ALL words using Google Vision
        all_words = await google_vision_service.extract_text_from_content(file_content)

        if not all_words:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No text found in image"
            )

        # Step 2: Simple limit to 100 words (no filtering)
        limited_words = all_words[:100]

        logger.info(f"Google Vision extracted {len(all_words)} words, using first {len(limited_words)}")

        # Step 3: Batch translate all words directly
        translations = await openai_service.batch_translate_to_uzbek(limited_words)

        # Step 4: Create response with all translated words
        translated_words = []
        for word in limited_words:
            translation = translations.get(word, word)  # Use original word if translation fails

            if translation and translation.strip():
                translated_words.append(ExtractedWord(
                    word=word.lower(),
                    translation=translation.strip(),
                    confidence=0.9
                ))

        # Fallback: if batch translation completely fails, try individual
        if not translated_words:
            logger.warning("Batch translation failed, trying individual translations")
            for word in limited_words[:20]:  # Limit to 20 for individual translation
                try:
                    translation = await openai_service.translate_to_uzbek(word)
                    if translation and translation.strip():
                        translated_words.append(ExtractedWord(
                            word=word.lower(),
                            translation=translation.strip(),
                            confidence=0.8
                        ))
                except Exception as e:
                    logger.warning(f"Individual translation failed for '{word}': {e}")
                    continue

        processing_time = round(time.time() - start_time, 2)

        logger.info(
            f"OCR completed: {len(all_words)} extracted → {len(translated_words)} translated in {processing_time}s for user {current_user.id}")

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


# Word CRUD operations
@router.post("/{folder_id}", response_model=AddWordResponse)
async def add_word(
        folder_id: int,
        request: AddWordRequest,
        current_user: User = Depends(get_current_user),
        db: Session = Depends(get_db)
):
    """Add word manually to folder"""
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


@router.get("/{word_id}", response_model=WordDetailResponse)
async def get_word_details(
        word_id: int,
        current_user: User = Depends(get_current_user),
        db: Session = Depends(get_db)
):
    """Get word details with stats"""
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
        word_stats = db.query(WordStats).filter(
            WordStats.word_id == word_id,
            WordStats.user_id == current_user.id
        ).first()

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
                "accuracy": (
                            word_stats.correct_attempts / word_stats.total_attempts * 100) if word_stats and word_stats.total_attempts > 0 else 0,
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
    """Edit word/translation/example"""
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
    """Delete word"""
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
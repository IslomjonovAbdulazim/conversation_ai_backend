from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File
from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import List, Dict, Optional
import logging
import time

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


# IMPORTANT: Add these endpoints BEFORE any parameterized routes (like /{folder_id})
# in your words.py file to avoid routing conflicts!

# Add these new Pydantic models to your existing models section in app/routers/words.py

class TranslateWordRequest(BaseModel):
    word: str


class TranslationOption(BaseModel):
    translation: str
    confidence: float


class TranslateWordResponse(BaseModel):
    word: str
    options: List[TranslationOption]
    total_options: int


class SaveTranslationRequest(BaseModel):
    word: str
    selected_translation: str
    folder_id: int
    generate_example: Optional[bool] = True


class SaveTranslationResponse(BaseModel):
    success: bool
    saved_word: WordResponse
    stats: Dict


# Add these new endpoint functions to your words router

@router.post("/translate", response_model=TranslateWordResponse)
async def translate_word(
        request: TranslateWordRequest,
        current_user: User = Depends(get_current_user),
        db: Session = Depends(get_db)
):
    """
    Get multiple translation options for an English word
    """
    try:
        if not request.word or len(request.word.strip()) == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Word cannot be empty"
            )

        word_clean = request.word.strip().lower()

        # Get multiple translation options
        translation_options = await get_translation_options(word_clean)

        logger.info(
            f"Generated {len(translation_options)} translation options for '{word_clean}' by user {current_user.id}")

        return TranslateWordResponse(
            word=word_clean,
            options=translation_options,
            total_options=len(translation_options)
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error translating word '{request.word}': {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to translate word"
        )


@router.post("/save-translation", response_model=SaveTranslationResponse)
async def save_selected_translation(
        request: SaveTranslationRequest,
        current_user: User = Depends(get_current_user),
        db: Session = Depends(get_db)
):
    """
    Save the user's selected translation to a folder
    """
    try:
        # Validate input
        if not request.word or len(request.word.strip()) == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Word cannot be empty"
            )

        if not request.selected_translation or len(request.selected_translation.strip()) == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Translation cannot be empty"
            )

        # Get folder
        folder = get_folder_or_404(request.folder_id, current_user.id, db)

        word_clean = request.word.strip().lower()
        translation_clean = request.selected_translation.strip()

        # Check if word already exists in this folder
        existing_word = db.query(Word).filter(
            Word.folder_id == request.folder_id,
            Word.word.ilike(word_clean)
        ).first()

        if existing_word:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Word already exists in this folder"
            )

        # Generate example sentence if requested
        example_sentence = None
        if request.generate_example:
            try:
                example_sentence = await generate_example_sentence(word_clean, translation_clean)
            except Exception as e:
                logger.warning(f"Failed to generate example for '{word_clean}': {str(e)}")
                # Continue without example sentence

        # Create new word
        word = Word(
            folder_id=request.folder_id,
            word=word_clean,
            translation=translation_clean,
            example_sentence=example_sentence
        )

        db.add(word)
        db.flush()  # Get word.id

        # Create word stats
        word_stats = create_word_stats(word.id, current_user.id, db)

        db.commit()
        db.refresh(word)

        logger.info(
            f"Saved translation '{word_clean}' -> '{translation_clean}' to folder {request.folder_id} by user {current_user.id}")

        return SaveTranslationResponse(
            success=True,
            saved_word=WordResponse(
                id=word.id,
                word=word.word,
                translation=word.translation,
                example_sentence=word.example_sentence,
                added_at=word.added_at.isoformat()
            ),
            stats={
                "category": word_stats.category,
                "total_attempts": word_stats.total_attempts,
                "correct_attempts": word_stats.correct_attempts
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error saving translation: {str(e)}")
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to save translation"
        )


# Helper function to generate multiple translation options
async def get_translation_options(english_word: str) -> List[TranslationOption]:
    """
    Generate multiple translation options for a word using OpenAI
    """
    try:
        import openai
        from app.config import settings

        client = openai.OpenAI(api_key=settings.openai_api_key)

        prompt = f"""
        Provide 3-4 different translation options for the English word "{english_word}" into Uzbek.

        Format your response as a numbered list of just the translations:
        1. [translation]
        2. [translation]
        3. [translation]

        Make sure translations are accurate and commonly used. Use plain text only, no bold or other formatting.
        """

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "You are a professional English-Uzbek translator. Provide multiple accurate translation options with usage contexts."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            max_tokens=300,
            temperature=0.3  # Some variation for different options
        )

        response_text = response.choices[0].message.content.strip()

        # Parse the response into TranslationOption objects
        options = parse_translation_options(response_text)

        # If parsing fails, fall back to single translation
        if not options:
            single_translation = await translate_to_uzbek(english_word)
            options = [TranslationOption(
                translation=single_translation,
                confidence=0.9
            )]

        return options

    except Exception as e:
        logger.error(f"Error getting translation options for '{english_word}': {str(e)}")
        # Fallback to single translation
        try:
            single_translation = await translate_to_uzbek(english_word)
            return [TranslationOption(
                translation=single_translation,
                confidence=0.8
            )]
        except:
            return [TranslationOption(
                translation=english_word,
                confidence=0.1
            )]


def parse_translation_options(response_text: str) -> List[TranslationOption]:
    """
    Parse OpenAI response into TranslationOption objects
    """
    options = []
    lines = response_text.strip().split('\n')

    for line in lines:
        line = line.strip()
        if not line or not any(char.isdigit() for char in line[:3]):
            continue

        try:
            # Remove number prefix (1. 2. etc.)
            clean_line = line
            for i in range(5):
                if clean_line.startswith(f"{i}.") or clean_line.startswith(f"{i})"):
                    clean_line = clean_line[2:].strip()
                    break

            # Split by dash or hyphen
            if ' - ' in clean_line:
                translation, context = clean_line.split(' - ', 1)
            elif ' – ' in clean_line:
                translation, context = clean_line.split(' – ', 1)
            else:
                translation = clean_line
                context = "Standard usage"

            translation = translation.strip()
            context = context.strip() if context else "Standard usage"

            if translation:
                # Assign confidence based on position (first option = highest confidence)
                confidence = max(0.95 - (len(options) * 0.1), 0.7)

                options.append(TranslationOption(
                    translation=translation,
                    confidence=confidence,
                    context=context,
                    usage_example=None
                ))

        except Exception as e:
            logger.warning(f"Failed to parse translation line: {line}")
            continue

    return options


# You'll also need to import this helper function if it doesn't exist
def create_word_stats(word_id: int, user_id: int, db: Session) -> WordStats:
    """
    Create initial word stats for a new word
    """
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


# Update the upload_photo_ocr function in app/routers/words.py

@router.post("/upload-photo", response_model=OCRResponse)
async def upload_photo_ocr(
        file: UploadFile = File(...),
        current_user: User = Depends(get_current_user)
):
    """
    Extract text from image using Google Vision API + AI word filtering
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

        # Check file size (max 10MB) by reading content
        file_content = await file.read()
        file_size = len(file_content)

        if file_size > 10 * 1024 * 1024:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="File too large. Maximum size is 10MB"
            )

        # Step 1: Extract ALL words using Google Vision (pass content directly)
        from app.services.google_vision import google_vision_service
        all_words = await google_vision_service.extract_text_from_content(file_content)

        if not all_words:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No text found in image"
            )

        logger.info(f"Google Vision extracted {len(all_words)} raw words")

        # Step 2: Use AI to filter best vocabulary words
        from app.services.openai_service import openai_service
        filtered_words = await openai_service.filter_best_vocabulary_words(all_words)

        if not filtered_words:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No suitable vocabulary words found in image"
            )

        logger.info(f"AI filtered down to {len(filtered_words)} quality words")
        from app.services.openai_service import openai_service
        translations = await openai_service.batch_translate_to_uzbek(filtered_words)
        # Step 3: Translate only the filtered words
        translated_words = []
        for word in filtered_words:
            try:
                translation = translations.get(word, word)  # Fallback to original word
                if translation and translation != word:
                    translated_words.append(ExtractedWord(
                        word=word.lower(),
                        translation=translation,
                        confidence=0.9
                    ))
            except Exception as e:
                logger.warning(f"Failed to translate word '{word}': {str(e)}")
                continue

        if not translated_words:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No words could be translated successfully"
            )

        processing_time = round(time.time() - start_time, 2)

        logger.info(
            f"OCR completed: {len(translated_words)} final words in {processing_time}s for user {current_user.id}")

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


@router.post("/generate-example", response_model=GenerateExampleResponse)
async def generate_example(
        request: GenerateExampleRequest,
        current_user: User = Depends(get_current_user),
        db: Session = Depends(get_db)
):
    """
    Generate example sentence for English word + Uzbek translation
    Provides variation if word already exists with examples
    """
    try:
        if not request.word or not request.translation:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Both word and translation are required"
            )

        word_clean = request.word.strip().lower()
        translation_clean = request.translation.strip()

        # Check if this word already exists in user's folders
        existing_words = db.query(Word).join(Folder).filter(
            Folder.user_id == current_user.id,
            Word.word.ilike(word_clean)
        ).all()

        existing_examples = []
        if existing_words:
            # Collect existing examples to avoid duplicates
            existing_examples = [
                word.example_sentence for word in existing_words
                if word.example_sentence and word.example_sentence.strip()
            ]

            logger.info(f"Found {len(existing_examples)} existing examples for '{word_clean}'")

        # Generate new example with variation
        example_sentence = await generate_example_sentence_with_variation(
            word_clean,
            translation_clean,
            existing_examples
        )

        if not example_sentence:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to generate example sentence"
            )

        logger.info(f"Generated example for word '{request.word}' by user {current_user.id}")

        return GenerateExampleResponse(
            example_sentence=example_sentence,
            alternatives=[]  # Keep empty as requested
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating example: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to generate example sentence"
        )


async def generate_example_sentence_with_variation(
        english_word: str,
        uzbek_translation: str,
        existing_examples: List[str] = None
) -> str:
    """
    Enhanced example generation with variation and duplicate prevention
    """
    try:
        # Build prompt with variation instructions
        existing_text = ""
        if existing_examples:
            existing_text = f"\n\nAvoid duplicating these existing examples:\n" + "\n".join(
                f"- {ex}" for ex in existing_examples)

        # Add randomness to prompt style
        import random
        variation_prompts = [
            "Create a simple, clear example sentence",
            "Make a practical, everyday example sentence",
            "Write a natural, conversational example sentence",
            "Generate a clear, educational example sentence"
        ]

        base_prompt = random.choice(variation_prompts)

        prompt = f"""
        {base_prompt} using the English word "{english_word}" (which means "{uzbek_translation}" in Uzbek).

        Requirements:
        - Use everyday, simple English suitable for language learners
        - Keep it under 15 words
        - Make sure the meaning of "{english_word}" is clear from context
        - Create a unique example different from any existing ones
        {existing_text}

        Example sentence:
        """

        import openai
        from app.config import settings

        client = openai.OpenAI(api_key=settings.openai_api_key)

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "You are an English teacher creating diverse example sentences for vocabulary learning. Always create unique examples that help students understand word usage in different contexts."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            max_tokens=100,
            temperature=0.7,  # Higher temperature for variation
            top_p=0.9  # Add randomness
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
        # Fallback: create simple sentence with some variation
        import random
        fallbacks = [
            f"I use {english_word} every day.",
            f"The {english_word} is very important.",
            f"We need a good {english_word} for this.",
            f"This {english_word} works perfectly."
        ]
        return random.choice(fallbacks)


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



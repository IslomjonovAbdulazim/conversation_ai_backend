from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import List, Dict, Optional
import random
import logging

from app.database import get_db
from app.models import User, Folder, Word, WordStats, QuizSession, QuizResult
from app.routers.auth import get_current_user
from app.config import settings
from app.services.openai_service import generate_quiz_question

logger = logging.getLogger(__name__)
router = APIRouter()


# Pydantic models
class StartQuizRequest(BaseModel):
    quiz_type: str  # anagram, translation_blitz, word_blitz, reading


class QuizQuestion(BaseModel):
    question_number: int
    word_id: int
    question: str
    options: Optional[List[str]] = None
    hint: Optional[str] = None
    time_limit: int


class StartQuizResponse(BaseModel):
    session: Dict
    question: QuizQuestion


class SubmitAnswerRequest(BaseModel):
    word_id: int
    answer: str
    time_taken: int


class SubmitAnswerResponse(BaseModel):
    is_correct: bool
    correct_answer: str
    explanation: Optional[str] = None
    next_question: Optional[QuizQuestion] = None


class CompleteQuizResponse(BaseModel):
    results: Dict
    updated_categories: List[Dict]
    performance: Dict


class QuizSessionManager:
    """Manage quiz sessions in memory"""

    def __init__(self):
        self.active_sessions = {}  # session_id -> quiz_data

    def create_session(self, session_id: str, quiz_data: dict):
        """Create new quiz session"""
        self.active_sessions[session_id] = quiz_data

    def get_session(self, session_id: str):
        """Get active quiz session"""
        return self.active_sessions.get(session_id)

    def update_session(self, session_id: str, quiz_data: dict):
        """Update quiz session"""
        self.active_sessions[session_id] = quiz_data

    def remove_session(self, session_id: str):
        """Remove quiz session"""
        if session_id in self.active_sessions:
            del self.active_sessions[session_id]


# Global quiz session manager
quiz_manager = QuizSessionManager()


def shuffle_letters(word: str) -> str:
    """Shuffle letters of a word for anagram"""
    letters = list(word)
    random.shuffle(letters)
    return ''.join(letters)


def get_wrong_options(correct_word: str, all_words: List[Word]) -> List[str]:
    """Get 2 wrong options for multiple choice"""
    other_words = [w.word for w in all_words if w.word != correct_word]
    random.shuffle(other_words)
    return other_words[:2]


def get_wrong_translations(correct_translation: str, all_words: List[Word]) -> List[str]:
    """Get 2 wrong translations for multiple choice"""
    other_translations = [w.translation for w in all_words if w.translation != correct_translation]
    random.shuffle(other_translations)
    return other_translations[:2]


@router.post("/{folder_id}/start", response_model=StartQuizResponse)
async def start_quiz(
        folder_id: int,
        request: StartQuizRequest,
        current_user: User = Depends(get_current_user),
        db: Session = Depends(get_db)
):
    """
    Start quiz session for specific folder
    """
    try:
        # Validate quiz type
        valid_quiz_types = ["anagram", "translation_blitz", "word_blitz", "reading"]
        if request.quiz_type not in valid_quiz_types:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid quiz type. Must be one of: {valid_quiz_types}"
            )

        # Get folder
        folder = db.query(Folder).filter(
            Folder.id == folder_id,
            Folder.user_id == current_user.id
        ).first()

        if not folder:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Folder not found"
            )

        # Get complete words (with example sentences) for quiz
        complete_words = [word for word in folder.words if word.is_complete]

        # Check minimum words requirement
        min_words = settings.min_words_for_quiz
        if request.quiz_type == "reading":
            min_words = settings.min_words_for_reading

        if len(complete_words) < min_words:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Need at least {min_words} words with example sentences to start {request.quiz_type} quiz"
            )

        # Create quiz session in database
        quiz_session = QuizSession(
            user_id=current_user.id,
            folder_id=folder_id,
            quiz_type=request.quiz_type
        )

        db.add(quiz_session)
        db.flush()  # Get session ID

        session_id = f"quiz_{quiz_session.id}"

        # Shuffle words for quiz
        quiz_words = complete_words.copy()
        random.shuffle(quiz_words)

        # Limit to 10 questions for regular quizzes
        if request.quiz_type != "reading":
            quiz_words = quiz_words[:10]
        else:
            quiz_words = quiz_words[:8]  # Reading uses fewer words

        # Create quiz session data
        quiz_data = {
            "session_id": quiz_session.id,
            "folder_id": folder_id,
            "quiz_type": request.quiz_type,
            "words": [{"id": w.id, "word": w.word, "translation": w.translation, "example": w.example_sentence} for w in
                      quiz_words],
            "all_words": [{"id": w.id, "word": w.word, "translation": w.translation} for w in complete_words],
            "current_question": 0,
            "answers": []
        }

        # Generate first question
        if request.quiz_type == "reading":
            # Generate reading passage
            reading_content = await generate_quiz_question(quiz_data["words"], "reading")
            if reading_content:
                quiz_data["reading_content"] = reading_content

            first_question = QuizQuestion(
                question_number=1,
                word_id=0,  # Special case for reading
                question="Fill in the blanks in the passage below:",
                time_limit=120  # 2 minutes for reading
            )
        else:
            # Generate first question for other quiz types
            first_question = generate_question(quiz_data, 0)

        # Store session
        quiz_manager.create_session(session_id, quiz_data)

        db.commit()

        logger.info(f"Started {request.quiz_type} quiz for folder {folder_id} by user {current_user.id}")

        return StartQuizResponse(
            session={
                "session_id": session_id,
                "quiz_type": request.quiz_type,
                "folder_id": folder_id,
                "total_questions": len(quiz_data["words"])
            },
            question=first_question
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error starting quiz: {str(e)}")
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to start quiz"
        )


def generate_question(quiz_data: dict, question_index: int) -> QuizQuestion:
    """Generate question based on quiz type"""
    word_data = quiz_data["words"][question_index]
    quiz_type = quiz_data["quiz_type"]

    if quiz_type == "anagram":
        # Anagram: Shuffle letters, show translation, user reorders
        shuffled = shuffle_letters(word_data["word"])
        return QuizQuestion(
            question_number=question_index + 1,
            word_id=word_data["id"],
            question=f"Rearrange these letters: {shuffled.upper()}",
            hint=word_data["translation"],
            time_limit=settings.quiz_time_limits["anagram"]
        )

    elif quiz_type == "translation_blitz":
        # Translation Blitz: Show Uzbek translation, pick English word
        wrong_options = get_wrong_options(word_data["word"], [{"word": w["word"]} for w in quiz_data["all_words"]])
        options = [word_data["word"]] + wrong_options[:2]
        random.shuffle(options)

        return QuizQuestion(
            question_number=question_index + 1,
            word_id=word_data["id"],
            question=f"What is the English word for: '{word_data['translation']}'?",
            options=options,
            time_limit=settings.quiz_time_limits["translation_blitz"]
        )

    elif quiz_type == "word_blitz":
        # Word Blitz: Show definition/example, pick English word
        wrong_options = get_wrong_options(word_data["word"], [{"word": w["word"]} for w in quiz_data["all_words"]])
        options = [word_data["word"]] + wrong_options[:2]
        random.shuffle(options)

        return QuizQuestion(
            question_number=question_index + 1,
            word_id=word_data["id"],
            question=f"Which word fits this example: '{word_data['example']}'?",
            options=options,
            time_limit=settings.quiz_time_limits["word_blitz"]
        )

    return None


@router.post("/{session_id}/answer", response_model=SubmitAnswerResponse)
async def submit_answer(
        session_id: str,
        request: SubmitAnswerRequest,
        current_user: User = Depends(get_current_user),
        db: Session = Depends(get_db)
):
    """
    Submit answer to question
    """
    try:
        # Get quiz session
        quiz_data = quiz_manager.get_session(session_id)
        if not quiz_data:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Quiz session not found"
            )

        current_question_index = quiz_data["current_question"]
        word_data = quiz_data["words"][current_question_index]

        # Check answer
        correct_answer = word_data["word"]
        is_correct = request.answer.lower().strip() == correct_answer.lower()

        # Save quiz result
        quiz_result = QuizResult(
            session_id=quiz_data["session_id"],
            word_id=request.word_id,
            is_correct=is_correct,
            time_taken=request.time_taken
        )

        db.add(quiz_result)

        # Update word stats
        word_stats = db.query(WordStats).filter(
            WordStats.word_id == request.word_id,
            WordStats.user_id == current_user.id
        ).first()

        if not word_stats:
            word_stats = WordStats(
                word_id=request.word_id,
                user_id=current_user.id
            )
            db.add(word_stats)

        word_stats.add_result(is_correct)

        # Store answer
        quiz_data["answers"].append({
            "word_id": request.word_id,
            "answer": request.answer,
            "is_correct": is_correct,
            "time_taken": request.time_taken
        })

        # Move to next question
        quiz_data["current_question"] += 1

        # Generate next question or finish quiz
        next_question = None
        if quiz_data["current_question"] < len(quiz_data["words"]):
            next_question = generate_question(quiz_data, quiz_data["current_question"])

        # Update session
        quiz_manager.update_session(session_id, quiz_data)

        db.commit()

        return SubmitAnswerResponse(
            is_correct=is_correct,
            correct_answer=correct_answer,
            explanation=f"{correct_answer} means '{word_data['translation']}' in Uzbek" if not is_correct else None,
            next_question=next_question
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error submitting answer: {str(e)}")
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to submit answer"
        )


@router.post("/{session_id}/complete", response_model=CompleteQuizResponse)
async def complete_quiz(
        session_id: str,
        current_user: User = Depends(get_current_user),
        db: Session = Depends(get_db)
):
    """
    Complete quiz session
    """
    try:
        # Get quiz session
        quiz_data = quiz_manager.get_session(session_id)
        if not quiz_data:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Quiz session not found"
            )

        # Update quiz session duration
        quiz_session = db.query(QuizSession).filter(
            QuizSession.id == quiz_data["session_id"]
        ).first()

        if quiz_session:
            total_time = sum(answer["time_taken"] for answer in quiz_data["answers"])
            quiz_session.duration = total_time

        # Calculate results
        total_questions = len(quiz_data["answers"])
        correct_answers = sum(1 for answer in quiz_data["answers"] if answer["is_correct"])
        accuracy = round((correct_answers / total_questions) * 100) if total_questions > 0 else 0

        # Get updated categories
        updated_categories = []
        for answer in quiz_data["answers"]:
            word_stats = db.query(WordStats).filter(
                WordStats.word_id == answer["word_id"],
                WordStats.user_id == current_user.id
            ).first()

            if word_stats:
                updated_categories.append({
                    "word_id": answer["word_id"],
                    "new_category": word_stats.category
                })

        # Performance analysis
        strongest_words = []
        needs_practice = []

        for answer in quiz_data["answers"]:
            word_data = next(w for w in quiz_data["words"] if w["id"] == answer["word_id"])
            if answer["is_correct"]:
                strongest_words.append(word_data["word"])
            else:
                needs_practice.append(word_data["word"])

        # Clean up session
        quiz_manager.remove_session(session_id)

        db.commit()

        logger.info(f"Completed quiz session {session_id} - {correct_answers}/{total_questions} correct")

        return CompleteQuizResponse(
            results={
                "session_id": session_id,
                "total_questions": total_questions,
                "correct_answers": correct_answers,
                "total_time": sum(answer["time_taken"] for answer in quiz_data["answers"]),
                "accuracy": accuracy,
                "quiz_type": quiz_data["quiz_type"]
            },
            updated_categories=updated_categories,
            performance={
                "improvement": f"+{accuracy}%",
                "strongest_words": strongest_words[:3],
                "needs_practice": needs_practice[:3]
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error completing quiz: {str(e)}")
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to complete quiz"
        )
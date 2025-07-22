from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, func, JSON
from sqlalchemy.orm import relationship
from app.database import Base
from typing import List


class Word(Base):
    __tablename__ = "words"

    id = Column(Integer, primary_key=True, index=True)
    folder_id = Column(Integer, ForeignKey("folders.id"), nullable=False)
    word = Column(String, nullable=False)  # English word
    translation = Column(String, nullable=False)  # Uzbek translation
    example_sentence = Column(String, nullable=True)  # Can be null
    added_at = Column(DateTime(timezone=True), server_default=func.now())

    # Relationships
    folder = relationship("Folder", back_populates="words")
    word_stats = relationship("WordStats", back_populates="word", cascade="all, delete-orphan")
    quiz_results = relationship("QuizResult", back_populates="word", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<Word(id={self.id}, word='{self.word}', translation='{self.translation}')>"

    @property
    def is_complete(self):
        """Check if word has example sentence (required for quizzes)"""
        return self.example_sentence is not None and self.example_sentence.strip() != ""

    def get_user_stats(self, user_id: int):
        """Get word statistics for specific user"""
        for stats in self.word_stats:
            if stats.user_id == user_id:
                return stats
        return None


class WordStats(Base):
    __tablename__ = "word_stats"

    id = Column(Integer, primary_key=True, index=True)
    word_id = Column(Integer, ForeignKey("words.id"), nullable=False)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    category = Column(String, default="not_known")  # not_known, normal, strong
    last_5_results = Column(JSON, default=list)  # [True, False, True, ...]
    total_attempts = Column(Integer, default=0)
    correct_attempts = Column(Integer, default=0)

    # Relationships
    word = relationship("Word", back_populates="word_stats")
    user = relationship("User", back_populates="word_stats")

    def __repr__(self):
        return f"<WordStats(word_id={self.word_id}, user_id={self.user_id}, category='{self.category}')>"

    @property
    def accuracy(self):
        """Calculate accuracy percentage"""
        if self.total_attempts == 0:
            return 0
        return round((self.correct_attempts / self.total_attempts) * 100)

    def add_result(self, is_correct: bool):
        """Add new quiz result and update category"""
        # Add to results list (keep only last 5)
        results = self.last_5_results or []
        results.append(is_correct)
        if len(results) > 5:
            results = results[-5:]
        self.last_5_results = results

        # Update counters
        self.total_attempts += 1
        if is_correct:
            self.correct_attempts += 1

        # Update category based on last 5 results
        self.update_category()

    def update_category(self):
        """Update word category based on last 5 quiz results"""
        if not self.last_5_results or len(self.last_5_results) < 3:
            self.category = "not_known"
            return

        # Calculate accuracy from last 5 results
        correct_count = sum(self.last_5_results)
        recent_accuracy = correct_count / len(self.last_5_results)

        if recent_accuracy >= 0.8:  # 80% or higher
            self.category = "strong"
        elif recent_accuracy >= 0.5:  # 50-79%
            self.category = "normal"
        else:  # Below 50%
            self.category = "not_known"
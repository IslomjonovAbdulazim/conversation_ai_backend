from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, Boolean, func
from sqlalchemy.orm import relationship
from app.database import Base


class QuizSession(Base):
    __tablename__ = "quiz_sessions"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    folder_id = Column(Integer, ForeignKey("folders.id"), nullable=False)
    quiz_type = Column(String, nullable=False)  # anagram, translation_blitz, word_blitz, reading
    duration = Column(Integer, nullable=True)  # seconds
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # Relationships
    user = relationship("User", back_populates="quiz_sessions")
    folder = relationship("Folder", back_populates="quiz_sessions")
    quiz_results = relationship("QuizResult", back_populates="session", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<QuizSession(id={self.id}, type='{self.quiz_type}', user_id={self.user_id})>"

    @property
    def total_questions(self):
        """Get total number of questions in this session"""
        return len(self.quiz_results)

    @property
    def correct_answers(self):
        """Get number of correct answers"""
        return len([result for result in self.quiz_results if result.is_correct])

    @property
    def accuracy(self):
        """Calculate accuracy percentage"""
        if self.total_questions == 0:
            return 0
        return round((self.correct_answers / self.total_questions) * 100)


class QuizResult(Base):
    __tablename__ = "quiz_results"

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(Integer, ForeignKey("quiz_sessions.id"), nullable=False)
    word_id = Column(Integer, ForeignKey("words.id"), nullable=False)
    is_correct = Column(Boolean, nullable=False)
    time_taken = Column(Integer, nullable=False)  # seconds

    # Relationships
    session = relationship("QuizSession", back_populates="quiz_results")
    word = relationship("Word", back_populates="quiz_results")

    def __repr__(self):
        return f"<QuizResult(word_id={self.word_id}, correct={self.is_correct}, time={self.time_taken})>"


class VoiceAgent(Base):
    __tablename__ = "voice_agents"

    id = Column(Integer, primary_key=True, index=True)
    topic = Column(String, nullable=False)  # cars, football, travel
    title = Column(String, nullable=False)
    description = Column(String, nullable=False)
    image_url = Column(String, nullable=False)
    agent_id = Column(String, nullable=False)  # ElevenLabs agent ID
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    def __repr__(self):
        return f"<VoiceAgent(id={self.id}, topic='{self.topic}', title='{self.title}')>"
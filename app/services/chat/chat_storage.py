"""
채팅 저장 서비스 - PostgreSQL 기반 채팅 히스토리 저장 및 분석

Neo4j 지식그래프 오염 방지를 위해 채팅 데이터는 별도 PostgreSQL에 저장
"""

import os
import uuid
import json
import logging
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field
from contextlib import contextmanager

from sqlalchemy import create_engine, Column, String, Text, DateTime, Integer, JSON, ForeignKey, Index, func
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship, Session as DBSession

logger = logging.getLogger(__name__)

Base = declarative_base()


# ============================================
# SQLAlchemy Models
# ============================================

class ChatSessionModel(Base):
    """채팅 세션 테이블"""
    __tablename__ = 'chat_sessions'

    id = Column(String(64), primary_key=True)
    brand_id = Column(String(64), nullable=False, index=True)
    user_id = Column(String(64), nullable=True, index=True)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, index=True)
    message_count = Column(Integer, default=0)
    metadata_ = Column('metadata', JSON, default=dict)

    messages = relationship("ChatMessageModel", back_populates="session", cascade="all, delete-orphan")

    __table_args__ = (
        Index('idx_session_brand_updated', 'brand_id', 'updated_at'),
    )


class ChatMessageModel(Base):
    """채팅 메시지 테이블"""
    __tablename__ = 'chat_messages'

    id = Column(String(64), primary_key=True)
    session_id = Column(String(64), ForeignKey('chat_sessions.id', ondelete='CASCADE'), nullable=False, index=True)
    role = Column(String(16), nullable=False, index=True)  # user, assistant, system
    content = Column(Text, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    metadata_ = Column('metadata', JSON, default=dict)

    # Analytics fields (denormalized for fast queries)
    grade = Column(String(2), nullable=True, index=True)  # A, B, C, D, F
    question_type = Column(String(64), nullable=True, index=True)
    response_time_ms = Column(Integer, nullable=True)

    session = relationship("ChatSessionModel", back_populates="messages")

    __table_args__ = (
        Index('idx_message_session_timestamp', 'session_id', 'timestamp'),
        Index('idx_message_grade', 'grade'),
    )


# ============================================
# Dataclasses (API Response)
# ============================================

@dataclass
class StoredMessage:
    """저장된 메시지"""
    id: str
    session_id: str
    role: str
    content: str
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "session_id": self.session_id,
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata
        }


@dataclass
class ChatSession:
    """채팅 세션"""
    id: str
    brand_id: str
    user_id: Optional[str]
    created_at: datetime
    updated_at: datetime
    message_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "brand_id": self.brand_id,
            "user_id": self.user_id,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "message_count": self.message_count,
            "metadata": self.metadata
        }


@dataclass
class ChatAnalytics:
    """채팅 분석 데이터"""
    total_sessions: int = 0
    total_messages: int = 0
    avg_messages_per_session: float = 0.0
    avg_response_time_ms: float = 0.0
    grade_distribution: Dict[str, int] = field(default_factory=dict)
    question_type_distribution: Dict[str, int] = field(default_factory=dict)
    daily_message_counts: List[Dict[str, Any]] = field(default_factory=list)
    top_topics: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_sessions": self.total_sessions,
            "total_messages": self.total_messages,
            "avg_messages_per_session": round(self.avg_messages_per_session, 2),
            "avg_response_time_ms": round(self.avg_response_time_ms, 2),
            "grade_distribution": self.grade_distribution,
            "question_type_distribution": self.question_type_distribution,
            "daily_message_counts": self.daily_message_counts,
            "top_topics": self.top_topics
        }


# ============================================
# Chat Storage Service
# ============================================

class ChatStorageService:
    """
    채팅 저장 서비스 (PostgreSQL)

    Neo4j 지식그래프와 분리하여 채팅 데이터를 별도 저장
    """

    def __init__(self):
        self.engine = self._create_engine()
        self.SessionLocal = sessionmaker(bind=self.engine)
        self._ensure_tables()

    def _create_engine(self):
        """PostgreSQL 엔진 생성"""
        host = os.getenv('POSTGRES_HOST', 'localhost')
        port = os.getenv('POSTGRES_PORT', '5432')
        db = os.getenv('POSTGRES_DB', 'ontix_chat')
        user = os.getenv('POSTGRES_USER', 'ontix')
        password = os.getenv('POSTGRES_PASSWORD', '')

        database_url = f"postgresql://{user}:{password}@{host}:{port}/{db}"

        logger.info(f"Connecting to PostgreSQL: {host}:{port}/{db}")

        return create_engine(
            database_url,
            pool_size=10,
            max_overflow=20,
            pool_pre_ping=True,
            echo=False
        )

    def _ensure_tables(self):
        """테이블 생성"""
        try:
            Base.metadata.create_all(self.engine)
            logger.info("Chat storage tables ensured")
        except Exception as e:
            logger.error(f"Failed to create tables: {e}")
            raise

    @contextmanager
    def _get_db(self):
        """DB 세션 컨텍스트 매니저"""
        db = self.SessionLocal()
        try:
            yield db
            db.commit()
        except Exception:
            db.rollback()
            raise
        finally:
            db.close()

    # ========================================
    # Session Management
    # ========================================

    def create_session(
        self,
        brand_id: str,
        user_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> ChatSession:
        """새 채팅 세션 생성"""
        session_id = f"session_{uuid.uuid4().hex[:12]}"
        now = datetime.utcnow()

        with self._get_db() as db:
            db_session = ChatSessionModel(
                id=session_id,
                brand_id=brand_id,
                user_id=user_id,
                created_at=now,
                updated_at=now,
                message_count=0,
                metadata_=metadata or {}
            )
            db.add(db_session)

        logger.info(f"Created chat session: {session_id} for brand {brand_id}")

        return ChatSession(
            id=session_id,
            brand_id=brand_id,
            user_id=user_id,
            created_at=now,
            updated_at=now,
            message_count=0,
            metadata=metadata or {}
        )

    def get_session(self, session_id: str) -> Optional[ChatSession]:
        """세션 조회"""
        with self._get_db() as db:
            result = db.query(ChatSessionModel).filter(
                ChatSessionModel.id == session_id
            ).first()

            if not result:
                return None

            return ChatSession(
                id=result.id,
                brand_id=result.brand_id,
                user_id=result.user_id,
                created_at=result.created_at,
                updated_at=result.updated_at,
                message_count=result.message_count,
                metadata=result.metadata_ or {}
            )

    def get_or_create_session(
        self,
        session_id: Optional[str],
        brand_id: str,
        user_id: Optional[str] = None
    ) -> ChatSession:
        """세션 조회 또는 생성"""
        if session_id:
            session = self.get_session(session_id)
            if session:
                return session

        return self.create_session(brand_id, user_id)

    def list_sessions(
        self,
        brand_id: str,
        user_id: Optional[str] = None,
        limit: int = 50,
        offset: int = 0
    ) -> List[ChatSession]:
        """세션 목록 조회"""
        with self._get_db() as db:
            query = db.query(ChatSessionModel).filter(
                ChatSessionModel.brand_id == brand_id
            )

            if user_id:
                query = query.filter(ChatSessionModel.user_id == user_id)

            results = query.order_by(
                ChatSessionModel.updated_at.desc()
            ).offset(offset).limit(limit).all()

            return [
                ChatSession(
                    id=r.id,
                    brand_id=r.brand_id,
                    user_id=r.user_id,
                    created_at=r.created_at,
                    updated_at=r.updated_at,
                    message_count=r.message_count,
                    metadata=r.metadata_ or {}
                )
                for r in results
            ]

    # ========================================
    # Message Management
    # ========================================

    def save_message(
        self,
        session_id: str,
        role: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> StoredMessage:
        """메시지 저장"""
        message_id = f"msg_{uuid.uuid4().hex[:12]}"
        now = datetime.utcnow()
        meta = metadata or {}

        with self._get_db() as db:
            # 메시지 저장
            db_message = ChatMessageModel(
                id=message_id,
                session_id=session_id,
                role=role,
                content=content,
                timestamp=now,
                metadata_=meta,
                grade=meta.get('grade'),
                question_type=meta.get('question_type'),
                response_time_ms=meta.get('response_time_ms')
            )
            db.add(db_message)

            # 세션 업데이트
            db.query(ChatSessionModel).filter(
                ChatSessionModel.id == session_id
            ).update({
                'updated_at': now,
                'message_count': ChatSessionModel.message_count + 1
            })

        logger.debug(f"Saved message: {message_id} in session {session_id}")

        return StoredMessage(
            id=message_id,
            session_id=session_id,
            role=role,
            content=content,
            timestamp=now,
            metadata=meta
        )

    def save_conversation_turn(
        self,
        session_id: str,
        user_message: str,
        assistant_message: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> tuple[StoredMessage, StoredMessage]:
        """대화 턴 저장"""
        user_msg = self.save_message(session_id, "user", user_message)
        assistant_msg = self.save_message(session_id, "assistant", assistant_message, metadata)
        return user_msg, assistant_msg

    def get_messages(
        self,
        session_id: str,
        limit: int = 100,
        before_timestamp: Optional[datetime] = None
    ) -> List[StoredMessage]:
        """세션의 메시지 조회"""
        with self._get_db() as db:
            query = db.query(ChatMessageModel).filter(
                ChatMessageModel.session_id == session_id
            )

            if before_timestamp:
                query = query.filter(ChatMessageModel.timestamp < before_timestamp)

            results = query.order_by(
                ChatMessageModel.timestamp.asc()
            ).limit(limit).all()

            return [
                StoredMessage(
                    id=m.id,
                    session_id=m.session_id,
                    role=m.role,
                    content=m.content,
                    timestamp=m.timestamp,
                    metadata=m.metadata_ or {}
                )
                for m in results
            ]

    def get_recent_context(
        self,
        session_id: str,
        max_messages: int = 10
    ) -> List[Dict[str, str]]:
        """최근 대화 컨텍스트 조회 (LLM용)"""
        with self._get_db() as db:
            results = db.query(
                ChatMessageModel.role,
                ChatMessageModel.content
            ).filter(
                ChatMessageModel.session_id == session_id
            ).order_by(
                ChatMessageModel.timestamp.desc()
            ).limit(max_messages).all()

            # 역순으로 반환 (가장 오래된 것부터)
            return [{"role": r.role, "content": r.content} for r in reversed(results)]

    # ========================================
    # Analytics
    # ========================================

    def get_analytics(
        self,
        brand_id: str,
        days: int = 30
    ) -> ChatAnalytics:
        """채팅 분석 데이터 조회"""
        since = datetime.utcnow() - timedelta(days=days)

        with self._get_db() as db:
            # 기본 통계
            session_count = db.query(func.count(ChatSessionModel.id)).filter(
                ChatSessionModel.brand_id == brand_id,
                ChatSessionModel.created_at >= since
            ).scalar() or 0

            message_count = db.query(func.count(ChatMessageModel.id)).join(
                ChatSessionModel
            ).filter(
                ChatSessionModel.brand_id == brand_id,
                ChatMessageModel.timestamp >= since
            ).scalar() or 0

            avg_messages = message_count / session_count if session_count > 0 else 0

            # 평균 응답 시간
            avg_response_time = db.query(
                func.avg(ChatMessageModel.response_time_ms)
            ).join(ChatSessionModel).filter(
                ChatSessionModel.brand_id == brand_id,
                ChatMessageModel.role == 'assistant',
                ChatMessageModel.timestamp >= since,
                ChatMessageModel.response_time_ms.isnot(None)
            ).scalar() or 0

            # Grade 분포
            grade_results = db.query(
                ChatMessageModel.grade,
                func.count(ChatMessageModel.id)
            ).join(ChatSessionModel).filter(
                ChatSessionModel.brand_id == brand_id,
                ChatMessageModel.role == 'assistant',
                ChatMessageModel.timestamp >= since,
                ChatMessageModel.grade.isnot(None)
            ).group_by(ChatMessageModel.grade).all()

            grade_distribution = {r[0]: r[1] for r in grade_results}

            # Question Type 분포
            qtype_results = db.query(
                ChatMessageModel.question_type,
                func.count(ChatMessageModel.id)
            ).join(ChatSessionModel).filter(
                ChatSessionModel.brand_id == brand_id,
                ChatMessageModel.role == 'assistant',
                ChatMessageModel.timestamp >= since,
                ChatMessageModel.question_type.isnot(None)
            ).group_by(ChatMessageModel.question_type).order_by(
                func.count(ChatMessageModel.id).desc()
            ).limit(10).all()

            question_type_distribution = {r[0]: r[1] for r in qtype_results}

            # 일별 메시지 수
            daily_results = db.query(
                func.date(ChatMessageModel.timestamp).label('date'),
                func.count(ChatMessageModel.id).label('count')
            ).join(ChatSessionModel).filter(
                ChatSessionModel.brand_id == brand_id,
                ChatMessageModel.timestamp >= since
            ).group_by(
                func.date(ChatMessageModel.timestamp)
            ).order_by(
                func.date(ChatMessageModel.timestamp).desc()
            ).limit(30).all()

            daily_message_counts = [
                {"date": str(r.date), "count": r.count}
                for r in daily_results
            ]

            return ChatAnalytics(
                total_sessions=session_count,
                total_messages=message_count,
                avg_messages_per_session=avg_messages,
                avg_response_time_ms=avg_response_time or 0,
                grade_distribution=grade_distribution,
                question_type_distribution=question_type_distribution,
                daily_message_counts=daily_message_counts,
                top_topics=[]
            )

    def get_session_summary(self, session_id: str) -> Dict[str, Any]:
        """세션 요약 정보"""
        with self._get_db() as db:
            session = db.query(ChatSessionModel).filter(
                ChatSessionModel.id == session_id
            ).first()

            if not session:
                return {}

            # 메시지 정보
            messages = db.query(ChatMessageModel).filter(
                ChatMessageModel.session_id == session_id
            ).all()

            user_messages = [m.content for m in messages if m.role == 'user'][:5]
            grades = [m.grade for m in messages if m.grade]

            grade_summary = {}
            for g in set(grades):
                grade_summary[g] = grades.count(g)

            return {
                "session_id": session_id,
                "brand_id": session.brand_id,
                "message_count": session.message_count,
                "user_messages": user_messages,
                "grade_summary": grade_summary,
                "created_at": session.created_at.isoformat(),
                "updated_at": session.updated_at.isoformat()
            }

    # ========================================
    # Cleanup
    # ========================================

    def delete_old_sessions(
        self,
        brand_id: str,
        days_old: int = 90
    ) -> int:
        """오래된 세션 삭제"""
        cutoff = datetime.utcnow() - timedelta(days=days_old)

        with self._get_db() as db:
            # CASCADE로 메시지도 함께 삭제됨
            count = db.query(ChatSessionModel).filter(
                ChatSessionModel.brand_id == brand_id,
                ChatSessionModel.updated_at < cutoff
            ).delete()

        logger.info(f"Deleted {count} old sessions for brand {brand_id}")
        return count


# ============================================
# In-Memory Fallback Storage
# ============================================

class InMemoryChatStorage:
    """
    PostgreSQL 연결 실패 시 사용되는 인메모리 폴백 스토리지

    서버 재시작 시 데이터 손실됨 (개발/테스트 용도)
    """

    def __init__(self):
        self._sessions: Dict[str, ChatSession] = {}
        self._messages: Dict[str, List[StoredMessage]] = {}
        logger.warning("Using in-memory chat storage (PostgreSQL unavailable)")

    def create_session(
        self,
        brand_id: str,
        user_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> ChatSession:
        session_id = str(uuid.uuid4())
        now = datetime.utcnow()
        session = ChatSession(
            id=session_id,
            brand_id=brand_id,
            user_id=user_id,
            created_at=now,
            updated_at=now,
            message_count=0,
            metadata=metadata or {}
        )
        self._sessions[session_id] = session
        self._messages[session_id] = []
        return session

    def get_session(self, session_id: str) -> Optional[ChatSession]:
        return self._sessions.get(session_id)

    def get_or_create_session(
        self,
        session_id: Optional[str],
        brand_id: str,
        user_id: Optional[str] = None
    ) -> ChatSession:
        if session_id and session_id in self._sessions:
            return self._sessions[session_id]
        return self.create_session(brand_id, user_id)

    def list_sessions(
        self,
        brand_id: str,
        user_id: Optional[str] = None,
        limit: int = 50,
        offset: int = 0
    ) -> List[ChatSession]:
        sessions = [s for s in self._sessions.values() if s.brand_id == brand_id]
        if user_id:
            sessions = [s for s in sessions if s.user_id == user_id]
        sessions.sort(key=lambda x: x.updated_at, reverse=True)
        return sessions[offset:offset + limit]

    def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> StoredMessage:
        message = StoredMessage(
            id=str(uuid.uuid4()),
            session_id=session_id,
            role=role,
            content=content,
            timestamp=datetime.utcnow(),
            metadata=metadata or {}
        )
        if session_id not in self._messages:
            self._messages[session_id] = []
        self._messages[session_id].append(message)

        # Update session
        if session_id in self._sessions:
            self._sessions[session_id].message_count += 1
            self._sessions[session_id].updated_at = datetime.utcnow()

        return message

    def save_conversation_turn(
        self,
        session_id: str,
        user_message: str,
        assistant_message: str,
        metadata: Optional[Dict[str, Any]] = None
    ):
        self.add_message(session_id, "user", user_message)
        self.add_message(session_id, "assistant", assistant_message, metadata)

    def get_messages(
        self,
        session_id: str,
        limit: int = 100
    ) -> List[StoredMessage]:
        messages = self._messages.get(session_id, [])
        return messages[-limit:]

    def get_recent_context(
        self,
        session_id: str,
        max_messages: int = 10
    ) -> List[Dict[str, str]]:
        messages = self._messages.get(session_id, [])
        recent = messages[-max_messages:]
        return [{"role": m.role, "content": m.content} for m in recent]

    def get_session_summary(self, session_id: str) -> Optional[Dict[str, Any]]:
        session = self._sessions.get(session_id)
        if not session:
            return None
        messages = self._messages.get(session_id, [])
        return {
            "session_id": session_id,
            "message_count": len(messages),
            "created_at": session.created_at.isoformat(),
            "updated_at": session.updated_at.isoformat()
        }

    def get_analytics(self, brand_id: str, days: int = 30) -> ChatAnalytics:
        return ChatAnalytics()


# Singleton instance
_chat_storage = None
_storage_type: str = "none"


def get_chat_storage():
    """
    채팅 저장 서비스 싱글톤

    PostgreSQL 연결 실패 시 자동으로 인메모리 폴백 사용
    """
    global _chat_storage, _storage_type

    if _chat_storage is None:
        # PostgreSQL 연결 시도
        try:
            _chat_storage = ChatStorageService()
            _storage_type = "postgresql"
            logger.info("Chat storage: PostgreSQL connected")
        except Exception as e:
            logger.warning(f"PostgreSQL unavailable: {e}")
            logger.warning("Falling back to in-memory chat storage")
            _chat_storage = InMemoryChatStorage()
            _storage_type = "memory"

    return _chat_storage


def get_storage_type() -> str:
    """현재 스토리지 타입 반환"""
    return _storage_type

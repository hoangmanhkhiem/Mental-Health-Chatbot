"""
PHẦN 2 – Personalization Layer (Cá nhân hóa theo người dùng)

Mục tiêu:
- Ghi nhớ thông tin quan trọng của mỗi người
- Trò chuyện theo phong cách họ thích
- Gợi ý phù hợp với tính cách & cảm xúc
"""

import logging
import json
from typing import Dict, Optional, List, Any
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)


class CommunicationStyle(Enum):
    """Phong cách giao tiếp ưa thích."""
    BRIEF = "ngắn gọn"
    DETAILED = "chi tiết"
    HUMOROUS = "hài hước"
    SERIOUS = "nghiêm túc"
    PROFESSIONAL = "chuyên nghiệp"
    FRIENDLY = "thân thiện"


class TherapyGoal(Enum):
    """Mục tiêu trị liệu."""
    REDUCE_STRESS = "giảm stress"
    MANAGE_ANXIETY = "quản lý lo âu"
    IMPROVE_MOOD = "cải thiện tâm trạng"
    BETTER_SLEEP = "ngủ ngon hơn"
    BUILD_CONFIDENCE = "xây dựng tự tin"
    IMPROVE_RELATIONSHIPS = "cải thiện mối quan hệ"
    WORK_LIFE_BALANCE = "cân bằng công việc cuộc sống"
    GRIEF_SUPPORT = "hỗ trợ mất mát"
    GENERAL_WELLBEING = "sức khỏe tổng quát"


@dataclass
class StaticProfile:
    """Thông tin tĩnh của người dùng."""
    age_range: Optional[str] = None  # "18-25", "26-35", etc.
    primary_goals: List[str] = field(default_factory=list)
    preferred_style: str = "friendly"
    preferred_language: str = "vi"
    occupation: Optional[str] = None
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())


@dataclass
class DynamicState:
    """Thông tin động - thay đổi theo thời gian."""
    current_mood: Optional[str] = None
    mood_history: List[Dict] = field(default_factory=list)  # [{mood, timestamp}]
    active_topics: List[str] = field(default_factory=list)
    therapy_progress: Dict = field(default_factory=dict)  # {goal: progress_notes}
    recent_concerns: List[str] = field(default_factory=list)
    last_interaction: Optional[str] = None
    interaction_count: int = 0
    crisis_history: List[Dict] = field(default_factory=list)


@dataclass  
class ConversationContext:
    """Ngữ cảnh cuộc hội thoại hiện tại."""
    session_start: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    messages_in_session: int = 0
    current_topic: Optional[str] = None
    emotion_trajectory: List[Dict] = field(default_factory=list)  # Track emotion changes
    follow_up_needed: List[str] = field(default_factory=list)


@dataclass
class UserMemory:
    """Bộ nhớ tổng hợp của người dùng."""
    user_id: str
    static_profile: StaticProfile = field(default_factory=StaticProfile)
    dynamic_state: DynamicState = field(default_factory=DynamicState)
    conversation_context: ConversationContext = field(default_factory=ConversationContext)
    
    def to_dict(self) -> Dict:
        """Chuyển thành dictionary."""
        return {
            "user_id": self.user_id,
            "static_profile": asdict(self.static_profile),
            "dynamic_state": asdict(self.dynamic_state),
            "conversation_context": asdict(self.conversation_context)
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'UserMemory':
        """Tạo từ dictionary."""
        return cls(
            user_id=data["user_id"],
            static_profile=StaticProfile(**data.get("static_profile", {})),
            dynamic_state=DynamicState(**data.get("dynamic_state", {})),
            conversation_context=ConversationContext(**data.get("conversation_context", {}))
        )


class UserMemoryStore:
    """
    Lưu trữ và quản lý bộ nhớ người dùng.
    Hỗ trợ cả in-memory cache và database persistence.
    """
    
    def __init__(self, db_session_factory=None):
        """
        Khởi tạo User Memory Store.
        
        Args:
            db_session_factory: SQLAlchemy session factory cho database persistence
        """
        self._memory_cache: Dict[str, UserMemory] = {}
        self._db_session_factory = db_session_factory
        logger.info("✓ UserMemoryStore initialized")
    
    def get_memory(self, user_id: str) -> UserMemory:
        """
        Lấy bộ nhớ của người dùng.
        
        Args:
            user_id: ID người dùng
            
        Returns:
            UserMemory object
        """
        if user_id not in self._memory_cache:
            # Try to load from database
            memory = self._load_from_db(user_id)
            if memory is None:
                memory = UserMemory(user_id=user_id)
            self._memory_cache[user_id] = memory
        
        return self._memory_cache[user_id]
    
    def update_memory(self, user_id: str, updates: Dict[str, Any]) -> UserMemory:
        """
        Cập nhật bộ nhớ người dùng.
        
        Args:
            user_id: ID người dùng
            updates: Dict chứa các cập nhật
            
        Returns:
            Updated UserMemory
        """
        memory = self.get_memory(user_id)
        
        # Update static profile
        if "static_profile" in updates:
            for key, value in updates["static_profile"].items():
                if hasattr(memory.static_profile, key):
                    setattr(memory.static_profile, key, value)
        
        # Update dynamic state
        if "dynamic_state" in updates:
            for key, value in updates["dynamic_state"].items():
                if hasattr(memory.dynamic_state, key):
                    setattr(memory.dynamic_state, key, value)
        
        # Update conversation context
        if "conversation_context" in updates:
            for key, value in updates["conversation_context"].items():
                if hasattr(memory.conversation_context, key):
                    setattr(memory.conversation_context, key, value)
        
        # Update last interaction
        memory.dynamic_state.last_interaction = datetime.utcnow().isoformat()
        memory.dynamic_state.interaction_count += 1
        
        # Persist to database
        self._save_to_db(memory)
        
        return memory
    
    def record_mood(self, user_id: str, mood: str, intensity: str = "medium") -> None:
        """
        Ghi lại tâm trạng hiện tại.
        
        Args:
            user_id: ID người dùng
            mood: Tâm trạng
            intensity: Mức độ (low/medium/high)
        """
        memory = self.get_memory(user_id)
        
        mood_record = {
            "mood": mood,
            "intensity": intensity,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        memory.dynamic_state.current_mood = mood
        memory.dynamic_state.mood_history.append(mood_record)
        
        # Keep only last 50 mood records
        if len(memory.dynamic_state.mood_history) > 50:
            memory.dynamic_state.mood_history = memory.dynamic_state.mood_history[-50:]
        
        # Track in conversation context
        memory.conversation_context.emotion_trajectory.append(mood_record)
        
        self._save_to_db(memory)
    
    def add_topic(self, user_id: str, topic: str) -> None:
        """
        Thêm chủ đề đang theo dõi.
        
        Args:
            user_id: ID người dùng
            topic: Chủ đề
        """
        memory = self.get_memory(user_id)
        
        if topic not in memory.dynamic_state.active_topics:
            memory.dynamic_state.active_topics.append(topic)
            
        # Keep only last 10 topics
        if len(memory.dynamic_state.active_topics) > 10:
            memory.dynamic_state.active_topics = memory.dynamic_state.active_topics[-10:]
        
        memory.conversation_context.current_topic = topic
        self._save_to_db(memory)
    
    def add_concern(self, user_id: str, concern: str) -> None:
        """
        Thêm mối quan tâm gần đây.
        
        Args:
            user_id: ID người dùng
            concern: Mối quan tâm
        """
        memory = self.get_memory(user_id)
        
        if concern not in memory.dynamic_state.recent_concerns:
            memory.dynamic_state.recent_concerns.insert(0, concern)
        
        # Keep only last 5 concerns
        if len(memory.dynamic_state.recent_concerns) > 5:
            memory.dynamic_state.recent_concerns = memory.dynamic_state.recent_concerns[:5]
        
        self._save_to_db(memory)
    
    def set_therapy_goal(self, user_id: str, goal: str, notes: str = "") -> None:
        """
        Đặt mục tiêu trị liệu.
        
        Args:
            user_id: ID người dùng
            goal: Mục tiêu
            notes: Ghi chú tiến triển
        """
        memory = self.get_memory(user_id)
        
        if goal not in memory.static_profile.primary_goals:
            memory.static_profile.primary_goals.append(goal)
        
        memory.dynamic_state.therapy_progress[goal] = {
            "notes": notes,
            "updated_at": datetime.utcnow().isoformat()
        }
        
        self._save_to_db(memory)
    
    def update_therapy_progress(self, user_id: str, goal: str, progress_notes: str) -> None:
        """
        Cập nhật tiến triển trị liệu.
        
        Args:
            user_id: ID người dùng
            goal: Mục tiêu
            progress_notes: Ghi chú tiến triển
        """
        memory = self.get_memory(user_id)
        
        if goal not in memory.dynamic_state.therapy_progress:
            memory.dynamic_state.therapy_progress[goal] = {}
        
        memory.dynamic_state.therapy_progress[goal]["notes"] = progress_notes
        memory.dynamic_state.therapy_progress[goal]["updated_at"] = datetime.utcnow().isoformat()
        
        self._save_to_db(memory)
    
    def record_crisis(self, user_id: str, details: str) -> None:
        """
        Ghi lại sự kiện khủng hoảng.
        
        Args:
            user_id: ID người dùng
            details: Chi tiết sự kiện
        """
        memory = self.get_memory(user_id)
        
        crisis_record = {
            "details": details,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        memory.dynamic_state.crisis_history.append(crisis_record)
        self._save_to_db(memory)
    
    def add_follow_up(self, user_id: str, follow_up_topic: str) -> None:
        """
        Thêm chủ đề cần theo dõi.
        
        Args:
            user_id: ID người dùng
            follow_up_topic: Chủ đề cần theo dõi
        """
        memory = self.get_memory(user_id)
        
        if follow_up_topic not in memory.conversation_context.follow_up_needed:
            memory.conversation_context.follow_up_needed.append(follow_up_topic)
        
        self._save_to_db(memory)
    
    def get_follow_ups(self, user_id: str) -> List[str]:
        """
        Lấy danh sách chủ đề cần theo dõi.
        
        Args:
            user_id: ID người dùng
            
        Returns:
            List các chủ đề cần theo dõi
        """
        memory = self.get_memory(user_id)
        return memory.conversation_context.follow_up_needed
    
    def clear_follow_up(self, user_id: str, topic: str) -> None:
        """
        Xóa chủ đề đã theo dõi.
        
        Args:
            user_id: ID người dùng
            topic: Chủ đề đã xử lý
        """
        memory = self.get_memory(user_id)
        
        if topic in memory.conversation_context.follow_up_needed:
            memory.conversation_context.follow_up_needed.remove(topic)
        
        self._save_to_db(memory)
    
    def get_personalization_context(self, user_id: str) -> Dict:
        """
        Tạo context cá nhân hóa cho LLM.
        
        Args:
            user_id: ID người dùng
            
        Returns:
            Dict chứa thông tin cá nhân hóa
        """
        memory = self.get_memory(user_id)
        
        # Calculate days since first interaction
        try:
            first_date = datetime.fromisoformat(memory.static_profile.created_at)
            days_known = (datetime.utcnow() - first_date).days
        except:
            days_known = 0
        
        # Get recent mood trend
        mood_trend = "ổn định"
        if len(memory.dynamic_state.mood_history) >= 3:
            recent_moods = memory.dynamic_state.mood_history[-3:]
            negative_moods = ["buồn", "lo âu", "căng thẳng", "tuyệt vọng", "trầm cảm"]
            negative_count = sum(1 for m in recent_moods if m.get("mood", "").lower() in negative_moods)
            if negative_count >= 2:
                mood_trend = "cần chú ý"
            elif negative_count == 0:
                mood_trend = "tích cực"
        
        return {
            "user_profile": {
                "age_range": memory.static_profile.age_range,
                "goals": memory.static_profile.primary_goals,
                "preferred_style": memory.static_profile.preferred_style,
                "days_known": days_known,
                "total_interactions": memory.dynamic_state.interaction_count
            },
            "current_state": {
                "mood": memory.dynamic_state.current_mood,
                "mood_trend": mood_trend,
                "active_topics": memory.dynamic_state.active_topics,
                "recent_concerns": memory.dynamic_state.recent_concerns
            },
            "therapy_context": {
                "goals": memory.static_profile.primary_goals,
                "progress": memory.dynamic_state.therapy_progress,
                "follow_ups": memory.conversation_context.follow_up_needed
            },
            "session_info": {
                "current_topic": memory.conversation_context.current_topic,
                "messages_count": memory.conversation_context.messages_in_session
            }
        }
    
    def should_check_in(self, user_id: str, hours_threshold: int = 24) -> bool:
        """
        Kiểm tra xem có nên gửi check-in không.
        
        Args:
            user_id: ID người dùng
            hours_threshold: Số giờ không tương tác
            
        Returns:
            True nếu nên gửi check-in
        """
        memory = self.get_memory(user_id)
        
        if not memory.dynamic_state.last_interaction:
            return False
        
        try:
            last_time = datetime.fromisoformat(memory.dynamic_state.last_interaction)
            hours_since = (datetime.utcnow() - last_time).total_seconds() / 3600
            return hours_since >= hours_threshold
        except:
            return False
    
    def start_new_session(self, user_id: str) -> None:
        """
        Bắt đầu phiên hội thoại mới.
        
        Args:
            user_id: ID người dùng
        """
        memory = self.get_memory(user_id)
        memory.conversation_context = ConversationContext()
        self._save_to_db(memory)
    
    def increment_session_messages(self, user_id: str) -> None:
        """
        Tăng số tin nhắn trong phiên.
        
        Args:
            user_id: ID người dùng
        """
        memory = self.get_memory(user_id)
        memory.conversation_context.messages_in_session += 1
        self._save_to_db(memory)
    
    def _load_from_db(self, user_id: str) -> Optional[UserMemory]:
        """Load user memory from database."""
        if not self._db_session_factory:
            return None
        
        try:
            # Implementation depends on database schema
            # For now, return None to use cache only
            return None
        except Exception as e:
            logger.error(f"❌ Error loading user memory: {e}")
            return None
    
    def _save_to_db(self, memory: UserMemory) -> None:
        """Save user memory to database."""
        if not self._db_session_factory:
            return
        
        try:
            # Implementation depends on database schema
            # For now, just keep in cache
            self._memory_cache[memory.user_id] = memory
        except Exception as e:
            logger.error(f"❌ Error saving user memory: {e}")


# Prompt templates for personalization
PERSONALIZATION_PROMPT = """Dựa trên hồ sơ cá nhân của người dùng và tâm trạng hiện tại của họ, 
hãy điều chỉnh giọng văn, mức độ chi tiết và cách diễn đạt sao cho phù hợp nhất.

HỒ SƠ NGƯỜI DÙNG:
- Độ tuổi: {age_range}
- Mục tiêu: {goals}
- Phong cách ưa thích: {preferred_style}
- Số ngày quen biết: {days_known}

TRẠNG THÁI HIỆN TẠI:
- Tâm trạng: {current_mood}
- Xu hướng tâm trạng: {mood_trend}
- Chủ đề đang quan tâm: {active_topics}
- Mối lo ngại gần đây: {recent_concerns}

TIẾN TRIỂN TRỊ LIỆU:
- Mục tiêu đang theo đuổi: {therapy_goals}
- Chủ đề cần theo dõi: {follow_ups}

Hãy phản hồi phù hợp với ngữ cảnh cá nhân này."""

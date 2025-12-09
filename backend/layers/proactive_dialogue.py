"""
PHẦN 8 – Proactive Dialogue Engine (Chatbot chủ động hỏi & dẫn dắt)

Mục tiêu:
- Không đợi người dùng hỏi
- Chủ động hỏi thăm, theo dõi tiến triển
- Giống therapist thực thụ
- Sử dụng dữ liệu lịch sử + RAG để đặt câu hỏi phù hợp

Modules:
8.1 - Conversation State Tracker
8.2 - Proactive Question Generator
8.3 - Contextual RAG Trigger
8.4 - Proactivity Rules
"""

import logging
from typing import Dict, Optional, List, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum

logger = logging.getLogger(__name__)


class ConversationState(Enum):
    """Trạng thái cuộc hội thoại."""
    INITIAL_GREETING = "initial_greeting"
    FOLLOW_UP = "follow_up"
    CHECK_IN = "check_in"
    DEEP_ISSUE_EXPLORATION = "deep_issue_exploration"
    SOLUTION_DISCUSSION = "solution_discussion"
    CLOSURE = "closure"
    RETURNING_USER = "returning_user"


class ProactiveRuleType(Enum):
    """Loại quy tắc chủ động."""
    AFTER_SHARING = "after_sharing"  # Sau khi chia sẻ cảm xúc
    NEGATIVE_EMOTION = "negative_emotion"  # Cảm xúc tiêu cực
    INACTIVE_24H = "inactive_24h"  # 24h không tương tác
    SHORT_RESPONSE = "short_response"  # Trả lời quá ngắn
    THERAPY_CYCLE = "therapy_cycle"  # Theo chu kỳ trị liệu
    FOLLOW_UP_TOPIC = "follow_up_topic"  # Theo dõi chủ đề trước


@dataclass
class ProactiveAction:
    """Hành động chủ động được đề xuất."""
    should_act: bool
    rule_type: ProactiveRuleType
    question: str
    context: Dict
    priority: int  # 1-5, cao hơn = ưu tiên hơn


@dataclass
class ConversationTracker:
    """Theo dõi trạng thái cuộc hội thoại."""
    user_id: str
    current_state: ConversationState = ConversationState.INITIAL_GREETING
    messages_count: int = 0
    last_message_time: Optional[datetime] = None
    last_user_emotion: Optional[str] = None
    last_topic: Optional[str] = None
    pending_follow_ups: List[str] = field(default_factory=list)
    session_topics: List[str] = field(default_factory=list)
    emotion_trajectory: List[Dict] = field(default_factory=list)


class ProactiveDialogueEngine:
    """
    Engine điều khiển dialogue chủ động.
    Kết hợp Conversation State Tracker, Question Generator, và Rules.
    """
    
    # Question templates theo state
    STATE_QUESTIONS = {
        ConversationState.INITIAL_GREETING: [
            "Hôm nay bạn thấy thế nào? 🌿",
            "Chào bạn! Bạn muốn chia sẻ điều gì với mình hôm nay?",
            "Xin chào! Tâm trạng của bạn hôm nay ra sao?"
        ],
        ConversationState.CHECK_IN: [
            "Hôm nay bạn thấy tâm trạng thế nào rồi?",
            "Bạn có muốn chia sẻ gì với mình hôm nay không?",
            "Chào bạn, mình muốn hỏi thăm - bạn khỏe không?"
        ],
        ConversationState.RETURNING_USER: [
            "Chào mừng bạn quay lại! 🌿 Bạn thấy thế nào từ lần trước chúng ta nói chuyện?",
            "Rất vui được gặp lại bạn! Có điều gì mới bạn muốn chia sẻ không?"
        ]
    }
    
    # Question templates theo rule
    RULE_QUESTIONS = {
        ProactiveRuleType.AFTER_SHARING: [
            "Điều gì khiến bạn cảm thấy như vậy?",
            "Bạn có thể chia sẻ thêm về điều đó không?",
            "Cảm xúc này xuất hiện từ khi nào?"
        ],
        ProactiveRuleType.NEGATIVE_EMOTION: [
            "Cảm giác này kéo dài bao lâu rồi?",
            "Có điều gì cụ thể gây ra cảm xúc này không?",
            "Bạn đã thử làm gì để cảm thấy tốt hơn chưa?"
        ],
        ProactiveRuleType.SHORT_RESPONSE: [
            "Ổn theo nghĩa nào nhỉ?",
            "Bạn có thể chia sẻ thêm không?",
            "Mình muốn hiểu hơn - bạn cảm thấy thế nào cụ thể?"
        ],
        ProactiveRuleType.THERAPY_CYCLE: [
            "Tuần này bạn có thử kỹ thuật thư giãn mình đã chia sẻ không?",
            "Tiến triển của bạn với mục tiêu đặt ra thế nào rồi?",
            "Bạn có nhận thấy thay đổi gì từ tuần trước không?"
        ],
        ProactiveRuleType.INACTIVE_24H: [
            "Hôm nay bạn thấy tâm trạng thế nào?",
            "Mình muốn hỏi thăm bạn - mọi thứ ổn chứ?",
            "Chào bạn, bạn khỏe không?"
        ]
    }
    
    # Question generation prompt
    QUESTION_GENERATION_PROMPT = """Dựa trên lịch sử trò chuyện và hồ sơ người dùng, 
hãy sinh ra một câu hỏi tự nhiên để tiếp tục cuộc trò chuyện theo hướng trị liệu.

LỊCH SỬ GẦN ĐÂY:
{conversation_history}

HỒ SƠ NGƯỜI DÙNG:
- Mục tiêu: {goals}
- Chủ đề đang theo dõi: {active_topics}
- Tâm trạng gần đây: {recent_mood}
- Mối quan tâm: {concerns}

CẢM XÚC HIỆN TẠI: {current_emotion}

YÊU CẦU:
- Câu hỏi phải ngắn gọn, nhẹ nhàng, mang tính lắng nghe
- Không được ép buộc người dùng trả lời
- Không hỏi về chủ đề nhạy cảm (lạm dụng, thuốc tâm thần, v.v.)
- Liên quan đến những gì đã thảo luận

Chỉ trả về câu hỏi, không có gì khác."""

    CONTEXTUAL_FOLLOW_UP_PROMPT = """Người dùng hôm qua đã chia sẻ:
"{previous_message}"

Hôm nay hãy tạo một câu follow-up tự nhiên kết hợp:
1. Nhắc đến vấn đề họ đã chia sẻ
2. Hỏi thăm tiến triển
3. Có thể đề cập kiến thức từ context sau (nếu phù hợp):
{rag_context}

Câu follow-up (thân thiện, ngắn gọn):"""

    def __init__(self, model=None, memory_store=None, safety_layer=None):
        """
        Khởi tạo Proactive Dialogue Engine.
        
        Args:
            model: Gemini model
            memory_store: UserMemoryStore instance
            safety_layer: SafetyEthicsLayer instance
        """
        self.model = model
        self.memory_store = memory_store
        self.safety_layer = safety_layer
        self._trackers: Dict[str, ConversationTracker] = {}
        logger.info("✓ ProactiveDialogueEngine initialized")
    
    def get_tracker(self, user_id: str) -> ConversationTracker:
        """
        Lấy hoặc tạo conversation tracker cho user.
        
        Args:
            user_id: ID người dùng
            
        Returns:
            ConversationTracker
        """
        if user_id not in self._trackers:
            self._trackers[user_id] = ConversationTracker(user_id=user_id)
        return self._trackers[user_id]
    
    def update_state(
        self,
        user_id: str,
        user_message: str,
        emotion: str = None,
        topic: str = None
    ) -> ConversationState:
        """
        Cập nhật trạng thái cuộc hội thoại.
        
        Args:
            user_id: ID người dùng
            user_message: Tin nhắn người dùng
            emotion: Cảm xúc phát hiện được
            topic: Chủ đề phát hiện được
            
        Returns:
            Trạng thái mới
        """
        tracker = self.get_tracker(user_id)
        now = datetime.utcnow()
        
        # Update basic info
        tracker.messages_count += 1
        tracker.last_message_time = now
        
        if emotion:
            tracker.last_user_emotion = emotion
            tracker.emotion_trajectory.append({
                "emotion": emotion,
                "timestamp": now.isoformat()
            })
        
        if topic:
            tracker.last_topic = topic
            if topic not in tracker.session_topics:
                tracker.session_topics.append(topic)
        
        # Determine state transition
        if tracker.messages_count == 1:
            # Check if returning user
            if self.memory_store:
                memory = self.memory_store.get_memory(user_id)
                if memory.dynamic_state.interaction_count > 1:
                    tracker.current_state = ConversationState.RETURNING_USER
                else:
                    tracker.current_state = ConversationState.INITIAL_GREETING
            else:
                tracker.current_state = ConversationState.INITIAL_GREETING
        
        elif tracker.messages_count <= 3:
            tracker.current_state = ConversationState.FOLLOW_UP
        
        elif emotion and emotion in ["buồn", "lo âu", "tuyệt vọng", "trầm cảm"]:
            tracker.current_state = ConversationState.DEEP_ISSUE_EXPLORATION
        
        elif any(word in user_message.lower() for word in ["làm sao", "cách nào", "giúp", "giải pháp"]):
            tracker.current_state = ConversationState.SOLUTION_DISCUSSION
        
        return tracker.current_state
    
    def should_be_proactive(self, user_id: str, user_message: str) -> List[ProactiveAction]:
        """
        Kiểm tra xem có nên chủ động không và theo quy tắc nào.
        
        Args:
            user_id: ID người dùng
            user_message: Tin nhắn người dùng
            
        Returns:
            List các ProactiveAction được đề xuất
        """
        tracker = self.get_tracker(user_id)
        actions = []
        
        message_lower = user_message.lower()
        
        # Rule 1: Sau câu chia sẻ mở
        sharing_indicators = ["em hơi", "tôi cảm thấy", "mình đang", "bị", "gặp phải"]
        if any(indicator in message_lower for indicator in sharing_indicators):
            actions.append(ProactiveAction(
                should_act=True,
                rule_type=ProactiveRuleType.AFTER_SHARING,
                question=self._select_question(ProactiveRuleType.AFTER_SHARING),
                context={"trigger": "sharing_detected"},
                priority=4
            ))
        
        # Rule 2: Cảm xúc tiêu cực
        negative_emotions = ["buồn", "chán", "lo", "sợ", "mệt", "stress", "căng thẳng"]
        if any(emotion in message_lower for emotion in negative_emotions):
            actions.append(ProactiveAction(
                should_act=True,
                rule_type=ProactiveRuleType.NEGATIVE_EMOTION,
                question=self._select_question(ProactiveRuleType.NEGATIVE_EMOTION),
                context={"trigger": "negative_emotion"},
                priority=5
            ))
        
        # Rule 4: Trả lời quá ngắn
        if len(user_message.strip()) < 10:
            short_replies = ["ổn", "ok", "được", "bình thường", "không", "có"]
            if any(reply == message_lower.strip().rstrip(".!") for reply in short_replies):
                actions.append(ProactiveAction(
                    should_act=True,
                    rule_type=ProactiveRuleType.SHORT_RESPONSE,
                    question=self._select_question(ProactiveRuleType.SHORT_RESPONSE),
                    context={"trigger": "short_response"},
                    priority=3
                ))
        
        # Rule 5: Follow-up từ lần trước
        if tracker.pending_follow_ups:
            topic = tracker.pending_follow_ups[0]
            actions.append(ProactiveAction(
                should_act=True,
                rule_type=ProactiveRuleType.FOLLOW_UP_TOPIC,
                question=f"Về vấn đề {topic} mà bạn nhắc đến, hôm nay bạn thấy thế nào?",
                context={"topic": topic},
                priority=2
            ))
        
        # Sort by priority
        actions.sort(key=lambda x: x.priority, reverse=True)
        
        return actions
    
    def check_inactive_users(self, hours_threshold: int = 24) -> Dict[str, ProactiveAction]:
        """
        Kiểm tra người dùng không hoạt động.
        
        Args:
            hours_threshold: Ngưỡng giờ không hoạt động
            
        Returns:
            Dict user_id -> ProactiveAction
        """
        inactive_actions = {}
        now = datetime.utcnow()
        
        for user_id, tracker in self._trackers.items():
            if tracker.last_message_time:
                hours_since = (now - tracker.last_message_time).total_seconds() / 3600
                
                if hours_since >= hours_threshold:
                    # Check if we should check in
                    question = self._generate_check_in_question(user_id, tracker)
                    
                    inactive_actions[user_id] = ProactiveAction(
                        should_act=True,
                        rule_type=ProactiveRuleType.INACTIVE_24H,
                        question=question,
                        context={
                            "hours_inactive": hours_since,
                            "last_emotion": tracker.last_user_emotion,
                            "last_topic": tracker.last_topic
                        },
                        priority=3
                    )
        
        return inactive_actions
    
    def _select_question(self, rule_type: ProactiveRuleType) -> str:
        """Chọn câu hỏi từ templates."""
        import random
        questions = self.RULE_QUESTIONS.get(rule_type, ["Bạn có muốn chia sẻ thêm không?"])
        return random.choice(questions)
    
    def _generate_check_in_question(
        self,
        user_id: str,
        tracker: ConversationTracker
    ) -> str:
        """Tạo câu hỏi check-in dựa trên context."""
        import random
        
        base_questions = self.RULE_QUESTIONS[ProactiveRuleType.INACTIVE_24H]
        
        # Personalize if possible
        if tracker.last_topic:
            personalized = f"Chào bạn! Lần trước chúng ta nói về {tracker.last_topic}. Hôm nay bạn thấy thế nào về điều đó?"
            return personalized
        
        if tracker.last_user_emotion:
            if tracker.last_user_emotion in ["buồn", "lo âu", "căng thẳng"]:
                return "Chào bạn! Mình muốn hỏi thăm - bạn thấy tốt hơn chưa? 🌿"
        
        return random.choice(base_questions)
    
    def generate_proactive_question(
        self,
        user_id: str,
        conversation_history: List[Dict] = None,
        rag_context: str = ""
    ) -> Optional[str]:
        """
        Tạo câu hỏi chủ động sử dụng LLM.
        
        Args:
            user_id: ID người dùng
            conversation_history: Lịch sử hội thoại
            rag_context: Context từ RAG
            
        Returns:
            Câu hỏi hoặc None
        """
        if not self.model:
            return self._get_fallback_question(user_id)
        
        try:
            # Get user context
            user_context = {}
            if self.memory_store:
                user_context = self.memory_store.get_personalization_context(user_id)
            
            tracker = self.get_tracker(user_id)
            
            # Format history
            history_text = ""
            if conversation_history:
                history_text = "\n".join([
                    f"{msg['role']}: {msg['content'][:100]}..."
                    for msg in conversation_history[-5:]
                ])
            
            prompt = self.QUESTION_GENERATION_PROMPT.format(
                conversation_history=history_text or "Không có lịch sử",
                goals=", ".join(user_context.get("therapy_context", {}).get("goals", [])) or "Chưa xác định",
                active_topics=", ".join(tracker.session_topics[-3:]) or "Chưa có",
                recent_mood=tracker.last_user_emotion or "Chưa biết",
                concerns=", ".join(user_context.get("current_state", {}).get("recent_concerns", [])) or "Chưa có",
                current_emotion=tracker.last_user_emotion or "Chưa xác định"
            )
            
            response = self.model.generate_content(prompt)
            question = response.text.strip()
            
            # Safety check
            if self.safety_layer:
                check = self.safety_layer.check_proactive_question(question)
                if not check.is_safe:
                    logger.warning(f"⚠️ Generated question blocked: {check.reason}")
                    return self._get_fallback_question(user_id)
            
            return question
            
        except Exception as e:
            logger.error(f"❌ Question generation error: {e}")
            return self._get_fallback_question(user_id)
    
    def generate_contextual_follow_up(
        self,
        user_id: str,
        previous_message: str,
        rag_context: str = ""
    ) -> Optional[str]:
        """
        Tạo câu follow-up kết hợp RAG context.
        
        Args:
            user_id: ID người dùng
            previous_message: Tin nhắn trước đó
            rag_context: Context từ RAG
            
        Returns:
            Câu follow-up
        """
        if not self.model:
            tracker = self.get_tracker(user_id)
            if tracker.last_topic:
                return f"Về vấn đề {tracker.last_topic} bạn đề cập, hôm nay bạn thấy thế nào?"
            return "Bạn có muốn chia sẻ thêm về điều đó không?"
        
        try:
            prompt = self.CONTEXTUAL_FOLLOW_UP_PROMPT.format(
                previous_message=previous_message[:200],
                rag_context=rag_context[:500] if rag_context else "Không có context cụ thể"
            )
            
            response = self.model.generate_content(prompt)
            follow_up = response.text.strip()
            
            # Safety check
            if self.safety_layer:
                check = self.safety_layer.check_proactive_question(follow_up)
                if not check.is_safe:
                    return "Hôm nay bạn thấy thế nào về điều đó?"
            
            return follow_up
            
        except Exception as e:
            logger.error(f"❌ Follow-up generation error: {e}")
            return "Bạn có muốn chia sẻ thêm về điều đó không?"
    
    def _get_fallback_question(self, user_id: str) -> str:
        """Lấy câu hỏi fallback."""
        import random
        tracker = self.get_tracker(user_id)
        
        state = tracker.current_state
        if state in self.STATE_QUESTIONS:
            return random.choice(self.STATE_QUESTIONS[state])
        
        return "Bạn có muốn chia sẻ thêm với mình không?"
    
    def add_follow_up_topic(self, user_id: str, topic: str) -> None:
        """
        Thêm chủ đề cần follow-up.
        
        Args:
            user_id: ID người dùng
            topic: Chủ đề
        """
        tracker = self.get_tracker(user_id)
        if topic not in tracker.pending_follow_ups:
            tracker.pending_follow_ups.append(topic)
    
    def clear_follow_up(self, user_id: str, topic: str = None) -> None:
        """
        Xóa chủ đề đã follow-up.
        
        Args:
            user_id: ID người dùng
            topic: Chủ đề (None = xóa tất cả)
        """
        tracker = self.get_tracker(user_id)
        if topic:
            if topic in tracker.pending_follow_ups:
                tracker.pending_follow_ups.remove(topic)
        else:
            tracker.pending_follow_ups.clear()
    
    def get_conversation_summary(self, user_id: str) -> Dict:
        """
        Lấy tóm tắt cuộc hội thoại.
        
        Args:
            user_id: ID người dùng
            
        Returns:
            Dict với thông tin tóm tắt
        """
        tracker = self.get_tracker(user_id)
        
        return {
            "user_id": user_id,
            "current_state": tracker.current_state.value,
            "messages_count": tracker.messages_count,
            "session_topics": tracker.session_topics,
            "last_emotion": tracker.last_user_emotion,
            "pending_follow_ups": tracker.pending_follow_ups,
            "emotion_trajectory": tracker.emotion_trajectory[-5:]
        }
    
    def should_include_proactive_element(
        self,
        user_id: str,
        bot_response: str
    ) -> Tuple[bool, Optional[str]]:
        """
        Kiểm tra xem có nên thêm yếu tố chủ động vào response không.
        
        Args:
            user_id: ID người dùng
            bot_response: Phản hồi của bot
            
        Returns:
            (should_include, proactive_addition)
        """
        tracker = self.get_tracker(user_id)
        
        # Don't add if response already ends with question
        if bot_response.strip().endswith("?"):
            return False, None
        
        # Add proactive element based on state
        if tracker.current_state == ConversationState.DEEP_ISSUE_EXPLORATION:
            return True, "\n\nBạn có muốn chia sẻ thêm về điều đó không?"
        
        if tracker.messages_count >= 3 and tracker.messages_count % 3 == 0:
            return True, "\n\nMình ở đây lắng nghe nếu bạn cần. 💚"
        
        return False, None


# Utility functions
def detect_topic_from_message(message: str) -> Optional[str]:
    """Phát hiện chủ đề từ tin nhắn."""
    topic_keywords = {
        "lo âu giao tiếp": ["lo âu", "giao tiếp", "nói chuyện", "người khác"],
        "stress công việc": ["stress", "công việc", "deadline", "sếp", "đồng nghiệp"],
        "mối quan hệ": ["bạn bè", "người yêu", "gia đình", "mối quan hệ"],
        "giấc ngủ": ["ngủ", "mất ngủ", "không ngủ được", "thức dậy"],
        "tự tin": ["tự tin", "tự ti", "không dám", "sợ thất bại"],
        "trầm cảm": ["trầm cảm", "buồn", "chán nản", "mất hứng thú"]
    }
    
    message_lower = message.lower()
    
    for topic, keywords in topic_keywords.items():
        if sum(1 for k in keywords if k in message_lower) >= 2:
            return topic
    
    return None

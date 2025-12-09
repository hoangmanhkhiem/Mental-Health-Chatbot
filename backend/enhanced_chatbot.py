"""
Enhanced Therapeutic Chatbot with All 8 Enhancement Layers
==========================================================

Tích hợp đầy đủ 8 module nâng cao từ doc.md:
1. Conversational Layer
2. Personalization Layer
3. Emotional Understanding Layer
4. Storytelling Therapy Mode
5. RAG Precision Boost
6. Reasoning Layer
7. Safety & Ethics Layer
8. Proactive Dialogue Engine

Pipeline:
User message or silence →
Emotion Analyzer →
Memory Reader →
Conversation State Tracker →
Proactive Rule Engine →
(If triggered) Proactive Question Generator →
Optional RAG Fetch →
Final Naturalized Question (tone friendly & empathetic)
"""

import os
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import google.generativeai as genai
from sqlalchemy.orm import Session

# Import all layers
from layers import (
    EmotionAnalyzer,
    UserMemoryStore,
    ConversationalLayer,
    StorytellingTherapy,
    RAGPrecisionBoost,
    ReasoningLayer,
    SafetyEthicsLayer,
    ProactiveDialogueEngine
)

# Import original components
from rag_system_v2 import TherapeuticRAG
from database import get_conversation_history, get_active_conversation, save_message

logger = logging.getLogger(__name__)


@dataclass
class EnhancedResponse:
    """Response object với metadata."""
    response: str
    is_crisis: bool
    emotion_detected: Optional[str]
    emotion_intensity: Optional[str]
    proactive_elements: List[str]
    used_rag: bool
    used_storytelling: bool
    safety_flags: List[str]
    user_id: Optional[str]


class EnhancedTherapeuticChatbot:
    """
    Chatbot trị liệu nâng cao với đầy đủ 8 layers.
    """
    
    # Crisis keywords (kept for compatibility)
    CRISIS_KEYWORDS = [
        "suicide", "suicidal", "kill myself", "end my life", "want to die",
        "self-harm", "hurt myself", "cutting", "overdose", "no reason to live",
        "better off dead", "hopeless", "can't go on",
        "tự tử", "muốn chết", "kết thúc cuộc sống", "tự làm đau"
    ]

    def __init__(
        self,
        google_api_key: str,
        rag_system: Optional[TherapeuticRAG] = None,
        db_session_factory=None,
        enable_all_layers: bool = True
    ):
        """
        Khởi tạo Enhanced Chatbot.
        
        Args:
            google_api_key: Google API key cho Gemini
            rag_system: RAG system instance
            db_session_factory: Database session factory
            enable_all_layers: Bật tất cả layers
        """
        # Initialize Gemini model
        genai.configure(api_key=google_api_key)
        
        models_to_try = [
            'gemini-2.0-flash'
        ]

        self.model = None
        for model_name in models_to_try:
            try:
                self.model = genai.GenerativeModel(model_name)
                logger.info(f"✓ Using model: {model_name}")
                break
            except Exception as e:
                logger.warning(f"⚠️ Model {model_name} not available: {e}")
        
        if self.model is None:
            self.model = genai.GenerativeModel('gemini-2.5-flash')
        
        self.rag = rag_system
        
        # Initialize all layers
        if enable_all_layers:
            self._init_all_layers(db_session_factory)
        else:
            self._init_minimal_layers()
        
        logger.info("✓ EnhancedTherapeuticChatbot initialized with all layers")

    def _init_all_layers(self, db_session_factory=None):
        """Khởi tạo tất cả layers."""
        # Layer 3: Emotion Analyzer
        self.emotion_analyzer = EmotionAnalyzer(model=self.model)
        
        # Layer 2: User Memory Store
        self.memory_store = UserMemoryStore(db_session_factory=db_session_factory)
        
        # Layer 1: Conversational Layer
        self.conversational = ConversationalLayer(model=self.model)
        
        # Layer 4: Storytelling Therapy
        self.storytelling = StorytellingTherapy(model=self.model)
        
        # Layer 5: RAG Precision Boost
        self.rag_precision = RAGPrecisionBoost(model=self.model, rag_system=self.rag)
        
        # Layer 6: Reasoning Layer
        self.reasoning = ReasoningLayer(model=self.model)
        
        # Layer 7: Safety & Ethics
        self.safety = SafetyEthicsLayer(model=self.model)
        
        # Layer 8: Proactive Dialogue
        self.proactive = ProactiveDialogueEngine(
            model=self.model,
            memory_store=self.memory_store,
            safety_layer=self.safety
        )

    def _init_minimal_layers(self):
        """Khởi tạo layers tối thiểu."""
        self.emotion_analyzer = EmotionAnalyzer()
        self.memory_store = UserMemoryStore()
        self.conversational = ConversationalLayer()
        self.storytelling = StorytellingTherapy()
        self.rag_precision = RAGPrecisionBoost(rag_system=self.rag)
        self.reasoning = ReasoningLayer()
        self.safety = SafetyEthicsLayer()
        self.proactive = ProactiveDialogueEngine(
            memory_store=self.memory_store,
            safety_layer=self.safety
        )

    def generate_response(
        self,
        db: Session,
        user_id: str,
        user_message: str,
        use_rag: bool = True,
        use_storytelling: bool = False,
        use_proactive: bool = True
    ) -> EnhancedResponse:
        """
        Tạo phản hồi với full pipeline.
        
        Args:
            db: Database session
            user_id: ID người dùng
            user_message: Tin nhắn người dùng
            use_rag: Sử dụng RAG
            use_storytelling: Sử dụng storytelling mode
            use_proactive: Sử dụng proactive elements
            
        Returns:
            EnhancedResponse
        """
        try:
            # ===== PHASE 1: Safety Check =====
            user_safety = self.safety.check_user_message(user_message)
            
            if user_safety.risk_level.value in ["high", "critical"]:
                logger.warning(f"⚠️ Crisis detected for user {user_id}")
                
                # Record crisis
                self.memory_store.record_crisis(user_id, user_message[:100])
                
                return EnhancedResponse(
                    response=self.safety.get_crisis_response(),
                    is_crisis=True,
                    emotion_detected="critical",
                    emotion_intensity="critical",
                    proactive_elements=[],
                    used_rag=False,
                    used_storytelling=False,
                    safety_flags=user_safety.flags,
                    user_id=user_id
                )
            
            # ===== PHASE 2: Emotion Analysis =====
            emotion_analysis = self.emotion_analyzer.analyze(user_message)
            emotion_str = emotion_analysis.primary_emotion.value
            intensity_str = emotion_analysis.intensity.value
            
            logger.info(f"Emotion: {emotion_str}, Intensity: {intensity_str}")
            
            # ===== PHASE 3: Memory & Personalization =====
            # Update memory
            self.memory_store.record_mood(user_id, emotion_str, intensity_str)
            self.memory_store.increment_session_messages(user_id)
            
            # Get personalization context
            personalization = self.memory_store.get_personalization_context(user_id)
            
            # ===== PHASE 4: Proactive State Update =====
            topic = self._detect_topic(user_message)
            if topic:
                self.memory_store.add_topic(user_id, topic)
            
            self.proactive.update_state(user_id, user_message, emotion_str, topic)
            
            # Check proactive actions
            proactive_actions = []
            if use_proactive:
                proactive_actions = self.proactive.should_be_proactive(user_id, user_message)
            
            # ===== PHASE 5: RAG Retrieval =====
            rag_context = ""
            rag_contexts = []
            used_rag = False
            
            if use_rag and self.rag:
                try:
                    results = self.rag_precision.retrieve_with_precision(
                        db, user_message, k=5,
                        use_multi_query=True,
                        use_hybrid=True,
                        use_rerank=True
                    )
                    
                    if results:
                        rag_context, citations = self.rag_precision.get_context_with_citations(results)
                        rag_contexts = [r.content for r in results]
                        used_rag = True
                        logger.info(f"✓ Retrieved {len(results)} RAG contexts")
                except Exception as e:
                    logger.warning(f"⚠️ RAG retrieval failed: {e}")
            
            # ===== PHASE 6: Response Generation =====
            if use_storytelling and emotion_analysis.requires_empathy:
                # Use storytelling therapy
                response = self._generate_storytelling_response(
                    user_message, rag_context, emotion_analysis
                )
                used_storytelling = True
            else:
                # Use reasoning layer with CoT
                response = self._generate_reasoned_response(
                    user_message, rag_context, rag_contexts,
                    emotion_analysis, personalization
                )
                used_storytelling = False
            
            # ===== PHASE 7: Conversational Transform =====
            style = self.conversational.select_style(emotion_analysis)
            
            if emotion_analysis.requires_empathy:
                response = self.conversational.add_empathy_prefix(emotion_analysis, response)
            
            # ===== PHASE 8: Proactive Elements =====
            proactive_elements = []
            if use_proactive and proactive_actions:
                top_action = proactive_actions[0]
                if top_action.should_act:
                    # Add proactive element to response
                    if not response.strip().endswith("?"):
                        response += f"\n\n{top_action.question}"
                        proactive_elements.append(top_action.question)
            
            # ===== PHASE 9: Safety Post-Check =====
            final_response, safety_check = self.safety.process_response(
                response, user_message
            )
            
            # ===== PHASE 10: Memory Update =====
            if topic:
                self.proactive.add_follow_up_topic(user_id, topic)
            
            return EnhancedResponse(
                response=final_response,
                is_crisis=False,
                emotion_detected=emotion_str,
                emotion_intensity=intensity_str,
                proactive_elements=proactive_elements,
                used_rag=used_rag,
                used_storytelling=used_storytelling,
                safety_flags=safety_check.flags,
                user_id=user_id
            )
            
        except Exception as e:
            logger.error(f"❌ Error generating response: {e}")
            return EnhancedResponse(
                response="Mình xin lỗi, có chút trục trặc. Bạn có thể chia sẻ lại được không?",
                is_crisis=False,
                emotion_detected=None,
                emotion_intensity=None,
                proactive_elements=[],
                used_rag=False,
                used_storytelling=False,
                safety_flags=[str(e)],
                user_id=user_id
            )

    def _generate_storytelling_response(
        self,
        user_message: str,
        rag_context: str,
        emotion_analysis
    ) -> str:
        """Tạo response với storytelling therapy."""
        # Determine therapy approach
        approach = self.storytelling.suggest_therapy_approach(user_message)
        
        # Generate story
        story = self.storytelling.generate_story(
            issue=user_message,
            context=rag_context,
            emotion=emotion_analysis.primary_emotion.value,
            approach=approach
        )
        
        return story

    def _generate_reasoned_response(
        self,
        user_message: str,
        rag_context: str,
        rag_contexts: List[str],
        emotion_analysis,
        personalization: Dict
    ) -> str:
        """Tạo response với reasoning layer."""
        # Build natural prompt
        prompt = self.conversational.build_natural_prompt(
            user_message=user_message,
            rag_context=rag_contexts,
            emotion_analysis=emotion_analysis,
            personalization_context=personalization
        )
        
        # Use reasoning layer
        result = self.reasoning.generate_with_reasoning(
            query=user_message,
            context=rag_context,
            contexts=rag_contexts if len(rag_contexts) > 1 else None,
            use_cot=True,
            use_refinement=True,
            use_synthesis=len(rag_contexts) > 1
        )
        
        return result.final_response

    def _detect_topic(self, message: str) -> Optional[str]:
        """Phát hiện chủ đề từ tin nhắn."""
        from layers.proactive_dialogue import detect_topic_from_message
        return detect_topic_from_message(message)

    def generate_greeting(self, user_id: str) -> str:
        """
        Tạo lời chào phù hợp.
        
        Args:
            user_id: ID người dùng
            
        Returns:
            Lời chào
        """
        from layers.conversational_layer import get_time_of_day
        
        memory = self.memory_store.get_memory(user_id)
        is_return = memory.dynamic_state.interaction_count > 1
        last_topic = memory.conversation_context.current_topic
        
        return self.conversational.generate_greeting(
            is_return_user=is_return,
            time_of_day=get_time_of_day(),
            last_topic=last_topic
        )

    def get_check_in_message(self, user_id: str) -> Optional[str]:
        """
        Tạo tin nhắn check-in nếu cần.
        
        Args:
            user_id: ID người dùng
            
        Returns:
            Tin nhắn check-in hoặc None
        """
        if self.memory_store.should_check_in(user_id, hours_threshold=24):
            memory = self.memory_store.get_memory(user_id)
            last_topic = memory.conversation_context.current_topic
            
            follow_up = self.proactive.generate_contextual_follow_up(
                user_id=user_id,
                previous_message=last_topic or "cuộc trò chuyện trước",
                rag_context=""
            )
            
            return follow_up
        
        return None

    def get_therapeutic_exercise(
        self,
        user_id: str,
        issue: str = None
    ) -> str:
        """
        Lấy bài tập trị liệu phù hợp.
        
        Args:
            user_id: ID người dùng
            issue: Vấn đề cụ thể
            
        Returns:
            Bài tập trị liệu
        """
        # Determine issue from memory if not provided
        if not issue:
            memory = self.memory_store.get_memory(user_id)
            if memory.dynamic_state.recent_concerns:
                issue = memory.dynamic_state.recent_concerns[0]
            else:
                issue = "stress"
        
        approach = self.storytelling.suggest_therapy_approach(issue)
        return self.storytelling.create_therapeutic_exercise(approach, issue)

    def get_conversation_summary(self, user_id: str) -> Dict:
        """
        Lấy tóm tắt cuộc hội thoại.
        
        Args:
            user_id: ID người dùng
            
        Returns:
            Dict tóm tắt
        """
        proactive_summary = self.proactive.get_conversation_summary(user_id)
        memory_context = self.memory_store.get_personalization_context(user_id)
        
        return {
            "proactive": proactive_summary,
            "personalization": memory_context
        }


# Simplified factory function
def create_enhanced_chatbot(
    google_api_key: str = None,
    rag_system: TherapeuticRAG = None,
    db_session_factory=None
) -> EnhancedTherapeuticChatbot:
    """
    Factory function để tạo Enhanced Chatbot.
    
    Args:
        google_api_key: API key (lấy từ env nếu None)
        rag_system: RAG system instance
        db_session_factory: Database session factory
        
    Returns:
        EnhancedTherapeuticChatbot instance
    """
    if not google_api_key:
        google_api_key = os.getenv("GOOGLE_API_KEY")
    
    if not google_api_key:
        raise ValueError("GOOGLE_API_KEY is required")
    
    return EnhancedTherapeuticChatbot(
        google_api_key=google_api_key,
        rag_system=rag_system,
        db_session_factory=db_session_factory,
        enable_all_layers=True
    )

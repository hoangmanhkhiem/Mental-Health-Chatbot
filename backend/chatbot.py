"""
Therapeutic Chatbot Logic (MVP Version)
--------------------------------------

This module implements a therapeutic chatbot with OpenAI GPT-4 integration.  
It supports:

- Crisis message detection
- GPT-4 response generation
- RAG-enhanced context retrieval
- Conversation continuity
- Robust error handling with graceful degradation
- Fallback responses when APIs fail
- Logging, metrics, and database failure-safe behavior

Enhancements:
- Improved error handling
- More stable API fallback system
- Handles missing DB columns safely (helpful during schema migration)
- Better logging for debugging & observability
"""

import os
import re
import time
import logging
import json
import requests
from typing import List, Dict, Optional

import google
import google.generativeai as genai
from google.api_core import retry
from sqlalchemy.orm import Session

from rag_system_v2 import TherapeuticRAG
from database import (
    get_or_create_user,
    get_active_conversation,
    save_message,
    get_conversation_history
)

logger = logging.getLogger(__name__)


class TherapeuticChatbot:
    """Main chatbot class for handling therapeutic conversations."""

    # Keywords that indicate a psychological crisis
    CRISIS_KEYWORDS = [
        "suicide", "suicidal", "kill myself", "end my life", "want to die",
        "self-harm", "hurt myself", "cutting", "overdose", "no reason to live",
        "better off dead", "hopeless", "can't go on"
    ]

    # Crisis response template
    CRISIS_RESPONSE = (
        "Tôi hiểu rằng bạn đang trải qua một thời gian vô cùng khó khăn, và tôi lo lắng về "
        "sự an toàn của bạn. Cuộc sống của bạn có ý nghĩa, và có những người muốn giúp đỡ bạn ngay bây giờ.\n\n"
        "**Vui lòng liên hệ ngay để được hỗ trợ:**\n\n"
        "🆘 Đường dây tự tử 24/7: 1800 123 456\n"
        "💬 Tư vấn khủng hoảng: 1900 232 389\n"
        "🌐 Trung tâm tâm lý: Gọi 028 3823 0808\n\n"
        "Những dịch vụ này miễn phí, bảo mật, và luôn sẵn sàng 24/7. Những nhân viên tư vấn đã được đào tạo sẵn sàng lắng nghe.\n\n"
        "Nếu bạn đang gặp nguy hiểm trực tiếp, vui lòng gọi cấp cứu 115 hoặc đến phòng khám cấp cứu gần nhất.\n\n"
        "Tôi ở đây để hỗ trợ bạn về sức khỏe tâm lý, nhưng những chuyên gia tư vấn chuyên nghiệp "
        "được trang bị tốt hơn để giúp đỡ những cảm xúc tuyệt vọng. Bạn có muốn chia sẻ điều gì khiến bạn liên hệ với tôi không?"
    )

    def __init__(self, google_api_key: str, rag_system: Optional[TherapeuticRAG] = None):
        """
        Initialize chatbot with Google Gemini API client and optional RAG system.

        Args:
            google_api_key: Google API key for Gemini
            rag_system: Optional RAG instance initialized in server.py
        """
        genai.configure(api_key=google_api_key)
        # Use available models in order of preference
        models_to_try = [
            'gemini-2.5-pro',        # Most capable
            'gemini-2.5-flash',      # Fast and good quality
            'gemini-pro-latest',     # Stable fallback
            'gemini-2.0-flash'       # Older but available
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
            # Final fallback to gemini-2.5-flash
            self.model = genai.GenerativeModel('gemini-2.5-flash')
            logger.warning("⚠️ Using fallback model gemini-2.5-flash")
        
        self.rag = rag_system
        logger.info("✓ Model Gemini initialized")

    # -------------------------------------------------------------------------
    # Crisis Detection
    # -------------------------------------------------------------------------

    def detect_crisis(self, message: str) -> bool:
        """Return True if the message contains crisis-related keywords."""
        message_lower = message.lower()
        return any(keyword in message_lower for keyword in self.CRISIS_KEYWORDS)

    # -------------------------------------------------------------------------
    # REST API method (Fallback)
    # -------------------------------------------------------------------------

    def _generate_response_via_rest(self, system_prompt: str, user_message: str) -> Optional[str]:
        """
        Generate response using REST API directly instead of SDK.
        Sometimes bypasses strict filtering that affects SDK calls.
        """
        try:
            # Try GEMINI_API_KEY first, then GOOGLE_API_KEY
            api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
            if not api_key:
                logger.warning("⚠️ API key (GEMINI_API_KEY or GOOGLE_API_KEY) not set")
                return None

            # Try multiple models in order of preference
            models_to_try = [
                'gemini-2.5-pro',
                'gemini-2.5-flash',
                'gemini-pro-latest',
                'gemini-2.0-flash'
            ]
            
            headers = {"Content-Type": "application/json"}
            
            payload = {
                "contents": [
                    {
                        "role": "user",
                        "parts": [
                            {"text": system_prompt},
                            {"text": user_message}
                        ]
                    }
                ],
                "generationConfig": {
                    "temperature": 0.7,
                    "maxOutputTokens": 10000
                }
            }

            for model_name in models_to_try:
                try:
                    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={api_key}"
                    response = requests.post(url, json=payload, headers=headers, timeout=30)
                    
                    if response.status_code == 200:
                        data = response.json()
                        if "candidates" in data and len(data["candidates"]) > 0:
                            candidate = data["candidates"][0]
                            if "content" in candidate and "parts" in candidate["content"]:
                                if len(candidate["content"]["parts"]) > 0:
                                    text = candidate["content"]["parts"][0].get("text", "").strip()
                                    if text:
                                        logger.info(f"✓ REST API ({model_name}) trả về response thành công")
                                        return text
                            # Check finish_reason
                            finish_reason = candidate.get("finishReason", "UNKNOWN")
                            logger.warning(f"⚠️ REST API ({model_name}) response bị chặn (finish_reason: {finish_reason})")
                    else:
                        logger.debug(f"⚠️ REST API model {model_name} lỗi {response.status_code}")
                        continue
                except Exception as e:
                    logger.debug(f"⚠️ REST API model {model_name} error: {e}")
                    continue
            
            logger.warning("⚠️ All REST API models blocked or failed")
            return None

        except Exception as e:
            logger.error(f"❌ Lỗi REST API: {e}")
            return None

    # User Handling (Safe Mode)
    # -------------------------------------------------------------------------

    def _get_or_create_user_safe(self, db: Session, whatsapp_number: str):
        """
        Safely retrieve or create a user object.
        Avoids breaking when DB columns are missing.

        Returns:
            User object, or a fallback MinimalUser object if DB query fails.
        """
        try:
            user = get_or_create_user(db, whatsapp_number)
            logger.info(f"✓ Người dùng được lấy/tạo: {whatsapp_number}")
            return user

        except Exception as e:
            logger.warning(f"⚠️ Không thể lấy/tạo người dùng: {e}")

            # Fallback minimal user representation
            class MinimalUser:
                def __init__(self, phone):
                    self.id = hash(phone)
                    self.whatsapp_number = phone
                    self.crisis_flag = False

            return MinimalUser(whatsapp_number)

    # -------------------------------------------------------------------------
    # Response Generation
    # -------------------------------------------------------------------------

    def generate_response(self, db: Session, whatsapp_number: str, user_message: str) -> Dict:
        """
        Generate a therapeutic response using GPT-4 + RAG (if available).

        Returns:
            {
              "response": <string>,
              "is_crisis": <bool>,
              "user_id": <string or None>
            }
        """
        start_time = time.time()

        try:
            # -------------------------
            # 1. Get User (Safe)
            # -------------------------
            user = self._get_or_create_user_safe(db, whatsapp_number)

            # -------------------------
            # 2. Crisis Detection
            # -------------------------
            is_crisis = self.detect_crisis(user_message)

            if is_crisis:
                logger.warning(f"⚠️ Phát hiện nội dung khủng hoảng từ {whatsapp_number}")

                # Attempt to set crisis flag
                try:
                    if hasattr(user, "crisis_flag"):
                        user.crisis_flag = True
                        db.commit()
                except Exception as e:
                    logger.warning(f"⚠️ Không thể lưu cờ khủng hoảng: {e}")

                # Save crisis message
                try:
                    conversation = get_active_conversation(db, user.id)
                    crisis_embedding = None

                    if self.rag:
                        try:
                            crisis_embedding = self.rag.create_embedding(self.CRISIS_RESPONSE)
                        except Exception:
                            crisis_embedding = None

                    save_message(
                        db, conversation.id, user.id, "assistant",
                        self.CRISIS_RESPONSE, crisis_embedding
                    )
                except Exception as e:
                    logger.warning(f"⚠️ Không thể lưu tin nhắn khủng hoảng: {e}")

                return {
                    "response": self.CRISIS_RESPONSE,
                    "is_crisis": True,
                    "user_id": str(user.id)
                }

            # -------------------------
            # 3. Normal Conversation Flow
            # -------------------------

            try:
                conversation = get_active_conversation(db, user.id)
            except Exception as e:
                logger.warning(f"⚠️ Không thể lấy cuộc hội thoại: {e}")
                conversation = None

            # Retrieve history
            try:
                if conversation:
                    history_messages = get_conversation_history(db, conversation.id, limit=10)
                    conversation_history = [
                        {"role": m.role, "content": m.content}
                        for m in history_messages
                    ]
                else:
                    conversation_history = []

            except Exception as e:
                logger.warning(f"⚠️ Lỗi khi lấy lịch sử: {e}")
                conversation_history = []

            # Retrieve RAG context
            relevant_contexts = []
            if self.rag:
                try:
                    relevant_contexts = self.rag.retrieve_relevant_context(
                        db, user_message, k=20
                    )
                except Exception as e:
                    logger.error(f"❌ Lỗi khi lấy bối cảnh RAG: {e}")

            # Build prompt
            try:
                if self.rag:
                    prompt = self.rag.build_prompt_with_context(
                        user_message, relevant_contexts, conversation_history
                    )
                else:
                    prompt = f"You are a compassionate digital wellness therapist. User message: {user_message}"
            except Exception:
                prompt = f"You are a compassionate digital wellness therapist. User message: {user_message}"

            # -------------------------
            # 4. Gemini Response (Retry Logic)
            # -------------------------

            bot_response = None

            for attempt in range(3):
                try:
                    # Build prompt with RAG context if available
                    if relevant_contexts:
                        context_text = "\n".join([f"- {ctx}" for ctx in relevant_contexts])
                        system_prompt = f"""Bạn là một chuyên gia tư vấn. Trả lời câu hỏi một cách hữu ích, thân thiện và tôn trọng.

Thông tin tham khảo:
{context_text}

Hướng dẫn: Hãy trả lời ngắn gọn (dưới 300 từ)."""
                    else:
                        system_prompt = """Bạn là một chuyên gia tư vấn. Trả lời câu hỏi một cách hữu ích, thân thiện và tôn trọng. Hãy trả lời ngắn gọn (dưới 300 từ)."""

                    # Use REST API on attempt >= 1 (fallback from SDK failures)
                    if attempt >= 1:
                        logger.info(f"🔄 Thử REST API (lần thử {attempt + 1})")
                        rest_response = self._generate_response_via_rest(system_prompt, user_message)
                        if rest_response:
                            bot_response = rest_response
                            break
                        else:
                            # REST API also failed, continue to retry
                            if attempt < 2:
                                time.sleep(1)
                            continue

                    # First attempt: use SDK
                    message_history = conversation_history.copy() if conversation_history else []
                    message_history.append({"role": "user", "content": user_message})

                    response = self.model.generate_content(
                        contents=[
                            {"role": "user", "parts": [system_prompt]},
                            {"role": "user", "parts": [user_message]}
                        ],
                        generation_config=genai.types.GenerationConfig(
                            temperature=0.7,
                            max_output_tokens=1000,
                        )
                    )

                    # Check if response has valid content before accessing .text
                    if response and response.candidates and len(response.candidates) > 0:
                        candidate = response.candidates[0]
                        if candidate.content and candidate.content.parts and len(candidate.content.parts) > 0:
                            bot_response = candidate.content.parts[0].text.strip()
                            break
                        else:
                            # Response blocked or empty - try REST API next
                            logger.warning(f"⚠️ Response bị chặn hoặc trống (finish_reason: {candidate.finish_reason})")
                            bot_response = None
                    else:
                        logger.warning("⚠️ API trả về response không có candidates")
                        bot_response = None

                except google.generativeai.types.BlockedPromptException as e:
                    logger.warning(f"⚠️ Lỗi bị chặn bởi bộ lọc an toàn: {e}")
                    bot_response = "Tôi đánh giá cao việc bạn liên hệ, nhưng tôi cần tiếp cận khác. Hãy tập trung vào hỗ trợ cụ thể bạn đang tìm kiếm ngay bây giờ."
                    break

                except Exception as e:
                    logger.error(f"❌ Lỗi API Gemini (lần thử {attempt + 1}): {e}")
                    if attempt < 2:
                        time.sleep(1 + attempt)  # Exponential backoff
                    elif attempt == 2:
                        # Final attempt failed
                        bot_response = None
                        break

            # Fallback response if Gemini fails
            if not bot_response:
                bot_response = self._get_fallback_response(user_message)

            # -------------------------
            # 5. Save Messages
            # -------------------------
            try:
                if conversation:
                    # Save user msg
                    try:
                        user_embed = self.rag.create_embedding(user_message) if self.rag else None
                    except Exception:
                        user_embed = None

                    save_message(
                        db, conversation.id, user.id,
                        "user", user_message, user_embed
                    )

                    # Save bot msg
                    try:
                        bot_embed = self.rag.create_embedding(bot_response) if self.rag else None
                    except Exception:
                        bot_embed = None

                    save_message(
                        db, conversation.id, user.id,
                        "assistant", bot_response, bot_embed
                    )
            except Exception as e:
                logger.warning(f"⚠️ Could not save messages: {e}")

            # Completed
            elapsed = time.time() - start_time
            logger.info(f"✓ Phản hồi được tạo trong {elapsed:.2f}s")

            return {
                "response": bot_response,
                "is_crisis": False,
                "user_id": str(user.id)
            }

        except Exception as e:
            logger.error(f"❌ Lỗi nghiêm trọng: {e}", exc_info=True)

            fallback = (
                "Xin lỗi — Tôi đang gặp sự cố kỹ thuật tạm thời.\n\n"
                "Nếu bạn đang gặp khủng hoảng, vui lòng liên hệ:\n"
                "- Đường dây tự tử 24/7: 1800 123 456\n"
                "- Tư vấn khủng hoảng: 1900 232 389\n\n"
                "Vui lòng thử lại sau một lát."
            )

            return {
                "response": fallback,
                "is_crisis": False,
                "user_id": None
            }

    # -------------------------------------------------------------------------
    # Fallback Responses
    # -------------------------------------------------------------------------

    def _get_fallback_response(self, user_message: str) -> str:
        """
        Generate contextual fallback if GPT-4 is unavailable.
        """

        text = user_message.lower()

        # Screen time concerns
        if any(word in text for word in ["screen", "phone", "instagram", "tiktok", "social media", "màn hình", "điện thoại", "mạng xã hội"]):
            return (
                "Tôi hiểu rằng bạn lo lắng về thời gian sử dụng màn hình. Đây là một kỹ thuật nhanh:\n\n"
                "Hãy thử phương pháp **Bốn Quân Xanh** — bắt đầu với *Nhận Thức*. Lưu ý các mô hình sử dụng điện thoại "
                "của bạn mà không phán xét. Điều gì thường kích hoạt bạn sử dụng điện thoại? 📱\n\n"
                "Bạn có thể xác định được một kích hoạt ngay bây giờ không?"
            )

        # Stress / anxiety
        if any(word in text for word in ["stress", "anxiety", "worried", "anxious", "căng thẳng", "lo âu", "lo lắng", "sợ hãi"]):
            return (
                "Nghe có vẻ bạn đang cảm thấy căng thẳng. Hãy thử kỹ thuật neo tâm lý đơn giản này:\n\n"
                "Thực hiện **ba hơi thở chậm** và tập trung vào những điều bạn *có thể* kiểm soát ngay bây giờ.\n"
                "Một điều nhỏ trong tầm kiểm soát của bạn hôm nay là gì? 🌿"
            )

        # Habit/addiction
        if any(word in text for word in ["addicted", "habit", "can't stop", "nghiện", "thói quen", "không thể dừng"]):
            return (
                "Phá vỡ thói quen thật khó. Bắt đầu với **Chấp Nhận** — công nhân thói quen "
                "mà không phán xét bản thân. Sau đó hỏi: một bước *nhỏ* nào bạn có thể thực hiện hôm nay?\n\n"
                "Hãy nhớ: tiến bộ hơn hoàn hảo. 💪"
            )

        # General fallback
        return (
            "Tôi ở đây để hỗ trợ bạn. Tôi đang gặp sự cố kỹ thuật tạm thời, nhưng cảm xúc của bạn là thật.\n\n"
            "Dành một chút thời gian cho bản thân — có thể đi ra ngoài hoặc thực hiện một vài hơi thở chậm.\n"
            "Một hành động chăm sóc bản thân nhỏ nào bạn có thể làm ngay bây giờ? 🌟"
        )

    # -------------------------------------------------------------------------
    # WhatsApp Formatting
    # -------------------------------------------------------------------------

    def format_whatsapp_message(self, text: str) -> str:
        """Clean and standardize message formatting for WhatsApp."""
        formatted = text.strip()
        formatted = re.sub(r"\n\n\n+", "\n\n", formatted)
        return formatted


# -------------------------------------------------------------------------
# Singleton Instance
# -------------------------------------------------------------------------

_chatbot_instance: Optional[TherapeuticChatbot] = None


def get_chatbot() -> TherapeuticChatbot:
    """Return singleton chatbot instance."""
    global _chatbot_instance

    if _chatbot_instance is None:
        api_key = os.getenv("GOOGLE_API_KEY")

        if not api_key or api_key == "your_google_api_key_here":
            raise ValueError("GOOGLE_API_KEY is not set")

        _chatbot_instance = TherapeuticChatbot(api_key)
        logger.info("Phiên bản Chatbot đã được tạo với Gemini")

    return _chatbot_instance


# -------------------------------------------------------------------------
# Standalone Test
# -------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    try:
        chatbot = get_chatbot()
        print("Chatbot đã khởi tạo thành công!")
    except Exception as e:
        print(f"Lỗi khi khởi tạo chatbot: {e}")

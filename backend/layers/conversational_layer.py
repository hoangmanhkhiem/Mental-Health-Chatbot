"""
PHẦN 1 – Conversational Layer (Biến nội dung RAG thành câu chuyện tự nhiên)

Mục tiêu:
- Không trả lời khô khan kiểu RAG ("theo tài liệu, …")
- Giống một bác sĩ tâm lý nói chuyện tự nhiên, đồng cảm
- Có thể kể chuyện, ví dụ minh họa, dẫn dắt nhẹ nhàng
"""

import logging
from typing import Dict, Optional, List
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ConversationStyle:
    """Định nghĩa phong cách hội thoại."""
    tone: str  # warm, professional, casual, empathetic
    formality: str  # formal, semi-formal, informal
    verbosity: str  # brief, moderate, detailed
    use_examples: bool
    use_metaphors: bool
    use_questions: bool


class ConversationalLayer:
    """
    Layer chuyển đổi nội dung RAG thành hội thoại tự nhiên.
    Biến kiến thức khô khan thành lời khuyên ấm áp, đồng cảm.
    """
    
    # System prompts cho các phong cách khác nhau
    NATURALIZATION_PROMPTS = {
        "default": """Bạn là một chuyên gia tâm lý giàu kinh nghiệm và đầy đồng cảm.
Dựa trên thông tin trong phần CONTEXT, hãy trả lời theo phong cách:
- Gần gũi, đời thường, có sự đồng cảm
- Không dùng ngôn ngữ học thuật quá mức
- Không liệt kê thông tin khô khan
- Chuyển hóa kiến thức thành câu chuyện, ví dụ, hoặc lời khuyên nhẹ nhàng
- Nói chuyện như một người bạn thấu hiểu

Quan trọng:
- Thể hiện sự lắng nghe và thấu hiểu trước khi đưa lời khuyên
- Sử dụng ngôn ngữ "mình/bạn" thân thiện
- Có thể dùng emoji nhẹ nhàng nếu phù hợp (🌿, 💚, ✨)
- Kết thúc bằng câu hỏi mở hoặc lời khích lệ nhẹ nhàng""",

        "empathetic": """Bạn là một người bạn tâm giao, luôn lắng nghe và thấu hiểu.
Người dùng đang trải qua cảm xúc mạnh. Hãy:
- ĐẦU TIÊN, thể hiện sự đồng cảm và công nhận cảm xúc của họ
- Không vội vàng đưa lời khuyên
- Sử dụng câu như "Mình hiểu...", "Cảm xúc đó hoàn toàn tự nhiên..."
- Tạo không gian an toàn để họ chia sẻ thêm
- Chỉ đề xuất giải pháp khi họ sẵn sàng""",

        "practical": """Bạn là một huấn luyện viên sức khỏe tâm lý thực tế.
Người dùng đang tìm kiếm giải pháp cụ thể. Hãy:
- Đưa ra các bước hành động rõ ràng
- Chia nhỏ thành các bước đơn giản
- Đưa ví dụ thực tế
- Nhưng vẫn giữ giọng điệu ấm áp, không máy móc
- Khích lệ họ thử từng bước nhỏ""",

        "gentle": """Bạn là một người đồng hành nhẹ nhàng và kiên nhẫn.
Người dùng có thể đang mong manh. Hãy:
- Sử dụng ngôn ngữ nhẹ nhàng, từ từ
- Không tạo áp lực phải làm gì
- Nhấn mạnh rằng họ không cô đơn
- Đề xuất những điều nhỏ, dễ làm
- Để họ tự quyết định tốc độ"""
    }

    TRANSFORMATION_TEMPLATE = """CONTEXT (Thông tin tham khảo):
{context}

CÂU HỎI CỦA NGƯỜI DÙNG:
{user_message}

THÔNG TIN CẢM XÚC:
- Cảm xúc chính: {emotion}
- Mức độ: {intensity}
- Giọng điệu đề xuất: {suggested_tone}

{personalization_context}

Dựa trên tất cả thông tin trên, hãy phản hồi một cách tự nhiên và đồng cảm.
Không đề cập đến "theo tài liệu" hay "dựa trên context". 
Hãy nói như thể đây là kiến thức tự nhiên của bạn."""

    GREETING_TEMPLATES = {
        "morning": [
            "Chào buổi sáng! 🌅 Mình rất vui được gặp bạn hôm nay.",
            "Buổi sáng tốt lành! ☀️ Bạn bắt đầu ngày mới thế nào rồi?"
        ],
        "afternoon": [
            "Chào bạn! 🌿 Chiều nay bạn thấy thế nào?",
            "Xin chào! Mình ở đây để lắng nghe bạn."
        ],
        "evening": [
            "Chào buổi tối! 🌙 Một ngày dài rồi nhỉ?",
            "Xin chào! 🌿 Mình sẵn sàng trò chuyện cùng bạn."
        ],
        "return_user": [
            "Chào mừng bạn quay lại! 💚 Mình vui vì được gặp lại bạn.",
            "Xin chào! 🌿 Thật tuyệt khi được trò chuyện với bạn lần nữa."
        ]
    }

    def __init__(self, model=None):
        """
        Khởi tạo Conversational Layer.
        
        Args:
            model: Gemini model instance (optional)
        """
        self.model = model
        logger.info("✓ ConversationalLayer initialized")

    def select_style(self, emotion_analysis=None, user_preference: str = None) -> str:
        """
        Chọn phong cách hội thoại phù hợp.
        
        Args:
            emotion_analysis: Kết quả phân tích cảm xúc
            user_preference: Phong cách người dùng ưa thích
            
        Returns:
            Tên phong cách
        """
        if user_preference and user_preference in self.NATURALIZATION_PROMPTS:
            return user_preference
        
        if emotion_analysis:
            # Chọn dựa trên cảm xúc và mức độ
            if hasattr(emotion_analysis, 'intensity'):
                intensity = emotion_analysis.intensity.value if hasattr(emotion_analysis.intensity, 'value') else emotion_analysis.intensity
                if intensity in ["cao", "nghiêm trọng", "high", "critical"]:
                    return "gentle"
            
            if hasattr(emotion_analysis, 'requires_empathy') and emotion_analysis.requires_empathy:
                return "empathetic"
        
        return "default"

    def get_system_prompt(self, style: str = "default") -> str:
        """
        Lấy system prompt theo phong cách.
        
        Args:
            style: Tên phong cách
            
        Returns:
            System prompt
        """
        return self.NATURALIZATION_PROMPTS.get(style, self.NATURALIZATION_PROMPTS["default"])

    def build_natural_prompt(
        self,
        user_message: str,
        rag_context: List[str],
        emotion_analysis=None,
        personalization_context: Dict = None,
        style: str = None
    ) -> str:
        """
        Xây dựng prompt tự nhiên từ các thành phần.
        
        Args:
            user_message: Tin nhắn người dùng
            rag_context: Context từ RAG
            emotion_analysis: Phân tích cảm xúc
            personalization_context: Context cá nhân hóa
            style: Phong cách (optional)
            
        Returns:
            Prompt đã được tối ưu
        """
        # Chọn phong cách
        if not style:
            style = self.select_style(emotion_analysis)
        
        # Chuẩn bị context
        context_text = "\n\n".join(rag_context) if rag_context else "Không có context cụ thể."
        
        # Thông tin cảm xúc
        if emotion_analysis:
            emotion = getattr(emotion_analysis, 'primary_emotion', None)
            if emotion:
                emotion = emotion.value if hasattr(emotion, 'value') else str(emotion)
            else:
                emotion = "chưa xác định"
                
            intensity = getattr(emotion_analysis, 'intensity', None)
            if intensity:
                intensity = intensity.value if hasattr(intensity, 'value') else str(intensity)
            else:
                intensity = "trung bình"
                
            suggested_tone = getattr(emotion_analysis, 'suggested_tone', 'thân thiện')
        else:
            emotion = "chưa xác định"
            intensity = "trung bình"
            suggested_tone = "thân thiện"
        
        # Personalization context
        personalization_text = ""
        if personalization_context:
            personalization_text = f"""
THÔNG TIN CÁ NHÂN HÓA:
- Phong cách ưa thích: {personalization_context.get('user_profile', {}).get('preferred_style', 'thân thiện')}
- Mục tiêu: {', '.join(personalization_context.get('therapy_context', {}).get('goals', []))}
- Chủ đề cần theo dõi: {', '.join(personalization_context.get('therapy_context', {}).get('follow_ups', []))}
"""
        
        return self.TRANSFORMATION_TEMPLATE.format(
            context=context_text,
            user_message=user_message,
            emotion=emotion,
            intensity=intensity,
            suggested_tone=suggested_tone,
            personalization_context=personalization_text
        )

    def transform_response(
        self,
        raw_response: str,
        emotion_analysis=None,
        add_follow_up: bool = True
    ) -> str:
        """
        Biến đổi phản hồi thô thành tự nhiên hơn.
        
        Args:
            raw_response: Phản hồi gốc
            emotion_analysis: Phân tích cảm xúc
            add_follow_up: Thêm câu hỏi theo dõi không
            
        Returns:
            Phản hồi đã được tự nhiên hóa
        """
        if not self.model:
            return raw_response
        
        try:
            transform_prompt = f"""Hãy chuyển đổi phản hồi sau thành giọng điệu tự nhiên, ấm áp hơn:

PHẢN HỒI GỐC:
{raw_response}

YÊU CẦU:
- Giữ nguyên nội dung chính
- Làm cho ngôn ngữ tự nhiên, như đang nói chuyện
- Bỏ các cụm từ học thuật không cần thiết
- Thêm sự đồng cảm nếu phù hợp
{"- Kết thúc bằng câu hỏi mở nhẹ nhàng" if add_follow_up else ""}

PHẢN HỒI TỰ NHIÊN:"""
            
            response = self.model.generate_content(transform_prompt)
            return response.text.strip()
        except Exception as e:
            logger.warning(f"⚠️ Transform failed: {e}")
            return raw_response

    def generate_greeting(
        self,
        is_return_user: bool = False,
        time_of_day: str = "afternoon",
        last_topic: str = None
    ) -> str:
        """
        Tạo lời chào phù hợp.
        
        Args:
            is_return_user: Người dùng quay lại
            time_of_day: Thời điểm trong ngày
            last_topic: Chủ đề lần trước
            
        Returns:
            Lời chào
        """
        import random
        
        if is_return_user:
            greeting = random.choice(self.GREETING_TEMPLATES["return_user"])
            if last_topic:
                greeting += f"\nLần trước chúng ta đã nói về {last_topic}. Bạn muốn tiếp tục hay có điều gì mới muốn chia sẻ?"
        else:
            templates = self.GREETING_TEMPLATES.get(time_of_day, self.GREETING_TEMPLATES["afternoon"])
            greeting = random.choice(templates)
        
        return greeting

    def add_empathy_prefix(self, emotion_analysis, response: str) -> str:
        """
        Thêm câu đồng cảm vào đầu phản hồi.
        
        Args:
            emotion_analysis: Phân tích cảm xúc
            response: Phản hồi gốc
            
        Returns:
            Phản hồi với prefix đồng cảm
        """
        empathy_phrases = {
            "lo âu": [
                "Mình hiểu cảm giác lo lắng đó có thể rất khó chịu. ",
                "Lo âu là cảm xúc mà nhiều người trải qua, và mình ở đây để cùng bạn. "
            ],
            "buồn": [
                "Mình hiểu bạn đang buồn, và điều đó hoàn toàn bình thường. ",
                "Cảm ơn bạn đã chia sẻ. Buồn bã cũng là một phần của cuộc sống. "
            ],
            "căng thẳng": [
                "Nghe có vẻ bạn đang chịu nhiều áp lực. Mình hiểu điều đó. ",
                "Stress có thể rất nặng nề. Mình ở đây để hỗ trợ bạn. "
            ],
            "tuyệt vọng": [
                "Mình rất trân trọng việc bạn chia sẻ điều này. Bạn không đơn độc. ",
                "Cảm xúc này rất nặng nề, và mình muốn bạn biết rằng mình ở đây. "
            ],
            "cô đơn": [
                "Cảm giác cô đơn có thể rất khó khăn. Mình ở đây lắng nghe bạn. ",
                "Bạn không cô đơn trong cuộc trò chuyện này. Mình ở đây. "
            ]
        }
        
        if emotion_analysis and hasattr(emotion_analysis, 'primary_emotion'):
            emotion = emotion_analysis.primary_emotion
            emotion_key = emotion.value if hasattr(emotion, 'value') else str(emotion)
            
            if emotion_key in empathy_phrases:
                import random
                prefix = random.choice(empathy_phrases[emotion_key])
                return prefix + response
        
        return response

    def format_with_suggestions(
        self,
        response: str,
        suggestions: List[str] = None,
        max_suggestions: int = 3
    ) -> str:
        """
        Thêm gợi ý hành động vào cuối phản hồi.
        
        Args:
            response: Phản hồi chính
            suggestions: Danh sách gợi ý
            max_suggestions: Số gợi ý tối đa
            
        Returns:
            Phản hồi với gợi ý
        """
        if not suggestions:
            return response
        
        formatted = response + "\n\n💡 Một vài điều bạn có thể thử:\n"
        for i, suggestion in enumerate(suggestions[:max_suggestions], 1):
            formatted += f"{i}. {suggestion}\n"
        
        formatted += "\nBạn cảm thấy điều nào phù hợp với mình nhất?"
        
        return formatted


# Utility functions
def get_time_of_day() -> str:
    """Lấy thời điểm trong ngày."""
    from datetime import datetime
    hour = datetime.now().hour
    
    if 5 <= hour < 12:
        return "morning"
    elif 12 <= hour < 18:
        return "afternoon"
    else:
        return "evening"

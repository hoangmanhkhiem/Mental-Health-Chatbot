"""
PHẦN 3 – Emotional Understanding Layer (Nhận diện cảm xúc)

Mục tiêu:
- Tự động nhận biết trạng thái cảm xúc
- Điều chỉnh giọng điệu phù hợp
- Tránh đưa lời khuyên sai bối cảnh
"""

import logging
from typing import Dict, Optional, List
from dataclasses import dataclass
from enum import Enum
import google.generativeai as genai

logger = logging.getLogger(__name__)


class EmotionType(Enum):
    """Các loại cảm xúc chính."""
    HAPPY = "vui vẻ"
    SAD = "buồn"
    ANXIOUS = "lo âu"
    ANGRY = "tức giận"
    FEARFUL = "sợ hãi"
    STRESSED = "căng thẳng"
    HOPELESS = "tuyệt vọng"
    CONFUSED = "bối rối"
    NEUTRAL = "trung lập"
    LONELY = "cô đơn"
    OVERWHELMED = "quá tải"
    FRUSTRATED = "bực bội"
    DEPRESSED = "trầm cảm"


class EmotionIntensity(Enum):
    """Mức độ cảm xúc."""
    LOW = "thấp"
    MEDIUM = "trung bình"
    HIGH = "cao"
    CRITICAL = "nghiêm trọng"


@dataclass
class EmotionAnalysis:
    """Kết quả phân tích cảm xúc."""
    primary_emotion: EmotionType
    intensity: EmotionIntensity
    secondary_emotions: List[EmotionType]
    confidence: float
    requires_empathy: bool
    suggested_tone: str
    raw_analysis: Dict


class EmotionAnalyzer:
    """
    Module phân tích cảm xúc từ tin nhắn người dùng.
    Sử dụng LLM để nhận diện cảm xúc và đề xuất giọng điệu phản hồi.
    """
    
    # Emotion keywords mapping (for quick detection)
    EMOTION_KEYWORDS = {
        EmotionType.ANXIOUS: [
            "lo lắng", "lo âu", "sợ", "bất an", "hoang mang", "hồi hộp",
            "nervous", "anxious", "worried", "panic"
        ],
        EmotionType.SAD: [
            "buồn", "đau khổ", "thất vọng", "tủi thân", "khóc", "đau lòng",
            "sad", "upset", "crying", "heartbroken"
        ],
        EmotionType.STRESSED: [
            "stress", "căng thẳng", "áp lực", "deadline", "quá tải", "mệt mỏi",
            "stressed", "pressure", "overwhelmed", "exhausted"
        ],
        EmotionType.ANGRY: [
            "tức giận", "bực", "ghét", "khó chịu", "điên", "cáu",
            "angry", "mad", "frustrated", "hate"
        ],
        EmotionType.HOPELESS: [
            "tuyệt vọng", "bế tắc", "không còn hy vọng", "vô nghĩa", "chán nản",
            "hopeless", "meaningless", "pointless", "give up"
        ],
        EmotionType.LONELY: [
            "cô đơn", "một mình", "không ai hiểu", "lẻ loi", "cô độc",
            "lonely", "alone", "isolated", "nobody understands"
        ],
        EmotionType.DEPRESSED: [
            "trầm cảm", "depression", "không muốn làm gì", "mất hứng thú",
            "depressed", "no motivation", "empty inside"
        ]
    }

    # Tone suggestions based on emotion
    TONE_MAPPING = {
        EmotionType.ANXIOUS: "nhẹ nhàng, trấn an, từ từ",
        EmotionType.SAD: "đồng cảm, ấm áp, thấu hiểu",
        EmotionType.STRESSED: "bình tĩnh, hỗ trợ, thực tế",
        EmotionType.ANGRY: "kiên nhẫn, không phán xét, lắng nghe",
        EmotionType.HOPELESS: "đồng cảm sâu sắc, hy vọng nhẹ nhàng, không ép buộc",
        EmotionType.LONELY: "ấm áp, hiện diện, kết nối",
        EmotionType.DEPRESSED: "nhẹ nhàng, không phán xét, từng bước nhỏ",
        EmotionType.CONFUSED: "rõ ràng, kiên nhẫn, hướng dẫn",
        EmotionType.HAPPY: "vui vẻ, tích cực, chia sẻ niềm vui",
        EmotionType.NEUTRAL: "thân thiện, chuyên nghiệp"
    }

    ANALYSIS_PROMPT = """Bạn là chuyên gia phân tích cảm xúc. Phân tích tin nhắn sau và trả về JSON:

TIN NHẮN: "{message}"

Trả về đúng định dạng JSON (không có text nào khác):
{{
    "primary_emotion": "<cảm xúc chính: vui vẻ|buồn|lo âu|tức giận|sợ hãi|căng thẳng|tuyệt vọng|bối rối|trung lập|cô đơn|quá tải|bực bội|trầm cảm>",
    "intensity": "<mức độ: thấp|trung bình|cao|nghiêm trọng>",
    "secondary_emotions": ["<cảm xúc phụ 1>", "<cảm xúc phụ 2>"],
    "confidence": <0.0-1.0>,
    "requires_empathy": <true|false>,
    "context_notes": "<ghi chú về ngữ cảnh>"
}}

Lưu ý:
- Phân tích cả ngôn ngữ trực tiếp và gián tiếp
- Nhận diện cảm xúc ẩn sau lời nói
- Đánh giá mức độ cần đồng cảm"""

    def __init__(self, model=None):
        """
        Khởi tạo Emotion Analyzer.
        
        Args:
            model: Gemini model instance (optional)
        """
        self.model = model
        logger.info("✓ EmotionAnalyzer initialized")

    def _quick_emotion_detection(self, message: str) -> Optional[EmotionType]:
        """
        Phát hiện nhanh cảm xúc dựa trên từ khóa.
        Dùng như fallback khi LLM không khả dụng.
        """
        message_lower = message.lower()
        
        for emotion, keywords in self.EMOTION_KEYWORDS.items():
            for keyword in keywords:
                if keyword in message_lower:
                    return emotion
        
        return EmotionType.NEUTRAL

    def _parse_emotion_type(self, emotion_str: str) -> EmotionType:
        """Parse chuỗi cảm xúc thành EmotionType."""
        emotion_map = {
            "vui vẻ": EmotionType.HAPPY,
            "buồn": EmotionType.SAD,
            "lo âu": EmotionType.ANXIOUS,
            "tức giận": EmotionType.ANGRY,
            "sợ hãi": EmotionType.FEARFUL,
            "căng thẳng": EmotionType.STRESSED,
            "tuyệt vọng": EmotionType.HOPELESS,
            "bối rối": EmotionType.CONFUSED,
            "trung lập": EmotionType.NEUTRAL,
            "cô đơn": EmotionType.LONELY,
            "quá tải": EmotionType.OVERWHELMED,
            "bực bội": EmotionType.FRUSTRATED,
            "trầm cảm": EmotionType.DEPRESSED,
        }
        return emotion_map.get(emotion_str.lower(), EmotionType.NEUTRAL)

    def _parse_intensity(self, intensity_str: str) -> EmotionIntensity:
        """Parse chuỗi mức độ thành EmotionIntensity."""
        intensity_map = {
            "thấp": EmotionIntensity.LOW,
            "trung bình": EmotionIntensity.MEDIUM,
            "cao": EmotionIntensity.HIGH,
            "nghiêm trọng": EmotionIntensity.CRITICAL
        }
        return intensity_map.get(intensity_str.lower(), EmotionIntensity.MEDIUM)

    async def analyze_async(self, message: str) -> EmotionAnalysis:
        """
        Phân tích cảm xúc bất đồng bộ.
        
        Args:
            message: Tin nhắn người dùng
            
        Returns:
            EmotionAnalysis object
        """
        return self.analyze(message)

    def analyze(self, message: str) -> EmotionAnalysis:
        """
        Phân tích cảm xúc từ tin nhắn.
        
        Args:
            message: Tin nhắn người dùng
            
        Returns:
            EmotionAnalysis object với thông tin cảm xúc
        """
        try:
            if self.model:
                return self._analyze_with_llm(message)
            else:
                return self._analyze_with_keywords(message)
        except Exception as e:
            logger.error(f"❌ Emotion analysis error: {e}")
            return self._analyze_with_keywords(message)

    def _analyze_with_llm(self, message: str) -> EmotionAnalysis:
        """Phân tích cảm xúc sử dụng LLM."""
        try:
            prompt = self.ANALYSIS_PROMPT.format(message=message)
            response = self.model.generate_content(prompt)
            
            # Parse JSON response
            import json
            response_text = response.text.strip()
            
            # Clean up response if needed
            if response_text.startswith("```json"):
                response_text = response_text[7:]
            if response_text.startswith("```"):
                response_text = response_text[3:]
            if response_text.endswith("```"):
                response_text = response_text[:-3]
            
            analysis_data = json.loads(response_text.strip())
            
            primary_emotion = self._parse_emotion_type(analysis_data.get("primary_emotion", "trung lập"))
            intensity = self._parse_intensity(analysis_data.get("intensity", "trung bình"))
            
            secondary_emotions = [
                self._parse_emotion_type(e) 
                for e in analysis_data.get("secondary_emotions", [])
            ]
            
            suggested_tone = self.TONE_MAPPING.get(
                primary_emotion, 
                "thân thiện, chuyên nghiệp"
            )
            
            return EmotionAnalysis(
                primary_emotion=primary_emotion,
                intensity=intensity,
                secondary_emotions=secondary_emotions,
                confidence=analysis_data.get("confidence", 0.7),
                requires_empathy=analysis_data.get("requires_empathy", True),
                suggested_tone=suggested_tone,
                raw_analysis=analysis_data
            )
            
        except Exception as e:
            logger.warning(f"⚠️ LLM emotion analysis failed: {e}, using keyword fallback")
            return self._analyze_with_keywords(message)

    def _analyze_with_keywords(self, message: str) -> EmotionAnalysis:
        """Phân tích cảm xúc dựa trên từ khóa (fallback)."""
        primary_emotion = self._quick_emotion_detection(message)
        
        # Determine intensity based on message characteristics
        intensity = EmotionIntensity.MEDIUM
        message_lower = message.lower()
        
        # High intensity indicators
        high_intensity_words = ["rất", "quá", "cực kỳ", "vô cùng", "không thể chịu được"]
        if any(word in message_lower for word in high_intensity_words):
            intensity = EmotionIntensity.HIGH
        
        # Critical indicators
        critical_words = ["chết", "tự tử", "kết thúc", "không muốn sống"]
        if any(word in message_lower for word in critical_words):
            intensity = EmotionIntensity.CRITICAL
            primary_emotion = EmotionType.HOPELESS
        
        suggested_tone = self.TONE_MAPPING.get(
            primary_emotion,
            "thân thiện, chuyên nghiệp"
        )
        
        return EmotionAnalysis(
            primary_emotion=primary_emotion,
            intensity=intensity,
            secondary_emotions=[],
            confidence=0.6,
            requires_empathy=intensity in [EmotionIntensity.HIGH, EmotionIntensity.CRITICAL],
            suggested_tone=suggested_tone,
            raw_analysis={"method": "keyword_based"}
        )

    def get_response_guidelines(self, analysis: EmotionAnalysis) -> Dict:
        """
        Tạo hướng dẫn phản hồi dựa trên phân tích cảm xúc.
        
        Args:
            analysis: Kết quả phân tích cảm xúc
            
        Returns:
            Dict với các hướng dẫn cho response generation
        """
        guidelines = {
            "tone": analysis.suggested_tone,
            "empathy_level": "high" if analysis.requires_empathy else "moderate",
            "response_style": "supportive",
            "avoid": [],
            "include": []
        }
        
        # Specific guidelines based on emotion
        if analysis.primary_emotion == EmotionType.ANXIOUS:
            guidelines["include"] = ["kỹ thuật thở", "trấn an", "từng bước nhỏ"]
            guidelines["avoid"] = ["thúc giục", "gây áp lực"]
        
        elif analysis.primary_emotion == EmotionType.SAD:
            guidelines["include"] = ["thừa nhận cảm xúc", "đồng cảm", "không gian an toàn"]
            guidelines["avoid"] = ["tích cực độc hại", "so sánh với người khác"]
        
        elif analysis.primary_emotion == EmotionType.HOPELESS:
            guidelines["include"] = ["lắng nghe", "có mặt", "hy vọng nhẹ nhàng"]
            guidelines["avoid"] = ["lời khuyên quá sớm", "phán xét", "ép buộc"]
            guidelines["crisis_check"] = True
        
        elif analysis.primary_emotion == EmotionType.STRESSED:
            guidelines["include"] = ["giải pháp thực tế", "ưu tiên hóa", "thư giãn"]
            guidelines["avoid"] = ["thêm việc", "phức tạp hóa"]
        
        elif analysis.primary_emotion == EmotionType.LONELY:
            guidelines["include"] = ["kết nối", "hiện diện", "ấm áp"]
            guidelines["avoid"] = ["đề xuất xã hội hóa ngay", "phán xét"]
        
        # Adjust for intensity
        if analysis.intensity == EmotionIntensity.CRITICAL:
            guidelines["priority"] = "safety_first"
            guidelines["response_style"] = "crisis_support"
        elif analysis.intensity == EmotionIntensity.HIGH:
            guidelines["response_style"] = "deep_empathy"
        
        return guidelines

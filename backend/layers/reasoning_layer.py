"""
PHẦN 6 – Reasoning Layer (Tăng khả năng suy luận nội bộ)

Mục tiêu:
- Tự suy luận khi thiếu RAG
- Tổng hợp khi có nhiều dữ liệu
- Chain-of-Thought reasoning
- Self-Refinement
"""

import logging
from typing import Dict, Optional, List, Tuple
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class ReasoningMode(Enum):
    """Chế độ suy luận."""
    IMPLICIT_COT = "implicit_chain_of_thought"  # Suy nghĩ ẩn
    EXPLICIT_COT = "explicit_chain_of_thought"  # Suy nghĩ hiện
    SELF_REFINEMENT = "self_refinement"  # Tự cải thiện
    SYNTHESIS = "synthesis"  # Tổng hợp từ nhiều nguồn


@dataclass
class ReasoningResult:
    """Kết quả suy luận."""
    final_response: str
    reasoning_steps: List[str]
    confidence: float
    refinements_made: int
    mode_used: ReasoningMode


class ReasoningLayer:
    """
    Layer suy luận nâng cao.
    Sử dụng Chain-of-Thought và Self-Refinement để cải thiện chất lượng phản hồi.
    """
    
    IMPLICIT_COT_PROMPT = """Bạn là chuyên gia tâm lý học lâm sàng.

HƯỚNG DẪN QUAN TRỌNG:
Trước khi trả lời, hãy suy nghĩ nội bộ theo từng bước (KHÔNG hiển thị ra ngoài):
1. Phân tích vấn đề người dùng đang gặp
2. Xác định cảm xúc chính
3. Đánh giá mức độ nghiêm trọng
4. Chọn phương pháp trị liệu phù hợp
5. Cân nhắc các lời khuyên từ context
6. Chọn lời khuyên phù hợp nhất

CONTEXT (tham khảo):
{context}

CÂU HỎI:
{query}

PHẢN HỒI (chỉ trả lời thân thiện, KHÔNG hiện suy nghĩ nội bộ):"""

    EXPLICIT_COT_PROMPT = """Bạn là chuyên gia tâm lý học.

Hãy phân tích và trả lời theo các bước sau:

## Bước 1: Hiểu vấn đề
- Người dùng đang nói về điều gì?
- Cảm xúc chính là gì?

## Bước 2: Phân tích
- Nguyên nhân có thể là gì?
- Mức độ nghiêm trọng?

## Bước 3: Đánh giá các lựa chọn
- Có những cách tiếp cận nào?
- Cách nào phù hợp nhất với người này?

## Bước 4: Kết luận và lời khuyên

CONTEXT:
{context}

CÂU HỎI:
{query}

PHÂN TÍCH VÀ TRẢ LỜI:"""

    SELF_REFINEMENT_PROMPT = """Đánh giá và cải thiện phản hồi sau đây:

PHẢN HỒI GỐC:
{draft_response}

CONTEXT GỐC:
{context}

CÂU HỎI GỐC:
{query}

Hãy đánh giá phản hồi theo các tiêu chí:
1. ĐỒng cảm: Có thể hiện sự thấu hiểu không? (1-10)
2. Chính xác: Thông tin có đúng với context không? (1-10)
3. Thực tế: Lời khuyên có thực hiện được không? (1-10)
4. Phù hợp: Có phù hợp với cảm xúc người dùng không? (1-10)

Nếu bất kỳ tiêu chí nào dưới 7, hãy viết lại phản hồi tốt hơn.

Trả về JSON:
{{
    "scores": {{"empathy": X, "accuracy": X, "practical": X, "appropriate": X}},
    "needs_improvement": true/false,
    "improved_response": "..." (nếu cần cải thiện)
}}"""

    SYNTHESIS_PROMPT = """Bạn cần tổng hợp thông tin từ nhiều nguồn để trả lời.

CONTEXT PIECES:
{contexts}

CÂU HỎI:
{query}

NHIỆM VỤ:
1. Xác định thông tin chính từ mỗi nguồn
2. Tìm điểm chung và bổ sung
3. Giải quyết mâu thuẫn (nếu có)
4. Tổng hợp thành một câu trả lời thống nhất, tự nhiên

PHẢN HỒI TỔNG HỢP (không đề cập đến các "nguồn"):"""

    KNOWLEDGE_GAP_PROMPT = """CONTEXT có sẵn:
{context}

CÂU HỎI:
{query}

Context có thể không đủ để trả lời hoàn toàn.
Hãy:
1. Sử dụng những gì có trong context
2. Bổ sung bằng kiến thức chung về tâm lý học
3. Rõ ràng về giới hạn của lời khuyên

PHẢN HỒI:"""

    def __init__(self, model=None):
        """
        Khởi tạo Reasoning Layer.
        
        Args:
            model: Gemini model instance
        """
        self.model = model
        self.max_refinements = 2
        logger.info("✓ ReasoningLayer initialized")
    
    def reason_with_cot(
        self,
        query: str,
        context: str = "",
        explicit: bool = False
    ) -> ReasoningResult:
        """
        Suy luận với Chain-of-Thought.
        
        Args:
            query: Câu hỏi
            context: Context từ RAG
            explicit: Hiển thị quá trình suy nghĩ không
            
        Returns:
            ReasoningResult
        """
        if not self.model:
            return ReasoningResult(
                final_response="",
                reasoning_steps=[],
                confidence=0.5,
                refinements_made=0,
                mode_used=ReasoningMode.IMPLICIT_COT
            )
        
        try:
            if explicit:
                prompt = self.EXPLICIT_COT_PROMPT.format(
                    context=context,
                    query=query
                )
                mode = ReasoningMode.EXPLICIT_COT
            else:
                prompt = self.IMPLICIT_COT_PROMPT.format(
                    context=context,
                    query=query
                )
                mode = ReasoningMode.IMPLICIT_COT
            
            response = self.model.generate_content(prompt)
            
            return ReasoningResult(
                final_response=response.text.strip(),
                reasoning_steps=["CoT reasoning applied"],
                confidence=0.8,
                refinements_made=0,
                mode_used=mode
            )
            
        except Exception as e:
            logger.error(f"❌ CoT reasoning error: {e}")
            return ReasoningResult(
                final_response="",
                reasoning_steps=[f"Error: {e}"],
                confidence=0.0,
                refinements_made=0,
                mode_used=ReasoningMode.IMPLICIT_COT
            )
    
    def self_refine(
        self,
        draft_response: str,
        query: str,
        context: str = ""
    ) -> ReasoningResult:
        """
        Tự đánh giá và cải thiện phản hồi.
        
        Args:
            draft_response: Phản hồi nháp
            query: Câu hỏi gốc
            context: Context
            
        Returns:
            ReasoningResult với phản hồi đã cải thiện
        """
        if not self.model:
            return ReasoningResult(
                final_response=draft_response,
                reasoning_steps=[],
                confidence=0.6,
                refinements_made=0,
                mode_used=ReasoningMode.SELF_REFINEMENT
            )
        
        current_response = draft_response
        refinements = 0
        steps = []
        
        for i in range(self.max_refinements):
            try:
                prompt = self.SELF_REFINEMENT_PROMPT.format(
                    draft_response=current_response,
                    context=context,
                    query=query
                )
                
                response = self.model.generate_content(prompt)
                response_text = response.text.strip()
                
                # Parse JSON
                import json
                if "```json" in response_text:
                    response_text = response_text.split("```json")[1].split("```")[0]
                
                data = json.loads(response_text.strip())
                
                scores = data.get("scores", {})
                steps.append(f"Refinement {i+1}: {scores}")
                
                if data.get("needs_improvement", False):
                    improved = data.get("improved_response", "")
                    if improved:
                        current_response = improved
                        refinements += 1
                        logger.info(f"✓ Applied refinement {i+1}")
                    else:
                        break
                else:
                    break
                    
            except Exception as e:
                logger.warning(f"⚠️ Refinement {i+1} failed: {e}")
                steps.append(f"Refinement {i+1} failed: {e}")
                break
        
        # Calculate confidence from scores
        avg_score = 0.7  # Default
        if steps:
            # Try to extract average from last successful refinement
            try:
                if "scores" in steps[-1]:
                    pass  # Could parse scores here
            except:
                pass
        
        return ReasoningResult(
            final_response=current_response,
            reasoning_steps=steps,
            confidence=min(0.9, 0.7 + refinements * 0.1),
            refinements_made=refinements,
            mode_used=ReasoningMode.SELF_REFINEMENT
        )
    
    def synthesize_contexts(
        self,
        query: str,
        contexts: List[str]
    ) -> ReasoningResult:
        """
        Tổng hợp thông tin từ nhiều context.
        
        Args:
            query: Câu hỏi
            contexts: Danh sách context
            
        Returns:
            ReasoningResult với phản hồi tổng hợp
        """
        if not contexts:
            return ReasoningResult(
                final_response="",
                reasoning_steps=["No contexts provided"],
                confidence=0.0,
                refinements_made=0,
                mode_used=ReasoningMode.SYNTHESIS
            )
        
        if not self.model:
            # Simple concatenation fallback
            combined = "\n\n".join(contexts)
            return ReasoningResult(
                final_response=combined,
                reasoning_steps=["Simple combination (no model)"],
                confidence=0.5,
                refinements_made=0,
                mode_used=ReasoningMode.SYNTHESIS
            )
        
        try:
            # Format contexts with labels
            formatted_contexts = "\n\n".join([
                f"[Nguồn {i+1}]:\n{ctx}"
                for i, ctx in enumerate(contexts)
            ])
            
            prompt = self.SYNTHESIS_PROMPT.format(
                contexts=formatted_contexts,
                query=query
            )
            
            response = self.model.generate_content(prompt)
            
            return ReasoningResult(
                final_response=response.text.strip(),
                reasoning_steps=[f"Synthesized {len(contexts)} contexts"],
                confidence=0.8,
                refinements_made=0,
                mode_used=ReasoningMode.SYNTHESIS
            )
            
        except Exception as e:
            logger.error(f"❌ Synthesis error: {e}")
            return ReasoningResult(
                final_response=contexts[0] if contexts else "",
                reasoning_steps=[f"Error: {e}"],
                confidence=0.4,
                refinements_made=0,
                mode_used=ReasoningMode.SYNTHESIS
            )
    
    def handle_knowledge_gap(
        self,
        query: str,
        context: str = ""
    ) -> ReasoningResult:
        """
        Xử lý khi context không đủ.
        
        Args:
            query: Câu hỏi
            context: Context có sẵn (có thể không đủ)
            
        Returns:
            ReasoningResult với phản hồi bổ sung
        """
        if not self.model:
            return ReasoningResult(
                final_response="Mình cần thêm thông tin để hỗ trợ bạn tốt hơn.",
                reasoning_steps=["No model available"],
                confidence=0.3,
                refinements_made=0,
                mode_used=ReasoningMode.IMPLICIT_COT
            )
        
        try:
            prompt = self.KNOWLEDGE_GAP_PROMPT.format(
                context=context if context else "Không có context cụ thể.",
                query=query
            )
            
            response = self.model.generate_content(prompt)
            
            return ReasoningResult(
                final_response=response.text.strip(),
                reasoning_steps=["Used general knowledge to fill gaps"],
                confidence=0.6,
                refinements_made=0,
                mode_used=ReasoningMode.IMPLICIT_COT
            )
            
        except Exception as e:
            logger.error(f"❌ Knowledge gap handling error: {e}")
            return ReasoningResult(
                final_response="",
                reasoning_steps=[f"Error: {e}"],
                confidence=0.0,
                refinements_made=0,
                mode_used=ReasoningMode.IMPLICIT_COT
            )
    
    def generate_with_reasoning(
        self,
        query: str,
        context: str = "",
        contexts: List[str] = None,
        use_cot: bool = True,
        use_refinement: bool = True,
        use_synthesis: bool = False
    ) -> ReasoningResult:
        """
        Tạo phản hồi với full reasoning pipeline.
        
        Args:
            query: Câu hỏi
            context: Context chính
            contexts: Danh sách contexts (cho synthesis)
            use_cot: Sử dụng Chain-of-Thought
            use_refinement: Sử dụng Self-Refinement
            use_synthesis: Sử dụng Synthesis
            
        Returns:
            ReasoningResult
        """
        steps = []
        
        # Step 1: Synthesis nếu có nhiều contexts
        if use_synthesis and contexts and len(contexts) > 1:
            synthesis_result = self.synthesize_contexts(query, contexts)
            context = synthesis_result.final_response
            steps.append(f"Synthesis: {synthesis_result.reasoning_steps}")
        
        # Step 2: CoT reasoning
        if use_cot:
            cot_result = self.reason_with_cot(query, context, explicit=False)
            draft_response = cot_result.final_response
            steps.append(f"CoT: {cot_result.reasoning_steps}")
        elif context:
            draft_response = context
        else:
            # Handle knowledge gap
            gap_result = self.handle_knowledge_gap(query, context)
            draft_response = gap_result.final_response
            steps.append(f"Knowledge gap handling: {gap_result.reasoning_steps}")
        
        # Step 3: Self-refinement
        if use_refinement and draft_response:
            final_result = self.self_refine(draft_response, query, context)
            steps.extend(final_result.reasoning_steps)
            
            return ReasoningResult(
                final_response=final_result.final_response,
                reasoning_steps=steps,
                confidence=final_result.confidence,
                refinements_made=final_result.refinements_made,
                mode_used=ReasoningMode.SELF_REFINEMENT
            )
        
        return ReasoningResult(
            final_response=draft_response,
            reasoning_steps=steps,
            confidence=0.7,
            refinements_made=0,
            mode_used=ReasoningMode.IMPLICIT_COT if use_cot else ReasoningMode.SYNTHESIS
        )

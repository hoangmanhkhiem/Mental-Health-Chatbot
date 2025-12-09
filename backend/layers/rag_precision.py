"""
PHẦN 5 – RAG Precision Boost (Cải thiện độ chính xác truy xuất)

Các kỹ thuật:
1. Multi-query Retrieval - Tạo nhiều truy vấn phụ
2. Hybrid Retrieval - Kết hợp Gemini embedding + BM25
3. Context Reranking - Sắp xếp lại kết quả
4. Semantic Chunking - Chunk theo ý nghĩa
"""

import logging
import re
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from collections import Counter
import math

logger = logging.getLogger(__name__)


@dataclass
class RetrievalResult:
    """Kết quả truy xuất với metadata."""
    content: str
    source: str
    page: Optional[int]
    book_title: str
    relevance_score: float
    retrieval_method: str  # "semantic", "keyword", "hybrid"


class BM25:
    """
    BM25 ranking algorithm cho keyword search.
    Dùng kết hợp với semantic search.
    """
    
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        """
        Khởi tạo BM25.
        
        Args:
            k1: Term saturation parameter
            b: Length normalization parameter
        """
        self.k1 = k1
        self.b = b
        self.doc_len = []
        self.avgdl = 0
        self.doc_freqs = []
        self.idf = {}
        self.doc_count = 0
        self._initialized = False
    
    def fit(self, documents: List[str]) -> None:
        """
        Fit BM25 với corpus.
        
        Args:
            documents: Danh sách documents
        """
        self.doc_count = len(documents)
        self.doc_len = []
        self.doc_freqs = []
        
        # Tokenize và đếm
        df = Counter()
        
        for doc in documents:
            tokens = self._tokenize(doc)
            self.doc_len.append(len(tokens))
            
            freqs = Counter(tokens)
            self.doc_freqs.append(freqs)
            
            # Document frequency
            for token in set(tokens):
                df[token] += 1
        
        self.avgdl = sum(self.doc_len) / self.doc_count if self.doc_count else 0
        
        # Calculate IDF
        for token, freq in df.items():
            self.idf[token] = math.log((self.doc_count - freq + 0.5) / (freq + 0.5) + 1)
        
        self._initialized = True
    
    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text đơn giản."""
        # Lowercase và split
        text = text.lower()
        # Remove punctuation và split
        tokens = re.findall(r'\b\w+\b', text)
        return tokens
    
    def score(self, query: str, doc_index: int) -> float:
        """
        Tính BM25 score cho một document.
        
        Args:
            query: Truy vấn
            doc_index: Index của document
            
        Returns:
            BM25 score
        """
        if not self._initialized:
            return 0.0
        
        query_tokens = self._tokenize(query)
        doc_freqs = self.doc_freqs[doc_index]
        doc_len = self.doc_len[doc_index]
        
        score = 0.0
        for token in query_tokens:
            if token not in doc_freqs:
                continue
            
            freq = doc_freqs[token]
            idf = self.idf.get(token, 0)
            
            # BM25 formula
            numerator = freq * (self.k1 + 1)
            denominator = freq + self.k1 * (1 - self.b + self.b * doc_len / self.avgdl)
            score += idf * numerator / denominator
        
        return score
    
    def search(self, query: str, top_k: int = 5) -> List[Tuple[int, float]]:
        """
        Tìm top-k documents.
        
        Args:
            query: Truy vấn
            top_k: Số kết quả
            
        Returns:
            List[(doc_index, score)]
        """
        if not self._initialized:
            return []
        
        scores = [(i, self.score(query, i)) for i in range(self.doc_count)]
        scores.sort(key=lambda x: x[1], reverse=True)
        
        return scores[:top_k]


class RAGPrecisionBoost:
    """
    Module nâng cao độ chính xác RAG.
    Kết hợp multi-query, hybrid retrieval, và reranking.
    """
    
    MULTI_QUERY_PROMPT = """Bạn là trợ lý tìm kiếm thông tin tâm lý học.
Với câu hỏi gốc sau, hãy tạo ra 3-5 câu hỏi/truy vấn phụ khác nhau 
để tìm kiếm thông tin đầy đủ hơn.

CÂU HỎI GỐC: {query}

Tạo các truy vấn đa dạng:
- Một số bằng tiếng Việt
- Một số có thể bằng tiếng Anh (thuật ngữ chuyên môn)
- Một số dùng từ đồng nghĩa
- Một số cụ thể hơn về triệu chứng
- Một số về phương pháp điều trị/giải pháp

Trả về JSON:
{{"queries": ["query1", "query2", "query3", ...]}}"""

    RERANK_PROMPT = """Đánh giá mức độ liên quan của đoạn văn sau với câu hỏi.

CÂU HỎI: {query}

ĐOẠN VĂN:
{passage}

Trả về JSON:
{{"relevance_score": <0.0-1.0>, "reason": "<lý do ngắn gọn>"}}

Tiêu chí đánh giá:
- 0.9-1.0: Trực tiếp trả lời câu hỏi
- 0.7-0.9: Rất liên quan, cung cấp thông tin hữu ích
- 0.5-0.7: Liên quan một phần
- 0.3-0.5: Liên quan gián tiếp
- 0.0-0.3: Không liên quan"""

    def __init__(self, model=None, rag_system=None):
        """
        Khởi tạo RAG Precision Boost.
        
        Args:
            model: Gemini model cho multi-query và reranking
            rag_system: RAG system gốc cho embedding search
        """
        self.model = model
        self.rag_system = rag_system
        self.bm25 = BM25()
        self._documents_cache = []
        self._documents_metadata = []
        logger.info("✓ RAGPrecisionBoost initialized")
    
    def initialize_bm25(self, documents: List[Dict]) -> None:
        """
        Khởi tạo BM25 index với documents.
        
        Args:
            documents: List of document dicts với 'content' key
        """
        contents = [doc.get('content', '') for doc in documents]
        self._documents_cache = contents
        self._documents_metadata = documents
        self.bm25.fit(contents)
        logger.info(f"✓ BM25 initialized with {len(documents)} documents")
    
    def generate_multi_queries(self, original_query: str) -> List[str]:
        """
        Tạo nhiều truy vấn từ câu hỏi gốc.
        
        Args:
            original_query: Câu hỏi gốc
            
        Returns:
            List các truy vấn (bao gồm câu gốc)
        """
        queries = [original_query]
        
        if not self.model:
            # Fallback: tạo các biến thể đơn giản
            queries.extend(self._simple_query_expansion(original_query))
            return queries[:5]
        
        try:
            prompt = self.MULTI_QUERY_PROMPT.format(query=original_query)
            response = self.model.generate_content(prompt)
            
            # Parse JSON
            import json
            response_text = response.text.strip()
            
            # Clean up response
            if "```json" in response_text:
                response_text = response_text.split("```json")[1].split("```")[0]
            elif "```" in response_text:
                response_text = response_text.split("```")[1].split("```")[0]
            
            data = json.loads(response_text.strip())
            additional_queries = data.get("queries", [])
            
            queries.extend(additional_queries)
            logger.info(f"✓ Generated {len(queries)} queries from original")
            
        except Exception as e:
            logger.warning(f"⚠️ Multi-query generation failed: {e}")
            queries.extend(self._simple_query_expansion(original_query))
        
        return queries[:6]  # Limit to 6 queries
    
    def _simple_query_expansion(self, query: str) -> List[str]:
        """Mở rộng truy vấn đơn giản không cần LLM."""
        expansions = []
        
        # Synonym mapping
        synonym_map = {
            "lo âu": ["anxiety", "lo lắng", "bất an", "hoang mang"],
            "stress": ["căng thẳng", "áp lực", "mệt mỏi"],
            "buồn": ["sad", "depression", "trầm cảm", "đau khổ"],
            "tức giận": ["angry", "bực bội", "khó chịu"],
            "sợ": ["fear", "lo sợ", "hoảng sợ"],
            "ngủ": ["sleep", "mất ngủ", "insomnia"],
            "CBT": ["liệu pháp nhận thức hành vi", "cognitive behavioral therapy"],
            "mindfulness": ["chánh niệm", "thiền", "meditation"]
        }
        
        query_lower = query.lower()
        for term, synonyms in synonym_map.items():
            if term in query_lower:
                for syn in synonyms[:2]:
                    expansions.append(query_lower.replace(term, syn))
        
        return expansions[:3]
    
    def hybrid_search(
        self,
        db,
        query: str,
        k: int = 10,
        semantic_weight: float = 0.7,
        keyword_weight: float = 0.3
    ) -> List[RetrievalResult]:
        """
        Tìm kiếm hybrid kết hợp semantic và keyword.
        
        Args:
            db: Database session
            query: Truy vấn
            k: Số kết quả
            semantic_weight: Trọng số cho semantic search
            keyword_weight: Trọng số cho keyword search
            
        Returns:
            List kết quả đã được kết hợp và sắp xếp
        """
        results = {}
        
        # 1. Semantic search
        if self.rag_system:
            try:
                semantic_results = self.rag_system.retrieve_relevant_context(db, query, k=k*2)
                for i, result in enumerate(semantic_results):
                    content = result.get('content', '') if isinstance(result, dict) else result
                    score = 1.0 - (i / len(semantic_results))  # Normalize by rank
                    
                    if content not in results:
                        results[content] = {
                            'content': content,
                            'semantic_score': score * semantic_weight,
                            'keyword_score': 0,
                            'metadata': result if isinstance(result, dict) else {}
                        }
                    else:
                        results[content]['semantic_score'] = score * semantic_weight
            except Exception as e:
                logger.warning(f"⚠️ Semantic search failed: {e}")
        
        # 2. BM25 keyword search
        if self.bm25._initialized:
            try:
                bm25_results = self.bm25.search(query, top_k=k*2)
                max_bm25_score = max(r[1] for r in bm25_results) if bm25_results else 1
                
                for doc_idx, bm25_score in bm25_results:
                    if doc_idx < len(self._documents_cache):
                        content = self._documents_cache[doc_idx]
                        normalized_score = (bm25_score / max_bm25_score) if max_bm25_score else 0
                        
                        if content not in results:
                            results[content] = {
                                'content': content,
                                'semantic_score': 0,
                                'keyword_score': normalized_score * keyword_weight,
                                'metadata': self._documents_metadata[doc_idx] if doc_idx < len(self._documents_metadata) else {}
                            }
                        else:
                            results[content]['keyword_score'] = normalized_score * keyword_weight
            except Exception as e:
                logger.warning(f"⚠️ BM25 search failed: {e}")
        
        # 3. Combine scores
        combined = []
        for content, data in results.items():
            total_score = data['semantic_score'] + data['keyword_score']
            combined.append(RetrievalResult(
                content=content,
                source=data['metadata'].get('source_file', 'unknown'),
                page=data['metadata'].get('page'),
                book_title=data['metadata'].get('book_title', 'Unknown'),
                relevance_score=total_score,
                retrieval_method='hybrid'
            ))
        
        # Sort by score
        combined.sort(key=lambda x: x.relevance_score, reverse=True)
        
        return combined[:k]
    
    def rerank_results(
        self,
        query: str,
        results: List[RetrievalResult],
        top_k: int = 5
    ) -> List[RetrievalResult]:
        """
        Rerank kết quả sử dụng LLM.
        
        Args:
            query: Truy vấn gốc
            results: Kết quả cần rerank
            top_k: Số kết quả cuối cùng
            
        Returns:
            Kết quả đã được rerank
        """
        if not self.model or not results:
            return results[:top_k]
        
        reranked = []
        
        for result in results:
            try:
                prompt = self.RERANK_PROMPT.format(
                    query=query,
                    passage=result.content[:500]  # Limit length
                )
                
                response = self.model.generate_content(prompt)
                
                # Parse score
                import json
                response_text = response.text.strip()
                
                if "```json" in response_text:
                    response_text = response_text.split("```json")[1].split("```")[0]
                
                data = json.loads(response_text.strip())
                new_score = data.get("relevance_score", 0.5)
                
                # Combine with original score
                combined_score = (result.relevance_score + new_score) / 2
                
                reranked.append(RetrievalResult(
                    content=result.content,
                    source=result.source,
                    page=result.page,
                    book_title=result.book_title,
                    relevance_score=combined_score,
                    retrieval_method=result.retrieval_method
                ))
                
            except Exception as e:
                logger.debug(f"⚠️ Rerank failed for one result: {e}")
                reranked.append(result)
        
        # Sort by new score
        reranked.sort(key=lambda x: x.relevance_score, reverse=True)
        
        return reranked[:top_k]
    
    def retrieve_with_precision(
        self,
        db,
        query: str,
        k: int = 5,
        use_multi_query: bool = True,
        use_hybrid: bool = True,
        use_rerank: bool = True
    ) -> List[RetrievalResult]:
        """
        Truy xuất với tất cả các kỹ thuật precision boost.
        
        Args:
            db: Database session
            query: Truy vấn
            k: Số kết quả
            use_multi_query: Sử dụng multi-query
            use_hybrid: Sử dụng hybrid search
            use_rerank: Sử dụng reranking
            
        Returns:
            Kết quả tối ưu
        """
        all_results = {}
        
        # 1. Generate queries
        if use_multi_query:
            queries = self.generate_multi_queries(query)
        else:
            queries = [query]
        
        # 2. Search with each query
        for q in queries:
            if use_hybrid:
                results = self.hybrid_search(db, q, k=k*2)
            elif self.rag_system:
                raw_results = self.rag_system.retrieve_relevant_context(db, q, k=k*2)
                results = [
                    RetrievalResult(
                        content=r.get('content', '') if isinstance(r, dict) else r,
                        source=r.get('source_file', 'unknown') if isinstance(r, dict) else 'unknown',
                        page=r.get('page') if isinstance(r, dict) else None,
                        book_title=r.get('book_title', 'Unknown') if isinstance(r, dict) else 'Unknown',
                        relevance_score=0.5,
                        retrieval_method='semantic'
                    )
                    for r in raw_results
                ]
            else:
                results = []
            
            # Merge results
            for result in results:
                if result.content not in all_results:
                    all_results[result.content] = result
                else:
                    # Boost score for documents found by multiple queries
                    all_results[result.content].relevance_score += result.relevance_score * 0.5
        
        # Convert to list
        merged_results = list(all_results.values())
        merged_results.sort(key=lambda x: x.relevance_score, reverse=True)
        
        # 3. Rerank top results
        if use_rerank:
            final_results = self.rerank_results(query, merged_results[:k*2], top_k=k)
        else:
            final_results = merged_results[:k]
        
        logger.info(f"✓ Retrieved {len(final_results)} results with precision boost")
        return final_results
    
    def get_context_with_citations(self, results: List[RetrievalResult]) -> Tuple[str, List[Dict]]:
        """
        Tạo context string với citations.
        
        Args:
            results: Kết quả truy xuất
            
        Returns:
            (context_string, citations_list)
        """
        context_parts = []
        citations = []
        
        for i, result in enumerate(results, 1):
            context_parts.append(result.content)
            citations.append({
                "index": i,
                "source": result.source,
                "page": result.page,
                "book_title": result.book_title,
                "relevance": result.relevance_score
            })
        
        context_string = "\n\n---\n\n".join(context_parts)
        
        return context_string, citations


class SemanticChunker:
    """
    Semantic chunking - chunk theo ý nghĩa thay vì ký tự.
    """
    
    def __init__(
        self,
        min_chunk_size: int = 300,
        max_chunk_size: int = 600,
        overlap_percentage: float = 0.15
    ):
        """
        Khởi tạo Semantic Chunker.
        
        Args:
            min_chunk_size: Kích thước chunk tối thiểu (tokens)
            max_chunk_size: Kích thước chunk tối đa (tokens)
            overlap_percentage: Phần trăm overlap
        """
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
        self.overlap_percentage = overlap_percentage
        logger.info("✓ SemanticChunker initialized")
    
    def _estimate_tokens(self, text: str) -> int:
        """Ước lượng số tokens (đơn giản)."""
        # Rough estimate: 1 token ≈ 4 characters cho tiếng Việt
        return len(text) // 4
    
    def _find_semantic_boundaries(self, text: str) -> List[int]:
        """Tìm các điểm ngắt ngữ nghĩa."""
        boundaries = [0]
        
        # Patterns for semantic boundaries
        patterns = [
            r'\n\n+',  # Double newlines (paragraphs)
            r'\n(?=[A-Z0-9])',  # Newline followed by capital letter
            r'\. (?=[A-Z])',  # Period followed by capital letter
            r'[.!?]\s+',  # End of sentence
        ]
        
        for pattern in patterns:
            for match in re.finditer(pattern, text):
                boundaries.append(match.end())
        
        boundaries.append(len(text))
        boundaries = sorted(set(boundaries))
        
        return boundaries
    
    def chunk(self, text: str) -> List[Dict]:
        """
        Chunk text theo ngữ nghĩa.
        
        Args:
            text: Text cần chunk
            
        Returns:
            List of chunk dicts
        """
        boundaries = self._find_semantic_boundaries(text)
        chunks = []
        current_chunk_start = 0
        current_chunk_text = ""
        
        for i, boundary in enumerate(boundaries[1:], 1):
            segment = text[boundaries[i-1]:boundary]
            potential_chunk = current_chunk_text + segment
            
            tokens = self._estimate_tokens(potential_chunk)
            
            if tokens > self.max_chunk_size:
                # Save current chunk
                if current_chunk_text.strip():
                    chunks.append({
                        "content": current_chunk_text.strip(),
                        "start": current_chunk_start,
                        "end": boundaries[i-1],
                        "tokens": self._estimate_tokens(current_chunk_text)
                    })
                
                # Calculate overlap
                overlap_size = int(len(current_chunk_text) * self.overlap_percentage)
                overlap_text = current_chunk_text[-overlap_size:] if overlap_size > 0 else ""
                
                current_chunk_start = boundaries[i-1] - len(overlap_text)
                current_chunk_text = overlap_text + segment
            else:
                current_chunk_text = potential_chunk
        
        # Add final chunk
        if current_chunk_text.strip():
            chunks.append({
                "content": current_chunk_text.strip(),
                "start": current_chunk_start,
                "end": len(text),
                "tokens": self._estimate_tokens(current_chunk_text)
            })
        
        logger.info(f"✓ Created {len(chunks)} semantic chunks")
        return chunks

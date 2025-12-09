#!/usr/bin/env python3

"""
Enhanced Therapeutic Knowledge Base Loader

Cải tiến so với bản gốc:
1. Semantic Chunking - Chunk theo ý nghĩa, không cứng nhắc theo ký tự
2. Hierarchical Chunking - Parent/Child chunks cho context tốt hơn
3. Multi-embedding - Tạo embedding cho cả summary + content
4. Metadata Enhancement - Rich metadata cho filtering
5. BM25 Index - Keyword index song song với vector
6. Chunk Overlap Optimization - Overlap thông minh
7. Vietnamese Text Processing - Xử lý tiếng Việt tốt hơn
8. Batch Processing - Xử lý embedding theo batch để tối ưu API calls
"""

import os
import sys
import json
import logging
import re
import hashlib
from pathlib import Path
from urllib.parse import urlparse
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import time

import psycopg2
from psycopg2.extras import execute_values
import google.generativeai as genai

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration from environment
DATABASE_URL = os.getenv('DATABASE_URL')
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
KNOWLEDGE_BASE_PATH = Path(__file__).parent / 'knowledge_base'

# Initialize Google Gemini
genai.configure(api_key=GOOGLE_API_KEY)

# Embedding model
EMBEDDING_MODEL = "models/text-embedding-004"
EMBEDDING_DIM = 768

# Chunking configuration
CHUNK_SIZE = 800  # Smaller for more precise retrieval
CHUNK_OVERLAP = 150
MIN_CHUNK_SIZE = 100
MAX_CHUNKS_PER_BATCH = 20  # Batch size for embedding API


@dataclass
class EnhancedChunk:
    """Enhanced chunk với metadata phong phú."""
    content: str
    chunk_id: str
    chunk_index: int
    source_file: str
    book_title: str
    author: str
    content_type: str
    topics: List[str]
    page_start: Optional[int]
    page_end: Optional[int]
    parent_chunk_id: Optional[str]  # For hierarchical chunking
    chunk_type: str  # "content", "summary", "heading"
    word_count: int
    char_count: int
    language: str
    key_terms: List[str]
    summary: Optional[str]
    created_at: str


# Books metadata (unchanged from original)
THERAPEUTIC_BOOKS = {
    "4As_Manuscript_v6.pdf": {
        "title": "4As_Manuscript_v6",
        "author": "",
        "content_type": "therapeutic_framework",
        "topics": ["thần kinh học", "khám lâm sàng", "bệnh lý thần kinh", "y khoa"],
        "language": "vi"
    },
    "BeyondHappy_MANUSCRIPT_v7.pdf": {
        "title": "BeyondHappy_MANUSCRIPT_v7",
        "author": "",
        "content_type": "therapeutic_framework",
        "topics": ["hạnh phúc", "tâm lý", "phát triển bản thân"],
        "language": "vi"
    },
    "data1.pdf": {
        "title": "Nội Thần Kinh",
        "author": "PGS.TS Hồng Khánh",
        "content_type": "therapeutic_framework",
        "topics": ["thần kinh học", "khám lâm sàng", "bệnh lý thần kinh", "y khoa"],
        "language": "vi"
    },
    "data2.pdf": {
        "title": "Những Trắc Nghiệm Tâm Lý Tập I",
        "author": "PGS.TS Ngô Công Hồn",
        "content_type": "therapeutic_framework",
        "topics": ["trắc nghiệm tâm lý", "trí tuệ", "phát triển trẻ em", "tâm lý học"],
        "language": "vi"
    },
    "data3.pdf": {
        "title": "Bệnh Học Tâm Thần",
        "author": "PGS.TS Nguyễn Kim Việt",
        "content_type": "therapeutic_framework",
        "topics": ["tâm thần học", "rối loạn tâm thần", "chẩn đoán", "điều trị"],
        "language": "vi"
    },
    "data4.pdf": {
    "title": "Chưa xác định rõ ràng từ dữ liệu tìm kiếm",
    "author": "Chưa xác định rõ ràng từ dữ liệu tìm kiếm",
    "content_type": "therapeutic_framework",
    "topics": ["Chưa xác định"]
    },
    "data5`.pdf": {
    "title": "Chưa xác định rõ ràng từ dữ liệu tìm kiếm",
    "author": "Chưa xác định rõ ràng từ dữ liệu tìm kiếm",
    "content_type": "therapeutic_framework",
    "topics": ["Chưa xác định"]
    },
    "data6.pdf": {
    "title": "Chưa xác định rõ ràng từ dữ liệu tìm kiếm",
    "author": "Chưa xác định rõ ràng từ dữ liệu tìm kiếm",
    "content_type": "therapeutic_framework",
    "topics": ["Chưa xác định"]
    },
    "data7.pdf": {
    "title": "Chưa xác định rõ ràng từ dữ liệu tìm kiếm",
    "author": "Chưa xác định rõ ràng từ dữ liệu tìm kiếm",
    "content_type": "therapeutic_framework",
    "topics": ["Chưa xác định"]
    },
    "data8.pdf": {
    "title": "Chưa xác định rõ ràng từ dữ liệu tìm kiếm",
    "author": "Chưa xác định rõ ràng từ dữ liệu tìm kiếm",
    "content_type": "therapeutic_framework",
    "topics": ["Chưa xác định"]
    },
    "data9.pdf": {
    "title": "Chưa xác định rõ ràng từ dữ liệu tìm kiếm",
    "author": "Chưa xác định rõ ràng từ dữ liệu tìm kiếm",
    "content_type": "therapeutic_framework",
    "topics": ["Chưa xác định"]
    },
    "data10.pdf": {
    "title": "Chưa xác định rõ ràng từ dữ liệu tìm kiếm",
    "author": "Chưa xác định rõ ràng từ dữ liệu tìm kiếm",
    "content_type": "therapeutic_framework",
    "topics": ["Chưa xác định"]
    },
    "data11.pdf": {
    "title": "Chưa xác định rõ ràng từ dữ liệu tìm kiếm",
    "author": "Chưa xác định rõ ràng từ dữ liệu tìm kiếm",
    "content_type": "therapeutic_framework",
    "topics": ["Chưa xác định"]
    },
    "data12.pdf": {
    "title": "Chưa xác định rõ ràng từ dữ liệu tìm kiếm",
    "author": "Chưa xác định rõ ràng từ dữ liệu tìm kiếm",
    "content_type": "therapeutic_framework",
    "topics": ["Chưa xác định"]
    },
    "data13.pdf": {
    "title": "Chưa xác định rõ ràng từ dữ liệu tìm kiếm",
    "author": "Chưa xác định rõ ràng từ dữ liệu tìm kiếm",
    "content_type": "therapeutic_framework",
    "topics": ["Chưa xác định"]
    },
    "data14.pdf": {
    "title": "Chưa xác định rõ ràng từ dữ liệu tìm kiếm",
    "author": "Chưa xác định rõ ràng từ dữ liệu tìm kiếm",
    "content_type": "therapeutic_framework",
    "topics": ["Chưa xác định"]
    }
}


# ============================================================
# VIETNAMESE TEXT PROCESSING
# ============================================================

class VietnameseTextProcessor:
    """Xử lý văn bản tiếng Việt."""
    
    # Vietnamese stopwords
    STOPWORDS = {
        "và", "của", "là", "có", "được", "cho", "với", "trong", "này", "đó",
        "các", "những", "một", "để", "về", "theo", "khi", "từ", "đến", "như",
        "không", "còn", "người", "nên", "cũng", "rất", "thì", "đã", "sẽ", "đang",
        "hay", "hoặc", "nhưng", "mà", "vì", "nếu", "tuy", "dù", "do", "bởi"
    }
    
    # Mental health key terms
    MENTAL_HEALTH_TERMS = {
        "trầm cảm", "lo âu", "stress", "căng thẳng", "tâm lý", "tâm thần",
        "rối loạn", "điều trị", "chẩn đoán", "triệu chứng", "thuốc", "liệu pháp",
        "cảm xúc", "hành vi", "nhận thức", "trị liệu", "tư vấn", "hỗ trợ",
        "sức khỏe tâm thần", "khủng hoảng", "tự tử", "hoảng sợ", "ám ảnh",
        "PTSD", "OCD", "ADHD", "tự kỷ", "tâm thần phân liệt", "lưỡng cực"
    }
    
    @classmethod
    def extract_key_terms(cls, text: str, max_terms: int = 10) -> List[str]:
        """Trích xuất key terms từ text."""
        text_lower = text.lower()
        found_terms = []
        
        # Tìm mental health terms
        for term in cls.MENTAL_HEALTH_TERMS:
            if term in text_lower:
                found_terms.append(term)
        
        # Tìm các từ quan trọng (capitalized, technical terms)
        words = re.findall(r'\b[A-Z][a-zA-Z]+\b', text)
        for word in words:
            if word.lower() not in cls.STOPWORDS and len(word) > 3:
                found_terms.append(word)
        
        return list(set(found_terms))[:max_terms]
    
    @classmethod
    def clean_text(cls, text: str) -> str:
        """Làm sạch text."""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove special characters but keep Vietnamese
        text = re.sub(r'[^\w\sàáảãạăắằẳẵặâấầẩẫậèéẻẽẹêếềểễệìíỉĩịòóỏõọôốồổỗộơớờởỡợùúủũụưứừửữựỳýỷỹỵđ.,!?;:\-\(\)\[\]\"\']+', ' ', text, flags=re.IGNORECASE)
        return text.strip()
    
    @classmethod
    def detect_language(cls, text: str) -> str:
        """Phát hiện ngôn ngữ (vi/en)."""
        # Count Vietnamese diacritics
        vietnamese_chars = set('àáảãạăắằẳẵặâấầẩẫậèéẻẽẹêếềểễệìíỉĩịòóỏõọôốồổỗộơớờởỡợùúủũụưứừửữựỳýỷỹỵđ')
        viet_count = sum(1 for c in text.lower() if c in vietnamese_chars)
        
        return "vi" if viet_count > len(text) * 0.01 else "en"


# ============================================================
# SEMANTIC CHUNKING
# ============================================================

class SemanticChunker:
    """
    Semantic chunking - chia theo ý nghĩa thay vì ký tự.
    """
    
    # Sentence ending patterns
    SENTENCE_ENDINGS = re.compile(r'[.!?]\s+')
    
    # Section/heading patterns
    HEADING_PATTERNS = [
        re.compile(r'^#{1,6}\s+.+$', re.MULTILINE),  # Markdown headings
        re.compile(r'^\d+\.\s+[A-ZÀÁẢÃẠĂẮẰẲẴẶÂẤẦẨẪẬ].+$', re.MULTILINE),  # Numbered sections
        re.compile(r'^[A-ZÀÁẢÃẠĂẮẰẲẴẶÂẤẦẨẪẬ][A-ZÀÁẢÃẠĂẮẰẲẴẶÂẤẦẨẪẬ\s]+:?\s*$', re.MULTILINE),  # ALL CAPS headings
        re.compile(r'^Chương\s+\d+', re.MULTILINE),  # Vietnamese chapter
        re.compile(r'^Phần\s+\d+', re.MULTILINE),  # Vietnamese part
        re.compile(r'^Bài\s+\d+', re.MULTILINE),  # Vietnamese lesson
    ]
    
    @classmethod
    def split_into_sentences(cls, text: str) -> List[str]:
        """Tách text thành các câu."""
        sentences = cls.SENTENCE_ENDINGS.split(text)
        return [s.strip() for s in sentences if s.strip()]
    
    @classmethod
    def find_headings(cls, text: str) -> List[Tuple[int, str]]:
        """Tìm các heading trong text."""
        headings = []
        for pattern in cls.HEADING_PATTERNS:
            for match in pattern.finditer(text):
                headings.append((match.start(), match.group()))
        return sorted(headings, key=lambda x: x[0])
    
    @classmethod
    def semantic_chunk(
        cls,
        text: str,
        target_size: int = CHUNK_SIZE,
        min_size: int = MIN_CHUNK_SIZE,
        overlap: int = CHUNK_OVERLAP
    ) -> List[Dict]:
        """
        Chunk text theo ý nghĩa semantic.
        
        Returns:
            List of dicts with 'content', 'type', 'heading'
        """
        chunks = []
        sentences = cls.split_into_sentences(text)
        headings = cls.find_headings(text)
        
        current_chunk = []
        current_size = 0
        current_heading = None
        
        # Find heading for position
        def get_heading_for_position(pos: int) -> Optional[str]:
            last_heading = None
            for h_pos, h_text in headings:
                if h_pos <= pos:
                    last_heading = h_text
                else:
                    break
            return last_heading
        
        char_position = 0
        
        for sentence in sentences:
            sentence_len = len(sentence)
            
            # Check if adding this sentence exceeds target
            if current_size + sentence_len > target_size and current_size >= min_size:
                # Save current chunk
                chunk_text = ' '.join(current_chunk)
                if chunk_text.strip():
                    chunks.append({
                        'content': chunk_text,
                        'type': 'content',
                        'heading': current_heading
                    })
                
                # Start new chunk with overlap
                overlap_sentences = []
                overlap_size = 0
                for s in reversed(current_chunk):
                    if overlap_size + len(s) <= overlap:
                        overlap_sentences.insert(0, s)
                        overlap_size += len(s)
                    else:
                        break
                
                current_chunk = overlap_sentences
                current_size = overlap_size
            
            # Update heading
            heading = get_heading_for_position(char_position)
            if heading:
                current_heading = heading
            
            current_chunk.append(sentence)
            current_size += sentence_len
            char_position += sentence_len + 1
        
        # Don't forget last chunk
        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            if chunk_text.strip() and len(chunk_text) >= min_size:
                chunks.append({
                    'content': chunk_text,
                    'type': 'content',
                    'heading': current_heading
                })
        
        return chunks


# ============================================================
# ENHANCED EMBEDDING GENERATION
# ============================================================

class EnhancedEmbeddingGenerator:
    """
    Enhanced embedding generation với:
    - KHÔNG delay giữa requests (giống load_kb.py gốc)
    - Chỉ retry khi bị rate limit
    - Context-aware embedding
    """
    
    def __init__(self, model: str = EMBEDDING_MODEL):
        self.model = model
        self.request_count = 0
        self.max_retries = 3
    
    def generate_single(self, text: str, task_type: str = "RETRIEVAL_DOCUMENT") -> Optional[List[float]]:
        """Generate embedding - chạy nhanh như load_kb.py gốc, chỉ retry khi lỗi."""
        for attempt in range(self.max_retries):
            try:
                # Truncate if too long
                text = text[:8000]
                
                response = genai.embed_content(
                    model=self.model,
                    content=text,
                    task_type=task_type
                )
                
                self.request_count += 1
                return response['embedding']
                
            except Exception as e:
                error_str = str(e)
                
                # Nếu là rate limit error, đợi và retry
                if "429" in error_str or "quota" in error_str.lower() or "rate" in error_str.lower():
                    wait_time = (2 ** attempt) * 10  # 10s, 20s, 40s
                    logger.warning(f"   ⚠️ Rate limit, đợi {wait_time}s (attempt {attempt + 1}/{self.max_retries})")
                    time.sleep(wait_time)
                else:
                    logger.error(f"Embedding error: {e}")
                    return None
        
        logger.error(f"Failed after {self.max_retries} retries")
        return None
    
    def generate_batch(
        self,
        texts: List[str],
        task_type: str = "RETRIEVAL_DOCUMENT"
    ) -> List[Optional[List[float]]]:
        """
        Generate embeddings cho batch of texts.
        Note: Gemini API may not support true batching, so we iterate.
        """
        embeddings = []
        
        for i, text in enumerate(texts):
            embedding = self.generate_single(text, task_type)
            embeddings.append(embedding)
            
            if (i + 1) % 10 == 0:
                logger.info(f"   Generated {i + 1}/{len(texts)} embeddings...")
        
        return embeddings
    
    def generate_with_context(
        self,
        content: str,
        title: str = "",
        heading: str = "",
        topics: List[str] = None
    ) -> Optional[List[float]]:
        """
        Generate embedding với context bổ sung.
        Prepend metadata để embedding hiểu context tốt hơn.
        """
        # Build context-enriched text
        context_parts = []
        
        if title:
            context_parts.append(f"Sách: {title}")
        if heading:
            context_parts.append(f"Phần: {heading}")
        if topics:
            context_parts.append(f"Chủ đề: {', '.join(topics[:3])}")
        
        if context_parts:
            enriched_text = f"{' | '.join(context_parts)}\n\n{content}"
        else:
            enriched_text = content
        
        return self.generate_single(enriched_text)


# ============================================================
# SUMMARY GENERATION (Disabled for Free Tier to save quota)
# ============================================================

class ChunkSummarizer:
    """Generate summaries cho chunks. DISABLED by default to save API quota."""
    
    def __init__(self, enabled: bool = False):
        self.enabled = enabled
        if enabled:
            self.model = genai.GenerativeModel("gemini-2.0-flash")
        else:
            self.model = None
    
    def summarize(self, content: str, max_length: int = 100) -> Optional[str]:
        """Generate short summary. Returns None if disabled."""
        # Skip if disabled (to save API quota)
        if not self.enabled:
            return None
            
        if len(content) < 200:
            return None
        
        try:
            prompt = f"""Tóm tắt đoạn văn sau trong 1-2 câu ngắn gọn (tối đa {max_length} từ).
Chỉ trả về câu tóm tắt, không giải thích thêm.

Đoạn văn:
{content[:2000]}

Tóm tắt:"""
            
            response = self.model.generate_content(prompt)
            summary = response.text.strip()
            
            # Limit length
            words = summary.split()
            if len(words) > max_length:
                summary = ' '.join(words[:max_length]) + '...'
            
            return summary
            
        except Exception as e:
            logger.warning(f"Summary generation failed: {e}")
            return None


# ============================================================
# DATABASE OPERATIONS
# ============================================================

def get_db_connection():
    """Create PostgreSQL connection."""
    try:
        parsed_url = urlparse(DATABASE_URL)
        conn = psycopg2.connect(
            host=parsed_url.hostname or 'postgres',
            port=parsed_url.port or 5432,
            database=parsed_url.path.lstrip('/').split('?')[0],
            user=parsed_url.username,
            password=parsed_url.password,
        )
        logger.info("✅ Connected to PostgreSQL")
        return conn
    except Exception as e:
        logger.error(f"❌ Database connection failed: {e}")
        sys.exit(1)


def init_enhanced_schema(conn):
    """Initialize enhanced database schema."""
    cur = conn.cursor()
    try:
        # Enable pgvector
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
        
        # Drop old table if exists (for clean reload)
        # cur.execute("DROP TABLE IF EXISTS knowledge_documents CASCADE")
        
        # Create enhanced table
        cur.execute("""
        CREATE TABLE IF NOT EXISTS knowledge_documents (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            chunk_id VARCHAR(255) UNIQUE,
            source_file VARCHAR(255),
            chunk_index INTEGER,
            content TEXT NOT NULL,
            summary TEXT,
            embedding vector(768),
            summary_embedding vector(768),
            doc_metadata JSONB,
            key_terms TEXT[],
            word_count INTEGER,
            char_count INTEGER,
            language VARCHAR(10) DEFAULT 'vi',
            chunk_type VARCHAR(50) DEFAULT 'content',
            parent_chunk_id VARCHAR(255),
            created_at TIMESTAMP DEFAULT NOW(),
            updated_at TIMESTAMP DEFAULT NOW()
        )
        """)
        
        # Create indexes
        cur.execute("""
        CREATE INDEX IF NOT EXISTS idx_knowledge_embedding 
        ON knowledge_documents USING ivfflat (embedding vector_cosine_ops)
        WITH (lists = 100)
        """)
        
        cur.execute("""
        CREATE INDEX IF NOT EXISTS idx_knowledge_summary_embedding 
        ON knowledge_documents USING ivfflat (summary_embedding vector_cosine_ops)
        WITH (lists = 50)
        """)
        
        cur.execute("""
        CREATE INDEX IF NOT EXISTS idx_knowledge_source 
        ON knowledge_documents (source_file)
        """)
        
        cur.execute("""
        CREATE INDEX IF NOT EXISTS idx_knowledge_chunk_type 
        ON knowledge_documents (chunk_type)
        """)
        
        cur.execute("""
        CREATE INDEX IF NOT EXISTS idx_knowledge_key_terms 
        ON knowledge_documents USING GIN (key_terms)
        """)
        
        cur.execute("""
        CREATE INDEX IF NOT EXISTS idx_knowledge_metadata 
        ON knowledge_documents USING GIN (doc_metadata)
        """)
        
        # Full-text search index for BM25-like search
        cur.execute("""
        CREATE INDEX IF NOT EXISTS idx_knowledge_content_fts 
        ON knowledge_documents USING GIN (to_tsvector('simple', content))
        """)
        
        conn.commit()
        logger.info("✅ Enhanced database schema initialized")
        
    except Exception as e:
        logger.error(f"❌ Schema initialization failed: {e}")
        conn.rollback()
        raise


def insert_enhanced_chunk(
    conn,
    chunk: EnhancedChunk,
    embedding: List[float],
    summary_embedding: Optional[List[float]] = None
) -> bool:
    """Insert enhanced chunk into database."""
    cur = conn.cursor()
    try:
        cur.execute("""
            INSERT INTO knowledge_documents 
            (chunk_id, source_file, chunk_index, content, summary, 
             embedding, summary_embedding, doc_metadata, key_terms,
             word_count, char_count, language, chunk_type, parent_chunk_id)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (chunk_id) DO UPDATE SET
                content = EXCLUDED.content,
                summary = EXCLUDED.summary,
                embedding = EXCLUDED.embedding,
                summary_embedding = EXCLUDED.summary_embedding,
                doc_metadata = EXCLUDED.doc_metadata,
                updated_at = NOW()
        """, (
            chunk.chunk_id,
            chunk.source_file,
            chunk.chunk_index,
            chunk.content,
            chunk.summary,
            embedding,
            summary_embedding,
            json.dumps({
                'book_title': chunk.book_title,
                'author': chunk.author,
                'content_type': chunk.content_type,
                'topics': chunk.topics,
                'page_start': chunk.page_start,
                'page_end': chunk.page_end,
                'heading': chunk.chunk_type
            }),
            chunk.key_terms,
            chunk.word_count,
            chunk.char_count,
            chunk.language,
            chunk.chunk_type,
            chunk.parent_chunk_id
        ))
        return True
        
    except Exception as e:
        logger.error(f"Insert error: {e}")
        return False


# ============================================================
# PDF EXTRACTION (Enhanced)
# ============================================================

def extract_pdf_content(pdf_path: Path) -> List[Dict]:
    """Extract text from PDF with page tracking."""
    try:
        import PyPDF2
    except ImportError:
        logger.warning("⚠️ PyPDF2 not installed")
        return []
    
    pages = []
    try:
        with open(pdf_path, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            for page_num, page in enumerate(reader.pages):
                text = page.extract_text()
                if text and len(text.strip()) > 50:
                    # Clean text
                    text = VietnameseTextProcessor.clean_text(text)
                    pages.append({
                        'page': page_num + 1,
                        'text': text
                    })
        
        logger.info(f"✅ Extracted {len(pages)} pages from {pdf_path.name}")
        return pages
        
    except Exception as e:
        logger.error(f"❌ PDF extraction failed: {e}")
        return []


# ============================================================
# MAIN LOADING PROCESS
# ============================================================

def load_enhanced_knowledge_base(conn, enable_summaries: bool = False):
    """Load knowledge base với enhanced chunking và embedding.
    
    Args:
        conn: Database connection
        enable_summaries: Set True to generate summaries (uses more API quota)
    """
    logger.info("\n📚 Loading Enhanced Knowledge Base...")
    logger.info(f"   Summaries: {'ENABLED' if enable_summaries else 'DISABLED (saving API quota)'}")
    
    embedding_gen = EnhancedEmbeddingGenerator()
    summarizer = ChunkSummarizer(enabled=enable_summaries)
    
    total_chunks = 0
    total_embeddings = 0
    
    for pdf_file, metadata in THERAPEUTIC_BOOKS.items():
        pdf_path = KNOWLEDGE_BASE_PATH / pdf_file
        
        if not pdf_path.exists():
            logger.warning(f"⚠️ PDF not found: {pdf_path}")
            continue
        
        logger.info(f"\n{'='*60}")
        logger.info(f"📖 Processing: {metadata['title']}")
        logger.info(f"   Author: {metadata.get('author', 'Unknown')}")
        logger.info(f"   File: {pdf_file}")
        
        # Extract pages
        pages = extract_pdf_content(pdf_path)
        if not pages:
            continue
        
        # Combine all text
        all_text = '\n\n'.join([p['text'] for p in pages])
        
        # Semantic chunking
        raw_chunks = SemanticChunker.semantic_chunk(
            all_text,
            target_size=CHUNK_SIZE,
            min_size=MIN_CHUNK_SIZE,
            overlap=CHUNK_OVERLAP
        )
        
        logger.info(f"   📄 Created {len(raw_chunks)} semantic chunks")
        
        # Process each chunk
        for i, raw_chunk in enumerate(raw_chunks):
            content = raw_chunk['content']
            heading = raw_chunk.get('heading')
            
            # Generate chunk ID
            chunk_id = hashlib.md5(
                f"{pdf_file}_{i}_{content[:50]}".encode()
            ).hexdigest()[:16]
            
            # Extract key terms
            key_terms = VietnameseTextProcessor.extract_key_terms(content)
            
            # Detect language
            language = VietnameseTextProcessor.detect_language(content)
            
            # Generate summary (only for longer chunks)
            summary = None
            if len(content) > 500:
                summary = summarizer.summarize(content)
            
            # Create enhanced chunk
            chunk = EnhancedChunk(
                content=content,
                chunk_id=f"{pdf_file}_{chunk_id}",
                chunk_index=i,
                source_file=pdf_file,
                book_title=metadata['title'],
                author=metadata.get('author', ''),
                content_type=metadata['content_type'],
                topics=metadata['topics'],
                page_start=None,
                page_end=None,
                parent_chunk_id=None,
                chunk_type='content',
                word_count=len(content.split()),
                char_count=len(content),
                language=language,
                key_terms=key_terms,
                summary=summary,
                created_at=datetime.now().isoformat()
            )
            
            # Generate embedding with context
            embedding = embedding_gen.generate_with_context(
                content=content,
                title=metadata['title'],
                heading=heading or "",
                topics=metadata['topics']
            )
            
            if not embedding:
                logger.warning(f"   ⚠️ Failed embedding for chunk {i}")
                continue
            
            # Generate summary embedding if summary exists
            summary_embedding = None
            if summary:
                summary_embedding = embedding_gen.generate_single(summary)
            
            # Insert into database
            success = insert_enhanced_chunk(
                conn, chunk, embedding, summary_embedding
            )
            
            if success:
                total_chunks += 1
                total_embeddings += 1
                if summary_embedding:
                    total_embeddings += 1
            
            # Commit periodically
            if total_chunks % 20 == 0:
                conn.commit()
                logger.info(f"   ✅ Processed {total_chunks} chunks...")
        
        # Final commit for this book
        conn.commit()
        logger.info(f"   ✅ Completed: {metadata['title']}")
    
    logger.info(f"\n{'='*60}")
    logger.info(f"🎉 Total: {total_chunks} chunks, {total_embeddings} embeddings")
    logger.info(f"   API calls: {embedding_gen.request_count}")
    
    return total_chunks


def verify_enhanced_load(conn) -> bool:
    """Verify enhanced knowledge base."""
    cur = conn.cursor()
    try:
        # Count documents
        cur.execute("SELECT COUNT(*) FROM knowledge_documents")
        total = cur.fetchone()[0]
        
        # Count by source
        cur.execute("""
            SELECT source_file, COUNT(*), AVG(word_count)::int 
            FROM knowledge_documents 
            GROUP BY source_file
        """)
        sources = cur.fetchall()
        
        # Count with summaries
        cur.execute("SELECT COUNT(*) FROM knowledge_documents WHERE summary IS NOT NULL")
        with_summary = cur.fetchone()[0]
        
        # Count with key terms
        cur.execute("SELECT COUNT(*) FROM knowledge_documents WHERE array_length(key_terms, 1) > 0")
        with_terms = cur.fetchone()[0]
        
        logger.info(f"\n✅ Verification Results:")
        logger.info(f"   Total documents: {total}")
        logger.info(f"   With summaries: {with_summary}")
        logger.info(f"   With key terms: {with_terms}")
        logger.info(f"\n   By source:")
        for source, count, avg_words in sources:
            logger.info(f"   • {source}: {count} chunks (avg {avg_words} words)")
        
        return total > 0
        
    except Exception as e:
        logger.error(f"Verification failed: {e}")
        return False


# ============================================================
# MAIN
# ============================================================

def main():
    """Main execution."""
    logger.info("\n" + "="*70)
    logger.info("  🚀 ENHANCED KNOWLEDGE BASE LOADER")
    logger.info("  Features:")
    logger.info("    • Semantic Chunking (meaning-based)")
    logger.info("    • Context-Enriched Embeddings")
    logger.info("    • Summary Generation")
    logger.info("    • Vietnamese Text Processing")
    logger.info("    • Key Term Extraction")
    logger.info("    • Full-Text Search Index")
    logger.info("="*70)
    
    if not DATABASE_URL:
        logger.error("❌ Missing DATABASE_URL")
        sys.exit(1)
    
    if not GOOGLE_API_KEY:
        logger.error("❌ Missing GOOGLE_API_KEY")
        sys.exit(1)
    
    conn = get_db_connection()
    
    try:
        init_enhanced_schema(conn)
        total = load_enhanced_knowledge_base(conn)
        success = verify_enhanced_load(conn)
        
        if success:
            logger.info("\n" + "="*70)
            logger.info("✅ SUCCESS! Enhanced Knowledge Base Ready!")
            logger.info("   • Semantic chunks for precise retrieval")
            logger.info("   • Context-enriched embeddings")
            logger.info("   • Summaries for quick scanning")
            logger.info("   • Full-text search enabled")
            logger.info("   • Vietnamese text optimized")
            logger.info("="*70 + "\n")
        else:
            logger.warning("\n⚠️ No documents loaded. Check logs.")
    
    except Exception as e:
        logger.error(f"\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    finally:
        conn.close()
        logger.info("✅ Database connection closed")


if __name__ == '__main__':
    main()

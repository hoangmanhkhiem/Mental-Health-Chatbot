#!/usr/bin/env python3

"""
Therapeutic Knowledge Base Loader for Azure PostgreSQL
Extracts PDFs, generates OpenAI embeddings, and loads into pgvector
Run AFTER deploying: python3 backend/load_kb.py
"""

import os
import sys
import json
import logging
from pathlib import Path
from urllib.parse import urlparse
import psycopg2
import google.generativeai as genai
from datetime import datetime

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

# Books metadata
THERAPEUTIC_BOOKS = {
    "4As_Manuscript_v6.pdf": {
    "title": "4As_Manuscript_v6",
    "author": "",
    "content_type": "therapeutic_framework",
    "topics": ["thần kinh học", "khám lâm sàng", "bệnh lý thần kinh", "y khoa", "sinh viên y khoa"]
   },
   "BeyondHappy_MANUSCRIPT_v7.pdf": {
    "title": "BeyondHappy_MANUSCRIPT_v7",
    "author": "",
    "content_type": "therapeutic_framework",
    "topics": ["thần kinh học", "khám lâm sàng", "bệnh lý thần kinh", "y khoa", "sinh viên y khoa"]
   },
    "data1.pdf": {
    "title": "Ni Thần Kinh",
    "author": "PGS.TS Hồng Khánh",
    "content_type": "therapeutic_framework",
    "topics": ["thần kinh học", "khám lâm sàng", "bệnh lý thần kinh", "y khoa", "sinh viên y khoa"]
   },
    "data2.pdf": {
    "title": "Những Trắc Nghiệm Tâm Lý Tập I",
    "author": "PGS.TS Ngô Công Hồn",
    "content_type": "therapeutic_framework",
    "topics": ["trắc nghiệm tâm lý", "trí tuệ", "phát triển trẻ em", "tâm lý học", "giáo dục"]
    },
    "data3.pdf": {
    "title": "Bệnh Học Tâm Thần",
    "author": "PGS.TS Nguyễn Kim Việt",
    "content_type": "therapeutic_framework",
    "topics": ["tâm thần học", "rối loạn tâm thần", "chẩn đoán", "điều trị", "y học"]
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

def get_db_connection():
    """Create PostgreSQL connection"""
    try:
        # Parse the SQLAlchemy connection string to psycopg2 format
        parsed_url = urlparse(DATABASE_URL)
        conn = psycopg2.connect(
            host=parsed_url.hostname or 'postgres',
            port=parsed_url.port or 5432,
            database=parsed_url.path.lstrip('/'),
            user=parsed_url.username,
            password=parsed_url.password,
    )
        logger.info("✅ Connected to PostgreSQL")
        return conn
    except Exception as e:
        logger.error(f"❌ Database connection failed: {e}")
        sys.exit(1)

def init_database(conn):
    """Initialize database schema"""
    cur = conn.cursor()
    try:
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
        logger.info("✅ pgvector extension ready")
        
        cur.execute("""
        CREATE TABLE IF NOT EXISTS knowledge_documents (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            source_file VARCHAR(255),
            chunk_index INTEGER,
            content TEXT NOT NULL,
            embedding vector(768),
            doc_metadata JSONB,
            created_at TIMESTAMP DEFAULT NOW()
        )
        """)
        
        cur.execute("""
        CREATE INDEX IF NOT EXISTS idx_knowledge_embedding 
        ON knowledge_documents USING ivfflat (embedding vector_cosine_ops)
        WITH (lists = 100)
        """)
        
        conn.commit()
        logger.info("✅ Database schema initialized")
    except Exception as e:
        logger.error(f"❌ Schema initialization failed: {e}")
        conn.rollback()
        raise

def extract_pdf_content(pdf_path):
    """Extract text from PDF using PyPDF2"""
    try:
        import PyPDF2
    except ImportError:
        logger.warning("⚠️ PyPDF2 not installed. Install with: pip install PyPDF2")
        return []
    
    text_content = []
    try:
        with open(pdf_path, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            for page_num, page in enumerate(reader.pages):
                text = page.extract_text()
                if text and len(text.strip()) > 100:
                    text_content.append({
                        'page': page_num + 1,
                        'text': text.strip()
                    })
        logger.info(f"✅ Extracted {len(text_content)} pages from {pdf_path.name}")
        return text_content
    except Exception as e:
        logger.error(f"❌ PDF extraction failed for {pdf_path}: {e}")
        return []

def chunk_text(text, chunk_size=1000, overlap=200):
    """Split text into overlapping chunks"""
    chunks = []
    for i in range(0, len(text), chunk_size - overlap):
        chunk = text[i:i + chunk_size]
        if len(chunk) > 150:
            chunks.append(chunk)
    return chunks

def generate_embedding(text):
    """Generate embedding using Google Gemini API"""
    try:
        response = genai.embed_content(
            model="models/text-embedding-004",
            content=text[:8191],
            task_type="RETRIEVAL_DOCUMENT"
        )
        embedding = response['embedding']
        return embedding
    except Exception as e:
        logger.error(f"❌ Embedding generation failed: {e}")
        return None

def load_knowledge_base(conn):
    """Load therapeutic PDFs into database"""
    logger.info("\n📚 Loading therapeutic knowledge base...")
    
    total_chunks = 0
    
    for pdf_file, metadata in THERAPEUTIC_BOOKS.items():
        pdf_path = KNOWLEDGE_BASE_PATH / pdf_file
        
        if not pdf_path.exists():
            logger.warning(f"⚠️ PDF not found: {pdf_path}")
            continue
        
        logger.info(f"\n📖 Processing: {metadata['title']}")
        logger.info(f"   File: {pdf_file}")
        
        pages = extract_pdf_content(pdf_path)
        if not pages:
            logger.warning(f"⚠️ Could not extract content from {pdf_file}")
            continue
        
        all_text = '\n\n'.join([p['text'] for p in pages])
        chunks = chunk_text(all_text)
        logger.info(f"   📄 Created {len(chunks)} chunks from {len(pages)} pages")
        
        cur = conn.cursor()
        chunk_index = 0
        inserted_count = 0
        
        for i, chunk in enumerate(chunks):
            try:
                embedding = generate_embedding(chunk)
                if not embedding:
                    logger.warning(f"   ⚠️ Failed to generate embedding for chunk {i}")
                    continue
                
                doc_metadata = {
                    'book_title': metadata['title'],
                    'author': metadata['author'],
                    'content_type': metadata['content_type'],
                    'chunk_id': f"{pdf_file}_{i}",
                    'topics': metadata['topics'],
                    'created_at': datetime.now().isoformat()
                }
                
                cur.execute("""
                    INSERT INTO knowledge_documents 
                    (source_file, chunk_index, content, embedding, doc_metadata)
                    VALUES (%s, %s, %s, %s, %s)
                """, (
                    pdf_file,
                    chunk_index,
                    chunk,
                    embedding,
                    json.dumps(doc_metadata)
                ))
                
                chunk_index += 1
                inserted_count += 1
                total_chunks += 1
                
                if inserted_count % 20 == 0:
                    conn.commit()
                    logger.info(f"   ✅ Inserted {inserted_count} chunks...")
                
            except Exception as e:
                logger.error(f"   ❌ Error processing chunk {i}: {e}")
                conn.rollback()
                continue
        
        try:
            conn.commit()
            logger.info(f"   ✅ Committed {inserted_count} chunks from {metadata['title']}")
        except Exception as e:
            logger.error(f"   ❌ Final commit failed: {e}")
            conn.rollback()
    
    logger.info(f"\n🎉 Knowledge base loaded: {total_chunks} total chunks")
    return total_chunks

def verify_load(conn):
    """Verify knowledge base was loaded"""
    cur = conn.cursor()
    try:
        cur.execute("SELECT COUNT(*) FROM knowledge_documents")
        count = cur.fetchone()[0]
        logger.info(f"\n✅ Database verification: {count} documents in knowledge base")
        
        cur.execute("""
            SELECT source_file, COUNT(*) as chunk_count 
            FROM knowledge_documents 
            GROUP BY source_file
        """)
        results = cur.fetchall()
        for source, chunk_count in results:
            logger.info(f"   • {source}: {chunk_count} chunks")
        
        return count > 0
    except Exception as e:
        logger.error(f"❌ Verification failed: {e}")
        return False

def main():
    """Main execution"""
    logger.info("\n" + "="*70)
    logger.info("  🚀 THERAPEUTIC KNOWLEDGE BASE LOADER")
    logger.info("  Loading PDFs into Azure PostgreSQL + pgvector")
    logger.info("  Generating Gemini embeddings automatically")
    logger.info("="*70)
    
    if not DATABASE_URL:
        logger.error("❌ Missing DATABASE_URL environment variable")
        sys.exit(1)
    
    if not GOOGLE_API_KEY:
        logger.error("❌ Missing GOOGLE_API_KEY environment variable")
        sys.exit(1)
    
    conn = get_db_connection()
    
    try:
        init_database(conn)
        total = load_knowledge_base(conn)
        success = verify_load(conn)
        
        if success:
            logger.info("\n" + "="*70)
            logger.info("✅ SUCCESS! Knowledge base is ready for production!")
            logger.info("   Your bot now has intelligent therapeutic responses.")
            logger.info("   Source citations are enabled.")
            logger.info("   RAG system is active with real embeddings.")
            logger.info("="*70 + "\n")
        else:
            logger.warning("\n⚠️ No documents were loaded. Check the logs above.")
    
    except Exception as e:
        logger.error(f"\n❌ Fatal error: {e}")
        sys.exit(1)
    
    finally:
        conn.close()
        logger.info("✅ Database connection closed")

if __name__ == '__main__':
    main()
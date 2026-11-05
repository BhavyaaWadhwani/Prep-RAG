import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Settings:
    gemini_api_key: str = os.getenv("GEMINI_API_KEY", "")
    llm_model: str = os.getenv("LLM_MODEL", "gemini-2.5-flash")
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "gemini-embedding-001")
    embedding_dim: int = int(os.getenv("EMBEDDING_DIM", "768"))
    admin_key: str = os.getenv("ADMIN_KEY", "")
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "800"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "120"))
    top_k: int = int(os.getenv("TOP_K", "5"))
    min_similarity: float = float(os.getenv("MIN_SIMILARITY", "0.30"))
    chroma_dir: str = os.getenv("CHROMA_DIR", str(BASE_DIR / "chroma_data"))
    knowledge_dir: str = os.getenv("KNOWLEDGE_DIR", str(BASE_DIR / "knowledge_base"))
    max_upload_mb: int = int(os.getenv("MAX_UPLOAD_MB", "10"))
    max_question_chars: int = 500


settings = Settings()
SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md"}

import os
import warnings
import time
from pathlib import Path
from dotenv import load_dotenv

warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    module=r"llama_index\.embeddings\.gemini\.base",
)

from llama_index.llms.google_genai import GoogleGenAI
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import StorageContext, load_index_from_storage
from llama_index.core.settings import Settings

load_dotenv(dotenv_path=Path(__file__).resolve().parents[1] / ".env")

# Đường dẫn tới thư mục lưu trữ vector (hỗ trợ cả cấu trúc cũ và mới)
VECTOR_STORE_CANDIDATES = [
    Path(__file__).resolve().parents[1] / "data" / "vector_db",
    Path(__file__).resolve().parents[1] / "vector_db",
]
EMBEDDING_MODEL_NAME = "keepitreal/vietnamese-sbert"


def _resolve_vector_store_dir() -> Path:
    for path in VECTOR_STORE_CANDIDATES:
        if path.exists():
            return path
    raise FileNotFoundError(
        "Vector store directory not found. Checked: "
        + ", ".join(str(p) for p in VECTOR_STORE_CANDIDATES)
    )


def get_default_api_key() -> str:
    """Lấy API key mặc định từ môi trường, hỗ trợ cả 2 biến tên cũ/mới."""
    return (os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or "").strip()


def _build_embed_model() -> HuggingFaceEmbedding:
    # Keep retrieval embeddings aligned with the embedding model used to build the index.
    return HuggingFaceEmbedding(
        model_name=EMBEDDING_MODEL_NAME,
        trust_remote_code=True,
    )

def get_query_engine(api_key: str, model_name: str = "gemini-2.5-flash-lite"):
    """
    Khởi tạo và trả về một query engine với API key và model được chỉ định.
    """
    print(f"Initializing query engine with model: {model_name}")
    started_at = time.perf_counter()
    
    if not api_key or not api_key.strip():
        raise ValueError("API key is empty.")

    # 1. Thiết lập LLM và Embedding model
    print("1/3 Initializing Gemini LLM client...")
    llm = GoogleGenAI(
        model=model_name,
        api_key=api_key,
    )
    print(f"2/3 Loading embedding model: {EMBEDDING_MODEL_NAME}...")
    embed_model = _build_embed_model()
    Settings.llm = llm
    Settings.embed_model = embed_model

    # 2. Load index từ storage
    print("3/3 Loading vector index from disk...")
    vector_store_dir = _resolve_vector_store_dir()
    storage_context = StorageContext.from_defaults(persist_dir=str(vector_store_dir))
    index = load_index_from_storage(storage_context)

    # 3. Tạo và trả về query engine
    query_engine = index.as_query_engine(similarity_top_k=6)
    elapsed = time.perf_counter() - started_at
    print(f"Query engine initialized successfully in {elapsed:.2f}s.")
    return query_engine


def initialize_default_query_engine():
    """
    Khởi tạo query engine mặc định sử dụng API key từ biến môi trường.
    """
    api_key = get_default_api_key()
    if not api_key:
        print("Warning: GEMINI_API_KEY/GOOGLE_API_KEY is not set. The default engine will not be available.")
        return None
    
    try:
        # Sử dụng model flash-lite mặc định để giảm chi phí và ổn định hạn mức
        return get_query_engine(api_key=api_key, model_name="gemini-2.5-flash-lite")
    except Exception as e:
        print(f"Failed to initialize default query engine. Error: {e}")
        return None

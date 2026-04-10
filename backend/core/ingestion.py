import os
from dotenv import load_dotenv
from llama_index.core import (
    SimpleDirectoryReader,
    VectorStoreIndex,
    StorageContext,
    load_index_from_storage,
    Settings
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
# from llama_index.readers.file import PyMuPDFReader # Dùng Local Reader thay vì Cloud OCR
from core.ocr_reader import OcrPdfReader

load_dotenv()

DATA_DIR = "./data"
PERSIST_DIR = "./vector_db"

def initialize_rag_pipeline():
    google_api_key = os.environ.get("GOOGLE_API_KEY")
    
    Settings.llm = GoogleGenAI(
        model="gemini-2.5-flash", 
        api_key=google_api_key,
        temperature=0.1
    )
    
    # NÃO TIẾNG VIỆT
    Settings.embed_model = HuggingFaceEmbedding(
        model_name="keepitreal/vietnamese-sbert",
        trust_remote_code=True
    )
    
    Settings.text_splitter = SentenceSplitter(chunk_size=1024, chunk_overlap=100)

def load_or_create_index():
    initialize_rag_pipeline()

    if not os.path.exists(os.path.join(PERSIST_DIR, "docstore.json")):
        print("🚀 Khởi động Luồng Ingestion (OCR thông minh)...")
        
        # Reader tùy chỉnh có khả năng OCR
        reader = SimpleDirectoryReader(
            DATA_DIR, 
            required_exts=[".pdf", ".txt"],
            file_extractor={".pdf": OcrPdfReader(tesseract_lang="vie")}
        )
        documents = reader.load_data()
        print(f"📄 Đã đọc xong {len(documents)} tài liệu.")
        
        # Băm tài liệu
        splitter = Settings.text_splitter
        raw_nodes = splitter.get_nodes_from_documents(documents)
        
        # Lọc nhiễu
        valid_nodes = [node for node in raw_nodes if len(node.get_content().strip()) > 20]
        print(f"🧩 Đã băm và lọc nhiễu, thu được {len(valid_nodes)} chunks chất lượng cao.")
        
        # Tạo Vector DB
        print("⚡ Bắt đầu tạo mã hóa nhúng vector (Local Embedding)...")
        index = VectorStoreIndex(valid_nodes, show_progress=True)
            
        index.storage_context.persist(persist_dir=PERSIST_DIR)
        print("✅ Đã xử lý trọn vẹn toàn bộ tài liệu và lưu Vector DB thành công!")
    else:
        print("⚡ Đã tìm thấy Vector DB local. Đang load dữ liệu...")
        storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
        index = load_index_from_storage(storage_context)
        print("✅ Load Vector DB thành công!")
        
    return index

if __name__ == "__main__":
    print("Bắt đầu chạy thử Ingestion Pipeline...")
    index = load_or_create_index()
    print("Sẵn sàng cho việc truy xuất (Retrieval)!")
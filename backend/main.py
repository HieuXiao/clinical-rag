from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.routes import router
from dotenv import load_dotenv
from pathlib import Path
import os

from contextlib import asynccontextmanager
from core.generator import initialize_default_query_engine

load_dotenv(dotenv_path=Path(__file__).resolve().parent / ".env")

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the RAG engine on startup with the default API key
    print("Khởi động RAG Engine mặc định cho API...")
    app.state.query_engine = initialize_default_query_engine()
    if app.state.query_engine:
        print("✅ RAG Engine mặc định đã sẵn sàng!")
    else:
        print("⚠️ Lỗi: Không thể khởi động RAG Engine mặc định. API key chưa được set (GEMINI_API_KEY hoặc GOOGLE_API_KEY).")
    yield
    # Clean up the models and release the resources
    print("Dọn dẹp và tắt RAG Engine...")
    app.state.query_engine = None

app = FastAPI(
    title="Clinical RAG API",
    description="API cung cấp dịch vụ trợ lý y khoa AI",
    version="1.0.0",
    lifespan=lifespan
)

# Cáº¥u hÃ¬nh CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api")

if __name__ == "__main__":
    import uvicorn
    reload_enabled = os.getenv("UVICORN_RELOAD", "false").lower() == "true"
    backend_host = os.getenv("BACKEND_HOST", "127.0.0.1")
    backend_port = int(os.getenv("BACKEND_PORT", "8001"))
    browser_host = "127.0.0.1" if backend_host == "0.0.0.0" else backend_host
    print(f"BE listening on: http://{browser_host}:{backend_port}")
    print(f"Health/models URL: http://{browser_host}:{backend_port}/api/models")
    uvicorn.run("main:app", host=backend_host, port=backend_port, reload=reload_enabled)

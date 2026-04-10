from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel
from typing import Optional
from datetime import date
from pathlib import Path
import json
import threading
from core.generator import get_query_engine, get_default_api_key

router = APIRouter()

MODEL_LIMITS = {
    "gemini-2.5-flash": 20,
    "gemini-2.5-flash-lite": 20,
}

USAGE_FILE = Path(__file__).resolve().parents[1] / "data" / "model_usage.json"
_usage_lock = threading.RLock()


def _default_usage_state():
    return {
        "usage_day": date.today().isoformat(),
        "model_usage": {name: 0 for name in MODEL_LIMITS},
    }


def _load_usage_state():
    if not USAGE_FILE.exists():
        state = _default_usage_state()
        USAGE_FILE.parent.mkdir(parents=True, exist_ok=True)
        USAGE_FILE.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
        return state["usage_day"], state["model_usage"]

    try:
        raw = json.loads(USAGE_FILE.read_text(encoding="utf-8"))
        usage_day = str(raw.get("usage_day", date.today().isoformat()))
        file_usage = raw.get("model_usage", {})
        merged_usage = {name: int(file_usage.get(name, 0)) for name in MODEL_LIMITS}
        return usage_day, merged_usage
    except Exception:
        state = _default_usage_state()
        USAGE_FILE.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
        return state["usage_day"], state["model_usage"]


def _save_usage_state():
    state = {
        "usage_day": _usage_day,
        "model_usage": _model_usage,
    }
    USAGE_FILE.parent.mkdir(parents=True, exist_ok=True)
    USAGE_FILE.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


_usage_day, _model_usage = _load_usage_state()


def _reset_usage_if_new_day() -> None:
    global _usage_day, _model_usage
    today = date.today().isoformat()
    with _usage_lock:
        if _usage_day != today:
            _usage_day = today
            _model_usage = {name: 0 for name in MODEL_LIMITS}
            _save_usage_state()


def _get_usage_percent(model_name: str) -> int:
    with _usage_lock:
        limit = MODEL_LIMITS.get(model_name, 0)
        used = _model_usage.get(model_name, 0)
    if limit <= 0:
        return 0
    return min(100, int((used / limit) * 100))


@router.get("/models")
def get_models():
    _reset_usage_if_new_day()
    models = []
    with _usage_lock:
        for model_name, limit in MODEL_LIMITS.items():
            used = _model_usage.get(model_name, 0)
            models.append(
                {
                    "name": model_name,
                    "used": used,
                    "limit": limit,
                    "usage_percent": _get_usage_percent(model_name),
                }
            )
    return {"models": models, "usage_day": _usage_day}

# Định dạng dữ liệu đầu vào từ Frontend
class ChatRequest(BaseModel):
    message: str
    model: Optional[str] = None
    api_key: Optional[str] = None

# Tạo endpoint POST /api/chat
@router.post("/chat")
def chat_endpoint(req: ChatRequest, request: Request):
    try:
        _reset_usage_if_new_day()
        query_engine = request.app.state.query_engine
        selected_model = (req.model or "gemini-2.5-flash-lite").strip()
        user_api_key = (req.api_key or "").strip()

        # Nếu người dùng cung cấp API key, tạo một query engine mới
        if user_api_key:
            print(f"Using user-provided API key and model: {selected_model}")
            try:
                query_engine = get_query_engine(
                    api_key=user_api_key,
                    model_name=selected_model,
                )
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Could not initialize RAG engine with the provided API key or model. Error: {e}")
        
        elif req.model:
            # Nếu người dùng chỉ chọn model mà không cung cấp API key
            print(f"Switching model to: {selected_model}")
            try:
                # Lấy API key mặc định từ môi trường
                default_api_key = get_default_api_key()
                if not default_api_key:
                    raise HTTPException(
                        status_code=400,
                        detail="Chưa cấu hình API key hệ thống (GEMINI_API_KEY/GOOGLE_API_KEY). Hãy dán API key vào ô nhập trên UI.",
                    )
                
                query_engine = get_query_engine(
                    api_key=default_api_key,
                    model_name=selected_model,
                )
            except HTTPException:
                raise
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Could not switch model. Error: {e}")


        if query_engine is None:
            raise HTTPException(status_code=503, detail="RAG Engine is not available.")

        # AI xử lý câu hỏi
        response = query_engine.query(req.message)

        if selected_model in _model_usage:
            with _usage_lock:
                _model_usage[selected_model] += 1
                _save_usage_state()
        
        answer_text = str(response).strip()
        if not answer_text or answer_text.lower() == "empty response":
            answer_text = "Hệ thống RAG đã truy xuất tài liệu nhưng không có câu trả lời nào được tạo ra. Vui lòng thử diễn đạt lại câu hỏi."
            
        # Trích xuất nguồn tài liệu (Citation)
        sources = []
        if hasattr(response, 'source_nodes'):
            for node in response.source_nodes:
                file_name = node.node.metadata.get('file_name', 'Tài liệu y khoa')
                snippet = node.node.get_content()[:200] + "..."
                sources.append(f"[{file_name}] {snippet}")

        # Trả về chuỗi JSON cho ReactJS
        return {
            "answer": answer_text,
            "sources": sources
        }
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

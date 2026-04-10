import os
from dotenv import load_dotenv
from llama_index.core import Settings
from llama_index.llms.google_genai import GoogleGenAI
from google.api_core import exceptions as google_exceptions

from core.ingestion import load_or_create_index
from core.config import AVAILABLE_MODELS

load_dotenv()

def initialize_query_engine(api_key: str = None):
    """
    Khởi tạo và trả về một query engine.
    Hàm này sẽ thử lần lượt các model trong AVAILABLE_MODELS.
    Nếu một model hết hạn ngạch, nó sẽ tự động thử model tiếp theo.

    Args:
        api_key (str, optional): API key do người dùng cung cấp. 
                                 Nếu là None, sẽ sử dụng key từ .env.

    Returns:
        Một instance của query_engine nếu thành công, ngược lại là None.
    """
    # Sử dụng key do người dùng cung cấp hoặc key mặc định từ .env
    final_api_key = api_key if api_key else os.environ.get("GOOGLE_API_KEY")

    if not final_api_key:
        print("Lỗi: Không tìm thấy GOOGLE_API_KEY trong file .env hoặc do người dùng cung cấp.")
        return None

    llm = None
    last_exception = None

    # Lặp qua danh sách model để tìm model hoạt động
    for model_name in AVAILABLE_MODELS:
        try:
            print(f"Đang thử khởi tạo với model: {model_name}...")
            # Cố gắng khởi tạo LLM với model hiện tại.
            # LlamaIndex sẽ tự động xử lý tên model (có hoặc không có tiền tố "models/").
            # Chúng ta sẽ bắt lỗi cụ thể nếu model không tìm thấy hoặc hết hạn ngạch.
            llm = GoogleGenAI(model=model_name, api_key=final_api_key, temperature=0.1)
            
            # Gửi một request nhỏ để kiểm tra xem model có hoạt động không
            llm.complete("test") 

            print(f"✅ Model '{model_name}' hoạt động thành công!")
            break  # Thoát khỏi vòng lặp nếu tìm thấy model hoạt động
        except google_exceptions.NotFound as e:
            print(f" Lỗi: Model '{model_name}' không tồn tại hoặc không thể truy cập (404). Đang thử model tiếp theo...")
            last_exception = e
            llm = None
            continue
        except google_exceptions.ResourceExhausted as e:
            print(f"⚠️ Model '{model_name}' đã hết hạn ngạch (429). Đang thử model tiếp theo...")
            last_exception = e
            llm = None
            continue
        except Exception as e:
            print(f" Lỗi không xác định khi khởi tạo model '{model_name}': {e}")
            last_exception = e
            llm = None
            continue
    
    if not llm:
        print("Lỗi: Không có model nào trong danh sách AVAILABLE_MODELS hoạt động.")
        # Nếu muốn, bạn có thể raise exception ở đây
        # raise last_exception
        return None

    # Gán LLM đã hoạt động thành công vào Settings
    Settings.llm = llm
    
    # Tải index và tạo query engine như bình thường
    index = load_or_create_index()
    query_engine = index.as_query_engine(similarity_top_k=6, streaming=True)
    return query_engine

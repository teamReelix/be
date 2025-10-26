import os
import logging
from model_v1.highlight_generator import export_highlight_from_full_mp4
from s3_utils import upload_to_s3
from progress_state import progress_data, progress_lock

logger = logging.getLogger(__name__)

def process_highlight(local_path: str, result_filename: str, target_minutes: int):
    """백그라운드에서 하이라이트 생성 및 S3 업로드"""
    try:
        # --- 하이라이트 생성 ---
        result = export_highlight_from_full_mp4(local_path, target_minutes)
        result_path = result[0] if isinstance(result, (list, tuple)) else result
        result_path = os.path.abspath(str(result_path))

        if not os.path.exists(result_path):
            raise FileNotFoundError(f"하이라이트 파일이 존재하지 않음: {result_path}")

        # --- S3 업로드 ---
        highlight_s3_key = f"exports/{result_filename}"
        highlight_s3_url = upload_to_s3(result_path, highlight_s3_key)

        # --- 로컬 파일 삭제 ---
        os.remove(local_path)
        os.remove(result_path)

        # 진행률 초기화
        with progress_lock:
            progress_data["total"] = 0
            progress_data["done"] = 0
            progress_data["current_start"] = 0

        logger.info(f"하이라이트 처리 완료: {highlight_s3_url}")
    except Exception as e:
        logger.error(f"하이라이트 처리 실패: {e}")

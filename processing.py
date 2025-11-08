import os
from importlib import import_module
from typing import Optional

from config import logger
from s3_utils import upload_to_s3
from model import get_model


def process_highlight(
    local_path: str,
    filename: str,
    target_minutes: int,
    result_filename: str,
    model_version: str = "v1",
    logo_path: Optional[str] = None  # 로고 경로
):
    """하이라이트 생성, S3 업로드, 로컬 파일 삭제를 수행하는 백그라운드 작업"""

    model = get_model(model_version)
    if model is None:
        logger.info(f"{model_version} 모델이 로드되지 않아 하이라이트 처리를 중단합니다.")
        return

    try:
        # --- 모델 버전에 따라 highlight_generator 불러오기 ---
        hg_module_name = f"model_{model_version}.highlight_generator"
        hg = import_module(hg_module_name)

        # --- 로컬 하이라이트 생성 ---
        if model_version == "v2":
            # 로고 경로를 리스트로 전달 (함수 내부에서 처리됨)
            logo_templates = [logo_path] if logo_path else []
            result = hg.export_highlight_from_full_mp4(
                local_path,
                target_minutes,
                logo_templates=logo_templates
            )
        else:
            result = hg.export_highlight_from_full_mp4(local_path, target_minutes)

        result_path = result[0] if isinstance(result, (list, tuple)) else result
        result_path = os.path.abspath(str(result_path))
        logger.info(f"[{model_version}] 하이라이트 결과 경로: {result_path}")

        if not os.path.exists(result_path):
            raise FileNotFoundError(f"하이라이트 파일이 생성되지 않음: {result_path}")

        logger.info(f"[{model_version}] 하이라이트 로컬 생성 완료")

        # --- S3 업로드 ---
        highlight_s3_key = f"exports/{result_filename}"
        highlight_s3_url = upload_to_s3(result_path, highlight_s3_key)
        logger.info(f"[{model_version}] 하이라이트 S3 업로드 완료: {highlight_s3_url}")

    except Exception as e:
        logger.info(f"[{model_version}] 하이라이트 처리 실패 (파일: {filename}): {e}")

    finally:
        # --- 로컬 파일 삭제 ---
        try:
            if os.path.exists(local_path):
                os.remove(local_path)
                logger.info(f"원본 로컬 파일 삭제: {local_path}")
            if 'result_path' in locals() and os.path.exists(result_path):
                os.remove(result_path)
                logger.info(f"하이라이트 로컬 파일 삭제: {result_path}")
        except Exception as e:
            logger.info(f"로컬 파일 삭제 실패: {e}")


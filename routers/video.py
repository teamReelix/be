import os
import shutil
import uuid
from typing import Optional
from fastapi import (
    APIRouter, UploadFile, File, BackgroundTasks,
    HTTPException, Form
)
from fastapi.responses import JSONResponse

from config import VIDEO_DIR, logger
from model import get_model
from progress_state import progress_data, progress_lock
from s3_utils import upload_to_s3, check_and_get_presigned_url
from processing import process_highlight

router = APIRouter()


@router.post("/upload-video/")
async def upload_video_for_highlight(
        background_tasks: BackgroundTasks,
        video: UploadFile = File(...),
        target_minutes: int = Form(5),
        model_version: str = Form("v1"),
        logo: Optional[UploadFile] = File(None)  # v2일 때만 업로드될 로고
):
    if get_model() is None:
        raise HTTPException(status_code=503, detail="모델이 로드되지 않아 서비스를 사용할 수 없습니다.")

    # --- 원본 영상 저장 ---
    unique_id = str(uuid.uuid4().hex[:8])
    base, ext = os.path.splitext(video.filename)
    safe_base = "".join(c for c in base if c.isalnum() or c in (' ', '_')).rstrip()
    input_filename = f"{safe_base}_{unique_id}{ext}"
    input_path = os.path.abspath(os.path.join(VIDEO_DIR, input_filename))

    try:
        with open(input_path, "wb") as buffer:
            shutil.copyfileobj(video.file, buffer)
        logger.info(f"로컬 영상 저장 완료: {input_path}")
    except Exception as e:
        logger.error(f"영상 저장 실패: {e}")
        raise HTTPException(status_code=500, detail="영상 파일 저장 중 오류")

    # --- 원본 영상 S3 업로드 ---
    original_s3_key = f"uploads/{input_filename}"
    original_s3_url = upload_to_s3(input_path, original_s3_key)

    if original_s3_url:
        logger.info(f"[S3 Upload] 업로드 완료 : {original_s3_url}")
    else:
        logger.error(f"[S3 Upload] 업로드 실패 : {original_s3_key}")

    # 모델 v2 로고 저장 경로
    ASSETS_DIR = os.path.join(os.path.dirname(__file__), "..", "model_v2", "assets")
    os.makedirs(ASSETS_DIR, exist_ok=True)

    # --- 로고 파일 처리 (모델 v2인 경우만, 서버 로컬에 저장) ---
    logo_path = None
    if model_version == "v2" and logo:
        try:
            logo_filename = f"{uuid.uuid4().hex}_{logo.filename}"
            logo_path = os.path.abspath(os.path.join(ASSETS_DIR, logo_filename))

            with open(logo_path, "wb") as buffer:
                shutil.copyfileobj(logo.file, buffer)

            logger.info(f"model_v2/assets에 로고 저장 완료: {logo_path}")
        except Exception as e:
            logger.warning(f"로고 저장 실패 (무시하고 진행): {e}")
            logo_path = None

    # --- 결과 파일 이름 ---
    result_filename = f"{os.path.splitext(input_filename)[0]}_HIGHLIGHT_{target_minutes}m.mp4"

    # --- 진행률 초기화 ---
    with progress_lock:
        progress_data["total"] = 0
        progress_data["done"] = 0
        progress_data["current_start"] = 0

    # --- 하이라이트 처리 (백그라운드) ---
    background_tasks.add_task(
        process_highlight,
        input_path,
        input_filename,
        target_minutes,
        result_filename,
        model_version,
        logo_path  # 서버 로컬 경로 전달
    )

    return JSONResponse(
        status_code=202,
        content={
            "message": "영상 업로드 성공! 하이라이트 생성 시작",
            "original_s3_url": original_s3_url,
            "result_filename": result_filename,
            "check_status_url": f"/get-highlight/{result_filename}"
        }
    )


@router.get("/get-highlight/{filename}")
async def get_highlight_video(filename: str):
    """S3에 업로드된 하이라이트 영상 확인"""

    url = check_and_get_presigned_url(filename)

    if url:
        return JSONResponse(status_code=200, content={"status": "done", "s3_url": url})
    else:
        return JSONResponse(
            status_code=404,
            content={"status": "processing", "message": "파일이 아직 처리 중이거나 존재하지 않습니다."}
        )

import os
import shutil
import uuid
import logging
from fastapi import FastAPI, UploadFile, File, BackgroundTasks, HTTPException, Request, Form
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from threading import Lock

import highlight_generator as hg
from dotenv import load_dotenv
import boto3

from progress_state import progress_data, progress_lock

# --- 로깅 설정 ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- FastAPI 앱 ---
app = FastAPI()

# --- 경로 설정 ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
hg.CKPT_DIR = os.path.join(BASE_DIR, "ckpts")
hg.OUT_DIR = os.path.join(BASE_DIR, "exports")
VIDEO_DIR = os.path.join(BASE_DIR, "videos")
STATIC_DIR = os.path.join(BASE_DIR, "static")

os.makedirs(hg.CKPT_DIR, exist_ok=True)
os.makedirs(hg.OUT_DIR, exist_ok=True)
os.makedirs(VIDEO_DIR, exist_ok=True)
os.makedirs(STATIC_DIR, exist_ok=True)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
templates = Jinja2Templates(directory=STATIC_DIR)

# --- AWS S3 설정 ---
load_dotenv()
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_BUCKET_NAME = os.getenv("AWS_BUCKET_NAME")
AWS_REGION = os.getenv("AWS_REGION", "ap-northeast-2")

s3_client = boto3.client(
    "s3",
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    region_name=AWS_REGION
)


def upload_to_s3(file_path: str, s3_key: str) -> str:
    """파일을 S3에 업로드하고 public URL 반환"""
    s3_client.upload_file(file_path, AWS_BUCKET_NAME, s3_key)
    url = f"https://{AWS_BUCKET_NAME}.s3.{AWS_REGION}.amazonaws.com/{s3_key}"
    logger.info(f"S3 업로드 완료: {url}")
    return url

# --- 모델 로드 ---
@app.on_event("startup")
def load_model_on_startup():
    logger.info("서버 시작... 모델 로드 중...")
    try:
        hg.model = hg.load_model()
        logger.info("모델 로드 완료.")
    except Exception as e:
        logger.error(f"모델 로드 실패: {e}")
        hg.model = None

# --- 엔드포인트 ---
@app.get("/", response_class=HTMLResponse)
async def read_index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/upload-video/")
async def upload_video_for_highlight(
        background_tasks: BackgroundTasks,
        video: UploadFile = File(...),
        target_minutes: int = Form(5)
):
    if hg.model is None:
        raise HTTPException(status_code=503, detail="모델이 로드되지 않아 서비스를 사용할 수 없습니다.")

    # --- 원본 영상 저장 ---
    unique_id = str(uuid.uuid4().hex[:8])
    base, ext = os.path.splitext(video.filename)
    safe_base = "".join(c for c in base if c.isalnum() or c in (' ', '_')).rstrip()
    input_filename = f"{safe_base}_{unique_id}{ext}"
    input_path = os.path.abspath(os.path.join(VIDEO_DIR, input_filename))  # ✅ 절대 경로

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

    # --- 결과 파일 이름 미리 계산 ---
    result_filename = f"{os.path.splitext(input_filename)[0]}_HIGHLIGHT_{target_minutes}m.mp4"

    # --- 하이라이트 처리 (백그라운드) ---
    def process_highlight(local_path: str, filename: str, target_minutes: int, result_filename: str):
        # 타입 체크
        if not isinstance(local_path, str):
            logger.error(f"파일 경로 타입 오류: {type(local_path)}")
            return
        if not isinstance(target_minutes, int):
            logger.error(f"하이라이트 길이 타입 오류: {type(target_minutes)}")
            return

        try:
            # --- 로컬 하이라이트 생성 ---
            result = hg.export_highlight_from_full_mp4(local_path, target_minutes)  # ✅ 수정: 반환값 전체 받기
            result_path = result[0] if isinstance(result, (list, tuple)) else result  # ✅ 수정: 실제 경로만 추출
            result_path = os.path.abspath(str(result_path))

            if not os.path.exists(result_path):
                raise FileNotFoundError(f"하이라이트 파일이 존재하지 않음: {result_path}")

            # --- S3 업로드 ---
            highlight_s3_key = f"exports/{result_filename}"  # ✅ 수정: 함수 안에서 result_filename 사용
            highlight_s3_url = upload_to_s3(result_path, highlight_s3_key)
            logger.info(f"하이라이트 S3 업로드 완료: {highlight_s3_url}")

            # --- 로컬 파일 삭제 ---
            os.remove(local_path)
            os.remove(result_path)

        except Exception as e:
            logger.error(f"하이라이트 처리 실패: {e}")

    # 수정: result_filename도 전달
    background_tasks.add_task(process_highlight, input_path, input_filename, target_minutes, result_filename)

    return JSONResponse(
        status_code=202,
        content={
            "message": "영상 업로드 성공! 하이라이트 생성 시작",
            "original_s3_url": original_s3_url,
            "result_filename": result_filename,
            "check_status_url": f"/get-highlight/{result_filename}"
        }
    )


@app.get("/get-highlight/{filename}")
async def get_highlight_video(filename: str):
    """S3에 업로드된 하이라이트 영상 확인"""
    s3_key = f"exports/{filename}"
    try:
        s3_client.head_object(Bucket=AWS_BUCKET_NAME, Key=s3_key)
        url = s3_client.generate_presigned_url(
            'get_object',
            Params={'Bucket': AWS_BUCKET_NAME, 'Key': s3_key},
            ExpiresIn=7200  # 2시간 유효
        )
        logger.info(f"S3 파일 존재 확인: {filename}")
        return JSONResponse(status_code=200, content={"status": "done", "s3_url": url})
    except s3_client.exceptions.ClientError:
        logger.warning(f"S3 파일 없음: {filename}")
        return JSONResponse(status_code=404, content={"status": "processing", "message": "파일이 아직 처리 중이거나 존재하지 않습니다."})

@app.get("/progress")
async def get_progress():
    with progress_lock:
        return JSONResponse(progress_data)
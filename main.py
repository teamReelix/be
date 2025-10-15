import os
import shutil
import uuid  # 고유한 파일명을 만들기 위해 import
import logging
from fastapi import FastAPI, UploadFile, File, BackgroundTasks, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

# 우리가 만든 하이라이트 생성 모듈 import
import highlight_generator as hg

# --- 로깅 설정 ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# --- 경로 설정 (highlight_generator와 동일하게) ---
# 이 경로들이 highlight_generator.py 내부의 경로 변수들과 연결됩니다.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
hg.CKPT_DIR = os.path.join(BASE_DIR, "ckpts")
hg.OUT_DIR = os.path.join(BASE_DIR, "exports")
VIDEO_DIR = os.path.join(BASE_DIR, "videos")
STATIC_DIR = os.path.join(BASE_DIR, "static")

os.makedirs(hg.CKPT_DIR, exist_ok=True)
os.makedirs(hg.OUT_DIR, exist_ok=True)
os.makedirs(VIDEO_DIR, exist_ok=True)
os.makedirs(STATIC_DIR, exist_ok=True)

# --- 정적 파일 및 템플릿 설정 ---
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
templates = Jinja2Templates(directory=STATIC_DIR)


# --- 모델 로드 ---
# 서버가 시작될 때 모델을 딱 한 번만 불러옵니다.
@app.on_event("startup")
def load_model_on_startup():
    logger.info("서버 시작... 모델을 로드합니다.")
    try:
        hg.model = hg.load_model()
        logger.info("모델 로드 완료.")
    except Exception as e:
        logger.error(f"모델 로드 실패: {e}")
        hg.model = None  # 모델 로드 실패 시 None으로 설정

# --- API 엔드포인트 정의 ---

@app.get("/", response_class=HTMLResponse)
async def read_index(request: Request):
    """
    메인 페이지 (index.html)를 렌더링합니다.
    """
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/upload-video/")
async def upload_video_for_highlight(
        background_tasks: BackgroundTasks,
        video: UploadFile = File(...)
):
    """
    사용자로부터 비디오를 업로드받아 백그라운드에서 하이라이트 생성을 시작합니다.
    """
    if hg.model is None:
        raise HTTPException(status_code=503, detail="모델이 로드되지 않아 서비스를 사용할 수 없습니다.")

    # 1. 고유한 파일명 생성 및 원본 영상 저장
    unique_id = str(uuid.uuid4().hex[:8])
    original_filename = video.filename
    base, ext = os.path.splitext(original_filename)

    # 한글 등 비-아스키 파일명을 안전하게 처리
    safe_base = "".join(c for c in base if c.isalnum() or c in (' ', '_')).rstrip()
    input_filename = f"{safe_base}_{unique_id}{ext}"
    input_path = os.path.join(VIDEO_DIR, input_filename)

    try:
        with open(input_path, "wb") as buffer:
            shutil.copyfileobj(video.file, buffer)
        logger.info(f"영상 저장 완료: {input_path}")
    except Exception as e:
        logger.error(f"영상 저장 실패: {e}")
        raise HTTPException(status_code=500, detail="영상 파일을 저장하는 중 오류가 발생했습니다.")


    # 2. 백그라운드 작업으로 하이라이트 생성 함수 실행
    target_minutes = 5  # 예시: 5분짜리 하이라이트
    background_tasks.add_task(
        hg.export_highlight_from_full_mp4,
        full_mp4_path=input_path,
        target_minutes=target_minutes
    )

    # 3. 사용자에게 즉시 응답 반환
    out_base = os.path.splitext(input_filename)[0]
    result_filename = f"{out_base}_HIGHLIGHT_{target_minutes}m.mp4"

    return JSONResponse(
        status_code=202,  # 202 Accepted: 요청이 접수되었고 처리 중임을 의미
        content={
            "message": "영상 업로드 성공! 하이라이트 생성을 시작합니다.",
            "original_filename": original_filename,
            "result_filename": result_filename,
            "check_status_url": f"/get-highlight/{result_filename}"
        }
    )


@app.get("/get-highlight/{filename}")
async def get_highlight_video(filename: str):
    """
    완성된 하이라이트 영상 파일을 다운로드하거나 처리 상태를 반환합니다.
    """
    file_path = os.path.join(hg.OUT_DIR, filename)

    if os.path.exists(file_path):
        logger.info(f"결과 파일 전송: {filename}")
        return FileResponse(file_path, media_type="video/mp4", filename=filename)
    else:
        logger.warning(f"결과 파일 없음 (아직 처리 중이거나 오류 발생 가능): {filename}")
        return JSONResponse(
            status_code=404,  # 404 Not Found: 파일을 아직 찾을 수 없음
            content={"status": "processing", "message": "파일을 찾을 수 없습니다. 아직 처리 중이거나 파일명에 오류가 있을 수 있습니다."}
        )

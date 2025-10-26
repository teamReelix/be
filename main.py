from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from config import setup_directories, STATIC_DIR, logger
from model import load_model_on_startup
from routers import general, video

# --- 디렉토리 설정 ---
# 앱이 시작되기 전에 필요한 디렉토리가 있는지 확인하고 생성합니다.
setup_directories()

# --- FastAPI 앱 ---
app = FastAPI()

# --- 모델 로드 (시작 이벤트) ---
@app.on_event("startup")
def on_startup():
    load_model_on_startup()

# --- 정적 파일 마운트 ---
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# --- 라우터 포함 ---
# 'routers' 디렉토리에서 가져온 라우터들을 앱에 포함시킵니다.
app.include_router(general.router)
app.include_router(video.router)

logger.info("FastAPI 애플리케이션 설정 완료")

# uvicorn으로 이 파일을 실행합니다: uvicorn main:app --reload
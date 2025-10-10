import os
import shutil
import uuid  # 고유한 파일명을 만들기 위해 import
from fastapi import FastAPI, UploadFile, File, BackgroundTasks, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from starlette.responses import FileResponse
# 우리가 만든 하이라이트 생성 모듈 import
import highlight_generator as hg

app = FastAPI()

# --- 경로 설정 (highlight_generator와 동일하게) ---
# 이 경로들이 highlight_generator.py 내부의 경로 변수들과 연결됩니다.
hg.CKPT_DIR = "./ckpts"
hg.OUT_DIR = "./exports"
VIDEO_DIR = "./videos"

os.makedirs(hg.CKPT_DIR, exist_ok=True)
os.makedirs(hg.OUT_DIR, exist_ok=True)
os.makedirs(VIDEO_DIR, exist_ok=True)

# --- 모델 로드 ---
# 서버가 시작될 때 모델을 딱 한 번만 불러옵니다.
print("서버 시작... 모델을 로드합니다.")
try:
    hg.model = hg.load_model()
    print("모델 로드 완료.")
except Exception as e:
    print(f"모델 로드 실패: {e}")
    hg.model = None  # 모델 로드 실패 시 None으로 설정


# --- API 엔드포인트 정의 ---

@app.post("/upload-video/")
async def upload_video_for_highlight(
        background_tasks: BackgroundTasks,
        video: UploadFile = File(...)
):
    """
    사용자로부터 비디오를 업로드받아 백그라운드에서 하이라이트 생성을 시작합니다.
    """
    if hg.model is None:
        raise HTTPException(status_code=500, detail="모델이 로드되지 않아 서비스를 사용할 수 없습니다.")

    # 1. 고유한 파일명 생성 및 원본 영상 저장
    # 사용자 파일명 중복을 피하기 위해 UUID 사용
    unique_id = str(uuid.uuid4())
    original_filename = video.filename
    base, ext = os.path.splitext(original_filename)

    # 한글 등 비-아스키 파일명을 안전하게 처리
    safe_base = "".join(c for c in base if c.isalnum() or c in (' ', '_')).rstrip()
    input_filename = f"{safe_base}_{unique_id}{ext}"
    input_path = os.path.join(VIDEO_DIR, input_filename)

    with open(input_path, "wb") as buffer:
        shutil.copyfileobj(video.file, buffer)

    # 2. 백그라운드 작업으로 하이라이트 생성 함수 실행
    background_tasks.add_task(
        hg.export_highlight_from_full_mp4,
        full_mp4_path=input_path,
        target_minutes=5  # 예시: 5분짜리 하이라이트
    )

    # 3. 사용자에게 즉시 응답 반환
    # 최종 결과 파일명을 미리 계산해서 알려줌
    out_base = os.path.splitext(input_filename)[0]
    result_filename = f"{out_base}_HIGHLIGHT_5m.mp4"

    return JSONResponse(
        status_code=202,  # 202 Accepted: 요청이 접수되었고 처리 중임을 의미
        content={
            "message": "영상 업로드 성공! 하이라이트 생성을 시작합니다.",
            "original_filename": original_filename,
            "result_filename": result_filename,
            "info": "결과 확인은 /get-highlight/{result_filename} 엔드포인트를 이용하세요."
        }
    )


@app.get("/get-highlight/{filename}")
async def get_highlight_video(filename: str):
    """
    완성된 하이라이트 영상 파일을 다운로드합니다.
    """
    file_path = os.path.join(hg.OUT_DIR, filename)

    if os.path.exists(file_path):
        return FileResponse(file_path, media_type="video/mp4", filename=filename)
    else:
        return JSONResponse(
            status_code=404,  # 404 Not Found: 파일을 아직 찾을 수 없음
            content={"message": "파일을 찾을 수 없습니다. 아직 처리 중이거나 파일명에 오류가 있을 수 있습니다."}
        )

    # "static" 폴더에 있는 파일들을 / 경로로 제공하겠다고 설정
    app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def read_index():
    # 사용자가 루트 경로로 접속하면 static/index.html 파일을 보내줌
    return FileResponse('static/index.html')
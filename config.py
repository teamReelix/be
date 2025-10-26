import os
import logging
import boto3
from dotenv import load_dotenv

# --- 로깅 설정 ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- 경로 설정 ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CKPT_DIR = os.path.join(BASE_DIR, "ckpts")
OUT_DIR = os.path.join(BASE_DIR, "exports")
VIDEO_DIR = os.path.join(BASE_DIR, "videos")
STATIC_DIR = os.path.join(BASE_DIR, "static")

# --- 디렉토리 생성 ---
def setup_directories():
    logger.info("필요한 디렉토리를 생성합니다...")
    os.makedirs(CKPT_DIR, exist_ok=True)
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(VIDEO_DIR, exist_ok=True)
    os.makedirs(STATIC_DIR, exist_ok=True)

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

# 상수
# NUM_FRAMES = 8
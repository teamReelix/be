import highlight_generator as hg
from config import CKPT_DIR, logger

# 로드된 모델을 저장할 전역 변수
model_instance = None

def load_model_on_startup():
    """서버 시작 시 모델을 로드합니다."""
    global model_instance
    logger.info("서버 시작... 모델 로드 중...")
    try:
        hg.CKPT_DIR = CKPT_DIR  # hg 모듈의 경로 설정
        model_instance = hg.load_model()
        hg.model = model_instance  # hg 모듈 내부에서도 참조할 수 있도록 설정
        logger.info("모델 로드 완료.")
    except Exception as e:
        logger.error(f"모델 로드 실패: {e}")
        model_instance = None

def get_model():
    """로드된 모델 인스턴스를 반환합니다."""
    return model_instance
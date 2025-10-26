from model_v1 import highlight_generator as hg
import logging

logger = logging.getLogger(__name__)

def load_model():
    logger.info("모델 로드 중...")
    try:
        hg.model = hg.load_model()
        logger.info("모델 로드 완료")
    except Exception as e:
        logger.error(f"모델 로드 실패: {e}")
        hg.model = None

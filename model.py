from config import CKPT_DIR, logger

# 버전별 로드된 모델을 저장할 전역 변수
_models = {}

def load_model_on_startup():
    """서버 시작 시 model_v1과 model_v2 모델을 모두 로드합니다."""
    from model_v1 import highlight_generator as hg_v1
    from model_v2 import highlight_generator as hg_v2

    logger.info("서버 시작... 모델 로드 중...")

    try:
        hg_v1.CKPT_DIR = CKPT_DIR
        _models["v1"] = hg_v1.load_model()
        hg_v1.model = _models["v1"]
        logger.info("model_v1 로드 완료.")
    except Exception as e:
        logger.error(f"model_v1 로드 실패: {e}")
        _models["v1"] = None

    try:
        hg_v2.CKPT_DIR = CKPT_DIR
        _models["v2"] = hg_v2.load_model()
        hg_v2.model = _models["v2"]
        logger.info("model_v2 로드 완료.")
    except Exception as e:
        logger.error(f"model_v2 로드 실패: {e}")
        _models["v2"] = None


def get_model(version: str = "v1"):
    """요청된 버전에 해당하는 모델 인스턴스를 반환합니다."""
    return _models.get(version)

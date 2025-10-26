from config import s3_client, AWS_BUCKET_NAME, AWS_REGION, logger
from typing import Optional
def upload_to_s3(file_path: str, s3_key: str) -> str:
    """파일을 S3에 업로드하고 public URL 반환"""
    try:
        s3_client.upload_file(file_path, AWS_BUCKET_NAME, s3_key)
        url = f"https://{AWS_BUCKET_NAME}.s3.{AWS_REGION}.amazonaws.com/{s3_key}"
        logger.info(f"S3 업로드 완료: {url}")
        return url
    except Exception as e:
        logger.error(f"S3 업로드 실패 ({s3_key}): {e}")
        raise

def check_and_get_presigned_url(filename: str, expires_in: int = 7200) -> Optional[str]:
    """S3에 파일이 있는지 확인하고 presigned URL을 반환합니다."""
    s3_key = f"exports/{filename}"
    try:
        s3_client.head_object(Bucket=AWS_BUCKET_NAME, Key=s3_key)
        url = s3_client.generate_presigned_url(
            'get_object',
            Params={'Bucket': AWS_BUCKET_NAME, 'Key': s3_key},
            ExpiresIn=expires_in  # 2시간 유효
        )
        logger.info(f"S3 파일 존재 확인: {filename}")
        return url
    except s3_client.exceptions.ClientError:
        logger.warning(f"S3 파일 없음: {filename}")
        return None
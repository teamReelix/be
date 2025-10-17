# 1. 베이스 이미지 선택
FROM python:3.10-slim

# 2. 컨테이너 내 작업 디렉토리 설정
WORKDIR /app

# 3. 의존성 설치 (빌드 속도 최적화를 위해 requirements.txt 먼저 복사)
COPY ./requirements.txt .
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# 4. 프로젝트 소스 코드 전체 복사
COPY . .

# 5. 컨테이너가 8000번 포트를 사용함을 명시
EXPOSE 8000

# 6. 컨테이너 시작 시 실행할 명령어
# --host 0.0.0.0 은 컨테이너 외부에서의 접속을 허용하기 위해 필수입니다.
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
const form = document.getElementById('upload-form');
const videoFileInput = document.getElementById('video-file');
const submitButton = document.getElementById('submit-button');
const statusDiv = document.getElementById('status');
const fileNameSpan = document.getElementById('file-name');
const dropArea = document.getElementById('drop-area');

videoFileInput.addEventListener('change', () => {
    const file = videoFileInput.files[0];
    submitButton.disabled = !file;
    fileNameSpan.textContent = file ? `선택된 파일: ${file.name}` : '선택된 파일 없음';
});

['dragenter', 'dragover', 'dragleave', 'drop'].forEach(ev => dropArea.addEventListener(ev, e => { e.preventDefault(); e.stopPropagation(); }, false));
['dragenter','dragover'].forEach(ev => dropArea.addEventListener(ev, () => dropArea.classList.add('dragover'), false));
['dragleave','drop'].forEach(ev => dropArea.addEventListener(ev, () => dropArea.classList.remove('dragover'), false));
dropArea.addEventListener('drop', e => { videoFileInput.files = e.dataTransfer.files; videoFileInput.dispatchEvent(new Event('change')); }, false);

form.addEventListener('submit', async (e) => {
    e.preventDefault();
    const formData = new FormData();
    formData.append('video', videoFileInput.files[0]);
    formData.append('target_minutes', document.getElementById('target-minutes').value);

    submitButton.disabled = true;
    submitButton.textContent = '업로드 중...';
    statusDiv.innerHTML = '<div class="spinner"></div><p>동영상을 서버로 업로드하는 중입니다...</p>';

    try {
        const response = await fetch('/upload-video/', { method: 'POST', body: formData });
        const result = await response.json();

        if (response.status === 202) {
            statusDiv.innerHTML = `<p>${result.message}<br>결과 파일명: <strong>${result.result_filename}</strong></p>`;
            startProgressPolling();
            pollForResult(result.check_status_url, result.result_filename);
        } else throw new Error(result.detail || '알 수 없는 오류가 발생했습니다.');
    } catch (err) {
        statusDiv.innerHTML = `<p style="color:#ff7675;"><strong>오류:</strong> ${err.message}</p>`;
        resetUI();
    }
});

function resetUI() {
    submitButton.textContent = '하이라이트 생성 시작';
    submitButton.disabled = true;
    fileNameSpan.textContent = '선택된 파일 없음';
}

// --- 진행률 폴링 ---
let progressInterval;
function startProgressPolling() {
    if (progressInterval) clearInterval(progressInterval);

    // 진행률 UI 초기화
    statusDiv.innerHTML = '<div class="spinner"></div><p>하이라이트 장면 추출 중</p>';
    const barContainer = document.createElement('div');
    barContainer.className = 'progress-bar-container';
    const barFill = document.createElement('div');
    barFill.className = 'progress-bar-fill';
    barContainer.appendChild(barFill);
    const progressText = document.createElement('p');
    progressText.style.marginTop = '5px';
    statusDiv.appendChild(barContainer);
    statusDiv.appendChild(progressText);

    progressInterval = setInterval(async () => {
        try {
            const res = await fetch('/progress');
            const data = await res.json();
            const percent = data.total ? (data.done/data.total)*100 : 0;
            barFill.style.width = percent.toFixed(1)+'%';
            progressText.textContent = `진행률: ${percent.toFixed(1)}% (윈도우 ${data.done}/${data.total}, ${data.current_start.toFixed(2)}s 처리 중)`;

            if (data.done >= data.total && data.total > 0) clearInterval(progressInterval);
        } catch(err) {
            console.error('진행률 폴링 오류:', err);
        }
    }, 1000);
}

// --- 결과 확인 폴링 ---
function pollForResult(url, filename) {
    const resultInterval = setInterval(async () => {
        try {
            const res = await fetch(url);
            if (res.status === 200) {
                clearInterval(resultInterval);
                const result = await res.json();
                if (result.status==="done" && result.s3_url) {
                    statusDiv.innerHTML = `<p style="color:#81c784; margin-bottom:15px;"><strong>하이라이트 생성 완료!</strong></p>
                    <a href="${result.s3_url}" download="${filename}" title="다운로드">
                        <button>다운로드</button>
                    </a>`;
                    resetUI();
                }
            }
        } catch(err) { console.error('결과 확인 오류:', err); }
    }, 5000);

}
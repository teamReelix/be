// main.js

const form = document.getElementById('upload-form');
const videoFileInput = document.getElementById('video-file');
const submitButton = document.getElementById('submit-button');
const statusDiv = document.getElementById('status');
const fileNameSpan = document.getElementById('file-name');
const dropArea = document.getElementById('drop-area');
const modelSelect = document.getElementById('model-version');

// 로고 모달 관련
const logoModal = document.getElementById('logo-modal');
const uploadLogoBtn = document.getElementById('upload-logo-btn');
const skipLogoBtn = document.getElementById('skip-logo-btn');
const cancelLogoBtn = document.getElementById('cancel-logo-btn');
const logoFileInput = document.getElementById('logo-file');

// ---------------- 파일 선택 / 드래그앤드롭 ----------------
videoFileInput.addEventListener('change', () => {
    const file = videoFileInput.files[0];
    submitButton.disabled = !file;
    fileNameSpan.textContent = file ? `선택된 파일: ${file.name}` : '선택된 파일 없음';
});

['dragenter','dragover','dragleave','drop'].forEach(ev =>
    dropArea.addEventListener(ev, e => { e.preventDefault(); e.stopPropagation(); })
);
['dragenter','dragover'].forEach(ev =>
    dropArea.addEventListener(ev, () => dropArea.classList.add('dragover'))
);
['dragleave','drop'].forEach(ev =>
    dropArea.addEventListener(ev, () => dropArea.classList.remove('dragover'))
);
dropArea.addEventListener('drop', e => {
    videoFileInput.files = e.dataTransfer.files;
    videoFileInput.dispatchEvent(new Event('change'));
});

// ---------------- 폼 제출 이벤트 ----------------
form.addEventListener('submit', e => {
    e.preventDefault();
    const selectedModel = modelSelect.value;

    if (selectedModel === 'v2') {
        logoModal.style.display = 'flex';
    } else {
        uploadVideo(null);
    }
});

// ---------------- 로고 모달 이벤트 ----------------
// uploadLogoBtn.addEventListener('click', () => {
//     const file = logoFileInput.files[0];
//     if (!file) return alert('로고 파일을 선택해주세요!');
//     logoModal.style.display = 'none';
//     uploadVideo(file);
// });
//
// skipLogoBtn.addEventListener('click', () => {
//     logoModal.style.display = 'none';
//     uploadVideo(null);
// });
//
// cancelLogoBtn.addEventListener('click', () => {
//     logoModal.style.display = 'none';
// });

// ---------------- UI 초기화 ----------------
function resetUI() {
    submitButton.textContent = '하이라이트 생성 시작';
    submitButton.disabled = true;
    fileNameSpan.textContent = '선택된 파일 없음';
}

// ---------------- 진행률 폴링 ----------------
let progressInterval;

function startProgressPolling() {
    if (progressInterval) clearInterval(progressInterval);

    // 진행률 UI 초기화
    statusDiv.innerHTML = `
        <div class="spinner"></div>
        <p>하이라이트 장면 추출 중</p>
        <div class="progress-bar-container">
            <div class="progress-bar-fill"></div>
        </div>
        <p style="margin-top:5px;"></p>
    `;

    const barFill = statusDiv.querySelector('.progress-bar-fill');
    const progressText = statusDiv.querySelectorAll('p')[1];

    progressInterval = setInterval(async () => {
        try {
            const res = await fetch('/progress');
            const data = await res.json();
            const percent = data.total ? (data.done / data.total) * 100 : 0;
            barFill.style.width = percent.toFixed(1) + '%';
            progressText.textContent = `진행률: ${percent.toFixed(1)}% (윈도우 ${data.done}/${data.total})`;
            // progressText.textContent = `진행률: ${percent.toFixed(1)}% (윈도우 ${data.done}/${data.total}, ${data.current_start.toFixed(0)}초 처리 중)`;

            if (data.done >= data.total && data.total > 0) clearInterval(progressInterval);
        } catch (err) {
            console.error('진행률 폴링 오류:', err);
        }
    }, 1000);
}

// ---------------- 영상 업로드 ----------------
async function uploadVideo(logoFile = null) {
    const formData = new FormData();
    formData.append('video', videoFileInput.files[0]);
    formData.append('target_minutes', document.getElementById('target-minutes').value);
    formData.append('model_version', modelSelect.value);
    if (logoFile) formData.append('logo', logoFile);

    submitButton.disabled = true;
    submitButton.textContent = '업로드 중...';

    try {
        const response = await fetch('/upload-video/', { method: 'POST', body: formData });
        const result = await response.json();

        if (response.status === 202) {
            statusDiv.querySelector('p').textContent = `${result.message} 결과 파일명: ${result.result_filename}`;
            const uploadForm = document.getElementById('upload-form');
            if (uploadForm) uploadForm.remove();

            // ---------------- 폴링 시작 ----------------
            startProgressPolling();
            pollForResult(result.check_status_url, result.result_filename);

        } else {
            throw new Error(result.detail || '알 수 없는 오류');
        }
    } catch (err) {
        statusDiv.innerHTML = `<p style="color:#ff7675;"><strong>오류:</strong> ${err.message}</p>`;
        resetUI();
    }
}

// ---------------- 결과 확인 폴링 ----------------
function pollForResult(url, filename) {
    const resultInterval = setInterval(async () => {
        try {
            const res = await fetch(url);
            if (res.status === 200) {
                const result = await res.json();
                if (result.status === 'done' && result.s3_url) {
                    clearInterval(resultInterval);

                    if (progressInterval) clearInterval(progressInterval);

                    // spinner/progress bar 제거
                    const spinner = statusDiv.querySelector('.spinner');
                    if (spinner) spinner.remove();
                    const bar = statusDiv.querySelector('.progress-bar-container');
                    if (bar) bar.remove();
                    statusDiv.querySelector('p').textContent = '하이라이트 생성 완료!';

                    // 결과 영역 표시
                    const resultDiv = document.getElementById('result');
                    resultDiv.style.display = 'block';
                    resultDiv.innerHTML = '';

                    const video = document.createElement('video');
                    video.controls = true;
                    video.style.width = '100%';
                    video.style.marginBottom = '10px';
                    const source = document.createElement('source');
                    source.src = result.s3_url;
                    source.type = 'video/mp4';
                    video.appendChild(source);
                    resultDiv.appendChild(video);

                    const downloadLink = document.createElement('a');
                    downloadLink.href = result.s3_url;
                    downloadLink.download = filename;
                    const downloadButton = document.createElement('button');
                    downloadButton.textContent = '다운로드';
                    downloadLink.appendChild(downloadButton);
                    resultDiv.appendChild(downloadLink);

                    resetUI();
                }
            }
        } catch (err) {
            console.error('결과 확인 오류:', err);
        }
    }, 5000);
}


// 로고 링크 선택
const homeLink = document.querySelector('header a');

// 클릭 시 경고
homeLink.addEventListener('click', (e) => {
    const leave = confirm("홈으로 돌아가면 진행 중인 작업이 사라질 수 있습니다. 계속 이동하시겠습니까?");
    if (!leave) {
        e.preventDefault(); // 사용자가 취소하면 이동 막기
    }
});

// 헤더에서 FAQ 링크 선택
const faqLink = document.querySelector('a[href="/faq"]');

faqLink.addEventListener('click', function(e) {
    const proceed = confirm("현재 페이지를 벗어나면 진행 중인 작업이 사라질 수 있습니다. 계속 이동하시겠습니까?");
    if (!proceed) {
        e.preventDefault(); // 이동 취소
    }
});
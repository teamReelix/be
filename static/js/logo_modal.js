// logo_modal.js

document.addEventListener("DOMContentLoaded", () => {
  // ---------------- 요소 선택 ----------------
  const startBtn = document.getElementById("submit-button");
  const logoModal = document.getElementById("logo-modal");
  const uploadBtn = document.getElementById("upload-logo-btn");
  const skipBtn = document.getElementById("skip-logo-btn");
  const cancelBtn = document.getElementById("cancel-logo-btn");
  const logoInput = document.getElementById("logo-file");
  const logoDropArea = document.getElementById("logo-drop-area");
  const fileNameSpan = document.getElementById("logo-file-name");
  const modelSelect = document.getElementById("model-version");
  const videoFileInput = document.getElementById("video-file");
  const targetMinutesInput = document.getElementById("target-minutes");
  const statusDiv = document.getElementById("status");

  // ---------------- 파일 선택 시 처리 ----------------
  logoInput.addEventListener("change", () => {
    const file = logoInput.files[0];
    fileNameSpan.textContent = file ? `선택된 파일: ${file.name}` : "선택된 파일 없음";
  });

  // ---------------- 드래그앤드롭 이벤트 ----------------
  ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(ev => {
    logoDropArea.addEventListener(ev, e => {
      e.preventDefault();
      e.stopPropagation();
    });
  });

  ['dragenter', 'dragover'].forEach(ev => {
    logoDropArea.addEventListener(ev, () => logoDropArea.classList.add('dragover'));
  });

  ['dragleave', 'drop'].forEach(ev => {
    logoDropArea.addEventListener(ev, () => logoDropArea.classList.remove('dragover'));
  });

  logoDropArea.addEventListener('drop', e => {
    if (e.dataTransfer.files.length) {
      logoInput.files = e.dataTransfer.files;
      logoInput.dispatchEvent(new Event('change')); // change 이벤트 재사용
    }
  });

  // ---------------- 하이라이트 생성 함수 ----------------
  async function startHighlightGeneration(logoFile) {
    const formData = new FormData();
    formData.append("video", videoFileInput.files[0]);
    formData.append("target_minutes", targetMinutesInput.value);
    formData.append("model_version", modelSelect.value);
    if (logoFile) formData.append("logo", logoFile);

    startBtn.disabled = true;
    startBtn.textContent = "업로드 중...";
    statusDiv.innerHTML = '<div class="spinner"></div><p>동영상을 서버로 업로드하는 중입니다...</p>';

    try {
      const res = await fetch("/upload-video/", { method: "POST", body: formData });
      const result = await res.json();

      if (res.status === 202) {
        statusDiv.innerHTML = `<p>${result.message}<br>결과 파일명: <strong>${result.result_filename}</strong></p>`;
        const uploadForm = document.getElementById('upload-form');
        if (uploadForm) uploadForm.remove();

        // 진행률 및 결과 폴링
        startProgressPolling();
        pollForResult(result.check_status_url, result.result_filename);
      } else {
        throw new Error(result.detail || "알 수 없는 오류가 발생했습니다.");
      }
    } catch (err) {
      statusDiv.innerHTML = `<p style="color:#ff7675;"><strong>오류:</strong> ${err.message}</p>`;
      startBtn.disabled = false;
      startBtn.textContent = "하이라이트 생성 시작";
    }
  }

  // ---------------- 제출 버튼 클릭 ----------------
  startBtn.addEventListener("click", (event) => {
    const selectedModel = modelSelect.value;

    if (selectedModel === "v2") {
      event.preventDefault();
      logoModal.style.display = "flex";
    } else {
      startHighlightGeneration(null);
    }
  });

  // ---------------- 로고 선택 후 진행 ----------------
  uploadBtn.addEventListener("click", () => {
    const file = logoInput.files[0];
    if (!file) {
      alert("로고 파일을 선택해주세요!");
      return;
    }
    logoModal.style.display = "none";
    startHighlightGeneration(file);
  });

  // ---------------- 로고 없이 진행 ----------------
  skipBtn.addEventListener("click", () => {
    logoModal.style.display = "none";
    startHighlightGeneration(null);
  });

  // ---------------- 취소 ----------------
  cancelBtn.addEventListener("click", () => {
    logoModal.style.display = "none";
  });
});

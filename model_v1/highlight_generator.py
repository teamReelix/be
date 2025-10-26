import os, re, glob, json, tempfile, subprocess, time
import numpy as np
import torch, torch.nn.functional as F
import torchaudio, librosa
from decord import VideoReader, cpu
import logging

from model_v1.models.av_fusion import AVFusion

from progress_state import progress_data, progress_lock

logger = logging.getLogger(__name__)

# === 경로 ===
CKPT_DIR = "./ckpts"  # 이미 학습된 체크포인트 폴더
OUT_DIR = "./exports"  # 하이라이트 결과물 저장 폴더
os.makedirs(OUT_DIR, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"

# === 전처리 하이퍼파라미터 (학습과 동일) ===
SR = 16000
MEL_BINS = 128
WIN_SEC = 15.0      # 한 윈도우 길이 (권장: 12~20초 사이)
STRIDE_SEC = 5.0    # 슬라이딩 간격
NUM_FRAMES = 8      # 영상 프레임 샘플 수 (학습과 동일)

def load_model(ckpt_dir=CKPT_DIR):
    model = AVFusion().to(device)
    alias = os.path.join(ckpt_dir, "best.pt")
    if os.path.exists(alias):
        ckpt = torch.load(alias, map_location=device, weights_only=False)
        logger.info("Loaded: %s", alias)
    else:
        cand = sorted(glob.glob(os.path.join(ckpt_dir, "best.pt")))
        assert cand, "ckpts 폴더에 가중치가 없습니다."
        def auc_from(p):
            m = re.search(r"auc([0-9.]+)\.pt$", os.path.basename(p))
            return float(m.group(1)) if m else -1.0
        best_path = max(cand, key=auc_from)
        ckpt = torch.load(best_path, map_location=device, weights_only=False)
        logger.info(f"Loaded: {best_path}")

    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state, strict=True)
    model.eval()
    return model

model = load_model()

def _safe_json_dump(obj, path):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False)
    os.replace(tmp, path)

def _safe_json_load(path, default=None):
    if not os.path.exists(path):
        return default
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default

def ffprobe_duration(path):
    try:
        out = subprocess.check_output([
            "ffprobe","-v","error","-show_entries","format=duration",
            "-of","default=nw=1:nk=1", path
        ], text=True)
        v = out.strip()
        return float(v) if v and v!="N/A" else None
    except Exception:
        return None

def load_video_frames(path, num_frames=NUM_FRAMES):
    try:
        vr = VideoReader(path, ctx=cpu(0))
        total = len(vr)
        if total <= 0:
            raise RuntimeError("no frames")
        idx = np.linspace(0, max(0,total-1), num_frames).astype(int)
        frames = vr.get_batch(idx).asnumpy().astype(np.float32)/255.0  # (T,H,W,3)
        frames = np.transpose(frames, (3,0,1,2))  # (3,T,H,W)
        return frames
    except Exception:
        return np.zeros((3, num_frames, 224, 224), dtype=np.float32)

def load_audio_mel(path, sr=SR, duration=WIN_SEC):
    """
    torchaudio.load()를 쓰지 않고
    ffmpeg로 WAV 추출 → librosa로 로드 → torchaudio.transforms로 mel 생성
    """
    need = int(sr * duration)

    # 1) ffmpeg로 WAV 임시 추출 (모노/리샘플/길이 제한)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_name = tmp.name
    cmd = [
        "ffmpeg","-hide_banner","-loglevel","error",
        "-i", path, "-vn",         # 비디오 무시
        "-ac","1","-ar", str(sr),  # 1ch, sr Hz
        "-t", str(duration),
        "-y", tmp_name
    ]
    subprocess.run(cmd, check=True)

    # 2) librosa로 WAV 로드
    y, _ = librosa.load(tmp_name, sr=sr, mono=True, duration=duration)
    try: os.remove(tmp_name)
    except: pass

    wav = torch.from_numpy(y).unsqueeze(0)  # (1, N)

    # 3) 길이 보정
    if wav.shape[1] < need:
        wav = F.pad(wav, (0, need - wav.shape[1]))
    else:
        wav = wav[:, :need]

    # 4) mel 변환 (torchaudio.transforms만 사용 → 경고 없음)
    mel = torchaudio.transforms.MelSpectrogram(
        sample_rate=sr, n_fft=1024, hop_length=320, n_mels=MEL_BINS
    )(wav)
    mel = torch.log(mel + 1e-6)
    return mel

@torch.no_grad()
def score_temp_clip(tmp_path, win_sec):
    # 비디오 8프레임 / 오디오(win_sec 길이)로 멀티모달 입력 생성
    xv = torch.from_numpy(load_video_frames(tmp_path, num_frames=NUM_FRAMES)).unsqueeze(0).to(device)  # (1,3,T,H,W)
    xa = load_audio_mel(tmp_path, duration=win_sec).unsqueeze(0).to(device)                            # (1,1,M,T)
    p  = torch.sigmoid(model(xv, xa)).item()
    return float(p)


def score_window(full_mp4, start_sec, win_sec=WIN_SEC):
    # --- 시간 측정 시작 ---
    t_start = time.time()

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        tmp_name = tmp.name

    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error",
        "-ss", f"{start_sec:.3f}", "-t", f"{win_sec:.3f}",
        "-i", full_mp4,
        "-c:v", "libx264", "-preset", "ultrafast", "-crf", "30",
        "-c:a", "aac", "-b:a", "96k",
        "-y", tmp_name
    ]

    t_ffmpeg_start = time.time()
    try:
        subprocess.run(cmd, check=True)
        t_ffmpeg_end = time.time()

        s = score_temp_clip(tmp_name, win_sec)
        t_model_end = time.time()

        # --- 시간 측정 결과 출력 ---
        logger.info(
            f"[{start_sec:.2f}s] FFMPEG 클립 생성: {t_ffmpeg_end - t_ffmpeg_start:.4f}초 / 모델 예측: {t_model_end - t_ffmpeg_end:.4f}초"
        )

    except subprocess.CalledProcessError:
        s = 0.0
    finally:
        try:
            os.remove(tmp_name)
        except:
            pass

    return s

def smooth_scores(scores, k=3):
    if k<=1: return scores
    out = []
    for i in range(len(scores)):
        L = max(0, i-(k//2))
        R = min(len(scores), i+(k//2)+1)
        out.append(float(np.mean(scores[L:R])))
    return out

def make_candidates(scores, starts, win_sec, nms_gap=5.0, top_k=None):
    """
    scores: 윈도우 점수 배열
    starts: 각 윈도우 시작초
    nms_gap: 선택된 후보와 너무 가까운(초) 다른 후보 제거 간격
    top_k: 상위 k개로 1차 제한(없으면 전체)
    """
    pairs = list(zip(starts, scores))
    pairs.sort(key=lambda x: x[1], reverse=True)
    if top_k is not None:
        pairs = pairs[:top_k]

    selected = []
    for st, sc in pairs:
        if not selected:
            selected.append((st, sc))
            continue
        # NMS: 시작점 간격이 너무 가깝다면 패스
        if min(abs(st - s0) for s0,_ in selected) < nms_gap:
            continue
        selected.append((st, sc))
    # 시작 시각 기준 정렬
    selected.sort(key=lambda x: x[0])
    # (start, end, score)
    cand = [(st, st+win_sec, sc) for st,sc in selected]
    return cand

def pick_segments_under_budget(cands, target_sec, min_gap=2.0,
                               onset_pre_sec=13.0, onset_post_sec=7.0, base_win_sec=15.0):
    """
    점수 높은 순으로 겹치지 않게 선택.
    예산/겹침 판단은 '최종 길이(= pre+post = 20초)' 기준.
    """
    picked=[]; used=0.0
    eff_len = onset_pre_sec + onset_post_sec  # = 20
    cands_sorted = sorted(cands, key=lambda x: x[2], reverse=True)

    def overlaps_expanded(a, b):
        # a,b: (st,ed). 실제 출력은 20초로 확장되므로 창길이 대비 확장량으로 겹침 판정
        base_len_a = a[1]-a[0]
        base_len_b = b[1]-b[0]
        half_ext_a = max(0.0, (eff_len - base_len_a)/2.0)
        half_ext_b = max(0.0, (eff_len - base_len_b)/2.0)
        A = (a[0]-half_ext_a-min_gap, a[1]+half_ext_a+min_gap)
        B = (b[0]-half_ext_b,         b[1]+half_ext_b)
        return not (A[1] <= B[0] or B[1] <= A[0])

    for st, ed, sc in cands_sorted:
        if used + eff_len > target_sec*1.03:
            continue
        bad=False
        for pst, ped, _ in picked:
            if overlaps_expanded((st,ed), (pst,ped)):
                bad=True; break
        if bad:
            continue
        picked.append((st, ed, sc))
        used += eff_len
        if used >= target_sec*0.98:
            break

    picked.sort(key=lambda x: x[0])
    return picked, used

def cut_segment(mp4, start, end, out_path):
    dur = max(0.1, end - start)
    cmd = [
        "ffmpeg","-hide_banner","-loglevel","error",
        "-ss", f"{start:.3f}", "-t", f"{dur:.3f}",
        "-i", mp4,
        "-c:v","libx264","-preset","veryfast","-crf","20",
        "-c:a","aac","-b:a","128k",
        "-y", out_path
    ]
    subprocess.run(cmd, check=True)

def concat_mp4s(paths, out_path):
    # ffmpeg concat demuxer (파일 리스트)
    # ‼️ utf-8 인코딩을 명시하고, delete=False로 설정
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as f:
        list_path = f.name
        for p in paths:
            # ‼️ 중요: 상대 경로를 절대 경로로 변환하여 기록
            abs_path = os.path.abspath(p)
            f.write(f"file '{abs_path}'\n")

    cmd = [
        "ffmpeg","-hide_banner","-loglevel","error",
        "-f","concat","-safe","0","-i", list_path,
        "-c","copy",  # 동일 코덱이면 copy가 빠름
        "-y", out_path
    ]
    # copy로 실패 시 재인코딩 백업 전략
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError:
        cmd2 = [
            "ffmpeg","-hide_banner","-loglevel","error",
            "-f","concat","-safe","0","-i", list_path,
            "-c:v","libx264","-preset","veryfast","-crf","20",
            "-c:a","aac","-b:a","128k",
            "-y", out_path
        ]
        subprocess.run(cmd2, check=True)
    finally:
        # with 블록이 끝나도 파일이 즉시 삭제되지 않으므로 수동으로 삭제
        try: os.remove(list_path)
        except: pass


# ----------------------------
# ✅ 오디오 온셋 기반 리센터링 (13초 + 7초)
# ----------------------------
def _extract_audio_chunk(full_mp4, start_sec, duration_sec, sr=SR):
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_name = tmp.name
    cmd = [
        "ffmpeg","-hide_banner","-loglevel","error",
        "-ss", f"{start_sec:.3f}", "-t", f"{duration_sec:.3f}",
        "-i", full_mp4, "-vn",
        "-ac","1","-ar",str(sr), "-y", tmp_name
    ]
    subprocess.run(cmd, check=True)
    y, _ = librosa.load(tmp_name, sr=sr, mono=True)
    try: os.remove(tmp_name)
    except: pass
    return y, sr

def _find_audio_onset_sec(y, sr, hop=512, frame_length=2048):
    rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop, center=True)[0]
    if len(rms) < 4:
        return None
    ker = 5
    rms_smooth = np.convolve(rms, np.ones(ker)/ker, mode="same")
    thr = np.median(rms_smooth) + 1.5*np.std(rms_smooth)
    diff = np.diff(rms_smooth, prepend=rms_smooth[:1])
    idxs = np.where((rms_smooth >= thr) & (diff > 0))[0]
    if len(idxs)==0:
        return None
    onset_frames = int(idxs[0])
    onset_sec = onset_frames * hop / sr
    return onset_sec



def _recenter_cut_times_by_audio(full_mp4, st, ed, duration,
                                 onset_pre_sec=13.0, onset_post_sec=7.0,
                                 search_back=10.0, search_fwd=8.0):
    search_start = max(0.0, st - search_back)
    search_end   = min(duration, ed + search_fwd)
    search_dur   = max(0.1, search_end - search_start)

    try:
        y, sr = _extract_audio_chunk(full_mp4, search_start, search_dur)
        onset_local = _find_audio_onset_sec(y, sr)
        if onset_local is None:
            raise RuntimeError("onset not found")
        onset_abs = search_start + onset_local
    except Exception:
        onset_abs = 0.5*(st+ed)  # 실패 시 창 중앙

    start_cut = onset_abs - onset_pre_sec
    end_cut   = onset_abs + onset_post_sec

    # 경계 보정
    if start_cut < 0:
        shift = -start_cut
        start_cut = 0.0
        end_cut = min(duration, end_cut + shift)
    if end_cut > duration:
        shift = end_cut - duration
        end_cut = duration
        start_cut = max(0.0, start_cut - shift)

    need = onset_pre_sec + onset_post_sec
    if end_cut - start_cut < need:
        lack = need - (end_cut - start_cut)
        start_cut = max(0.0, start_cut - lack/2)
        end_cut   = min(duration, end_cut + lack/2)

    return float(start_cut), float(end_cut)

def export_highlight_from_full_mp4(full_mp4_path, target_minutes=8,
                                   win_sec=15.0, stride_sec=7.0,
                                   smooth_k=3, nms_gap=12.0, top_k=None,
                                   min_gap_between=2.0,
                                   align_to_audio_onset=True,
                                   onset_pre_sec=13.0,  # ← 13초
                                   onset_post_sec=7.0,  # ← 7초 (총 20초)
                                   search_back_sec=10.0,
                                   search_fwd_sec=8.0):
    os.makedirs(OUT_DIR, exist_ok=True)
    base = os.path.splitext(os.path.basename(full_mp4_path))[0]
    out_mp4 = os.path.join(OUT_DIR, f"{base}_HIGHLIGHT_{int(target_minutes)}m.mp4")

    duration = ffprobe_duration(full_mp4_path)
    if duration is None or duration <= 0:
        raise RuntimeError(f"duration 읽기 실패. 경로/코덱/마운트 상태를 확인하세요.\n{full_mp4_path}")

    # 1) 시작점
    starts=[]; t=0.0
    while t + win_sec <= duration:
        starts.append(round(t, 3))
        t += stride_sec
    if not starts:
        raise RuntimeError("영상 길이가 너무 짧아 윈도우가 생성되지 않았습니다.")
    index_map = {st:i for i,st in enumerate(starts)}

    # 2) 스코어 캐시(Resume)
    job_prefix = os.path.join(OUT_DIR, f"{base}_w{int(win_sec)}_s{int(stride_sec)}")
    scores_path = job_prefix + "_scores.json"
    score_cache = _safe_json_load(scores_path, default={})
    scores = [None]*len(starts)
    todo = []
    for st in starts:
        key = f"{st:.3f}"
        if key in score_cache:
            scores[index_map[st]] = float(score_cache[key])
        else:
            todo.append(st)

    logger.info(f"scoring windows... total={len(starts)}  todo={len(todo)} (win={win_sec}s, stride={stride_sec}s)")
    for i, st in enumerate(todo, 1):
        s = score_window(full_mp4_path, st, win_sec)
        key = f"{st:.3f}"
        score_cache[key] = float(s)
        scores[index_map[st]] = float(s)

        # 진행률 로그
        progress = (i / len(todo)) * 100
        logger.info(f"[{st:.2f}s] scored {i}/{len(todo)} ({progress:.1f}%)")

        # --- 전역 진행률 갱신 ---
        with progress_lock:
            progress_data["done"] = i
            progress_data["total"] = len(todo)
            progress_data["current_start"] = st

        if (i % 20)==0 or i==len(todo):
            _safe_json_dump(score_cache, scores_path)
            logger.info(f"  - scored {i}/{len(todo)} (checkpoint saved)")

    # 3) 스무딩 → 후보 → 20초 기준 선택
    s_smooth = smooth_scores(scores, k=smooth_k)
    cands = make_candidates(s_smooth, starts, win_sec, nms_gap=nms_gap, top_k=top_k)
    target_sec = target_minutes * 60.0
    picked, used = pick_segments_under_budget(
        cands, target_sec, min_gap=min_gap_between,
        onset_pre_sec=onset_pre_sec, onset_post_sec=onset_post_sec, base_win_sec=win_sec
    )
    logger.info(
        f"후보 {len(cands)}개 → 최종 {len(picked)}개, 총 {used:.1f}s (총 {onset_pre_sec + onset_post_sec:.0f}s 기준)"
    )

    if not picked:
        raise RuntimeError("선택된 세그먼트가 없습니다. 파라미터를 조정해보세요.")

    # 4) 컷(Resume: 이미 있으면 스킵)
    tmp_parts=[]; cuts_meta=[]
    for k, (st, ed, sc) in enumerate(picked, 1):
        if align_to_audio_onset:
            st_cut, ed_cut = _recenter_cut_times_by_audio(
                full_mp4_path, st, ed, duration,
                onset_pre_sec=onset_pre_sec, onset_post_sec=onset_post_sec,
                search_back=search_back_sec, search_fwd=search_fwd_sec
            )
        else:
            center = 0.5*(st+ed)
            st_cut = max(0.0, center - onset_pre_sec)
            ed_cut = min(duration, center + onset_post_sec)
            need = onset_pre_sec + onset_post_sec
            if ed_cut - st_cut < need:
                lack = need - (ed_cut - st_cut)
                st_cut = max(0.0, st_cut - lack/2)
                ed_cut = min(duration, ed_cut + lack/2)

        part = os.path.join(OUT_DIR, f"{base}_part{k:03d}.mp4")
        if os.path.exists(part) and os.path.getsize(part) > 0:
            logger.info(f"  • skip cut (exists): {os.path.basename(part)}")
        else:
            cut_segment(full_mp4_path, st_cut, ed_cut, part)
        tmp_parts.append(part)
        cuts_meta.append({"k":k, "st":st, "ed":ed, "score":sc, "st_cut":st_cut, "ed_cut":ed_cut})

    # 5) concat (Resume)
    if os.path.exists(out_mp4) and os.path.getsize(out_mp4) > 0:
        logger.info(f"skip concat (exists): {out_mp4}")
    else:
        concat_mp4s(tmp_parts, out_mp4)

    # 6) 메타 저장
    picked_meta_path = job_prefix + f"_target{int(target_minutes)}m_picked.json"
    meta = {
        "full": full_mp4_path,
        "win_sec": win_sec, "stride_sec": stride_sec,
        "target_min": target_minutes,
        "onset_pre_sec": onset_pre_sec, "onset_post_sec": onset_post_sec,
        "picked": cuts_meta,
        "parts": [os.path.basename(p) for p in tmp_parts],
        "out": out_mp4
    }
    _safe_json_dump(meta, picked_meta_path)

    logger.info(f"DONE: {out_mp4}")
    return out_mp4, picked, list(zip(starts, scores)), s_smooth

# 이 스크립트가 직접 실행될 때만 아래 테스트 코드가 동작하도록 수정합니다.
if __name__ == "__main__":
    # 이 블록 안의 코드는 `import highlight_generator`로는 실행되지 않습니다.
    # 스크립트를 직접 테스트하고 싶을 때 `python highlight_generator.py`로 실행하면 됩니다.
    FULL_MP4 = "./test_videos/2017-01-14_-_18-15_Barcelona_5_-_0_Las_Palmas.mkv"

    # 테스트 비디오 파일이 없으면 에러를 발생시키지 않도록 존재 여부 확인
    if os.path.exists(FULL_MP4):
        logger.debug("--- Running Test ---")
        out_path, picked_segments, raw_scores, smooth_scores_arr = export_highlight_from_full_mp4(
            full_mp4_path=FULL_MP4,
            target_minutes=6,      # 6분
            win_sec=15.0,          # 점수용 창
            stride_sec=10.0,
            smooth_k=3,
            nms_gap=14.0,
            top_k=None,
            min_gap_between=2.0,
            align_to_audio_onset=True,
            onset_pre_sec=15.0,    # onset 이전 13초
            onset_post_sec=10.0,    # onset 이후 7초 (총 20초)
            search_back_sec=10.0,
            search_fwd_sec=10.0
        )
        logger.debug("DONE:", out_path)
    else:
        logger.debug(f"--- Test Skipped: Video file not found at {FULL_MP4} ---")

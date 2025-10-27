import glob
import os, json, shlex, tempfile, subprocess, pathlib, math, random, csv
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio, librosa, cv2
from decord import VideoReader, cpu
import logging

logger = logging.getLogger(__name__)

CKPT_DIR   = "./ckpts"
ASSETS_DIR = "./assets"
OUT_DIR    = "./exports"

device = "cuda" if torch.cuda.is_available() else "cpu"

# 전처리 하이퍼(학습과 동일)
SR         = 16000
MEL_BINS   = 128
WIN_SEC    = 15.0
STRIDE_SEC = 10.0
NUM_FRAMES = 12

def load_model(ckpt_dir=CKPT_DIR):
    """
    모델 2용 가중치 로드 함수
    ckpts 디렉토리 안에서 best_logo_1023.pt 파일을 불러옴.
    """
    model = AVFusionMTL().to(device)
    ckpt_path = os.path.join(ckpt_dir, "best_logo_1023.pt")

    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        logger.info(f"Loaded: {ckpt_path}")
    else:
        cand = sorted(glob.glob(os.path.join(ckpt_dir, "*.pt")))
        assert cand, "ckpts 폴더에 가중치 파일이 없습니다."
        ckpt_path = cand[-1]
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        logger.warning(f"best_logo_1023.pt을 찾을 수 없어, 대신 {ckpt_path} 로드함.")

    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


'''ffprobe/ffmpeg 유틸'''
def ffprobe_duration(path: str):
    try:
        out = subprocess.check_output(
            ["ffprobe","-v","error","-show_entries","format=duration","-of","default=nw=1:nk=1", path],
            text=True
        ).strip()
        return float(out) if out and out!="N/A" else None
    except Exception:
        return None

def make_proxy_mp4(in_path: str, force=False, height=540, fps=None, crf=30, preset="ultrafast"):
    p = pathlib.Path(in_path)
    proxy = p.with_name(p.stem + f"_proxy{height}.mp4")
    if (not force) and proxy.exists() and proxy.stat().st_size>0:
        return str(proxy)
    vf = [f"scale=-2:{height}"]
    if fps: vf.append(f"fps={fps}")
    cmd = [
        "ffmpeg","-y","-hide_banner","-loglevel","error",
        "-i", in_path, "-an",
        "-c:v","libx264","-preset", preset, "-crf", str(crf), "-pix_fmt","yuv420p",
        "-vf", ",".join(vf),
        str(proxy)
    ]
    subprocess.run(cmd, check=True)
    return str(proxy)

def cut_segment(full_mp4, start, end, out_path):
    dur = max(0.1, end-start)
    cmd = [
        "ffmpeg","-hide_banner","-loglevel","error",
        "-ss", f"{start:.3f}", "-t", f"{dur:.3f}",
        "-i", full_mp4,
        "-c:v","libx264","-preset","veryfast","-crf","20",
        "-c:a","aac","-b:a","128k","-y", out_path
    ]
    subprocess.run(cmd, check=True)

def concat_mp4s(paths: List[str], out_path: str):
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        lst = f.name
        for p in paths:
            # 경로를 절대 경로로 변환
            abs_p = os.path.abspath(p)
            # (Windows 호환성을 위해) 역슬래시를 슬래시로 변경해주는 것이 더 안전합니다.
            abs_p_ffmpeg = abs_p.replace("\\", "/")
            f.write(f"file '{abs_p_ffmpeg}'\n")
    try:
        cmd = ["ffmpeg","-hide_banner","-loglevel","error","-f","concat","-safe","0","-i", lst,"-c","copy","-y", out_path]
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError:
        cmd = ["ffmpeg","-hide_banner","-loglevel","error","-f","concat","-safe","0","-i", lst,
               "-c:v","libx264","-preset","veryfast","-crf","20","-c:a","aac","-b:a","128k","-y", out_path]
        subprocess.run(cmd, check=True)
    finally:
        try: os.remove(lst)
        except: pass

'''입력 로딩 (비디오/오디오)'''
def load_video_frames(path: str, num_frames=NUM_FRAMES):
    try:
        vr = VideoReader(path, ctx=cpu(0))
        total = len(vr)
        if total<=0: raise RuntimeError("no frames")
        idx = np.linspace(0, max(0,total-1), num_frames).astype(int)
        frames = vr.get_batch(idx).asnumpy().astype(np.float32)/255.0
        frames = np.transpose(frames, (3,0,1,2))

        C,T,H,W = frames.shape
        if H<224 or W<224:
            pad_h = max(0, 224-H); pad_w = max(0, 224-W)
            frames = np.pad(frames, ((0,0),(0,0),(0,pad_h),(0,pad_w)), mode="edge")
        return frames[:, :, :224, :224]
    except Exception:

        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            return np.zeros((3,num_frames,224,224), np.float32)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if total<=0:
            return np.zeros((3,num_frames,224,224), np.float32)
        idxs = np.linspace(0, total-1, num_frames).astype(int)
        ims=[]
        for i in idxs:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
            ok, fr = cap.read()
            if not ok:
                ims.append(np.zeros((224,224,3), np.float32))
            else:
                fr = cv2.resize(fr, (224,224))
                ims.append(fr[...,::-1]/255.0)
        cap.release()
        arr = np.stack(ims, 0).astype(np.float32)
        arr = np.transpose(arr, (3,0,1,2))
        return arr

def load_audio_mel_ffmpeg(path: str, start_sec: float, duration: float, sr=SR):
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_name = tmp.name
    cmd = [
        "ffmpeg","-hide_banner","-loglevel","error",
        "-ss", f"{start_sec:.3f}", "-t", f"{duration:.3f}",
        "-i", path, "-vn","-ac","1","-ar",str(sr), "-y", tmp_name
    ]
    try:
        subprocess.run(cmd, check=True)
        y, _ = librosa.load(tmp_name, sr=sr, mono=True, duration=duration)
    finally:
        try: os.remove(tmp_name)
        except: pass
    wav = torch.from_numpy(y).unsqueeze(0)
    need = int(sr*duration)
    if wav.shape[1] < need:
        wav = F.pad(wav, (0, need-wav.shape[1]))
    else:
        wav = wav[:, :need]
    mel = torchaudio.transforms.MelSpectrogram(
        sample_rate=sr, n_fft=1024, hop_length=320, n_mels=MEL_BINS
    )(wav)
    return torch.log(mel + 1e-6)

'''체크포인트 로드'''
ALL_EVENT_LABELS = sorted(list({
    "Goal","Penalty","Red Card","Yellow -> Red Card","Shots on target",
    "Shots off target","Foul","Offside","Throw-in","Ball out of play"
}))

class SmallAudioCNN(nn.Module):
    def __init__(self, in_ch=1, emb=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, 1, 1),    nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, 1, 1),   nn.ReLU(), nn.AdaptiveAvgPool2d((1,1)),
        )
        self.fc = nn.Linear(128, emb)
    def forward(self, x):
        return self.fc(self.net(x).flatten(1))

class SmallVideoCNN(nn.Module):
    def __init__(self, emb=256):
        super().__init__()
        in_ch = 3*NUM_FRAMES
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 64, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, 1, 1),   nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, 1, 1),  nn.ReLU(), nn.AdaptiveAvgPool2d((1,1)),
        )
        self.fc = nn.Linear(256, emb)
    def forward(self, x):
        B,C,T,H,W = x.shape
        x = x.permute(0,2,1,3,4).contiguous().view(B, C*T, H, W)
        return self.fc(self.net(x).flatten(1))

class AVFusionMTL(nn.Module):
    def __init__(self, vemb=256, aemb=128, n_events=len(ALL_EVENT_LABELS)):
        super().__init__()
        self.v = SmallVideoCNN(vemb)
        self.a = SmallAudioCNN(1, aemb)
        self.fuse = nn.Sequential(nn.Linear(vemb+aemb, 256), nn.ReLU())
        self.head_main   = nn.Linear(256, 1)          # highlight
        self.head_replay = nn.Linear(256, 1)          # replay
        self.head_event  = nn.Linear(256, n_events)   # multi-label
    def forward(self, xv, xa):
        v = self.v(xv); a = self.a(xa)
        z = self.fuse(torch.cat([v,a], dim=1))
        return self.head_main(z).squeeze(1), self.head_replay(z).squeeze(1), self.head_event(z)


model = AVFusionMTL().to(device)
ckpt_path = os.path.join(CKPT_DIR, "best_logo_1023.pt")
assert os.path.exists(ckpt_path), f"best_logo_1023.pt not found in {CKPT_DIR}"
ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
sd = ckpt["model"] if (isinstance(ckpt, dict) and "model" in ckpt) else ckpt
missing, unexpected = model.load_state_dict(sd, strict=True)
model.eval()
logger.info(f"weights loaded on {device}")

'''윈도우 스코어링'''
@torch.no_grad()
def score_window_proxy(full_mp4, proxy_mp4, start_sec, win_sec, model, device):

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmpx:
        tmpx_name = tmpx.name
    try:
        cmd_v = [
            "ffmpeg","-hide_banner","-loglevel","error",
            "-ss", f"{start_sec:.3f}", "-t", f"{win_sec:.3f}",
            "-i", proxy_mp4, "-c:v","libx264","-preset","ultrafast","-crf","30","-an","-y", tmpx_name
        ]
        subprocess.run(cmd_v, check=True)

        xv = torch.from_numpy(load_video_frames(tmpx_name, num_frames=NUM_FRAMES)).unsqueeze(0).to(device)
        xa = load_audio_mel_ffmpeg(full_mp4, start_sec, win_sec).unsqueeze(0).to(device)
        logit_main, _, _ = model(xv, xa)
        p = torch.sigmoid(logit_main).item()
    except subprocess.CalledProcessError:
        p = 0.0
    finally:
        try: os.remove(tmpx_name)
        except: pass
    return float(p)

def smooth_scores(scores, k=3):
    if k<=1: return scores
    out=[]
    for i in range(len(scores)):
        L=max(0,i-k//2); R=min(len(scores), i+k//2+1)
        out.append(float(np.mean(scores[L:R])))
    return out

def make_candidates(scores, starts, win_sec, nms_gap=12.0, top_k=None):
    pairs = list(zip(starts, scores))
    pairs.sort(key=lambda x: x[1], reverse=True)
    if top_k is not None: pairs = pairs[:top_k]
    selected=[]
    for st,sc in pairs:
        if not selected:
            selected.append((st,sc)); continue
        if min(abs(st-s0) for s0,_ in selected) < nms_gap:
            continue
        selected.append((st,sc))
    selected.sort(key=lambda x: x[0])
    return [(st, st+win_sec, sc) for st,sc in selected]

def pick_segments_under_budget(cands, target_sec, min_gap=2.0, eff_len=None):
    if eff_len is None: eff_len = cands[0][1] - cands[0][0]
    picked, used = [], 0.0
    cands_sorted = sorted(cands, key=lambda x: x[2], reverse=True)
    def overlaps(a, b): return not (a[1] <= b[0] or b[1] <= a[0])
    for st, ed, sc in cands_sorted:
        if used + eff_len > target_sec * 1.03: continue
        bad = any(overlaps((st-min_gap, ed+min_gap), (pst,ped)) for pst,ped,_ in picked)
        if bad: continue
        picked.append((st, ed, sc)); used += eff_len
        if used >= target_sec * 0.98: break
    picked.sort(key=lambda x: x[0])
    return picked, used

'''오디오 온셋 정렬'''
def recenter_by_audio_onset(full_mp4_path, st, ed, duration,
                            onset_pre_sec=15.0, onset_post_sec=10.0,
                            search_back=10.0, search_fwd=10.0):
    center = 0.5*(st+ed)
    s0 = max(0.0, center - search_back)
    e0 = min(duration, center + search_fwd)
    span = max(0.1, e0 - s0)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_name = tmp.name
    cmd = [
        "ffmpeg","-hide_banner","-loglevel","error",
        "-ss", f"{s0:.3f}", "-t", f"{span:.3f}",
        "-i", full_mp4_path, "-vn","-ac","1","-ar", str(SR), "-y", tmp_name
    ]
    subprocess.run(cmd, check=True)
    y, sr = librosa.load(tmp_name, sr=SR, mono=True)
    try: os.remove(tmp_name)
    except: pass
    S = np.abs(librosa.stft(y, n_fft=1024, hop_length=320))
    oenv = librosa.onset.onset_strength(S=S, sr=sr, hop_length=320)
    onsets = librosa.onset.onset_detect(onset_envelope=oenv, sr=sr, hop_length=320, units="time")
    onset_abs = (s0 + float(onsets[0])) if onsets.size>0 else center
    st_cut = max(0.0, onset_abs - onset_pre_sec)
    ed_cut = min(duration, onset_abs + onset_post_sec)
    need = onset_pre_sec + onset_post_sec
    if (ed_cut - st_cut) < need:
        lack = need - (ed_cut - st_cut)
        st_cut = max(0.0, st_cut - lack/2)
        ed_cut = min(duration, ed_cut + lack/2)
    return float(st_cut), float(ed_cut)

'''로고 탐지, 클립 내부 트리밍'''
def _ms__load_gray(p):
    img = cv2.imread(p, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(p)
    if img.ndim == 3 and img.shape[2] == 4:
        b, g, r, a = cv2.split(img)
        img = cv2.merge([b, g, r])
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def _ms__prep(gray):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    g = clahe.apply(gray)
    e = cv2.Canny(g, 80, 160)
    return g, e


def _ms__match_best(crop_g, crop_e, tpl_g, tpl_e):
    best = 0.0
    for (A, B) in [(crop_g, tpl_g), (crop_e, tpl_e)]:
        if A.shape[0] < B.shape[0] or A.shape[1] < B.shape[1]:
            continue
        v = float(cv2.matchTemplate(A, B, cv2.TM_CCOEFF_NORMED).max())
        if v > best:
            best = v
    return best


def scan_logo_range_multiscale(
    mp4_path,
    t_start,
    t_end,
    templates,
    region="full",
    step=0.25,
    thr=0.50,
    min_hold=0.10,
    sigma=3,
    scales=None,
    out_dir=None,
    topk_debug=8,
):
    if scales is None:
        scales = [0.50, 0.60, 0.75, 0.90, 1.00, 1.15, 1.30, 1.50, 1.70]

    tpls = [_ms__load_gray(p) for p in (templates or [])]
    if not tpls:
        return [], [], []

    tbank = []
    for tpl in tpls:
        tg, te = _ms__prep(tpl)
        for s in scales:
            ts = cv2.resize(
                tg, None, fx=s, fy=s,
                interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC
            )
            es = cv2.resize(
                te, None, fx=s, fy=s,
                interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC
            )
            tbank.append((ts, es))

    cap = cv2.VideoCapture(mp4_path)
    if not cap.isOpened():
        return [], [], []

    def grab(t):
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
        ok, fr = cap.read()
        return ok, fr

    times = []
    scores_raw = []
    t = float(max(0.0, t_start))
    t_end = float(max(t, t_end))

    while t <= t_end:
        ok, frame = grab(t)
        if not ok:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        crop = gray if region == "full" else gray
        cg, ce = _ms__prep(crop)

        best = 0.0
        for tg, te in tbank:
            v = _ms__match_best(cg, ce, tg, te)
            if v > best:
                best = v

        times.append(t)
        scores_raw.append(float(best))
        t += step

    if out_dir:
        try:
            os.makedirs(out_dir, exist_ok=True)
        except:
            pass

        idxs = np.argsort(scores_raw)[::-1][:topk_debug]
        cap2 = cv2.VideoCapture(mp4_path)
        for rank, i in enumerate(idxs, 1):
            tt = times[i]
            cap2.set(cv2.CAP_PROP_POS_MSEC, tt * 1000.0)
            ok, fr = cap2.read()
            if ok:
                cv2.imwrite(os.path.join(out_dir, f"peak_{rank:02d}_{tt:.2f}s.jpg"), fr)
        cap2.release()


    scores = scores_raw[:]
    if sigma and sigma > 1 and sigma % 2 == 1:
        r = sigma // 2
        scores = [
            float(np.mean(scores_raw[max(0, i - r):min(len(scores_raw), i + r + 1)]))
            for i in range(len(scores_raw))
        ]


    mask = [1 if (s_raw >= thr or s_s >= thr) else 0 for s_raw, s_s in zip(scores_raw, scores)]

    spans = []
    i = 0
    while i < len(mask):
        if mask[i] == 0:
            i += 1
            continue
        j = i + 1
        while j < len(mask) and mask[j] == 1:
            j += 1
        st = times[i]
        ed = times[j - 1] + step
        if ed - st >= min_hold:
            spans.append((round(st, 2), round(ed, 2)))
        i = j

    cap.release()
    return times, scores_raw, spans


def trim_by_logos_inside_clip_ms(
    proxy_mp4,
    st_cut,
    ed_cut,
    duration,
    logo_templates,
    step=0.16,
    thr=0.50,
    min_hold=0.10,
    sigma=3,
    start_pad=0.18,
    end_pad=0.00,
    lead_window=15.0,
    lead_prelook=10.0,
    lead_end_tol=8.0,
    tail_window=15.0,
):
    st_new, ed_new = float(st_cut), float(ed_cut)


    t0 = max(0.0, st_cut - float(lead_prelook))
    t1 = min(duration, st_cut + max(2.0, lead_window))
    _, _, spans = scan_logo_range_multiscale(
        proxy_mp4, t0, t1, logo_templates, region="full",
        step=step, thr=thr, min_hold=min_hold, sigma=sigma
    )

    cand_end = None
    best_key = None
    if spans:
        for a, b in spans:
            if a <= st_cut <= b:
                key = (0, abs(b - st_cut))
            elif 0.0 <= (st_cut - b) <= lead_end_tol:
                key = (0.5, (st_cut - b))
            elif st_cut <= a <= (st_cut + lead_window + 0.3):
                key = (1, a - st_cut)
            else:
                continue
            if (best_key is None) or (key < best_key):
                best_key = key
                cand_end = b

    if cand_end is not None:
        st_new = min(cand_end + start_pad, ed_new - 0.2)


    t0 = max(0.0, ed_cut - max(2.0, tail_window))
    t1 = min(duration, ed_cut + 2.0)
    times_t, scores_t, spans_t = scan_logo_range_multiscale(
        proxy_mp4, t0, t1, logo_templates, region="full",
        step=step, thr=thr, min_hold=min_hold, sigma=sigma
    )

    # 디버그
    logger.info(
        f"[DEBUG] window=({t0:.2f},{t1:.2f}) "
        f"max_score={max(scores_t or [0]):.3f} spans={spans_t[:3]}"
    )

    cand_start = None
    if spans_t:
        for a, b in spans_t:
            if st_new <= a <= ed_cut + 0.1:
                cand_start = a
                break

    if cand_start is not None:
        ed_new = max(st_new + 0.2, cand_start - end_pad)

    st_new = max(0.0, st_new)
    ed_new = min(duration, ed_new)
    return st_new, ed_new


def merge_intervals(cuts, join_gap=0.5):
    """
    cuts: 리스트[ { "k":int, "st_cut":float, "ed_cut":float, "score":float, ... } ]
    join_gap: 두 구간 사이 간격이 이 값 이하이면 하나로 합침(초)
    return: 병합된 리스트 (동일 형식, score 등은 max로 유지)
    """
    if not cuts:
        return []

    cuts_sorted = sorted(cuts, key=lambda x: x["st_cut"])
    merged = [cuts_sorted[0].copy()]
    merged[0]["sources"] = [cuts_sorted[0]["k"]]

    for c in cuts_sorted[1:]:
        cur = merged[-1]
        if c["st_cut"] <= cur["ed_cut"] + join_gap:
            cur["ed_cut"] = max(cur["ed_cut"], c["ed_cut"])
            cur["duration"] = cur["ed_cut"] - cur["st_cut"]
            cur["score"] = max(cur.get("score", 0.0), c.get("score", 0.0))  # 보수적으로 최대점
            cur.setdefault("sources", []).append(c["k"])
        else:
            nxt = c.copy()
            nxt["sources"] = [c["k"]]
            merged.append(nxt)

    return merged

'''하이라이트 내보내기'''
def export_highlight_from_full_mp4(
    full_mp4_path,
    target_minutes=8,
    win_sec=WIN_SEC,
    stride_sec=STRIDE_SEC,
    smooth_k=3,
    nms_gap=12.0,
    top_k=None,
    min_gap_between=2.0,
    align_to_audio_onset=True,
    onset_pre_sec=20.0,
    onset_post_sec=20.0,
    logo_templates=None,
    proxy_height=540,
):
    base = pathlib.Path(full_mp4_path).stem
    out_mp4 = os.path.join(OUT_DIR, f"{base}_HIGHLIGHT_{int(target_minutes)}m.mp4")

    duration = ffprobe_duration(full_mp4_path)
    assert duration and duration > 0, "duration 읽기 실패"

    proxy_mp4 = make_proxy_mp4(
        full_mp4_path, height=proxy_height, fps=None, crf=30, preset="ultrafast"
    )
    logger.info(f"[proxy] {proxy_mp4}")


    starts = []
    t = 0.0
    while t + win_sec <= duration:
        starts.append(round(t, 3))
        t += stride_sec
    assert starts, "윈도우 없음"


    job_prefix = os.path.join(OUT_DIR, f"{base}_w{int(win_sec)}_s{int(stride_sec)}")
    score_path = job_prefix + "_scores.json"

    def _safe_json_load(path, default=None):
        try:
            with open(path, "r") as f:
                return json.load(f)
        except:
            return default

    def _safe_json_dump(obj, path):
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)
        os.replace(tmp, path)

    cache = _safe_json_load(score_path, {})
    scores = [None] * len(starts)
    todo = []
    for i, st in enumerate(starts):
        key = f"{st:.3f}"
        if key in cache:
            scores[i] = float(cache[key])
        else:
            todo.append((i, st))

    logger.info(f"[INFO] scoring: total={len(starts)} todo={len(todo)} (win={win_sec}s, stride={stride_sec}s)")
    for k, (i, st) in enumerate(todo, 1):
        s = score_window_proxy(full_mp4_path, proxy_mp4, st, win_sec, model, device)
        scores[i] = float(s)
        cache[f"{st:.3f}"] = float(s)
        if (k % 20) == 0 or k == len(todo):
            _safe_json_dump(cache, score_path)
            logger.info(f" - scored {k}/{len(todo)} (checkpoint saved)")

    # 후보 선택
    s_smooth = smooth_scores(scores, k=smooth_k)
    cands = make_candidates(s_smooth, starts, win_sec, nms_gap=nms_gap, top_k=top_k)

    # 예산 기준(최종 길이) 선택
    target_sec = target_minutes * 60.0
    eff_len = onset_pre_sec + onset_post_sec
    picked, used = pick_segments_under_budget(
        cands, target_sec, min_gap=min_gap_between, eff_len=eff_len
    )
    logger.info(f"[INFO] 후보 {len(cands)} → 최종 {len(picked)}개, 총 {used:.1f}s (클립 기준 {eff_len}s)")

    # 컷 생성 & 로고 트리밍
    PARTS_DIR = os.path.join(OUT_DIR, f"{base}_parts")
    os.makedirs(PARTS_DIR, exist_ok=True)


    raw_cuts = []
    for k, (st, ed, sc) in enumerate(picked, 1):
        # 1) 오디오 온셋 정렬
        if align_to_audio_onset:
            st_cut, ed_cut = recenter_by_audio_onset(
                full_mp4_path, st, ed, duration,
                onset_pre_sec=onset_pre_sec, onset_post_sec=onset_post_sec,
                search_back=10.0, search_fwd=10.0
            )
        else:
            center = 0.5 * (st + ed)
            st_cut = max(0.0, center - onset_pre_sec)
            ed_cut = min(duration, center + onset_post_sec)



        if logo_templates:
            t0_rule = max(0.0, st_cut)
            t1_rule = min(duration, ed_cut)
            _, _, spans_rl = scan_logo_range_multiscale(
                proxy_mp4, t0_rule, t1_rule, logo_templates,
                region="full", step=0.20, thr=0.60, min_hold=0.10, sigma=3
            )

            if len(spans_rl) >= 2:
                a1, b1 = spans_rl[0]
                if a1 > st_cut + 0.2:
                    ed_cut = max(st_cut + 0.2, a1 - 0.00)
                if (ed_cut - st_cut) < 15.0:
                    st_cut = max(0.0, st_cut - 10.0)
                logger.info(f"[DOUBLE-LOGO RULE(pre)] spans={spans_rl[:2]} -> keep first-play only: "
                      f"{st_cut:.2f} ~ {ed_cut:.2f} (len={ed_cut-st_cut:.2f}s)")
            else:
                st_cut, ed_cut = trim_by_logos_inside_clip_ms(
                    proxy_mp4, st_cut, ed_cut, duration, logo_templates,
                    step=0.20, thr=0.60, min_hold=0.10, sigma=3,
                    start_pad=0.18, end_pad=0.00,
                    lead_window=15.0, lead_prelook=12.0, lead_end_tol=10.0,
                    tail_window=15.0
                )


        st_cut = max(0.0, st_cut)
        ed_cut = min(duration, ed_cut)
        dur_cut = ed_cut - st_cut
        logger.info(f"[CUT {k:02d}] {st_cut:.2f} ~ {ed_cut:.2f} (dur={dur_cut:.2f}s)")
        if dur_cut < 1.0:
            continue

        raw_cuts.append({
            "k": k,
            "st": float(st),
            "ed": float(ed),
            "score": float(sc),
            "st_cut": float(st_cut),
            "ed_cut": float(ed_cut),
            "duration": float(dur_cut),
        })

    merged_cuts = merge_intervals(raw_cuts, join_gap=0.5)
    logger.info(f"[MERGE] raw={len(raw_cuts)} → merged={len(merged_cuts)}")


    tmp_parts = []
    cuts_meta = []
    for i, c in enumerate(merged_cuts, 1):
        part = os.path.join(PARTS_DIR, f"{base}_part{i:02d}.mp4")
        cut_segment(full_mp4_path, c["st_cut"], c["ed_cut"], part)
        tmp_parts.append(part)
        cuts_meta.append(c)

    assert tmp_parts, "생성된 파트가 없습니다."
    concat_mp4s(tmp_parts, out_mp4)


    for p in tmp_parts:
        try:
            os.remove(p)
        except:
            pass
    try:
        os.rmdir(PARTS_DIR)
    except:
        pass


    meta = {
        "full": full_mp4_path,
        "proxy": proxy_mp4,
        "win_sec": win_sec,
        "stride_sec": stride_sec,
        "target_min": target_minutes,
        "onset_pre_sec": onset_pre_sec,
        "onset_post_sec": onset_post_sec,
        "picked": cuts_meta,
        "out": out_mp4,
    }
    with open(os.path.join(OUT_DIR, f"{base}_target{int(target_minutes)}m_meta.json"), "w") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    logger.info(f"DONE: {out_mp4}")
    return out_mp4, picked, list(zip(starts, scores)), s_smooth
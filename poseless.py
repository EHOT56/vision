"""
Face Biometrics (InsightFace)
UI: cv2.imshow (fast)

Controls:
  Q - quit
"""

import cv2
import time
import uuid
import threading
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union

from insightface.app import FaceAnalysis

BBox = Tuple[int, int, int, int]


# ==========================
# CONFIG
# ==========================
CAMERA_SOURCE: Union[int, str] = 0

MODEL_PACK_NAME = "buffalo_s"
USE_GPU = False
DET_SIZE = (640, 640)

INFER_HZ = 6
SCALE = 1.0  # если хочешь быстрее: 0.75/0.6/0.5

WINDOW_NAME = "Face Biometrics Demo"

SIM_THRESHOLD = 0.58
SOFT_MATCH_THRESHOLD = 0.50

ENROLL_MIN_SAMPLES = 6
ENROLL_MAX_SECONDS = 3.0
ENROLL_MIN_INTERNAL_SIM = 0.40

MAX_EMBS_PER_UID = 40
MERGE_SKIP_IF_MAXSIM_ABOVE = 0.93

# ВАЖНО: фильтр для "безопасного merge"
MERGE_FILTER_THRESHOLD = 0.55

MIN_FACE_SIZE = 40

SESSION_IOU_MATCH = 0.30
SESSION_TTL_SEC = 1.2

AVATAR_SZ = 56
FOCUS_THICK = 4
NORMAL_THICK = 2

BOTTOM_BAR_H = 120
BOTTOM_MARGIN = 10
BOTTOM_CARD_H = 90
BOTTOM_CARD_W = 230
BOTTOM_GAP = 10
BOTTOM_MAX_CARDS = 5

# Enrollment quality gate
ENROLL_USE_QUALITY_FILTER = True
MIN_BLUR_VAR = 35.0
MIN_BRIGHTNESS = 40.0
MAX_BRIGHTNESS = 220.0


# ==========================
# Low-level utils (fast)
# ==========================
def clamp_int(v: float, lo: int, hi: int) -> int:
    if v < lo:
        return lo
    if v > hi:
        return hi
    return int(v)

def bbox_area(bb: BBox) -> int:
    x1, y1, x2, y2 = bb
    w = x2 - x1
    h = y2 - y1
    return (w * h) if (w > 0 and h > 0) else 0

def iou(a: BBox, b: BBox) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = ax1 if ax1 > bx1 else bx1
    iy1 = ay1 if ay1 > by1 else by1
    ix2 = ax2 if ax2 < bx2 else bx2
    iy2 = ay2 if ay2 < by2 else by2
    iw = ix2 - ix1
    ih = iy2 - iy1
    if iw <= 0 or ih <= 0:
        return 0.0
    inter = iw * ih
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    denom = area_a + area_b - inter
    return (inter / denom) if denom > 0 else 0.0

def safe_crop(img: np.ndarray, bb: BBox) -> Optional[np.ndarray]:
    x1, y1, x2, y2 = bb
    H, W = img.shape[:2]
    x1 = 0 if x1 < 0 else (W - 1 if x1 >= W else x1)
    x2 = 0 if x2 < 0 else (W if x2 > W else x2)
    y1 = 0 if y1 < 0 else (H - 1 if y1 >= H else y1)
    y2 = 0 if y2 < 0 else (H if y2 > H else y2)
    if x2 <= x1 or y2 <= y1:
        return None
    return img[y1:y2, x1:x2]

def l2_normalize(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n <= 1e-12:
        return v.astype(np.float32, copy=False)
    return (v / n).astype(np.float32, copy=False)

def quality_ok_for_enroll(crop_bgr: np.ndarray) -> bool:
    if crop_bgr is None or crop_bgr.size == 0:
        return False
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    mean = float(gray.mean())
    if mean < MIN_BRIGHTNESS or mean > MAX_BRIGHTNESS:
        return False
    blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    return blur >= MIN_BLUR_VAR


# ==========================
# Diagnostics + shared state
# ==========================
@dataclass
class Diagnostics:
    lock: threading.Lock = field(default_factory=threading.Lock)
    face_last_at: float = 0.0
    face_last_err: str = ""

    def set(self, **kwargs):
        with self.lock:
            for k, v in kwargs.items():
                setattr(self, k, v)

    def snapshot(self) -> dict:
        with self.lock:
            return {
                "face_last_at": self.face_last_at,
                "face_last_err": self.face_last_err,
            }

@dataclass
class SharedState:
    lock: threading.Lock = field(default_factory=threading.Lock)
    vis: List[dict] = field(default_factory=list)
    infer_ms: float = 0.0


# ==========================
# Async camera (zero-copy read)
# ==========================
class AsyncCameraStream:
    """
    get_latest_frame() возвращает ССЫЛКУ на np.ndarray (без copy).
    UI рисует на frame.copy(), воркеры читают.
    """
    def __init__(self, source, reconnect_delay: float = 1.0):
        self.source = source
        self.reconnect_delay = reconnect_delay
        self.lock = threading.Lock()
        self._frame: Optional[np.ndarray] = None
        self._stop = threading.Event()
        self._cap = None
        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()

    def _open(self):
        if isinstance(self.source, int):
            cap = cv2.VideoCapture(self.source, cv2.CAP_DSHOW)
            if not cap.isOpened():
                cap.release()
                cap = cv2.VideoCapture(self.source)
        else:
            cap = cv2.VideoCapture(self.source, cv2.CAP_FFMPEG)
            if not cap.isOpened():
                cap.release()
                cap = cv2.VideoCapture(self.source)

        if not cap.isOpened():
            cap.release()
            return None

        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass
        return cap

    def _worker(self):
        while not self._stop.is_set():
            if self._cap is None:
                self._cap = self._open()
                if self._cap is None:
                    time.sleep(self.reconnect_delay)
                    continue

            ret, frame = self._cap.read()
            if not ret or frame is None:
                try:
                    self._cap.release()
                except Exception:
                    pass
                self._cap = None
                time.sleep(self.reconnect_delay)
                continue

            with self.lock:
                self._frame = frame

    def get_latest_frame(self) -> Optional[np.ndarray]:
        with self.lock:
            return self._frame

    def stop(self):
        self._stop.set()
        self.thread.join(timeout=2.0)
        if self._cap is not None:
            try:
                self._cap.release()
            except Exception:
                pass
            self._cap = None


# ==========================
# InsightFace init
# ==========================
def init_insightface():
    providers = (["CUDAExecutionProvider", "CPUExecutionProvider"] if USE_GPU else ["CPUExecutionProvider"])
    app = FaceAnalysis(
        name=MODEL_PACK_NAME,
        providers=providers,
        allowed_modules=["detection", "recognition"]
    )
    app.prepare(ctx_id=(0 if USE_GPU else -1), det_size=DET_SIZE)
    return app


# ==========================
# Gallery DB (vectorized)
# ==========================
@dataclass
class Identity:
    uid: str
    embs: List[np.ndarray] = field(default_factory=list)        # list of (D,)
    emb_mat: Optional[np.ndarray] = None                        # (N, D)
    centroid: Optional[np.ndarray] = None                       # (D,)
    avatar: Optional[np.ndarray] = None                         # (AV, AV, 3)

    def _rebuild_mat(self):
        if not self.embs:
            self.emb_mat = None
            return
        self.emb_mat = np.stack(self.embs, axis=0).astype(np.float32, copy=False)

    def recompute_centroid(self) -> None:
        if not self.embs:
            self.centroid = None
            self.emb_mat = None
            return
        self._rebuild_mat()
        c = self.emb_mat.mean(axis=0)
        self.centroid = l2_normalize(c)

    def best_sim(self, emb_norm: np.ndarray) -> float:
        best = -1.0
        if self.centroid is not None:
            best = float(self.centroid @ emb_norm)
        if self.emb_mat is not None and self.emb_mat.shape[0] > 0:
            sims = self.emb_mat @ emb_norm
            m = float(sims.max())
            if m > best:
                best = m
        return best


class GalleryDB:
    def __init__(self):
        self.lock = threading.Lock()
        self.ids: Dict[str, Identity] = {}

    def find_best(self, emb_norm: np.ndarray) -> Tuple[Optional[str], float]:
        with self.lock:
            if not self.ids:
                return None, -1.0
            best_uid = None
            best_score = -1.0
            for uid, ident in self.ids.items():
                s = ident.best_sim(emb_norm)
                if s > best_score:
                    best_score = s
                    best_uid = uid
            return best_uid, best_score

    def create_identity_from_enroll(self, enroll_embs: List[np.ndarray], avatar: Optional[np.ndarray]) -> str:
        uid = str(uuid.uuid4())
        if len(enroll_embs) > MAX_EMBS_PER_UID:
            enroll_embs = enroll_embs[-MAX_EMBS_PER_UID:]
        ident = Identity(uid=uid, embs=list(enroll_embs), avatar=avatar)
        ident.recompute_centroid()
        with self.lock:
            self.ids[uid] = ident
        return uid

    def merge_enroll_into_uid(self, uid: str, enroll_embs: List[np.ndarray]) -> int:
        if not enroll_embs:
            return 0
        with self.lock:
            ident = self.ids.get(uid)
            if ident is None:
                ident = Identity(uid=uid, embs=list(enroll_embs))
                if len(ident.embs) > MAX_EMBS_PER_UID:
                    ident.embs = ident.embs[-MAX_EMBS_PER_UID:]
                ident.recompute_centroid()
                self.ids[uid] = ident
                return len(enroll_embs)

            added = 0
            if ident.emb_mat is None and ident.embs:
                ident._rebuild_mat()

            for emb in enroll_embs:
                if ident.emb_mat is not None and ident.emb_mat.shape[0] > 0:
                    max_sim = float((ident.emb_mat @ emb).max())
                else:
                    max_sim = -1.0
                if max_sim >= MERGE_SKIP_IF_MAXSIM_ABOVE:
                    continue
                ident.embs.append(emb)
                added += 1
                if len(ident.embs) > MAX_EMBS_PER_UID:
                    ident.embs.pop(0)

            if added > 0:
                ident.recompute_centroid()
            return added

    def filter_enroll_for_uid(self, uid: str, enroll_embs: List[np.ndarray], min_sim: float) -> Tuple[List[np.ndarray], int]:
        if not enroll_embs:
            return [], 0

        with self.lock:
            ident = self.ids.get(uid)
            if ident is None:
                return [], len(enroll_embs)

            if ident.centroid is None and ident.emb_mat is None and ident.embs:
                ident.recompute_centroid()
            elif ident.emb_mat is None and ident.embs:
                ident._rebuild_mat()

            mat = np.stack(enroll_embs, axis=0).astype(np.float32, copy=False)

            sims_best = None
            if ident.centroid is not None:
                sims_best = mat @ ident.centroid  # (M,)

            if ident.emb_mat is not None and ident.emb_mat.shape[0] > 0:
                sims_all = mat @ ident.emb_mat.T  # (M,N)
                sims_max = sims_all.max(axis=1)   # (M,)
                sims_best = sims_max if sims_best is None else np.maximum(sims_best, sims_max)

            if sims_best is None:
                return [], len(enroll_embs)

            keep_mask = sims_best >= float(min_sim)
            kept = [enroll_embs[i] for i in range(len(enroll_embs)) if bool(keep_mask[i])]
            dropped = int(len(enroll_embs) - len(kept))
            return kept, dropped

    def set_avatar_if_missing(self, uid: str, avatar: Optional[np.ndarray]) -> None:
        if avatar is None:
            return
        with self.lock:
            ident = self.ids.get(uid)
            if ident and ident.avatar is None:
                ident.avatar = avatar

    def get_avatar(self, uid: str) -> Optional[np.ndarray]:
        with self.lock:
            ident = self.ids.get(uid)
            return ident.avatar if ident else None

    def size(self) -> int:
        with self.lock:
            return len(self.ids)


# ==========================
# Sessions (greedy IoU match)
# ==========================
@dataclass
class FaceSession:
    sid: int
    bbox: BBox
    last_seen: float
    uid: Optional[str] = None

    enroll_started: float = 0.0
    enroll_embs: List[np.ndarray] = field(default_factory=list)
    enroll_best_crop: Optional[np.ndarray] = None

    first_seen: float = 0.0

class SessionManager:
    def __init__(self):
        self.next_sid = 1
        self.sessions: List[FaceSession] = []

    def update(self, det_bboxes: List[BBox], now: float) -> Dict[int, FaceSession]:
        alive = []
        for s in self.sessions:
            if (now - s.last_seen) <= SESSION_TTL_SEC:
                alive.append(s)
        self.sessions = alive

        assigned: Dict[int, FaceSession] = {}
        used_sessions = set()

        for i, bb in enumerate(det_bboxes):
            best_s = None
            best_i = 0.0
            for s in self.sessions:
                if s.sid in used_sessions:
                    continue
                sc = iou(s.bbox, bb)
                if sc > best_i:
                    best_i = sc
                    best_s = s

            if best_s is not None and best_i >= SESSION_IOU_MATCH:
                best_s.bbox = bb
                best_s.last_seen = now
                assigned[i] = best_s
                used_sessions.add(best_s.sid)
            else:
                snew = FaceSession(
                    sid=self.next_sid,
                    bbox=bb,
                    last_seen=now,
                    uid=None,
                    first_seen=now
                )
                self.next_sid += 1
                self.sessions.append(snew)
                assigned[i] = snew
                used_sessions.add(snew.sid)

        return assigned


# ==========================
# Face inference worker
# ==========================
class FaceWorker:
    def __init__(self, app: FaceAnalysis, cam: AsyncCameraStream, db: GalleryDB, sess_mgr: SessionManager,
                 shared: SharedState, diag: Diagnostics, stop_evt: threading.Event):
        self.app = app
        self.cam = cam
        self.db = db
        self.sess_mgr = sess_mgr
        self.shared = shared
        self.diag = diag
        self.stop_evt = stop_evt
        self.thread = threading.Thread(target=self.run, daemon=True)

    def start(self):
        self.thread.start()

    def _merge_session_buffer_to_uid_safe(self, s: FaceSession, target_uid: str) -> Tuple[int, int]:
        if not s.enroll_embs:
            return 0, 0
        kept, dropped = self.db.filter_enroll_for_uid(target_uid, s.enroll_embs, min_sim=MERGE_FILTER_THRESHOLD)
        if not kept:
            return 0, dropped
        added = self.db.merge_enroll_into_uid(target_uid, kept)
        return added, dropped

    def run(self):
        self.diag.set(face_last_err="", face_last_at=0.0)
        period = 1.0 / max(1, INFER_HZ)

        try:
            while not self.stop_evt.is_set():
                t_loop = time.time()
                now = t_loop

                frame = self.cam.get_latest_frame()
                if frame is None:
                    time.sleep(0.01)
                    continue

                H, W = frame.shape[:2]

                if SCALE != 1.0:
                    interp = cv2.INTER_AREA if SCALE < 1.0 else cv2.INTER_LINEAR
                    small = cv2.resize(frame, None, fx=SCALE, fy=SCALE, interpolation=interp)
                else:
                    small = frame

                if small.dtype != np.uint8:
                    small = small.astype(np.uint8, copy=False)

                t0 = time.time()
                faces = self.app.get(small)
                infer_ms = (time.time() - t0) * 1000.0

                dets: List[Tuple[BBox, np.ndarray, Optional[np.ndarray]]] = []
                det_bboxes: List[BBox] = []

                for f in faces:
                    bb = f.bbox
                    if SCALE != 1.0:
                        bb = bb / SCALE
                    bb = bb.astype(np.int32, copy=False)

                    x1 = clamp_int(bb[0], 0, W - 1)
                    y1 = clamp_int(bb[1], 0, H - 1)
                    x2 = clamp_int(bb[2], 0, W - 1)
                    y2 = clamp_int(bb[3], 0, H - 1)

                    if x2 <= x1 or y2 <= y1:
                        continue
                    if (x2 - x1) < MIN_FACE_SIZE or (y2 - y1) < MIN_FACE_SIZE:
                        continue

                    emb = getattr(f, "embedding", None)
                    if emb is None:
                        continue

                    emb_norm = l2_normalize(np.asarray(emb, dtype=np.float32))
                    bbox = (x1, y1, x2, y2)

                    crop = safe_crop(frame, bbox)
                    dets.append((bbox, emb_norm, crop))
                    det_bboxes.append(bbox)

                idx_to_sess = self.sess_mgr.update(det_bboxes, now)

                vis_out: List[dict] = []

                for idx, (bb, emb_norm, crop) in enumerate(dets):
                    s = idx_to_sess.get(idx)
                    if s is None:
                        s = FaceSession(sid=-1, bbox=bb, last_seen=now, first_seen=now)

                    label = ""
                    color = (0, 0, 255)
                    status = "unknown"
                    score = -1.0

                    best_uid, best_score = self.db.find_best(emb_norm)
                    score = best_score

                    if best_uid is not None and best_score >= SIM_THRESHOLD:
                        if s.enroll_embs:
                            added, dropped = self._merge_session_buffer_to_uid_safe(s, best_uid)
                            if added > 0 or dropped > 0:
                                print(f">>> MERGE(buffer->KNOWN) {best_uid[:6]}.. +{added} drop={dropped} buf={len(s.enroll_embs)}")
                            s.enroll_started = 0.0
                            s.enroll_embs.clear()
                            s.enroll_best_crop = None

                        s.uid = best_uid
                        status = "known"
                        label = f"{best_uid[:6]}.. ({best_score:.2f})"
                        color = (0, 255, 0)

                        if crop is not None and ((not ENROLL_USE_QUALITY_FILTER) or quality_ok_for_enroll(crop)):
                            av = cv2.resize(crop, (AVATAR_SZ, AVATAR_SZ), interpolation=cv2.INTER_AREA)
                            self.db.set_avatar_if_missing(best_uid, av)

                    elif best_uid is not None and best_score >= SOFT_MATCH_THRESHOLD:
                        if s.enroll_embs:
                            added, dropped = self._merge_session_buffer_to_uid_safe(s, best_uid)
                            if added > 0 or dropped > 0:
                                print(f">>> MERGE(buffer->SOFT)  {best_uid[:6]}.. +{added} drop={dropped} buf={len(s.enroll_embs)}")
                            s.enroll_started = 0.0
                            s.enroll_embs.clear()
                            s.enroll_best_crop = None

                        s.uid = best_uid
                        status = "soft"
                        label = f"{best_uid[:6]}.. (~{best_score:.2f})"
                        color = (0, 200, 255)

                        if crop is not None and ((not ENROLL_USE_QUALITY_FILTER) or quality_ok_for_enroll(crop)):
                            av = cv2.resize(crop, (AVATAR_SZ, AVATAR_SZ), interpolation=cv2.INTER_AREA)
                            self.db.set_avatar_if_missing(best_uid, av)

                    else:
                        if s.enroll_started == 0.0:
                            s.enroll_started = now
                            s.enroll_embs.clear()
                            s.enroll_best_crop = None

                        can_add = True
                        if ENROLL_USE_QUALITY_FILTER:
                            can_add = (crop is not None) and quality_ok_for_enroll(crop)

                        if can_add:
                            s.enroll_embs.append(emb_norm)
                            if s.enroll_best_crop is None and crop is not None:
                                s.enroll_best_crop = crop

                        if (now - s.enroll_started) > ENROLL_MAX_SECONDS:
                            s.enroll_started = now
                            s.enroll_embs = [emb_norm] if can_add else []
                            s.enroll_best_crop = crop if can_add else None

                        created_uid = None
                        merged_uid = None
                        merged_added = 0
                        merged_dropped = 0
                        merged_score = -1.0

                        n = len(s.enroll_embs)
                        if n >= ENROLL_MIN_SAMPLES:
                            mat = np.stack(s.enroll_embs, axis=0).astype(np.float32, copy=False)
                            c = l2_normalize(mat.mean(axis=0))
                            sims = mat @ c
                            min_sim = float(sims.min())

                            if min_sim >= ENROLL_MIN_INTERNAL_SIM:
                                uid_c, sc = self.db.find_best(c)
                                merged_score = float(sc)

                                if uid_c is not None and sc >= SOFT_MATCH_THRESHOLD:
                                    kept, dropped = self.db.filter_enroll_for_uid(uid_c, s.enroll_embs, min_sim=MERGE_FILTER_THRESHOLD)
                                    merged_dropped = dropped
                                    if kept:
                                        merged_uid = uid_c
                                        merged_added = self.db.merge_enroll_into_uid(uid_c, kept)

                                        if s.enroll_best_crop is not None:
                                            av = cv2.resize(s.enroll_best_crop, (AVATAR_SZ, AVATAR_SZ), interpolation=cv2.INTER_AREA)
                                            self.db.set_avatar_if_missing(uid_c, av)

                                        s.uid = uid_c
                                        s.enroll_started = 0.0
                                        s.enroll_embs.clear()
                                        s.enroll_best_crop = None

                                        print(
                                            f">>> MERGE-BUFFER -> {uid_c[:6]}.. "
                                            f"score={sc:.2f} +{merged_added} drop={merged_dropped} buf={n} min_internal={min_sim:.2f}"
                                        )
                                    else:
                                        merged_uid = None

                                if merged_uid is None:
                                    avatar = None
                                    if s.enroll_best_crop is not None:
                                        avatar = cv2.resize(s.enroll_best_crop, (AVATAR_SZ, AVATAR_SZ), interpolation=cv2.INTER_AREA)
                                    created_uid = self.db.create_identity_from_enroll(s.enroll_embs, avatar=avatar)

                                    s.uid = created_uid
                                    s.enroll_started = 0.0
                                    s.enroll_embs.clear()
                                    s.enroll_best_crop = None

                                    print(f">>> NEW UID: {created_uid[:6]}.. (min_internal_sim={min_sim:.2f})")

                        if merged_uid is not None:
                            if merged_score >= SIM_THRESHOLD:
                                status = "known"
                                label = f"{merged_uid[:6]}.. ({merged_score:.2f}) +M{merged_added}"
                                color = (0, 255, 0)
                            else:
                                status = "soft"
                                label = f"{merged_uid[:6]}.. (~{merged_score:.2f}) +M{merged_added}"
                                color = (0, 200, 255)

                        elif created_uid is not None:
                            status = "new"
                            label = f"NEW {created_uid[:6]}.."
                            color = (255, 0, 255)

                        else:
                            status = "enroll"
                            label = f"ENROLL {len(s.enroll_embs)}/{ENROLL_MIN_SAMPLES}"
                            color = (0, 0, 255)

                    s.bbox = bb
                    s.last_seen = now

                    vis_out.append({
                        "bbox": bb,
                        "sid": s.sid,
                        "uid": s.uid,
                        "label": label,
                        "color": color,
                        "status": status,
                        "score": float(score),
                        "first_seen": float(s.first_seen),
                        "enroll_n": int(len(s.enroll_embs)),
                        "area": int(bbox_area(bb)),
                    })

                with self.shared.lock:
                    self.shared.vis = vis_out
                    self.shared.infer_ms = float(infer_ms)

                self.diag.set(face_last_at=time.time(), face_last_err="")

                sleep_left = period - (time.time() - t_loop)
                if sleep_left > 0:
                    time.sleep(sleep_left)

        except Exception:
            import traceback
            err = traceback.format_exc()
            print(">>> FACE THREAD CRASH:\n", err)
            self.diag.set(face_last_err=err, face_last_at=time.time())
        finally:
            self.diag.set(face_last_at=time.time())


# ==========================
# UI drawing
# ==========================
def draw_transparent_rect(img, x1, y1, x2, y2, alpha=0.6):
    overlay = img.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 0), -1)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, dst=img)

def draw_bar(img, x, y, w, h, frac, color=(0, 255, 0), bg=(50, 50, 50)):
    frac = 0.0 if frac < 0 else (1.0 if frac > 1.0 else float(frac))
    cv2.rectangle(img, (x, y), (x + w, y + h), bg, -1)
    cv2.rectangle(img, (x, y), (x + int(w * frac), y + h), color, -1)
    cv2.rectangle(img, (x, y), (x + w, y + h), (200, 200, 200), 1)

def draw_card(img: np.ndarray, x1: int, y1: int, x2: int, y2: int,
              alpha=0.55, border=(255, 255, 255), border_th=1):
    draw_transparent_rect(img, x1, y1, x2, y2, alpha=alpha)
    if border_th > 0:
        cv2.rectangle(img, (x1, y1), (x2, y2), border, border_th)

def choose_focus(vis: List[dict]) -> Optional[dict]:
    if not vis:
        return None
    def key(item):
        pri = 0
        st = item.get("status", "")
        if st == "known":
            pri = 4
        elif st == "soft":
            pri = 3
        elif st == "enroll":
            pri = 1
        return (pri, item.get("area", 0), item.get("score", -1.0))
    return max(vis, key=key)

def draw_bottom_db_bar(display: np.ndarray, vis: List[dict], db: GalleryDB, focus: Optional[dict]):
    H, W = display.shape[:2]
    bar_h = min(BOTTOM_BAR_H, H - 60)
    y0 = H - bar_h
    if y0 < 46:
        return

    draw_transparent_rect(display, 0, y0, W, H, alpha=0.55)
    cv2.line(display, (0, y0), (W, y0), (255, 255, 255), 1)

    cv2.putText(display, "DB", (12, y0 + 26), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1)
    cv2.putText(display, f"IDs {db.size()}", (62, y0 + 26), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

    focus_sid = focus["sid"] if focus else None

    def sort_key(it):
        pri = 0
        if it["sid"] == focus_sid:
            pri = 5
        elif it["status"] == "known":
            pri = 4
        elif it["status"] == "soft":
            pri = 3
        elif it["status"] == "enroll":
            pri = 1
        return (-pri, -it.get("area", 0))

    rows = sorted(vis, key=sort_key)

    y_card = y0 + 34
    h_card = min(BOTTOM_CARD_H, bar_h - 44)
    if h_card < 60:
        return

    max_cards_fit = (W - 2 * BOTTOM_MARGIN + BOTTOM_GAP) // (BOTTOM_CARD_W + BOTTOM_GAP)
    max_cards = int(max(1, min(BOTTOM_MAX_CARDS, max_cards_fit)))

    x = BOTTOM_MARGIN
    for it in rows[:max_cards]:
        uid = it.get("uid")
        status = it.get("status", "unknown")
        score = float(it.get("score", -1.0))
        first_seen = float(it.get("first_seen", 0.0))

        x1 = x
        y1 = y_card
        x2 = min(W - BOTTOM_MARGIN, x1 + BOTTOM_CARD_W)
        y2 = y1 + h_card

        border = (255, 255, 255) if it["sid"] == focus_sid else (140, 140, 140)
        draw_card(display, x1, y1, x2, y2, alpha=0.42, border=border, border_th=1)

        ax = x1 + 10
        ay = y1 + 12

        if uid:
            av = db.get_avatar(uid)
            if av is not None:
                ah, aw = av.shape[:2]
                if ay + ah <= H and ax + aw <= W:
                    display[ay:ay + ah, ax:ax + aw] = av
                cv2.rectangle(display, (ax, ay), (ax + AVATAR_SZ, ay + AVATAR_SZ), (200, 200, 200), 1)
            else:
                cv2.rectangle(display, (ax, ay), (ax + AVATAR_SZ, ay + AVATAR_SZ), (75, 75, 75), -1)
        else:
            cv2.rectangle(display, (ax, ay), (ax + AVATAR_SZ, ay + AVATAR_SZ), (65, 65, 65), -1)

        tx = ax + AVATAR_SZ + 10
        name = (uid[:6] + "..") if uid else f"Session {it['sid']}"
        cv2.putText(display, name, (tx, ay + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 1)

        seen_time = time.strftime("%H:%M:%S", time.localtime(first_seen)) if first_seen else "--:--:--"
        cv2.putText(display, f"first seen: {seen_time}", (tx, ay + 36), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (200, 200, 200), 1)

        if status in ("known", "soft"):
            cv2.putText(display, f"{status} {score:.2f}", (tx, ay + 54), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (210, 210, 210), 1)
        elif status == "enroll":
            cv2.putText(display, f"enroll {it.get('enroll_n', 0)}/{ENROLL_MIN_SAMPLES}",
                        (tx, ay + 54), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (210, 210, 210), 1)

        x = x2 + BOTTOM_GAP
        if x >= W - BOTTOM_MARGIN:
            break

def draw_top_debug_bar(display: np.ndarray, ui_fps: float, infer_ms: float, faces: int, ids: int, diag: Diagnostics):
    H, W = display.shape[:2]
    bar_h = 34
    draw_transparent_rect(display, 0, 0, W, bar_h, alpha=0.55)
    cv2.line(display, (0, bar_h), (W, bar_h), (255, 255, 255), 1)

    d = diag.snapshot()
    line1 = f"UI {ui_fps:.1f}fps | Face {infer_ms:.0f}ms | Faces {faces} | IDs {ids}"
    cv2.putText(display, line1, (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (255, 255, 255), 1)

    err = d.get("face_last_err", "")
    if err:
        first = err.strip().splitlines()[0][:160]
        cv2.putText(display, f"ERR: {first}", (12, bar_h + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 255, 255), 1)


# ==========================
# Main
# ==========================
def main():
    cv2.setUseOptimized(True)

    print(">>> Starting...")
    cam = AsyncCameraStream(CAMERA_SOURCE)
    time.sleep(0.2)

    diag = Diagnostics()
    shared = SharedState()

    try:
        print(">>> Init InsightFace...")
        app = init_insightface()
        print(">>> InsightFace OK.")
    except Exception:
        import traceback
        err = traceback.format_exc()
        print(">>> InsightFace INIT FAILED:\n", err)
        diag.set(face_last_err=err)
        app = None

    db = GalleryDB()
    sess_mgr = SessionManager()
    stop_evt = threading.Event()

    if app is not None:
        face_worker = FaceWorker(app, cam, db, sess_mgr, shared, diag, stop_evt)
        face_worker.start()
    else:
        print(">>> Face thread not started (InsightFace init failed).")

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, 1280, 720)

    fps_t0 = time.time()
    fps_frames = 0
    ui_fps = 0.0

    print("Controls: Q - quit")

    try:
        while True:
            frame = cam.get_latest_frame()
            if frame is None:
                img = np.zeros((540, 960, 3), dtype=np.uint8)
                cv2.putText(img, "WAITING FOR CAMERA...", (30, 270),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
                cv2.imshow(WINDOW_NAME, img)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                continue

            now = time.time()
            with shared.lock:
                vis = list(shared.vis)
                infer_ms = float(shared.infer_ms)

            display = frame.copy()
            focus = choose_focus(vis)

            for item in vis:
                x1, y1, x2, y2 = item["bbox"]
                color = item["color"]
                label = item["label"]

                thick = FOCUS_THICK if (focus is not None and item["sid"] == focus["sid"]) else NORMAL_THICK
                cv2.rectangle(display, (x1, y1), (x2, y2), color, thick)
                cv2.putText(display, label, (x1, 24 if (y1 - 10) < 24 else (y1 - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)

                if item["status"] == "enroll":
                    bar_h = min(BOTTOM_BAR_H, display.shape[0] - 60)
                    bar_y0 = display.shape[0] - bar_h

                    bx = x1
                    by = y2 + 6
                    if by > bar_y0 - 14:
                        by = 78 if (y1 - 14) < 78 else (y1 - 14)
                    n = item.get("enroll_n", 0)
                    frac = n / float(max(1, ENROLL_MIN_SAMPLES))
                    draw_bar(display, bx, by, max(40, x2 - x1), 8, frac, color=(0, 0, 255))

            draw_bottom_db_bar(display, vis, db, focus)

            fps_frames += 1
            if now - fps_t0 >= 1.0:
                ui_fps = fps_frames / (now - fps_t0)
                fps_t0 = now
                fps_frames = 0

            draw_top_debug_bar(display, ui_fps=ui_fps, infer_ms=infer_ms, faces=len(vis), ids=db.size(), diag=diag)

            cv2.imshow(WINDOW_NAME, display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

    finally:
        stop_evt.set()
        cam.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

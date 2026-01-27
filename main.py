import cv2
import time
import threading
import uuid
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union
from numpy.linalg import norm
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
SCALE = 0.5

WINDOW_NAME = "Face Biometrics (enroll merge to known)"

# Узнавание (когда считаем, что это точно он)
SIM_THRESHOLD = 0.58
# "мягкое" узнавание — считаем, что это он (чтобы не плодить UID),
# и если был enroll-буфер — присоединим его к этому uid.
SOFT_MATCH_THRESHOLD = 0.50

# Enrollment: сколько эмбеддингов собрать, чтобы создать НОВЫЙ uid
ENROLL_MIN_SAMPLES = 6
ENROLL_MAX_SECONDS = 3.0
ENROLL_MIN_INTERNAL_SIM = 0.40  # качество буфера (чтобы не смешать 2 разных)

# Галерея: несколько эмбеддингов на uid
MAX_EMBS_PER_UID = 40
# при bulk-merge не добавляем почти дубликаты
MERGE_SKIP_IF_MAXSIM_ABOVE = 0.93

MIN_FACE_SIZE = 40

# "Сессии" (очень лёгкая связка по bbox IoU)
SESSION_IOU_MATCH = 0.30
SESSION_TTL_SEC = 1.2

# ==========================
# Utils
# ==========================
def l2_normalize(v: np.ndarray) -> np.ndarray:
    n = norm(v)
    if n <= 1e-12:
        return v.astype(np.float32)
    return (v / n).astype(np.float32)

def cosine(a_norm: np.ndarray, b_norm: np.ndarray) -> float:
    return float(np.dot(a_norm, b_norm))

def clamp_int(v: float, lo: int, hi: int) -> int:
    return max(lo, min(int(v), hi))

def iou(a: BBox, b: BBox) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    denom = area_a + area_b - inter
    return (inter / denom) if denom > 0 else 0.0

def bbox_center(bb: BBox) -> Tuple[float, float]:
    x1, y1, x2, y2 = bb
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0

# ==========================
# Gallery DB: uid -> multiple embeddings
# ==========================
@dataclass
class Identity:
    uid: str
    embs: List[np.ndarray] = field(default_factory=list)  # normalized
    centroid: Optional[np.ndarray] = None                 # normalized

    def recompute_centroid(self) -> None:
        if not self.embs:
            self.centroid = None
            return
        c = np.mean(np.stack(self.embs, axis=0), axis=0)
        self.centroid = l2_normalize(c)

class GalleryDB:
    def __init__(self):
        self.lock = threading.Lock()
        self.ids: Dict[str, Identity] = {}

    def _best_sim_to_identity(self, emb_norm: np.ndarray, ident: Identity) -> float:
        best = -1.0
        # быстрый фильтр: centroid
        if ident.centroid is not None:
            best = cosine(emb_norm, ident.centroid)
        # точный по всем эмбеддингам
        for e in ident.embs:
            s = cosine(emb_norm, e)
            if s > best:
                best = s
        return best

    def find_best(self, emb_norm: np.ndarray) -> Tuple[Optional[str], float]:
        with self.lock:
            if not self.ids:
                return None, -1.0

            best_uid = None
            best_score = -1.0
            for uid, ident in self.ids.items():
                s = self._best_sim_to_identity(emb_norm, ident)
                if s > best_score:
                    best_score = s
                    best_uid = uid
            return best_uid, best_score

    def create_identity_from_enroll(self, enroll_embs: List[np.ndarray]) -> str:
        uid = str(uuid.uuid4())
        with self.lock:
            ident = Identity(uid=uid, embs=list(enroll_embs))
            # ограничение
            if len(ident.embs) > MAX_EMBS_PER_UID:
                ident.embs = ident.embs[-MAX_EMBS_PER_UID:]
            ident.recompute_centroid()
            self.ids[uid] = ident
        return uid

    def merge_enroll_into_uid(self, uid: str, enroll_embs: List[np.ndarray]) -> int:
        """
        Добавить эмбеддинги из enroll к существующему uid.
        Возвращает: сколько реально добавили (после фильтра дублей).
        """
        if not enroll_embs:
            return 0

        with self.lock:
            ident = self.ids.get(uid)
            if ident is None:
                # если uid вдруг исчез — создадим как новый
                self.ids[uid] = Identity(uid=uid, embs=list(enroll_embs))
                self.ids[uid].recompute_centroid()
                return len(enroll_embs)

            added = 0
            for emb in enroll_embs:
                # фильтр дублей: если слишком похож на уже имеющиеся, пропускаем
                max_sim = -1.0
                for e in ident.embs:
                    s = cosine(emb, e)
                    if s > max_sim:
                        max_sim = s

                if max_sim >= MERGE_SKIP_IF_MAXSIM_ABOVE:
                    continue

                ident.embs.append(emb)
                added += 1

                if len(ident.embs) > MAX_EMBS_PER_UID:
                    ident.embs.pop(0)

            if added > 0:
                ident.recompute_centroid()

            return added

    def size(self) -> int:
        with self.lock:
            return len(self.ids)

    def emb_count(self, uid: str) -> int:
        with self.lock:
            ident = self.ids.get(uid)
            return len(ident.embs) if ident else 0


# ==========================
# Light session buffer (not tracking UI) for enroll
# ==========================
@dataclass
class FaceSession:
    sid: int
    last_bbox: BBox
    last_seen: float
    uid: Optional[str] = None

    enroll_embs: List[np.ndarray] = field(default_factory=list)
    enroll_started: float = 0.0

class SessionManager:
    def __init__(self):
        self.next_sid = 1
        self.sessions: List[FaceSession] = []

    def _match_session(self, bb: BBox) -> Optional[FaceSession]:
        best = None
        best_iou = 0.0
        for s in self.sessions:
            sc = iou(s.last_bbox, bb)
            if sc > best_iou:
                best_iou = sc
                best = s
        if best is not None and best_iou >= SESSION_IOU_MATCH:
            return best
        return None

    def update_sessions(self, det_bboxes: List[BBox], now: float) -> List[FaceSession]:
        # purge old
        self.sessions = [s for s in self.sessions if (now - s.last_seen) <= SESSION_TTL_SEC]

        updated: List[FaceSession] = []
        used_sid = set()

        for bb in det_bboxes:
            s = self._match_session(bb)
            if s is None or s.sid in used_sid:
                s = FaceSession(
                    sid=self.next_sid,
                    last_bbox=bb,
                    last_seen=now,
                    uid=None,
                    enroll_embs=[],
                    enroll_started=0.0
                )
                self.next_sid += 1
                self.sessions.append(s)
            else:
                s.last_bbox = bb
                s.last_seen = now

            used_sid.add(s.sid)
            updated.append(s)

        return updated


# ==========================
# Async camera
# ==========================
class AsyncCameraStream:
    def __init__(self, source, reconnect_delay: float = 1.0):
        self.source = source
        self.reconnect_delay = reconnect_delay
        self.lock = threading.Lock()
        self._frame = None
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

    def get_latest_frame(self):
        with self.lock:
            return self._frame.copy() if self._frame is not None else None

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
    providers = (["CUDAExecutionProvider", "CPUExecutionProvider"]
                 if USE_GPU else ["CPUExecutionProvider"])
    app = FaceAnalysis(
        name=MODEL_PACK_NAME,
        providers=providers,
        allowed_modules=["detection", "recognition"]
    )
    app.prepare(ctx_id=(0 if USE_GPU else -1), det_size=DET_SIZE)
    return app


# ==========================
# Shared inference results (for UI)
# ==========================
state_lock = threading.Lock()
latest_vis = []          # list of dicts: bbox, label, color
latest_infer_ms = 0.0
latest_infer_at = 0.0


def infer_worker(app: FaceAnalysis, cam: AsyncCameraStream, db: GalleryDB,
                 sess_mgr: SessionManager, stop_evt: threading.Event):
    global latest_vis, latest_infer_ms, latest_infer_at

    period = 1.0 / max(1, INFER_HZ)

    while not stop_evt.is_set():
        t_start = time.time()
        now = time.time()

        frame = cam.get_latest_frame()
        if frame is None:
            time.sleep(0.01)
            continue

        small = cv2.resize(frame, None, fx=SCALE, fy=SCALE, interpolation=cv2.INTER_LINEAR) if SCALE != 1.0 else frame

        t0 = time.time()
        faces = app.get(small)
        infer_ms = (time.time() - t0) * 1000.0

        H, W = frame.shape[:2]

        dets: List[Tuple[BBox, np.ndarray]] = []
        det_bboxes: List[BBox] = []

        for f in faces:
            bb = f.bbox
            if SCALE != 1.0:
                bb = bb / SCALE
            bb = bb.astype(int)

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
            dets.append((bbox, emb_norm))
            det_bboxes.append(bbox)

        # sessions by bbox IoU
        sessions = sess_mgr.update_sessions(det_bboxes, now)

        # map each detection bbox -> its session (best IoU)
        bbox_to_session: Dict[BBox, FaceSession] = {}
        for bb, _ in dets:
            best_s = None
            best_i = 0.0
            for s in sessions:
                sc = iou(s.last_bbox, bb)
                if sc > best_i:
                    best_i = sc
                    best_s = s
            if best_s is not None and best_i >= SESSION_IOU_MATCH:
                bbox_to_session[bb] = best_s

        vis = []

        for bb, emb_norm in dets:
            s = bbox_to_session.get(bb)
            if s is None:
                # не должно часто случаться, но на всякий
                s = FaceSession(sid=-1, last_bbox=bb, last_seen=now)

            # Если у сессии есть uid — просто проверим (по желанию можно пропустить проверку)
            if s.uid is not None:
                best_uid, best_score = db.find_best(emb_norm)
                if best_uid == s.uid and best_score >= SOFT_MATCH_THRESHOLD:
                    label = f"{s.uid[:6]}.. ({best_score:.2f})"
                    color = (0, 255, 0)
                else:
                    # потеряли связь — вернёмся к unknown/enroll
                    s.uid = None
                    s.enroll_embs.clear()
                    s.enroll_started = 0.0

            # Если uid нет — пробуем узнать
            if s.uid is None:
                best_uid, best_score = db.find_best(emb_norm)

                if best_uid is not None and best_score >= SIM_THRESHOLD:
                    # Узнали уверенно: если был enroll-буфер — присоединяем его к этому uid
                    if s.enroll_embs:
                        merged = db.merge_enroll_into_uid(best_uid, s.enroll_embs)
                        print(f">>> MERGE enroll -> {best_uid[:6]}.. : +{merged} embs (buffer={len(s.enroll_embs)})")
                        s.enroll_embs.clear()
                        s.enroll_started = 0.0

                    s.uid = best_uid
                    label = f"{best_uid[:6]}.. ({best_score:.2f})"
                    color = (0, 255, 0)

                elif best_uid is not None and best_score >= SOFT_MATCH_THRESHOLD:
                    # Мягко: считаем это он же (анти-дубликаты), и если был enroll — тоже мерджим
                    if s.enroll_embs:
                        merged = db.merge_enroll_into_uid(best_uid, s.enroll_embs)
                        print(f">>> MERGE(enroll-soft) -> {best_uid[:6]}.. : +{merged} embs (buffer={len(s.enroll_embs)})")
                        s.enroll_embs.clear()
                        s.enroll_started = 0.0

                    s.uid = best_uid
                    label = f"{best_uid[:6]}.. (~{best_score:.2f})"
                    color = (0, 200, 255)

                else:
                    # Unknown -> enrollment buffer
                    if s.enroll_started == 0.0:
                        s.enroll_started = now
                        s.enroll_embs = []

                    s.enroll_embs.append(emb_norm)

                    # time trim
                    if (now - s.enroll_started) > ENROLL_MAX_SECONDS:
                        s.enroll_started = now
                        s.enroll_embs = [emb_norm]

                    created_uid = None
                    if len(s.enroll_embs) >= ENROLL_MIN_SAMPLES:
                        c = l2_normalize(np.mean(np.stack(s.enroll_embs, axis=0), axis=0))
                        min_sim = min(cosine(e, c) for e in s.enroll_embs)
                        if min_sim >= ENROLL_MIN_INTERNAL_SIM:
                            created_uid = db.create_identity_from_enroll(s.enroll_embs)
                            s.uid = created_uid
                            s.enroll_embs = []
                            s.enroll_started = 0.0
                            print(f">>> NEW UID: {created_uid} (enroll={ENROLL_MIN_SAMPLES}, min_internal_sim={min_sim:.2f})")

                    if created_uid is not None:
                        label = f"NEW {created_uid[:6]}.."
                        color = (0, 0, 255)
                    else:
                        label = f"ENROLL {len(s.enroll_embs)}/{ENROLL_MIN_SAMPLES}"
                        color = (0, 0, 255)

            # UI overlay
            vis.append({"bbox": bb, "label": label, "color": color})

            # update session
            s.last_bbox = bb
            s.last_seen = now

        with state_lock:
            latest_vis = vis
            latest_infer_ms = infer_ms
            latest_infer_at = time.time()

        elapsed = time.time() - t_start
        sleep_left = period - elapsed
        if sleep_left > 0:
            time.sleep(sleep_left)


def main():
    print(">>> Init InsightFace (detection + recognition)...")
    app = init_insightface()

    print(">>> Opening camera stream...")
    cam = AsyncCameraStream(CAMERA_SOURCE)

    db = GalleryDB()
    sess_mgr = SessionManager()

    stop_evt = threading.Event()
    t = threading.Thread(target=infer_worker, args=(app, cam, db, sess_mgr, stop_evt), daemon=True)
    t.start()

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, 960, 540)

    fps_t0 = time.time()
    fps_frames = 0
    ui_fps = 0.0

    try:
        while True:
            frame = cam.get_latest_frame()
            if frame is None:
                img = np.zeros((540, 960, 3), dtype=np.uint8)
                cv2.putText(img, "WAITING FOR FRAMES...", (30, 270),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
                cv2.imshow(WINDOW_NAME, img)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                continue

            now = time.time()
            with state_lock:
                vis = list(latest_vis)
                infer_ms = float(latest_infer_ms)
                infer_at = float(latest_infer_at)

            infer_age = (now - infer_at) if infer_at else 999.0

            display = frame.copy()

            for item in vis:
                x1, y1, x2, y2 = item["bbox"]
                color = item["color"]
                label = item["label"]

                cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)
                cv2.putText(display, label, (x1, max(20, y1 - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)

            fps_frames += 1
            if now - fps_t0 >= 1.0:
                ui_fps = fps_frames / (now - fps_t0)
                fps_t0 = now
                fps_frames = 0

            stats = (
                f"UI FPS:{ui_fps:.1f} | Infer:{infer_ms:.0f}ms @ {INFER_HZ}Hz "
                f"| faces:{len(vis)} | infer_age:{infer_age*1000:.0f}ms | ids:{db.size()}"
            )
            cv2.putText(display, stats, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

            cv2.imshow(WINDOW_NAME, display)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        stop_evt.set()
        cam.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

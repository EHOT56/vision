import cv2
import numpy as np
import threading
import time
import uuid
from numpy.linalg import norm
from insightface.app import FaceAnalysis

# ==============================================================================
# КОНФИГУРАЦИЯ СИСТЕМЫ
# ==============================================================================

# 'buffalo_l' - точнее, но медленнее. 'buffalo_s' - быстрее, но менее точно.
MODEL_PACK_NAME = 'buffalo_l'

# Размер детекции. (640, 640) оптимально для CPU.
DET_SIZE = (640, 640)

# Порог косинусного сходства.
SIMILARITY_THRESHOLD = 0.55

# Источник видео:
# 0 - ваша веб-камера (для теста)
# 'http://192.168.1.100:8080/video' - IP камера
CAMERA_SOURCE = "http://192.168.0.178:8080/video"


# ==============================================================================
# МОДУЛЬ 1: IN-MEMORY ВЕКТОРНАЯ БАЗА ДАННЫХ (RAM ONLY)
# ==============================================================================
class VolatileVectorDB:
    def __init__(self):
        self.db = {}
        self.lock = threading.Lock()

    def find_closest(self, new_embedding):
        with self.lock:
            if not self.db:
                return None, 0.0

            # Принудительная нормализация входного вектора
            new_emb_norm = new_embedding / norm(new_embedding)

            best_score = -1.0
            best_id = None

            for uid, stored_emb in self.db.items():
                # Нормализация сохраненного вектора
                stored_emb_norm = stored_emb / norm(stored_emb)

                # Косинусное сходство
                score = np.dot(new_emb_norm, stored_emb_norm)

                if score > best_score:
                    best_score = score
                    best_id = uid

            return best_id, best_score

    def register_new_identity(self, embedding):
        new_uid = str(uuid.uuid4())
        with self.lock:
            self.db[new_uid] = embedding
        return new_uid


# ==============================================================================
# МОДУЛЬ 2: АСИНХРОННЫЙ ЗАХВАТ ВИДЕОПОТОКА
# ==============================================================================
class AsyncCameraStream:
    def __init__(self, source):
        self.source = source
        self.cap = cv2.VideoCapture(source)
        self.current_frame = None
        self.running = True
        self.lock = threading.Lock()

        if not self.cap.isOpened():
            raise ValueError(f"Не удалось открыть камеру: {source}")

        self.thread = threading.Thread(target=self._update, daemon=True)
        self.thread.start()

    def _update(self):
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                with self.lock:
                    self.current_frame = frame
            else:
                # Если поток прервался (особенно для IP камер), ждем перед реконнектом
                time.sleep(0.1)

    def get_latest_frame(self):
        with self.lock:
            return self.current_frame.copy() if self.current_frame is not None else None

    def stop(self):
        self.running = False
        self.thread.join()
        self.cap.release()


# ==============================================================================
# ГЛАВНЫЙ ИСПОЛНЯЕМЫЙ МОДУЛЬ
# ==============================================================================
def main():
    print(">>> Инициализация системы распознавания...")

    # Инициализация InsightFace
    # Пытаемся использовать GPU (CUDA), если нет - CPU.
    app = FaceAnalysis(name=MODEL_PACK_NAME,
                       providers=['CPUExecutionProvider'])

    # ctx_id=0 (GPU) или -1 (CPU). Оставляем 0, библиотека обычно сама делает fallback,
    # но если будет ошибка CUDA, поставьте ctx_id=-1
    app.prepare(ctx_id=0, det_size=DET_SIZE)

    vector_db = VolatileVectorDB()

    try:
        camera = AsyncCameraStream(CAMERA_SOURCE)
    except ValueError as e:
        print(f"Ошибка: {e}")
        return

    print(">>> Система готова. Нажмите 'q' для выхода.")

    try:
        while True:
            frame = camera.get_latest_frame()
            if frame is None:
                time.sleep(0.01)
                continue

            # Инференс
            t0 = time.time()
            faces = app.get(frame)
            inference_time = time.time() - t0

            display_img = frame.copy()

            for face in faces:
                embedding = face.embedding
                bbox = face.bbox.astype(int)
                # bbox - это массив [x1, y1, x2, y2]

                identity, score = vector_db.find_closest(embedding)

                if score > SIMILARITY_THRESHOLD:
                    label = f"ID: {identity[:4]}.. ({score:.2f})"
                    color = (0, 255, 0)
                else:
                    new_id = vector_db.register_new_identity(embedding)
                    label = f"NEW: {new_id[:4]}.."
                    color = (0, 0, 255)
                    print(f">>> New Identity: {new_id}")

                # Исправлена отрисовка: используем индексы массива bbox
                cv2.rectangle(display_img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
                cv2.putText(display_img, label, (bbox[0], bbox[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            fps = 1.0 / (inference_time + 0.001)
            stats = f"Inf: {inference_time * 1000:.0f}ms | FPS: {fps:.1f} | Faces: {len(faces)}"
            cv2.putText(display_img, stats, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

            cv2.imshow('Face Recognition', display_img)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        print(">>> Остановка...")
    finally:
        camera.stop()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
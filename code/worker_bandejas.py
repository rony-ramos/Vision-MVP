"""
worker_bandejas.py — Worker 1: Inspección de posición de bandejas (YOLO + Geometría).

Proceso independiente que captura video, detecta la bandeja usando YOLOv8 (Inteligencia Artificial)
y luego valida matemáticamente su ángulo de inclinación usando OpenCV (Geometría).

Ejecución: python worker_bandejas.py
"""

import sys
import time
import logging
import collections
import cv2
import numpy as np

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None
    print("CRÍTICO: Librería 'ultralytics' no instalada. Ejecuta 'pip install ultralytics'")

import config
import db
from hal import crear_actuador
from streamer import VideoStreamingServer

# =============================================
# Logging
# =============================================
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [BANDEJAS] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)


def preprocesar_y_detectar_contorno(roi: np.ndarray) -> tuple:
    """
    Aplica thresholding sobre el recorte de YOLO para encontrar el contorno matemático exacto.
    """
    gris = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gris, config.BANDEJA_BLUR_KERNEL, 0)
    thresh = cv2.adaptiveThreshold(
        blur, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        config.BANDEJA_THRESH_BLOCK_SIZE,
        config.BANDEJA_THRESH_C
    )

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None, 0

    # Nos quedamos con el contorno más grande dentro de la caja de YOLO
    mejor = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(mejor)
    
    # Filtro básico de ruido
    if area < 500:
        return None, 0
        
    return mejor, area


def evaluar_geometria(contorno: np.ndarray) -> dict:
    """
    Evalúa la rectitud usando el contorno detectado mediante una caja rotada (minAreaRect).
    """
    if contorno is None or len(contorno) < 5:
        return {
            'resultado': 'DEFECTO',
            'angulo': 0,
            'rect': None,
            'detalle': 'Contorno inválido o ruido'
        }

    # minAreaRect devuelve ( center (x,y), (width, height), angle of rotation )
    rect = cv2.minAreaRect(contorno)
    angle = rect[2]

    # Normalizar ángulo: OpenCV minAreaRect ángulo depende de las proporciones.
    # El ángulo suele estar en el rango [-90, 0)
    # Queremos saber la desviación respecto a 0 (recto).
    # Normalizamos a [-45, 45]
    if angle < -45:
        angle += 90
        
    desviacion = abs(angle)

    if desviacion > config.BANDEJA_MAX_ANGLE_TOLERANCE:
        resultado = 'DEFECTO'
        detalle = f"Inclinación: {desviacion:.1f}° > {config.BANDEJA_MAX_ANGLE_TOLERANCE}°"
    else:
        resultado = 'OK'
        detalle = f"Alineación correcta ({desviacion:.1f}°)"

    return {
        'resultado': resultado,
        'angulo': desviacion,
        'rect': rect,
        'detalle': detalle
    }


def dibujar_overlay(frame: np.ndarray, yolo_box: tuple, resultado: dict, contorno: np.ndarray) -> np.ndarray:
    """Dibuja la detección de YOLO y la matemática geométrica."""
    if yolo_box is None:
        cv2.putText(frame, "BUSCANDO BANDEJA...", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 165, 255), 2)
        return frame

    x1, y1, x2, y2 = yolo_box
    es_ok = resultado['resultado'] == 'OK'
    color = (0, 200, 0) if es_ok else (0, 0, 255)

    # 1. Dibujar la "Búsqueda" (Caja YOLO) en Naranja
    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 144, 30), 2)
    cv2.putText(frame, "YOLO Vision", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 144, 30), 1)

    # 2. Dibujar la "Medición" (Geometría)
    if contorno is not None and resultado.get('rect'):
        # Offset al contorno para dibujarlo en el frame original
        contorno_offset = contorno.copy()
        contorno_offset[:, :, 0] += x1
        contorno_offset[:, :, 1] += y1
        cv2.drawContours(frame, [contorno_offset], -1, (255, 255, 255), 1)

        # Dibujar la caja rotada (minAreaRect)
        rect = resultado['rect']
        # El rect está referenciado a la sub-imagen (ROI). Necesitamos sumarle el offset (x1, y1) al centro
        centro_x, centro_y = rect[0]
        rect_ajustado = ((centro_x + x1, centro_y + y1), rect[1], rect[2])
        
        box = cv2.boxPoints(rect_ajustado)
        box = np.intp(box)
        cv2.drawContours(frame, [box], 0, color, 3)

    label = f"{resultado['resultado']}: {resultado.get('angulo', 0):.1f} grados"
    cv2.putText(frame, label, (x1, y2 + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    return frame


class WorkerBandejas:
    def __init__(self):
        self.frame_count = 0
        self.historial = collections.deque(maxlen=config.DEQUE_MAXLEN)
        self.ultimo_estado_registrado = None
        self.actuador = crear_actuador()
        self.heartbeat_interval = 30
        self.cap = None
        self.streamer = VideoStreamingServer(config.STREAM_PORT_BANDEJAS)
        
        # Cargar modelo YOLO
        if YOLO is None:
            logger.error("No se puede iniciar el worker sin ultralytics.")
            sys.exit(1)
        logger.info(f"Cargando modelo YOLO: {config.BANDEJA_YOLO_MODEL}")
        self.yolo_model = YOLO(config.BANDEJA_YOLO_MODEL)

    def inicializar_camara(self):
        configs = db.obtener_config_sistema()
        idx = configs.get('cam_bandejas_index', config.CAM_BANDEJAS_INDEX)
        
        self.cap = cv2.VideoCapture(idx)
        if not self.cap.isOpened():
            logger.error(f"No se pudo abrir la cámara {idx}")
            sys.exit(1)
        logger.info(f"Cámara {idx} abierta. Iniciando pipeline híbrido YOLO+Geometría...")

    def _gestionar_alertas(self, resultado: dict, area: float):
        resultado_actual = resultado['resultado']
        if resultado_actual != self.ultimo_estado_registrado:
            if resultado_actual == 'DEFECTO':
                self.actuador.trigger(f"Bandeja chueca: {resultado['detalle']}")
                db.insertar_evento_calidad('DEFECTO', area, resultado['detalle'])
                logger.warning(f"Calidad DEFECTO: {resultado['detalle']}")
            else:
                db.insertar_evento_calidad('OK', area, resultado['detalle'])
                logger.info("Calidad OK.")
            self.ultimo_estado_registrado = resultado_actual

    def procesar_frame(self, frame) -> bool:
        self.frame_count += 1
        frame_con_overlay = frame.copy()
        
        # ETAPA 1: Búsqueda con IA (YOLO)
        # Filtramos por la clase proxy (bajamos la confianza a 0.15 para objetos difíciles)
        results = self.yolo_model.predict(frame, classes=[config.BANDEJA_YOLO_CLASS], conf=0.15, verbose=False)
        
        yolo_box = None
        contorno = None
        area = 0
        resultado = {
            'resultado': 'DEFECTO',
            'angulo': 0,
            'detalle': 'No se detectó el objeto (YOLO)'
        }

        if len(results) > 0 and len(results[0].boxes) > 0:
            # Tomamos la detección con mayor confianza
            box = results[0].boxes[0].xyxy[0].cpu().numpy().astype(int)
            x1, y1, x2, y2 = box
            
            # Asegurar límites dentro del frame
            h_f, w_f = frame.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w_f, x2), min(h_f, y2)
            yolo_box = (x1, y1, x2, y2)
            
            roi = frame[y1:y2, x1:x2]
            
            if roi.shape[0] > 10 and roi.shape[1] > 10:
                # ETAPA 2: Medición con Geometría (OpenCV)
                contorno, area = preprocesar_y_detectar_contorno(roi)
                resultado = evaluar_geometria(contorno)
                
                self.historial.append(resultado['resultado'])
                self._gestionar_alertas(resultado, area)

        if self.frame_count % self.heartbeat_interval == 0:
            db.actualizar_heartbeat("worker_bandejas")

        frame_con_overlay = dibujar_overlay(frame_con_overlay, yolo_box, resultado, contorno)
        self.streamer.set_frame(frame_con_overlay)

        if config.DEBUG_MODE:
            cv2.imshow("Vision-MVP: Inspeccion Hibrida (YOLO+Geometria)", frame_con_overlay)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                return False

        return True

    def run_loop(self):
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    time.sleep(0.5)
                    continue

                if not self.procesar_frame(frame):
                    break
        except KeyboardInterrupt:
            logger.info("Worker detenido")
        except Exception as e:
            logger.exception(f"Error crítico: {e}")
            raise
        finally:
            if self.cap: self.cap.release()
            cv2.destroyAllWindows()
            self.streamer.stop()
            self.actuador.cleanup()
            logger.info("Worker finalizado.")

def run():
    db.init_db()
    worker = WorkerBandejas()
    worker.inicializar_camara()
    worker.run_loop()

if __name__ == "__main__":
    run()

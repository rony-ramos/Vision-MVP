"""
worker_bandejas.py — Worker 1: Inspección de posición de bandejas.

Proceso independiente que captura video, detecta bandejas via contornos OpenCV
y valida que estén correctamente posicionadas dentro de la ROI.

Pipeline:
  1. Captura → cv2.VideoCapture(CAM_BANDEJAS_INDEX)
  2. ROI → Recorte a la región de interés
  3. Preprocesamiento → Gris + Blur + Threshold adaptativo
  4. Detección → cv2.findContours()
  5. Validación → Área + Centrado dentro de la ROI
  6. Acción → Si DEFECTO: actuador.trigger() + log a SQLite

Ejecución: python worker_bandejas.py
           o via watchdog: watchdog.bat worker_bandejas.py
"""

import sys
import time
import logging
import collections
import cv2
import numpy as np

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


def preprocesar_roi(frame: np.ndarray) -> tuple:
    """
    Extrae la ROI del frame y la preprocesa para detección de contornos.
    """
    x, y, w, h = config.BANDEJA_ROI
    roi = frame[y:y+h, x:x+w]

    gris = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gris, config.BANDEJA_BLUR_KERNEL, 0)
    thresh = cv2.adaptiveThreshold(
        blur, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        config.BANDEJA_THRESH_BLOCK_SIZE,
        config.BANDEJA_THRESH_C
    )
    return roi, thresh


def detectar_bandeja(thresh: np.ndarray) -> tuple:
    """
    Encuentra el contorno principal que podría ser una bandeja.
    """
    contours, _ = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        return None, 0

    candidatos = [
        c for c in contours
        if config.MIN_CONTOUR_AREA <= cv2.contourArea(c) <= config.MAX_CONTOUR_AREA
    ]

    if not candidatos:
        return None, 0

    mejor = max(candidatos, key=cv2.contourArea)
    return mejor, cv2.contourArea(mejor)


def evaluar_posicion(contorno: np.ndarray, roi_shape: tuple) -> dict:
    """
    Evalúa si la bandeja está correctamente posicionada en la ROI.
    """
    h_roi, w_roi = roi_shape[:2]
    area_roi = h_roi * w_roi
    area_contorno = cv2.contourArea(contorno)
    cobertura = area_contorno / area_roi

    M = cv2.moments(contorno)
    if M["m00"] == 0:
        return {
            'resultado': 'DEFECTO',
            'cobertura': 0,
            'desviacion_centro': float('inf'),
            'detalle': 'Contorno sin masa detectada'
        }

    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])
    centro_roi = (w_roi // 2, h_roi // 2)
    desviacion = np.sqrt((cx - centro_roi[0])**2 + (cy - centro_roi[1])**2)

    posicion_ok = (
        cobertura >= config.BANDEJA_COBERTURA_MIN and
        desviacion <= config.BANDEJA_CENTRO_TOLERANCIA
    )

    detalle_parts = []
    if cobertura < config.BANDEJA_COBERTURA_MIN:
        detalle_parts.append(f"Cobertura baja: {cobertura:.1%} < {config.BANDEJA_COBERTURA_MIN:.0%}")
    if desviacion > config.BANDEJA_CENTRO_TOLERANCIA:
        detalle_parts.append(f"Descentrada: {desviacion:.0f}px > {config.BANDEJA_CENTRO_TOLERANCIA}px")

    return {
        'resultado': 'OK' if posicion_ok else 'DEFECTO',
        'cobertura': cobertura,
        'desviacion_centro': desviacion,
        'detalle': '; '.join(detalle_parts) if detalle_parts else 'Posición correcta'
    }


def dibujar_overlay(frame: np.ndarray, resultado: dict, contorno: np.ndarray, area: float) -> np.ndarray:
    """Dibuja la ROI, contorno detectado y estado en el frame."""
    x, y, w, h = config.BANDEJA_ROI
    es_ok = resultado['resultado'] == 'OK'
    color = (0, 200, 0) if es_ok else (0, 0, 255)

    cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)

    if contorno is not None:
        contorno_offset = contorno.copy()
        contorno_offset[:, :, 0] += x
        contorno_offset[:, :, 1] += y
        cv2.drawContours(frame, [contorno_offset], -1, color, 2)

    label = f"BANDEJA: {resultado['resultado']}"
    cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    info = f"Area: {area:.0f}px | Cob: {resultado['cobertura']:.0%}"
    cv2.putText(frame, info, (x, y + h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    return frame


class WorkerBandejas:
    """Clase principal para encapsular el estado y ejecución de la inspección de bandejas."""
    
    def __init__(self):
        self.frame_count = 0
        self.historial = collections.deque(maxlen=config.DEQUE_MAXLEN)
        self.ultimo_estado_registrado = None
        self.actuador = crear_actuador()
        self.heartbeat_interval = 30
        self.cap = None
        self.streamer = VideoStreamingServer(config.STREAM_PORT_BANDEJAS)

    def inicializar_camara(self):
        """Prepara la captura de video."""
        # Leer configuración dinámica desde la DB
        configs = db.obtener_config_camaras()
        idx = configs.get('cam_bandejas_index', config.CAM_BANDEJAS_INDEX)
        
        self.cap = cv2.VideoCapture(idx)
        if not self.cap.isOpened():
            logger.error(f"No se pudo abrir la cámara {idx}")
            sys.exit(1)
        logger.info(f"Cámara {idx} abierta. Procesando frames...")

    def _gestionar_alertas(self, resultado: dict, area: float):
        """Maneja los cambios de estado y registros en base de datos."""
        resultado_actual = resultado['resultado']
        if resultado_actual != self.ultimo_estado_registrado:
            if resultado_actual == 'DEFECTO':
                self.actuador.trigger(f"Bandeja mal posicionada: {resultado['detalle']}")
                db.insertar_evento_calidad('DEFECTO', area, resultado['detalle'])
                logger.warning(f"Bandeja DEFECTO: {resultado['detalle']}")
            else:
                db.insertar_evento_calidad('OK', area, resultado['detalle'])
                logger.info("Bandeja OK.")
            self.ultimo_estado_registrado = resultado_actual

    def procesar_frame(self, frame) -> bool:
        """Procesa un frame individual de cámara."""
        self.frame_count += 1
        roi, thresh = preprocesar_roi(frame)
        contorno, area = detectar_bandeja(thresh)

        if contorno is not None:
            resultado = evaluar_posicion(contorno, roi.shape)
        else:
            resultado = {
                'resultado': 'DEFECTO',
                'cobertura': 0,
                'desviacion_centro': float('inf'),
                'detalle': 'No se detectó bandeja en la ROI'
            }

        self.historial.append(resultado['resultado'])
        self._gestionar_alertas(resultado, area)

        if self.frame_count % self.heartbeat_interval == 0:
            db.actualizar_heartbeat("worker_bandejas")

        # Siempre enviamos el frame con overlay al streamer
        frame_con_overlay = frame.copy()
        frame_con_overlay = dibujar_overlay(frame_con_overlay, resultado, contorno, area)
        self.streamer.set_frame(frame_con_overlay)

        if config.DEBUG_MODE:
            cv2.imshow("Vision-MVP: Inspeccion de Bandejas", frame_con_overlay)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                logger.info("Salida solicitada por usuario (tecla 'q')")
                return False

        return True

    def run_loop(self):
        """Bucle infinito de procesamiento de video."""
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    logger.warning("Frame no capturado, reintentando...")
                    time.sleep(0.5)
                    continue

                if not self.procesar_frame(frame):
                    break
        except KeyboardInterrupt:
            logger.info("Worker detenido por Ctrl+C")
        except Exception as e:
            logger.exception(f"Error crítico: {e}")
            raise
        finally:
            if self.cap:
                self.cap.release()
            cv2.destroyAllWindows()
            self.streamer.stop()
            self.actuador.cleanup()
            logger.info("Recursos liberados. Worker finalizado.")


def run():
    """Punto de entrada."""
    logger.info("Iniciando Worker de Bandejas...")
    db.init_db()
    configs = db.obtener_config_camaras()
    idx = configs.get('cam_bandejas_index', config.CAM_BANDEJAS_INDEX)
    logger.info(f"Cámara: index={idx}")
    logger.info(f"ROI: {config.BANDEJA_ROI}")

    worker = WorkerBandejas()
    worker.inicializar_camara()
    worker.run_loop()


if __name__ == "__main__":
    run()

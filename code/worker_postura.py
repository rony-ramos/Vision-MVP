"""
worker_postura.py — Worker 2: Monitoreo ergonómico del operario.

Proceso independiente que captura video degradado a 5-7 FPS, detecta pose
con MediaPipe y evalúa ángulos de espalda y cuello contra umbrales
parametrizables.

Pipeline:
  1. Captura degradada → 640x480, time.sleep(0.15)
  2. Detección → MediaPipe Pose (Lightweight)
  3. Cálculo angular → SHOULDER-HIP-KNEE (espalda), EAR-SHOULDER-HIP (cuello)
  4. Evaluación temporal → N frames consecutivos en alerta → registro
  5. Acción → Si alerta sostenida: actuador.trigger() + log a SQLite

Ejecución: python worker_postura.py
           o via watchdog: watchdog.bat worker_postura.py
"""

import sys
import time
import logging
import collections

import cv2
import numpy as np
import mediapipe as mp

import config
import db
from hal import crear_actuador

# =============================================
# Logging
# =============================================
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [POSTURA] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)

# =============================================
# MediaPipe setup
# =============================================
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles


def calcular_angulo(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """
    Calcula el ángulo en el punto b formado por los segmentos ba y bc.

    Args:
        a, b, c: Arrays de coordenadas [x, y] de los 3 puntos.

    Returns:
        Ángulo en grados (0-180).
    """
    ba = np.array(a) - np.array(b)
    bc = np.array(c) - np.array(b)
    cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    return float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))


def obtener_landmark(landmarks, idx, w: int, h: int) -> tuple | None:
    """
    Extrae coordenadas de un landmark de MediaPipe.

    Returns:
        (x_px, y_px, visibility) o None si visibilidad es muy baja.
    """
    lm = landmarks[idx]
    if lm.visibility < 0.3:
        return None
    return (int(lm.x * w), int(lm.y * h), lm.visibility)


def evaluar_postura(landmarks, frame_shape: tuple) -> dict:
    """
    Evalúa la postura del operario basándose en ángulos articulares.

    Returns:
        dict con 'alerta', 'angulo_espalda', 'angulo_cuello', 'detalle'
    """
    h, w = frame_shape[:2]
    pose_lm = mp_pose.PoseLandmark

    puntos = {}
    nombres = {
        'ear_l': pose_lm.LEFT_EAR, 'ear_r': pose_lm.RIGHT_EAR,
        'shoulder_l': pose_lm.LEFT_SHOULDER, 'shoulder_r': pose_lm.RIGHT_SHOULDER,
        'hip_l': pose_lm.LEFT_HIP, 'hip_r': pose_lm.RIGHT_HIP,
        'knee_l': pose_lm.LEFT_KNEE, 'knee_r': pose_lm.RIGHT_KNEE,
    }

    for nombre, idx in nombres.items():
        pt = obtener_landmark(landmarks, idx, w, h)
        if pt is not None:
            puntos[nombre] = np.array(pt[:2])

    angulos_espalda = []
    angulos_cuello = []

    if all(k in puntos for k in ['shoulder_l', 'hip_l', 'knee_l']):
        angulos_espalda.append(calcular_angulo(puntos['shoulder_l'], puntos['hip_l'], puntos['knee_l']))
    if all(k in puntos for k in ['ear_l', 'shoulder_l', 'hip_l']):
        angulos_cuello.append(calcular_angulo(puntos['ear_l'], puntos['shoulder_l'], puntos['hip_l']))

    if all(k in puntos for k in ['shoulder_r', 'hip_r', 'knee_r']):
        angulos_espalda.append(calcular_angulo(puntos['shoulder_r'], puntos['hip_r'], puntos['knee_r']))
    if all(k in puntos for k in ['ear_r', 'shoulder_r', 'hip_r']):
        angulos_cuello.append(calcular_angulo(puntos['ear_r'], puntos['shoulder_r'], puntos['hip_r']))

    if not angulos_espalda and not angulos_cuello:
        return {
            'alerta': False,
            'angulo_espalda': None,
            'angulo_cuello': None,
            'detalle': 'Visibilidad de landmarks insuficiente para evaluar'
        }

    desviacion_espalda = min([abs(180 - a) for a in angulos_espalda]) if angulos_espalda else 0
    desviacion_cuello = min([abs(180 - a) for a in angulos_cuello]) if angulos_cuello else 0

    alerta_espalda = desviacion_espalda > config.MAX_BACK_INCLINATION
    alerta_cuello = desviacion_cuello > config.MAX_NECK_FLEXION

    detalle_parts = []
    if alerta_espalda:
        detalle_parts.append(f"Espalda inclinada: {desviacion_espalda:.1f}° > {config.MAX_BACK_INCLINATION}°")
    if alerta_cuello:
        detalle_parts.append(f"Cuello flexionado: {desviacion_cuello:.1f}° > {config.MAX_NECK_FLEXION}°")

    return {
        'alerta': alerta_espalda or alerta_cuello,
        'angulo_espalda': round(desviacion_espalda, 1),
        'angulo_cuello': round(desviacion_cuello, 1),
        'detalle': '; '.join(detalle_parts) if detalle_parts else 'Postura correcta'
    }


def dibujar_overlay(frame: np.ndarray, results, evaluacion: dict) -> np.ndarray:
    """Dibuja la pose detectada y el estado ergonómico."""
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            frame,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
        )

    es_alerta = evaluacion.get('alerta', False)
    color = (0, 0, 255) if es_alerta else (0, 200, 0)
    estado = "⚠ ALERTA ERGONOMICA" if es_alerta else "✓ Postura OK"

    cv2.putText(frame, estado, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    ang_e = evaluacion.get('angulo_espalda')
    ang_c = evaluacion.get('angulo_cuello')
    if ang_e is not None:
        cv2.putText(frame, f"Espalda: {ang_e:.1f} deg", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    if ang_c is not None:
        cv2.putText(frame, f"Cuello: {ang_c:.1f} deg", (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    cv2.putText(frame, evaluacion.get('detalle', ''), (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
    return frame


class WorkerPostura:
    """Clase principal para encapsular el estado y ejecución del monitoreo ergonómico."""
    
    def __init__(self):
        self.frames_en_alerta = 0
        self.ultimo_estado_registrado = None
        self.frame_count = 0
        self.historial = collections.deque(maxlen=config.DEQUE_MAXLEN)
        self.actuador = crear_actuador()
        self.heartbeat_interval = 20
        self.pose = mp_pose.Pose(
            min_detection_confidence=config.POSTURA_MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=config.POSTURA_MIN_TRACKING_CONFIDENCE,
            model_complexity=0
        )
        self.cap = None

    def inicializar_camara(self):
        """Prepara la captura de video."""
        # Leer configuración dinámica desde la DB
        configs = db.obtener_config_camaras()
        idx = configs.get('cam_postura_index', config.CAM_POSTURA_INDEX)

        self.cap = cv2.VideoCapture(idx)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.POSTURA_RESOLUTION[0])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.POSTURA_RESOLUTION[1])

        if not self.cap.isOpened():
            logger.error(f"No se pudo abrir la cámara {idx}")
            sys.exit(1)
        logger.info(f"Cámara {idx} abierta. Procesando poses...")

    def _gestionar_alertas(self, evaluacion: dict):
        """Maneja los cambios de estado y registros en base de datos."""
        if evaluacion['alerta']:
            self.frames_en_alerta += 1
        else:
            self.frames_en_alerta = 0

        if self.frames_en_alerta >= config.POSTURA_FRAMES_ALERTA:
            estado_consolidado = True
        elif self.frames_en_alerta == 0:
            estado_consolidado = False
        else:
            estado_consolidado = self.ultimo_estado_registrado if self.ultimo_estado_registrado is not None else False

        if estado_consolidado != self.ultimo_estado_registrado:
            if estado_consolidado:
                self.actuador.trigger(f"Postura riesgosa sostenida: {evaluacion['detalle']}")
                db.insertar_evento_postura(True, evaluacion['angulo_espalda'], evaluacion['angulo_cuello'], evaluacion['detalle'])
                logger.warning(f"Alerta ergonómica registrada: {evaluacion['detalle']}")
            else:
                db.insertar_evento_postura(False, evaluacion['angulo_espalda'], evaluacion['angulo_cuello'], 'Postura correcta')
                logger.info("Postura en estado OK.")
            self.ultimo_estado_registrado = estado_consolidado

    def _renderizar(self, frame, results, evaluacion) -> bool:
        """Visualiza los resultados en modo debug o aplica delay intencional."""
        if config.DEBUG_MODE:
            frame = dibujar_overlay(frame, results, evaluacion)
            cv2.imshow("Vision-MVP: Monitoreo Ergonomico", frame)
            delay_ms = int(config.POSTURA_FPS_DELAY * 1000)
            if cv2.waitKey(delay_ms) & 0xFF == ord('q'):
                logger.info("Salida solicitada por usuario (tecla 'q')")
                return False
        else:
            time.sleep(config.POSTURA_FPS_DELAY)
        return True

    def procesar_frame(self, frame) -> bool:
        """Procesa un frame individual de cámara."""
        self.frame_count += 1
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_rgb.flags.writeable = False
        results = self.pose.process(frame_rgb)
        frame_rgb.flags.writeable = True

        if results.pose_landmarks:
            evaluacion = evaluar_postura(results.pose_landmarks.landmark, frame.shape)
        else:
            evaluacion = {
                'alerta': False,
                'angulo_espalda': None,
                'angulo_cuello': None,
                'detalle': 'Persona no detectada'
            }

        self._gestionar_alertas(evaluacion)
        self.historial.append(evaluacion['alerta'])

        if self.frame_count % self.heartbeat_interval == 0:
            db.actualizar_heartbeat("worker_postura")

        return self._renderizar(frame, results, evaluacion)

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
            self.pose.close()
            self.actuador.cleanup()
            logger.info("Recursos liberados. Worker finalizado.")


def run():
    """Punto de entrada."""
    logger.info("Iniciando Worker de Postura Ergonómica...")
    db.init_db()
    configs = db.obtener_config_camaras()
    idx = configs.get('cam_postura_index', config.CAM_POSTURA_INDEX)

    logger.info(f"Cámara: index={idx}")
    logger.info(f"Resolución: {config.POSTURA_RESOLUTION}")
    logger.info(f"FPS target: ~{1/config.POSTURA_FPS_DELAY:.0f} FPS")
    logger.info(f"Umbrales: espalda={config.MAX_BACK_INCLINATION}°, cuello={config.MAX_NECK_FLEXION}°")

    worker = WorkerPostura()
    worker.inicializar_camara()
    worker.run_loop()


if __name__ == "__main__":
    run()

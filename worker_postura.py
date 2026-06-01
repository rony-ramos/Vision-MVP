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

# ─────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [POSTURA] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# MediaPipe setup
# ─────────────────────────────────────────────
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


def obtener_landmark(landmarks, idx, w: int, h: int) -> tuple:
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

    Ángulos calculados:
    - Espalda: Desviación lateral del tronco (SHOULDER → HIP → KNEE)
      Un ángulo bajo indica inclinación excesiva.
    - Cuello: Flexión cervical (EAR → SHOULDER → HIP)
      Un ángulo bajo indica flexión excesiva.

    Los ángulos se evalúan bilateralmente (izq + der) y se toma el peor caso.

    Returns:
        dict con 'alerta', 'angulo_espalda', 'angulo_cuello', 'detalle'
    """
    h, w = frame_shape[:2]
    PoseLM = mp_pose.PoseLandmark

    # Extraer landmarks relevantes (lado izquierdo y derecho)
    puntos = {}
    nombres = {
        'ear_l': PoseLM.LEFT_EAR, 'ear_r': PoseLM.RIGHT_EAR,
        'shoulder_l': PoseLM.LEFT_SHOULDER, 'shoulder_r': PoseLM.RIGHT_SHOULDER,
        'hip_l': PoseLM.LEFT_HIP, 'hip_r': PoseLM.RIGHT_HIP,
        'knee_l': PoseLM.LEFT_KNEE, 'knee_r': PoseLM.RIGHT_KNEE,
    }

    for nombre, idx in nombres.items():
        pt = obtener_landmark(landmarks, idx, w, h)
        if pt is not None:
            puntos[nombre] = np.array(pt[:2])

    angulos_espalda = []
    angulos_cuello = []

    # Validar lado izquierdo
    if all(k in puntos for k in ['shoulder_l', 'hip_l', 'knee_l']):
        angulos_espalda.append(calcular_angulo(puntos['shoulder_l'], puntos['hip_l'], puntos['knee_l']))
    if all(k in puntos for k in ['ear_l', 'shoulder_l', 'hip_l']):
        angulos_cuello.append(calcular_angulo(puntos['ear_l'], puntos['shoulder_l'], puntos['hip_l']))

    # Validar lado derecho
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

    # Calcula desviacion con los ángulos disponibles (180 - ángulo)
    desviacion_espalda = min([abs(180 - a) for a in angulos_espalda]) if angulos_espalda else 0
    desviacion_cuello = min([abs(180 - a) for a in angulos_cuello]) if angulos_cuello else 0

    # Evaluar contra umbrales
    alerta_espalda = desviacion_espalda > config.MAX_BACK_INCLINATION
    alerta_cuello = desviacion_cuello > config.MAX_NECK_FLEXION

    detalle_parts = []
    if alerta_espalda:
        detalle_parts.append(
            f"Espalda inclinada: {desviacion_espalda:.1f}° > {config.MAX_BACK_INCLINATION}°"
        )
    if alerta_cuello:
        detalle_parts.append(
            f"Cuello flexionado: {desviacion_cuello:.1f}° > {config.MAX_NECK_FLEXION}°"
        )

    return {
        'alerta': alerta_espalda or alerta_cuello,
        'angulo_espalda': round(desviacion_espalda, 1),
        'angulo_cuello': round(desviacion_cuello, 1),
        'detalle': '; '.join(detalle_parts) if detalle_parts else 'Postura correcta'
    }


def dibujar_overlay(frame: np.ndarray, results, evaluacion: dict) -> np.ndarray:
    """Dibuja la pose detectada y el estado ergonómico."""

    # Dibujar esqueleto de MediaPipe
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            frame,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
        )

    # Estado
    es_alerta = evaluacion.get('alerta', False)
    color = (0, 0, 255) if es_alerta else (0, 200, 0)
    estado = "⚠ ALERTA ERGONOMICA" if es_alerta else "✓ Postura OK"

    cv2.putText(frame, estado, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    # Ángulos
    ang_e = evaluacion.get('angulo_espalda')
    ang_c = evaluacion.get('angulo_cuello')
    if ang_e is not None:
        cv2.putText(frame, f"Espalda: {ang_e:.1f} deg", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    if ang_c is not None:
        cv2.putText(frame, f"Cuello: {ang_c:.1f} deg", (10, 85),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # Detalle
    cv2.putText(frame, evaluacion.get('detalle', ''), (10, 110),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

    return frame


def run():
    """Bucle principal del worker de postura."""
    logger.info("Iniciando Worker de Postura Ergonómica...")
    logger.info(f"Cámara: index={config.CAM_POSTURA_INDEX}")
    logger.info(f"Resolución: {config.POSTURA_RESOLUTION}")
    logger.info(f"FPS target: ~{1/config.POSTURA_FPS_DELAY:.0f} FPS")
    logger.info(
        f"Umbrales: espalda={config.MAX_BACK_INCLINATION}°, "
        f"cuello={config.MAX_NECK_FLEXION}°"
    )

    # Inicializar DB
    db.init_db()

    # Crear actuador
    actuador = crear_actuador()

    # Historial con límite de memoria
    historial = collections.deque(maxlen=config.DEQUE_MAXLEN)

    # Contador de frames consecutivos en alerta
    frames_en_alerta = 0
    alerta_registrada = False  # Evitar spam de alertas

    # Abrir cámara con resolución reducida
    cap = cv2.VideoCapture(config.CAM_POSTURA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.POSTURA_RESOLUTION[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.POSTURA_RESOLUTION[1])

    if not cap.isOpened():
        logger.error(
            f"No se pudo abrir la cámara {config.CAM_POSTURA_INDEX}"
        )
        sys.exit(1)

    logger.info("Cámara abierta. Procesando poses...")
    frame_count = 0
    heartbeat_interval = 20  # Cada ~3s a 7 FPS

    pose = mp_pose.Pose(
        min_detection_confidence=config.POSTURA_MIN_DETECTION_CONFIDENCE,
        min_tracking_confidence=config.POSTURA_MIN_TRACKING_CONFIDENCE,
        model_complexity=0  # Lightweight para performance
    )

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                logger.warning("Frame no capturado, reintentando...")
                time.sleep(0.5)
                continue

            frame_count += 1

            # 1. Convertir BGR → RGB para MediaPipe
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_rgb.flags.writeable = False  # Optimización de performance

            # 2. Procesar pose
            results = pose.process(frame_rgb)
            frame_rgb.flags.writeable = True

            # 3. Evaluar postura
            if results.pose_landmarks:
                evaluacion = evaluar_postura(
                    results.pose_landmarks.landmark,
                    frame.shape
                )
            else:
                evaluacion = {
                    'alerta': False,
                    'angulo_espalda': None,
                    'angulo_cuello': None,
                    'detalle': 'Persona no detectada'
                }

            # 4. Lógica de alerta temporal (N frames consecutivos)
            if evaluacion['alerta']:
                frames_en_alerta += 1
            else:
                frames_en_alerta = 0
                alerta_registrada = False

            # 5. Acción si alerta sostenida
            if (frames_en_alerta >= config.POSTURA_FRAMES_ALERTA
                    and not alerta_registrada):
                actuador.trigger(
                    f"Postura riesgosa sostenida: {evaluacion['detalle']}"
                )
                db.insertar_evento_postura(
                    alerta=True,
                    angulo_espalda=evaluacion['angulo_espalda'],
                    angulo_cuello=evaluacion['angulo_cuello'],
                    detalle=evaluacion['detalle']
                )
                alerta_registrada = True
                logger.warning(
                    f"Alerta ergonómica registrada: {evaluacion['detalle']}"
                )

            # Registrar OK periódicamente
            if frame_count % 50 == 0 and not evaluacion['alerta']:
                db.insertar_evento_postura(
                    alerta=False,
                    angulo_espalda=evaluacion['angulo_espalda'],
                    angulo_cuello=evaluacion['angulo_cuello'],
                    detalle=evaluacion['detalle']
                )

            # 6. Historial (memoria acotada)
            historial.append(evaluacion['alerta'])

            # 7. Heartbeat
            if frame_count % heartbeat_interval == 0:
                db.actualizar_heartbeat("worker_postura")

            # 8. Visualización (Sólo en DEBUG_MODE)
            if config.DEBUG_MODE:
                frame = dibujar_overlay(frame, results, evaluacion)
                cv2.imshow("Vision-MVP: Monitoreo Ergonomico", frame)

                # Salir con 'q' y aplicar throttle de forma nativa con OpenCV
                delay_ms = int(config.POSTURA_FPS_DELAY * 1000)
                if cv2.waitKey(delay_ms) & 0xFF == ord('q'):
                    logger.info("Salida solicitada por usuario (tecla 'q')")
                    break
            else:
                # 9. Throttle intencional (~5-7 FPS) cuando no hay GUI
                time.sleep(config.POSTURA_FPS_DELAY)

    except KeyboardInterrupt:
        logger.info("Worker detenido por Ctrl+C")
    except Exception as e:
        logger.error(f"Error crítico: {e}", exc_info=True)
        raise
    finally:
        cap.release()
        cv2.destroyAllWindows()
        pose.close()
        actuador.cleanup()
        logger.info("Recursos liberados. Worker finalizado.")


if __name__ == "__main__":
    run()

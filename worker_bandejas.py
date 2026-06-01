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

# ─────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [BANDEJAS] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)


def preprocesar_roi(frame: np.ndarray) -> tuple:
    """
    Extrae la ROI del frame y la preprocesa para detección de contornos.

    Returns:
        (roi_original, roi_procesada) — ROI en color y binarizada
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

    Returns:
        (contorno, area) o (None, 0) si no se detecta
    """
    contours, _ = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        return None, 0

    # Filtrar por área válida
    candidatos = [
        c for c in contours
        if config.MIN_CONTOUR_AREA <= cv2.contourArea(c) <= config.MAX_CONTOUR_AREA
    ]

    if not candidatos:
        return None, 0

    # Tomar el contorno más grande como candidato principal
    mejor = max(candidatos, key=cv2.contourArea)
    return mejor, cv2.contourArea(mejor)


def evaluar_posicion(contorno: np.ndarray, roi_shape: tuple) -> dict:
    """
    Evalúa si la bandeja está correctamente posicionada en la ROI.

    Criterios:
    - Cobertura: El contorno debe cubrir al menos BANDEJA_COBERTURA_MIN de la ROI
    - Centrado: El centroide del contorno debe estar cerca del centro de la ROI

    Returns:
        dict con 'resultado', 'cobertura', 'desviacion_centro', 'detalle'
    """
    h_roi, w_roi = roi_shape[:2]
    area_roi = h_roi * w_roi
    area_contorno = cv2.contourArea(contorno)

    # Cobertura
    cobertura = area_contorno / area_roi

    # Centrado (centroide del contorno vs centro de la ROI)
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

    # Evaluación
    posicion_ok = (
        cobertura >= config.BANDEJA_COBERTURA_MIN and
        desviacion <= config.BANDEJA_CENTRO_TOLERANCIA
    )

    detalle_parts = []
    if cobertura < config.BANDEJA_COBERTURA_MIN:
        detalle_parts.append(
            f"Cobertura baja: {cobertura:.1%} < {config.BANDEJA_COBERTURA_MIN:.0%}"
        )
    if desviacion > config.BANDEJA_CENTRO_TOLERANCIA:
        detalle_parts.append(
            f"Descentrada: {desviacion:.0f}px > {config.BANDEJA_CENTRO_TOLERANCIA}px"
        )

    return {
        'resultado': 'OK' if posicion_ok else 'DEFECTO',
        'cobertura': cobertura,
        'desviacion_centro': desviacion,
        'detalle': '; '.join(detalle_parts) if detalle_parts else 'Posición correcta'
    }


def dibujar_overlay(frame: np.ndarray, resultado: dict,
                    contorno: np.ndarray, area: float) -> np.ndarray:
    """Dibuja la ROI, contorno detectado y estado en el frame."""
    x, y, w, h = config.BANDEJA_ROI
    es_ok = resultado['resultado'] == 'OK'
    color = (0, 200, 0) if es_ok else (0, 0, 255)

    # Dibujar ROI
    cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)

    # Dibujar contorno detectado (offset por la posición de la ROI)
    if contorno is not None:
        contorno_offset = contorno.copy()
        contorno_offset[:, :, 0] += x
        contorno_offset[:, :, 1] += y
        cv2.drawContours(frame, [contorno_offset], -1, color, 2)

    # Texto de estado
    label = f"BANDEJA: {resultado['resultado']}"
    cv2.putText(frame, label, (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    # Info adicional
    info = f"Area: {area:.0f}px | Cob: {resultado['cobertura']:.0%}"
    cv2.putText(frame, info, (x, y + h + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    return frame


def run():
    """Bucle principal del worker de bandejas."""
    logger.info("Iniciando Worker de Bandejas...")
    logger.info(f"Cámara: index={config.CAM_BANDEJAS_INDEX}")
    logger.info(f"ROI: {config.BANDEJA_ROI}")

    # Inicializar DB
    db.init_db()

    # Crear actuador
    actuador = crear_actuador()

    # Historial con límite de memoria
    historial = collections.deque(maxlen=config.DEQUE_MAXLEN)

    # Abrir cámara
    cap = cv2.VideoCapture(config.CAM_BANDEJAS_INDEX)
    if not cap.isOpened():
        logger.error(
            f"No se pudo abrir la cámara {config.CAM_BANDEJAS_INDEX}"
        )
        sys.exit(1)

    logger.info("Cámara abierta. Procesando frames...")
    frame_count = 0
    heartbeat_interval = 30  # Actualizar heartbeat cada N frames

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                logger.warning("Frame no capturado, reintentando...")
                time.sleep(0.5)
                continue

            frame_count += 1

            # 1. Preprocesar ROI
            roi, thresh = preprocesar_roi(frame)

            # 2. Detectar bandeja
            contorno, area = detectar_bandeja(thresh)

            # 3. Evaluar posición
            if contorno is not None:
                resultado = evaluar_posicion(contorno, roi.shape)
            else:
                resultado = {
                    'resultado': 'DEFECTO',
                    'cobertura': 0,
                    'desviacion_centro': float('inf'),
                    'detalle': 'No se detectó bandeja en la ROI'
                }

            # 4. Registrar en historial (memoria acotada)
            historial.append(resultado['resultado'])

            # 5. Acción si defecto
            if resultado['resultado'] == 'DEFECTO':
                actuador.trigger(f"Bandeja mal posicionada: {resultado['detalle']}")
                db.insertar_evento_calidad(
                    resultado='DEFECTO',
                    area=area,
                    detalle=resultado['detalle']
                )
            else:
                # Registrar OK periódicamente (cada 30 frames para no saturar DB)
                if frame_count % 30 == 0:
                    db.insertar_evento_calidad(
                        resultado='OK',
                        area=area,
                        detalle=resultado['detalle']
                    )

            # 6. Heartbeat
            if frame_count % heartbeat_interval == 0:
                db.actualizar_heartbeat("worker_bandejas")

            # 7. Visualización (Sólo en DEBUG_MODE)
            if config.DEBUG_MODE:
                frame = dibujar_overlay(frame, resultado, contorno, area)
                cv2.imshow("Vision-MVP: Inspeccion de Bandejas", frame)
                # Salir con 'q'
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    logger.info("Salida solicitada por usuario (tecla 'q')")
                    break

    except KeyboardInterrupt:
        logger.info("Worker detenido por Ctrl+C")
    except Exception as e:
        logger.error(f"Error crítico: {e}", exc_info=True)
        raise
    finally:
        cap.release()
        cv2.destroyAllWindows()
        actuador.cleanup()
        logger.info("Recursos liberados. Worker finalizado.")


if __name__ == "__main__":
    run()

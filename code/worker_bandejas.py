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
    from ultralytics import YOLOWorld
    import torch
except ImportError:
    YOLOWorld = None
    torch = None
    print("CRÍTICO: Librería 'ultralytics' o 'torch' no instalada. Ejecuta 'pip install ultralytics'")

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


def evaluar_geometria_hough(roi: np.ndarray) -> dict:
    """
    Evalúa la rectitud de la caja usando Hough Lines y PCA como fallback.
    """
    gris = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    
    h_roi, w_roi = gris.shape
    if h_roi <= 10 or w_roi <= 10:
        return {'resultado': 'DEFECTO', 'angulo': 0.0, 'detalle': 'ROI inválido'}

    # Desenfoque suave y Canny
    blur = cv2.GaussianBlur(gris, (5, 5), 0)
    mediana = np.median(blur)
    low = int(max(0, 0.5 * mediana))
    high = int(min(255, 1.5 * mediana))
    edges = cv2.Canny(blur, low, high)
    
    # Dilatación leve para unir bordes rotos
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    edges = cv2.dilate(edges, kernel, iterations=1)

    angulo = None
    metodo = ""
    lines_to_draw = []

    # INTENTO 1: Hough Lines
    # Buscamos líneas que sean al menos el 20% del ancho del ROI
    min_line = max(30, int(w_roi * 0.2))
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=40, minLineLength=min_line, maxLineGap=15)
    
    if lines is not None and len(lines) > 0:
        angle_len_list = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            a = np.degrees(np.arctan2(y2 - y1, x2 - x1))
            
            # Normalizar a [-45, 45)
            while a <= -45: a += 90
            while a > 45: a -= 90
                
            angle_len_list.append((a, length, line[0]))
            
        # Ordenamos por longitud de línea y tomamos las 5 más largas
        angle_len_list.sort(key=lambda x: x[1], reverse=True)
        top_lines = angle_len_list[:5]
        
        # Calcular mediana de los ángulos para ser robustos ante outliers
        angulo = abs(np.median([item[0] for item in top_lines]))
        metodo = "Hough"
        lines_to_draw = [item[2] for item in top_lines]

    # INTENTO 2: PCA (Fallback)
    if angulo is None:
        y, x = np.where(edges > 0)
        if len(x) > 20:
            points = np.column_stack((x, y)).astype(np.float32)
            mean, eigenvectors = cv2.PCACompute(points, mean=None)
            vector = eigenvectors[0]
            a = np.degrees(np.arctan2(vector[1], vector[0]))
            
            while a <= -45: a += 90
            while a > 45: a -= 90
            
            angulo = abs(a)
            metodo = "PCA"

    if angulo is None:
        return {
            'resultado': 'DEFECTO',
            'angulo': 0.0,
            'detalle': 'Fallo Geometría (No se encontraron bordes)'
        }

    desviacion = angulo
    if desviacion > config.BANDEJA_MAX_ANGLE_TOLERANCE:
        resultado = 'DEFECTO'
        detalle = f"Inclinación ({metodo}): {desviacion:.1f}° > {config.BANDEJA_MAX_ANGLE_TOLERANCE}°"
    else:
        resultado = 'OK'
        detalle = f"Alineación correcta ({metodo}: {desviacion:.1f}°)"

    return {
        'resultado': resultado,
        'angulo': desviacion,
        'detalle': detalle,
        'metodo': metodo,
        'lines': lines_to_draw
    }


def dibujar_overlay(frame: np.ndarray, yolo_box: tuple, resultado: dict, debug_info: str = None, roi_offset: tuple = None) -> np.ndarray:
    """Dibuja la detección de YOLO y la matemática geométrica (Hough/PCA)."""
    if yolo_box is None:
        cv2.putText(frame, "BUSCANDO CONTENEDOR...", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 165, 255), 2)
        if config.DEBUG_MODE and debug_info:
            cv2.putText(frame, f"[DEBUG] {debug_info}", (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
        return frame

    x1, y1, x2, y2 = yolo_box
    ox, oy = roi_offset if roi_offset else (x1, y1)
    
    es_ok = resultado['resultado'] == 'OK'
    color = (0, 200, 0) if es_ok else (0, 0, 255)

    if config.DEBUG_MODE and debug_info:
        cv2.putText(frame, f"[DEBUG] {debug_info}", (x1, y1 - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)

    # Dibujar la caja base (YOLO)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 144, 30), 2)

    # Dibujar las líneas de Hough usadas para el cálculo
    lines = resultado.get('lines', [])
    for line in lines:
        lx1, ly1, lx2, ly2 = line
        # Desplazar por el offset del ROI
        cv2.line(frame, (lx1 + ox, ly1 + oy), (lx2 + ox, ly2 + oy), (255, 255, 255), 2)
        # Resaltar puntos finales
        cv2.circle(frame, (lx1 + ox, ly1 + oy), 3, color, -1)
        cv2.circle(frame, (lx2 + ox, ly2 + oy), 3, color, -1)

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
        
        # Cargar modelo YOLO-World
        if YOLOWorld is None or torch is None:
            logger.error("No se puede iniciar el worker sin ultralytics/torch.")
            sys.exit(1)
            
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.info(f"Dispositivo de IA detectado: {device.upper()}")
        if device == 'cuda':
            logger.info(f"GPU detectada: {torch.cuda.get_device_name(0)}")
            
        logger.info(f"Cargando modelo YOLO-World: {config.BANDEJA_YOLO_MODEL}")
        self.yolo_model = YOLOWorld(config.BANDEJA_YOLO_MODEL)
        self.yolo_model.set_classes(config.BANDEJA_YOLO_PROMPTS)
        self.yolo_model.to(device)

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
        
        # 1. Verificar si el historial reciente está lleno de defectos (Consenso temporal)
        consenso_defectos = self.historial.count('DEFECTO')
        UMBRAL_CONSENSO = 3 # Exigir 3 frames seguidos de error
        
        # 2. Si hay error sostenido y no lo hemos registrado aún
        if consenso_defectos >= UMBRAL_CONSENSO and self.ultimo_estado_registrado != 'DEFECTO':
            self.actuador.trigger(f"Bandeja chueca: {resultado['detalle']}")
            db.insertar_evento_calidad('DEFECTO', area, resultado['detalle'])
            logger.warning(f"Calidad DEFECTO (Confirmado): {resultado['detalle']}")
            self.ultimo_estado_registrado = 'DEFECTO'
            
        # 3. Si la lectura vuelve a ser perfecta, reseteamos el estado
        elif resultado_actual == 'OK' and self.ultimo_estado_registrado != 'OK':
            db.insertar_evento_calidad('OK', area, resultado['detalle'])
            logger.info("Calidad OK (Restablecida).")
            self.ultimo_estado_registrado = 'OK'

    def procesar_frame(self, frame) -> bool:
        self.frame_count += 1
        frame_con_overlay = frame.copy()
        
        # ETAPA 1: Búsqueda con IA (YOLO-World)
        results = self.yolo_model.predict(frame, conf=config.BANDEJA_YOLO_CONFIDENCE, verbose=False)
        
        yolo_box = None
        roi_offset = None
        area = 0
        resultado = {
            'resultado': 'DEFECTO',
            'angulo': 0.0,
            'detalle': 'No se detectó el objeto (YOLO)'
        }
        debug_info = None

        if len(results) > 0 and len(results[0].boxes) > 0:
            if config.DEBUG_MODE:
                todas_detecciones = []
                for b in results[0].boxes:
                    c_id = int(b.cls[0].item())
                    c_name = self.yolo_model.names[c_id]
                    c_conf = b.conf[0].item()
                    todas_detecciones.append(f"{c_name} {c_conf*100:.0f}%")
                
                # Unimos todas las detecciones separadas por comas
                debug_info = "Detectado: " + " | ".join(todas_detecciones)
                print(f"[DEBUG YOLO] {debug_info}")

            # Tomamos la detección con mayor confianza
            box_obj = results[0].boxes[0]
            box = box_obj.xyxy[0].cpu().numpy().astype(int)
            x1, y1, x2, y2 = box
            
            # Asegurar límites dentro del frame
            h_f, w_f = frame.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w_f, x2), min(h_f, y2)
            yolo_box = (x1, y1, x2, y2)
            
            area = (x2 - x1) * (y2 - y1)
            
            # Ampliar la ROI un poco para que el objeto no toque los bordes de la imagen
            # y el algoritmo de Hough pueda encontrar todas las líneas bien
            padding = 15
            px1 = max(0, x1 - padding)
            py1 = max(0, y1 - padding)
            px2 = min(w_f, x2 + padding)
            py2 = min(h_f, y2 + padding)
            
            roi_offset = (px1, py1)
            roi = frame[py1:py2, px1:px2]
            
            if roi.shape[0] > 10 and roi.shape[1] > 10:
                # ETAPA 2: Medición con Geometría (OpenCV)
                resultado = evaluar_geometria_hough(roi)
                
                self.historial.append(resultado['resultado'])
                self._gestionar_alertas(resultado, area)
        else:
            if config.DEBUG_MODE and len(results) > 0:
                debug_info = "Ningún objeto detectado con confianza > 10%"

        if self.frame_count % self.heartbeat_interval == 0:
            db.actualizar_heartbeat("worker_bandejas")

        frame_con_overlay = dibujar_overlay(frame_con_overlay, yolo_box, resultado, debug_info, roi_offset)
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
                
                # LIMITADOR TÉRMICO OBLIGATORIO
                time.sleep(config.BANDEJA_FPS_DELAY) 
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

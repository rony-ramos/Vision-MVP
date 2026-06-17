"""
config.py — Configuración central del sistema Vision-MVP.

Todas las constantes, umbrales y parámetros operativos del sistema.
Diseñado para ser el único punto de modificación ante cambios de entorno.
"""

import os

# =============================================
# Base de Datos
# =============================================
DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "vision_mvp.db")

# =============================================
# Cámaras
# =============================================
CAM_BANDEJAS_INDEX = 0      # Cámara para inspección de bandejas
CAM_POSTURA_INDEX = 1       # Cámara para monitoreo ergonómico

# =============================================
# Worker de Bandejas (Visión Clásica + YOLO)
# =============================================
# Configuración YOLO
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BANDEJA_YOLO_MODEL = os.path.join(BASE_DIR, 'assets', 'yolov8x.pt')
BANDEJA_YOLO_CLASS = 45 # 45 es 'bowl' en COCO (Proxy para la bandeja real detectada en p1.jpeg)
BANDEJA_MAX_ANGLE_TOLERANCE = 5.0 # Grados de tolerancia máxima de rectitud

# =============================================
# General
# =============================================
DEBUG_MODE = False          # Si es True, usa cv2.imshow nativo (no recomendado para dashboard)

# =============================================
# Streaming de Video (MJPEG)
# =============================================
STREAM_PORT_BANDEJAS = 8001
STREAM_PORT_POSTURA = 8002

# =============================================
# Worker Bandejas — Detección de Posición
# =============================================
# ROI (Region of Interest) en formato (x, y, ancho, alto)
# Define la zona esperada donde debe estar la bandeja.
# Ajustar según la posición física de la cámara.
BANDEJA_ROI = (100, 80, 440, 340)

# Limitador térmico y de FPS (Aprox. 2-3 FPS por defecto)
BANDEJA_FPS_DELAY = 0.05

# Umbrales de contorno para detección de bandeja
MIN_CONTOUR_AREA = 5000         # Área mínima para considerar un contorno como bandeja
MAX_CONTOUR_AREA = 200000       # Área máxima (filtrar ruido grande)

# Porcentaje de la ROI que debe estar cubierta por la bandeja para ser "correcta"
BANDEJA_COBERTURA_MIN = 0.30    # 30% mínimo de cobertura en la ROI

# Tolerancia de centrado: desviación máxima permitida del centro (en px)
BANDEJA_CENTRO_TOLERANCIA = 80

# Preprocesamiento
BANDEJA_BLUR_KERNEL = (5, 5)
BANDEJA_THRESH_BLOCK_SIZE = 11
BANDEJA_THRESH_C = 2

# =============================================
# Worker Postura — MediaPipe Pose
# =============================================
POSTURA_FPS_DELAY = 0.025                  # Sin retraso, procesa a máxima velocidad
POSTURA_RESOLUTION = (1280, 720)          # Resolución HD (720p) para mayor nitidez
POSTURA_MIN_DETECTION_CONFIDENCE = 0.70   # 70% de seguridad para detectar la postura
POSTURA_MIN_TRACKING_CONFIDENCE = 0.70    # 70% de seguridad para seguir los movimientos
POSTURA_MODEL_ASSET = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'assets', 'pose_landmarker_heavy.task') # Modelo Heavy de MediaPipe

# ROI de Postura: Region of Interest por defecto (x, y, ancho, alto)
# Centrado para una resolución de 1280x720 con tamaño 640x640.
POSTURA_ROI = (320, 40, 640, 640)

# Configuración visual del ROI
POSTURA_DRAW_ROI = True
POSTURA_ROI_COLOR = (0, 255, 255)  # BGR (Amarillo por defecto)
POSTURA_ROI_THICKNESS = 2
POSTURA_ROI_TEXT = "ROI (Zona de Proceso)"

# Umbrales ergonómicos (PARAMETRIZABLES)
# Ángulo de espalda: medido entre SHOULDER-HIP-KNEE
# Valores < umbral indican inclinación excesiva
MAX_BACK_INCLINATION = 20.0     # Grados de inclinación lateral máxima

# Ángulo de cuello: medido entre EAR-SHOULDER-HIP
# Valores > umbral indican flexión excesiva
MAX_NECK_FLEXION = 30.0         # Grados de flexión cervical máxima

# Frames consecutivos en alerta antes de registrar evento
POSTURA_FRAMES_ALERTA = 10     # ~1.5 segundos a 7 FPS

# =============================================
# Configuración Global del Sistema
# =============================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEQUE_MAXLEN = 100              # Tamaño máximo de historiales en memoria

# =============================================
# Dashboard (HMI)
# =============================================
DASHBOARD_REFRESH_MS = 2000     # Intervalo de polling (ms)
DASHBOARD_EVENTOS_LIMIT = 50    # Cantidad de eventos recientes a mostrar

# =============================================
# Heartbeat
# =============================================
HEARTBEAT_TIMEOUT_S = 10        # Segundos sin heartbeat → worker considerado muerto

# =============================================
# HAL — Hardware Abstraction Layer
# =============================================
ACTUADOR_MODO = "mock"          # "mock" | "arduino"
ARDUINO_PORT = "COM3"           # Puerto serial para Fase 2
ARDUINO_BAUDRATE = 9600

# Sonido de alerta (ActuadorMock - Windows)
ALERTA_FRECUENCIA_HZ = 1000    # Frecuencia del beep
ALERTA_DURACION_MS = 500        # Duración del beep

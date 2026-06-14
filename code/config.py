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
# General
# =============================================
DEBUG_MODE = True          # Si es True, usa cv2.imshow nativo (no recomendado para dashboard)

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
POSTURA_FPS_DELAY = 0.0015                  # Sin retraso, procesa a máxima velocidad
POSTURA_RESOLUTION = (1280, 720)          # Resolución HD (720p) para mayor nitidez
POSTURA_MIN_DETECTION_CONFIDENCE = 0.7   # 70% de seguridad para detectar la postura
POSTURA_MIN_TRACKING_CONFIDENCE = 0.7    # 70% de seguridad para seguir los movimientos

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
# Resiliencia (Gestión de Memoria)
# =============================================
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

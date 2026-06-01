# Vision-MVP 🏭

**Sistema de Inspección Automatizada por Visión Artificial**

Prototipo Edge modular para inspección de calidad de posicionamiento de bandejas de pollo y monitoreo ergonómico del operario en línea de empaque.

## Topología del Sistema

```
[Terminal 1] worker_bandejas.py  ──┐
[Terminal 2] worker_postura.py   ──┼──> [SQLite + WAL] <── [Terminal 3] dashboard.py
                                   │
                                   └──> [HAL: ActuadorMock] ---> (Fase 2: Arduino)
```

Procesos independientes aislados por SO para evitar bloqueos por GIL. Si un nodo cae, el resto sobrevive.

## Requisitos

- **Python** 3.10+
- **Cámaras** 2x (una para bandejas, una para postura)
- **SO**: Windows (winsound para alertas en Fase 1)

## Instalación

```bash
# 1. Clonar repositorio
git clone <url> && cd Vision-MVP

# 2. Crear entorno virtual (recomendado)
python -m venv .venv
.venv\Scripts\activate

# 3. Instalar dependencias
pip install -r requirements.txt
```

## Ejecución

Abrir **3 terminales** en la carpeta del proyecto:

```bash
# Terminal 1: Inspección de bandejas
python worker_bandejas.py

# Terminal 2: Monitoreo ergonómico
python worker_postura.py

# Terminal 3: Dashboard
streamlit run dashboard.py
```

### Con Auto-Healing (Watchdog)

```bash
# Terminal 1: Worker con reinicio automático
watchdog.bat worker_bandejas.py

# Terminal 2: Worker con reinicio automático
watchdog.bat worker_postura.py

# Terminal 3: Dashboard
streamlit run dashboard.py
```

## Estructura del Proyecto

```
Vision-MVP/
├── config.py              # Constantes y umbrales parametrizables
├── db.py                  # Persistencia SQLite + WAL
├── hal.py                 # Hardware Abstraction Layer (Adapter)
├── worker_bandejas.py     # Worker 1: Inspección de posición
├── worker_postura.py      # Worker 2: Monitoreo ergonómico
├── dashboard.py           # HMI: Streamlit dashboard
├── watchdog.bat           # Auto-healing wrapper
├── requirements.txt       # Dependencias Python
├── README.md              # Esta documentación
└── tests/
    ├── test_db.py         # Tests de persistencia
    └── test_hal.py        # Tests del HAL
```

## Configuración

Todos los parámetros se ajustan en [`config.py`](config.py):

| Parámetro | Default | Descripción |
|-----------|---------|-------------|
| `CAM_BANDEJAS_INDEX` | `0` | Índice de cámara para bandejas |
| `CAM_POSTURA_INDEX` | `1` | Índice de cámara para postura |
| `BANDEJA_ROI` | `(100,80,440,340)` | Región de interés (x,y,w,h) |
| `MAX_BACK_INCLINATION` | `20.0°` | Umbral de inclinación de espalda |
| `MAX_NECK_FLEXION` | `30.0°` | Umbral de flexión cervical |
| `POSTURA_FRAMES_ALERTA` | `10` | Frames consecutivos para alerta |
| `ACTUADOR_MODO` | `"mock"` | `"mock"` o `"arduino"` |

## Decisiones Arquitectónicas (ADR)

### A. Visión Artificial
- **Worker Bandejas**: OpenCV clásico. ROI fija + `cv2.findContours`. Sin redes neuronales.
- **Worker Postura**: MediaPipe Pose. Captura degradada a 5-7 FPS. Resolución 640x480.

### B. Persistencia
- SQLite3 con `PRAGMA journal_mode=WAL` para concurrencia (2 writers + 1 reader).
- Tabla unificada `eventos` con columna `tipo_evento`.

### C. HAL (Puertos y Adaptadores)
- Patrón Adapter. `actuador.trigger(motivo)` es la interfaz genérica.
- Fase 1: `ActuadorMock` (beep Windows). Fase 2: `ArduinoActuador` (UART pyserial).

### D. Dashboard
- Streamlit con `st_autorefresh(interval=2000)`. Modelo Pull/Polling.
- Cero estado en frontend. Puro visor de SQLite.

### E. Resiliencia (8+ horas)
- `collections.deque(maxlen=100)` — sin `list.append()` global.
- `cap.release()` + `cv2.destroyAllWindows()` en `finally`.
- Watchdog `.bat` para auto-restart ante crashes.

## Tests

```bash
python -m pytest tests/ -v
```

## Roadmap

- [x] Fase 1: Prototipo con ActuadorMock
- [ ] Fase 2: Integración Arduino (pyserial + UART)
- [ ] Fase 3: Modelo IA para validación de sellado

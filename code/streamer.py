"""
streamer.py — Módulo para servir frames de OpenCV vía HTTP (MJPEG).

Permite que múltiples clientes web (como el Dashboard de Streamlit)
consuman un flujo de video en tiempo real sin bloquear el worker de OpenCV.
"""

import cv2
import time
import logging
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from socketserver import ThreadingMixIn

logger = logging.getLogger(__name__)


class ThreadingHTTPServer(ThreadingMixIn, HTTPServer):
    """Servidor HTTP que maneja requests en hilos independientes (daemon_threads = True)."""
    daemon_threads = True  # CRÍTICO: Previene que el servidor bloquee el cierre del programa principal
    allow_reuse_address = True  # CRÍTICO: Evita "Address already in use" al reiniciar


class MJPEGHandler(BaseHTTPRequestHandler):
    """Manejador HTTP que emite el stream MJPEG del último frame."""
    
    def do_GET(self):
        if self.path == '/stream':
            self.send_response(200)
            self.send_header('Age', 0)
            self.send_header('Cache-Control', 'no-cache, private')
            self.send_header('Pragma', 'no-cache')
            self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=FRAME')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()

            try:
                while True:
                    frame = self.server.video_source.get_frame()
                    if frame is not None:
                        # Codificar a JPEG
                        ret, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
                        if ret:
                            self.wfile.write(b'--FRAME\r\n')
                            self.send_header('Content-Type', 'image/jpeg')
                            self.send_header('Content-Length', len(jpeg))
                            self.end_headers()
                            self.wfile.write(jpeg.tobytes())
                            self.wfile.write(b'\r\n')
                    
                    # Evitar saturar la CPU
                    time.sleep(0.05)
            except Exception as e:
                # Se lanza al desconectar el cliente web
                pass
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        # Evitar inundar la terminal con peticiones HTTP
        return


class VideoStreamingServer:
    """Clase principal para instanciar el servidor MJPEG desde los workers."""
    
    def __init__(self, port: int):
        self.port = port
        self._current_frame = None
        self._lock = threading.Lock()
        
        self.server = ThreadingHTTPServer(('0.0.0.0', self.port), MJPEGHandler)
        self.server.video_source = self  # Inyectar referencia al propio objeto

        self.thread = threading.Thread(target=self.server.serve_forever)
        self.thread.daemon = True  # CRÍTICO
        self.thread.start()
        logger.info(f"MJPEG Streamer iniciado en puerto {self.port}")

    def set_frame(self, frame):
        """Actualiza el último frame disponible para streaming."""
        with self._lock:
            self._current_frame = frame.copy() if frame is not None else None

    def get_frame(self):
        """Devuelve el último frame disponible de manera thread-safe."""
        with self._lock:
            return self._current_frame

    def stop(self):
        """Detiene el servidor HTTP."""
        self.server.shutdown()
        self.server.server_close()
        logger.info(f"MJPEG Streamer en puerto {self.port} detenido.")

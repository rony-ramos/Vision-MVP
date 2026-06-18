import os
import cv2

# Monkeypatch cv2.imshow and cv2.waitKey to prevent UI hanging
cv2.imshow = lambda *args, **kwargs: None
cv2.waitKey = lambda *args, **kwargs: 1

import config
import db
from worker_bandejas import WorkerBandejas

config.DEBUG_MODE = True
out_dir = r"C:\Users\ramos\.gemini\antigravity-ide\brain\68c18349-7ec3-4d3f-a4f9-84dcdebaf649"

class DummyStreamer:
    def __init__(self, filename):
        self.filename = filename
    def set_frame(self, frame):
        filepath = os.path.join(out_dir, self.filename)
        cv2.imwrite(filepath, frame)
        print(f"Imagen procesada y guardada en: {filepath}")
    def stop(self):
        pass

def main():
    print("Inicializando BD...")
    db.init_db()
    
    print("Inicializando Worker...")
    worker = WorkerBandejas()
    
    print("Procesando p1.jpeg...")
    worker.streamer = DummyStreamer('out_p1.jpg')
    frame1 = cv2.imread(r'a:\USIL CS\2026\2026-1\capstone taz\Vision-MVP\code\assets\p1.jpeg')
    if frame1 is not None:
        worker.procesar_frame(frame1)
    else:
        print("Error al leer p1.jpeg")
        
    print("Procesando p3.png...")
    worker.streamer = DummyStreamer('out_p3.jpg')
    frame3 = cv2.imread(r'a:\USIL CS\2026\2026-1\capstone taz\Vision-MVP\code\assets\p3.png')
    if frame3 is not None:
        worker.procesar_frame(frame3)
        # Añadir guardado de YOLO crudo
        results_raw = worker.yolo_model.predict(frame3, conf=0.1)
        if len(results_raw) > 0:
            cv2.imwrite(os.path.join(out_dir, 'raw_yolo_p3.jpg'), results_raw[0].plot())
    else:
        print("Error al leer p3.png")

if __name__ == '__main__':
    main()

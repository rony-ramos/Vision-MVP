import os
import cv2
import numpy as np
import config
import db
from worker_bandejas import WorkerBandejas, preprocesar_y_detectar_contorno, evaluar_geometria

config.DEBUG_MODE = True
out_dir = r"C:\Users\ramos\.gemini\antigravity-ide\brain\68c18349-7ec3-4d3f-a4f9-84dcdebaf649"

def debug_opencv_steps(frame, yolo_box, out_prefix):
    x1, y1, x2, y2 = yolo_box
    h_f, w_f = frame.shape[:2]
    
    padding = 15
    px1 = max(0, x1 - padding)
    py1 = max(0, y1 - padding)
    px2 = min(w_f, x2 + padding)
    py2 = min(h_f, y2 + padding)
    
    roi = frame[py1:py2, px1:px2]
    
    gris = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gris, (11, 11), 0)
    edges = cv2.Canny(blur, 20, 80)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    bordes_cerrados = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    cv2.imwrite(os.path.join(out_dir, f"{out_prefix}_1_roi.jpg"), roi)
    cv2.imwrite(os.path.join(out_dir, f"{out_prefix}_2_blur.jpg"), blur)
    cv2.imwrite(os.path.join(out_dir, f"{out_prefix}_3_canny.jpg"), edges)
    cv2.imwrite(os.path.join(out_dir, f"{out_prefix}_4_clausura.jpg"), bordes_cerrados)

    contours, _ = cv2.findContours(bordes_cerrados, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        mejor = max(contours, key=cv2.contourArea)
        hull = cv2.convexHull(mejor)
        
        # Draw on a copy of ROI
        roi_debug = roi.copy()
        cv2.drawContours(roi_debug, [hull], -1, (0, 255, 0), 2)
        
        # Draw minAreaRect
        rect = cv2.minAreaRect(hull)
        box = cv2.boxPoints(rect)
        box = np.intp(box)
        cv2.drawContours(roi_debug, [box], 0, (0, 0, 255), 2)
        
        cv2.imwrite(os.path.join(out_dir, f"{out_prefix}_5_hull_and_rect.jpg"), roi_debug)

def main():
    db.init_db()
    worker = WorkerBandejas()
    
    frame2 = cv2.imread(r'a:\USIL CS\2026\2026-1\capstone taz\Vision-MVP\code\assets\p2.png')
    if frame2 is not None:
        # Predecir sin filtro de clases
        results = worker.yolo_model.predict(frame2, conf=0.10, verbose=False)
        
        if len(results) > 0 and len(results[0].boxes) > 0:
            print(f"Todas las detecciones (sin filtro):")
            for b in results[0].boxes:
                c_id = int(b.cls[0].item())
                c_name = worker.yolo_model.names[c_id]
                c_conf = b.conf[0].item()
                box = b.xyxy[0].cpu().numpy().astype(int)
                print(f" - {c_name} ({c_conf*100:.1f}%) en {box}")
        else:
            print("YOLO no detectó nada en p2.png con las clases dadas.")
    else:
        print("Error al leer p2.png")

if __name__ == '__main__':
    main()

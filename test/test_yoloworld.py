"""
test_yoloworld.py — Prueba de YOLO-World (zero-shot) con las imágenes del escenario real.
Utiliza la geometría de Hough + PCA (misma de worker_bandejas).
"""
import os
import time
import cv2
import numpy as np
from ultralytics import YOLOWorld

cv2.imshow = lambda *args, **kwargs: None
cv2.waitKey = lambda *args, **kwargs: 1

out_dir = r"C:\Users\ramos\.gemini\antigravity-ide\brain\68c18349-7ec3-4d3f-a4f9-84dcdebaf649"
assets = r"a:\USIL CS\2026\2026-1\capstone taz\Vision-MVP\code\assets"
BANDEJA_YOLO_PROMPTS = ["box", "cardboard box", "open box", "tray", "container"]


def evaluar_geometria_hough(roi: np.ndarray) -> dict:
    gris = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    h_roi, w_roi = gris.shape
    if h_roi <= 10 or w_roi <= 10:
        return {'resultado': 'DEFECTO', 'angulo': 0.0, 'detalle': 'ROI inválido'}

    blur = cv2.GaussianBlur(gris, (5, 5), 0)
    mediana = np.median(blur)
    low = int(max(0, 0.5 * mediana))
    high = int(min(255, 1.5 * mediana))
    edges = cv2.Canny(blur, low, high)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    edges = cv2.dilate(edges, kernel, iterations=1)

    angulo = None
    metodo = ""
    lines_to_draw = []

    min_line = max(30, int(w_roi * 0.2))
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=40, minLineLength=min_line, maxLineGap=15)
    
    if lines is not None and len(lines) > 0:
        angle_len_list = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            a = np.degrees(np.arctan2(y2 - y1, x2 - x1))
            while a <= -45: a += 90
            while a > 45: a -= 90
            angle_len_list.append((a, length, line[0]))
            
        angle_len_list.sort(key=lambda x: x[1], reverse=True)
        top_lines = angle_len_list[:5]
        angulo = abs(np.median([item[0] for item in top_lines]))
        metodo = "Hough"
        lines_to_draw = [item[2] for item in top_lines]

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
        return {'resultado': 'DEFECTO', 'angulo': 0.0, 'detalle': 'Fallo Geometría'}

    desviacion = angulo
    if desviacion > 5.0:
        resultado = 'DEFECTO'
    else:
        resultado = 'OK'

    return {
        'resultado': resultado,
        'angulo': desviacion,
        'metodo': metodo,
        'lines': lines_to_draw
    }


def procesar_imagen(model, frame, nombre):
    t0 = time.perf_counter()
    results = model.predict(frame, conf=0.05, verbose=False)
    t_yolo = (time.perf_counter() - t0) * 1000
    
    if len(results) == 0 or len(results[0].boxes) == 0:
        print(f"  {nombre}: NO DETECTADO ({t_yolo:.0f}ms)")
        return
    
    box_obj = results[0].boxes[0]
    box = box_obj.xyxy[0].cpu().numpy().astype(int)
    conf = box_obj.conf[0].item()
    cls_name = model.names[int(box_obj.cls[0].item())]
    
    x1, y1, x2, y2 = box
    padding = 15
    px1, py1 = max(0, x1 - padding), max(0, y1 - padding)
    px2, py2 = min(frame.shape[1], x2 + padding), min(frame.shape[0], y2 + padding)
    roi = frame[py1:py2, px1:px2]
    
    t1 = time.perf_counter()
    resultado = evaluar_geometria_hough(roi)
    t_geom = (time.perf_counter() - t1) * 1000
    t_total = t_yolo + t_geom
    
    estado = resultado['resultado']
    angulo = resultado['angulo']
    metodo = resultado.get('metodo', 'None')
    
    # Dibujar
    img_res = frame.copy()
    color = (0, 200, 0) if estado == "OK" else (0, 0, 255)
    
    cv2.rectangle(img_res, (x1, y1), (x2, y2), (255, 144, 30), 2)
    cv2.putText(img_res, f"{cls_name} {conf*100:.0f}%", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 144, 30), 1)
    
    for line in resultado.get('lines', []):
        lx1, ly1, lx2, ly2 = line
        cv2.line(img_res, (lx1 + px1, ly1 + py1), (lx2 + px1, ly2 + py1), (255, 255, 255), 2)
        cv2.circle(img_res, (lx1 + px1, ly1 + py1), 3, color, -1)
        cv2.circle(img_res, (lx2 + px1, ly2 + py1), 3, color, -1)
        
    label = f"{estado}: {angulo:.1f} grados ({metodo})"
    cv2.putText(img_res, label, (x1, y2 + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    cv2.putText(img_res, f"YOLO+Hough ({t_total:.0f}ms)", (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
    
    cv2.imwrite(os.path.join(out_dir, f'test_final_{nombre}.jpg'), img_res)
    print(f"    => {estado} | Angulo: {angulo:.1f}° ({metodo}) | YOLO: {t_yolo:.0f}ms | Geom: {t_geom:.0f}ms")


def main():
    model = YOLOWorld(os.path.join(assets, "yolov8s-worldv2.pt"))
    model.set_classes(BANDEJA_YOLO_PROMPTS)
    
    imagenes = [
        ('p4', os.path.join(assets, 'p4.png'), 'Caja amarilla (OK)'),
        ('p5', os.path.join(assets, 'p5.jpeg'), 'Escenario real - recta (OK)'),
        ('p6', os.path.join(assets, 'p6.jpeg'), 'Escenario real - leve inclinacion'),
        ('p7', os.path.join(assets, 'p7.jpeg'), 'Escenario real - inclinada (DEFECTO)'),
        ('p9', os.path.join(assets, 'p9.jpeg'), 'Escenario real - camara corregida 1'),
        ('p10', os.path.join(assets, 'p10.jpeg'), 'Escenario real - camara corregida 2'),
    ]
    
    for nombre, path, desc in imagenes:
        print(f"[{nombre}] {desc}")
        frame = cv2.imread(path)
        if frame is not None:
            procesar_imagen(model, frame, nombre)

if __name__ == '__main__':
    main()

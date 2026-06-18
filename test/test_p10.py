import cv2
from ultralytics import YOLOWorld

def main():
    model = YOLOWorld("yolov8s-worldv2")
    model.set_classes(["box", "cardboard box", "open box", "tray", "container"])
    results = model.predict(r"a:\USIL CS\2026\2026-1\capstone taz\Vision-MVP\code\assets\p10.jpeg", conf=0.01, verbose=True)
    
    print("Detecciones p10:")
    for i, b in enumerate(results[0].boxes):
        c_name = model.names[int(b.cls[0].item())]
        c_conf = b.conf[0].item()
        print(f"[{i}] {c_name} ({c_conf*100:.1f}%)")

if __name__ == '__main__':
    main()

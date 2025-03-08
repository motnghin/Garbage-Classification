from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from ultralytics import YOLO
from PIL import Image
import io
import cv2
import numpy as np

app = FastAPI()

# Load mô hình YOLOv8
model = YOLO(r"C:\Users\ACER\OneDrive\Máy tính\project_4\runs\detect\train20\weights\best.pt")

# Danh sách loại rác
waste_types = {0: "Nhựa", 1: "Giấy", 2: "Kim loại", 3: "Thủy tinh", 4: "Hữu cơ", 5: "Khác"}

# Màu sắc từng loại
colors = {0: (255, 0, 0), 1: (0, 255, 0), 2: (0, 0, 255), 3: (255, 255, 0), 4: (255, 0, 255), 5: (0, 255, 255)}

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    image = Image.open(io.BytesIO(await file.read()))
    img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)  # Chuyển sang OpenCV

    # Chạy mô hình nhận diện
    results = model(img_cv)
    detections = []

    for result in results:
        for box in result.boxes:
            class_id = int(box.cls)
            class_name = waste_types.get(class_id, "Không xác định")
            color = colors.get(class_id, (255, 255, 255))

            x1, y1, x2, y2 = map(int, box.xyxy.tolist()[0])
            confidence = float(box.conf.item())  # Chuyển thành số thực

            # Vẽ hộp giới hạn
            cv2.rectangle(img_cv, (x1, y1), (x2, y2), color, 3)

            # Vẽ nhãn
            label = f"{class_name} ({confidence:.2f})"
            cv2.putText(img_cv, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            detections.append({"class": class_name, "confidence": confidence, "bbox": [x1, y1, x2, y2]})

    # Chuyển ảnh OpenCV về JPEG
    _, img_encoded = cv2.imencode(".jpg", img_cv)
    
    return JSONResponse(content={
        "detections": detections,
        "image": img_encoded.tobytes().decode("latin1")  # Encode ảnh thành chuỗi
    })

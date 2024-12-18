import cv2
import torch
from pathlib import Path
from models.experimental import attempt_load
from utils.general import non_max_suppression, scale_coords
from utils.datasets import letterbox
from utils.plots import plot_one_box
import numpy as np

# بارگذاری مدل YOLOv7
model = attempt_load('yolov7.pt', map_location='cuda')  # وزن‌های از پیش‌آموزش‌داده‌شده

# نام کلاس‌ها (کلاس‌های COCO)
names = model.module.names if hasattr(model, 'module') else model.names

# دریافت تصویر از دوربین
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # پیش‌پردازش تصویر
    img = letterbox(frame, new_shape=640)[0]
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR به RGB
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img).float()
    img /= 255.0  # تغییر مقیاس به [0, 1]
    img = img.unsqueeze(0).cuda()

    # اجرای مدل
    pred = model(img)[0]
    pred = non_max_suppression(pred, 0.25, 0.45)

    # پردازش خروجی‌ها و ترسیم جعبه‌ها
    for i, det in enumerate(pred):  # هر تصویر
        if len(det):
            # تغییر مقیاس مختصات به اندازه تصویر اصلی
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], frame.shape).round()

            # رسم هر جعبه در تصویر
            for *xyxy, conf, cls in reversed(det):
                label = f'{names[int(cls)]} {conf:.2f}'
                plot_one_box(xyxy, frame, label=label, color=(255, 0, 0), line_thickness=2)

    # نمایش تصویر با جعبه‌ها
    cv2.imshow('YOLOv7 Detection', frame)

    # فشردن کلید 'q' برای خروج از برنامه
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

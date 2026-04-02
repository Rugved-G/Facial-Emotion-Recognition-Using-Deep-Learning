import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
import json
from collections import deque

# -------------------------
# Load labels
# -------------------------
with open("labels.json", "r") as f:
    emotions = json.load(f)

# -------------------------
# Load trained grayscale model
# -------------------------
model = tf.keras.models.load_model("fer_MobileNetV2_gray.keras", compile=False)

# -------------------------
# Mediapipe face detection
# -------------------------
mp_face_detection = mp.solutions.face_detection

# -------------------------
# IP webcam setup
# -------------------------
# Replace <IP_ADDRESS> with your phone's IP from IP Webcam app
url =0
cap = cv2.VideoCapture(url)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 480)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 640)

# -------------------------
# Prediction smoothing
# -------------------------
pred_buffer = deque(maxlen=5)

# -------------------------
# Bar colors
# -------------------------
colors = [(0,0,255),(0,165,255),(255,0,0),(0,255,0),(255,255,0),(255,0,255),(128,0,128)]

with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5) as face_detection:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            continue

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_detection.process(rgb_frame)

        if results.detections:
            for det in results.detections:
                bbox = det.location_data.relative_bounding_box
                ih, iw, _ = frame.shape
                x, y, w, h = int(bbox.xmin*iw), int(bbox.ymin*ih), int(bbox.width*iw), int(bbox.height*ih)
                x, y = max(0,x), max(0,y)
                w, h = min(iw-x,w), min(ih-y,h)
                face = frame[y:y+h, x:x+w]

                if face.size>0:
                    face_gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
                    face_resized = cv2.resize(face_gray, (48,48))/255.0
                    face_input = np.expand_dims(face_resized, axis=(0,-1))

                    preds = model.predict(face_input, verbose=0)[0]
                    pred_buffer.append(preds)
                    avg_preds = np.mean(pred_buffer, axis=0)

                    top_idx = np.argmax(avg_preds)
                    top_emotion = emotions[top_idx]
                    top_conf = avg_preds[top_idx]*100

                    # Red rectangle + top emotion
                    cv2.rectangle(frame, (x,y), (x+w,y+h), (0,0,255), 2)
                    cv2.putText(frame, f"{top_emotion} ({top_conf:.1f}%)", (x,y-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 2)

                    # Emotion bars with probabilities
                    bar_x, bar_y, bar_h = x, y+h+10, 12
                    for i,e in enumerate(emotions):
                        length = int(avg_preds[i]*w)
                        color = colors[i % len(colors)]
                        cv2.rectangle(frame, (bar_x, bar_y+i*(bar_h+2)), (bar_x+length, bar_y+i*(bar_h+2)+bar_h), color, -1)
                        cv2.putText(frame, f"{e}: {avg_preds[i]*100:.1f}%",
                                    (bar_x+length+5, bar_y+i*(bar_h+2)+bar_h-1),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255) if i==top_idx else (0,0,0), 1)

        cv2.imshow("Emotion Detection (IP Webcam)", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()

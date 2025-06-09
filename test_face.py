# test_face.py

import cv2
import face_recognition

# ───── 이 부분만 본인의 실제 업로드 비디오 경로로 바꿔 주세요. ─────
video_path = r"C:\Users\kkati\Desktop\test1.mp4"

# ────────────────────────────────────────────────────────────────

cap = cv2.VideoCapture(video_path)
print("▶ Video Opened?:", cap.isOpened())

success, frame = cap.read()
print("▶ First frame is None?:", frame is None)

if frame is not None:
    rgb = frame[:, :, ::-1]  # BGR → RGB
    try:
        faces = face_recognition.face_locations(rgb)
        print("▶ face_recognition.face_locations 결과:", faces)
    except Exception as e:
        print("▶ face_recognition 예외 발생:", e)

cap.release()

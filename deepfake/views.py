# backend/deepfake/views.py

import os
import cv2
import face_recognition
from django.conf import settings
from django.http import JsonResponse
from django.core.files.storage import FileSystemStorage
from django.views.decorators.csrf import csrf_exempt
from .models import Video, PreprocessedFrame

@csrf_exempt
def upload_file(request):
    if request.method == 'POST' and request.FILES.get('file'):
        # 1. 파일 저장
        uploaded_file = request.FILES['file']
        fs = FileSystemStorage()
        filename = fs.save(uploaded_file.name, uploaded_file)
        file_url = fs.url(filename)  # 예: /media/your.mp4

        # 2. Video 객체에 저장
        video = Video(video_file=uploaded_file)
        video.save()

        # 3. 프레임 추출 및 전처리
        video_path = fs.path(filename)
        cap = cv2.VideoCapture(video_path)

        frame_folder = os.path.join(settings.MEDIA_ROOT, 'frames')
        os.makedirs(frame_folder, exist_ok=True)

        saved_frame_count = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # 얼굴 검출
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face_locations = face_recognition.face_locations(rgb_frame)

            if face_locations:
                for (top, right, bottom, left) in face_locations:
                    face = frame[top:bottom, left:right]
                    if face.size == 0:
                        continue

                    # 크기 전처리
                    face = cv2.resize(face, (112, 112))

                    # 파일로 저장
                    saved_path = os.path.join(frame_folder, f'{video.id}_{saved_frame_count}.jpg')
                    cv2.imwrite(saved_path, face)

                    # DB에 저장
                    PreprocessedFrame.objects.create(
                        video=video,
                        frame_image=os.path.relpath(saved_path, settings.MEDIA_ROOT)
                    )

                    saved_frame_count += 1

        cap.release()

        # 4. 프레임 개수 검사
        if saved_frame_count < 100:
            video.delete()
            return JsonResponse({
                'error': 'Less than 100 face frames detected. Upload failed.'
            }, status=400)

        # 5. 결과 반환 (모델 연동 전 임시값)
        return JsonResponse({
            'file_url': file_url,
            'video_id': video.id,
            'result': 'Fake',
            'confidence': 87.2,
            'explanation': '눈 깜빡임이 비정상적이고, 입 모양과 소리가 어긋납니다.'
        })

    return JsonResponse({'error': 'No file uploaded'}, status=400)

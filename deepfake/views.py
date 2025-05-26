# backend/deepfake/views.py

import os
import cv2
import traceback
import face_recognition
from django.conf import settings
from django.http import JsonResponse
from django.core.files.storage import FileSystemStorage
from django.views.decorators.csrf import csrf_exempt
from .models import Video, PreprocessedFrame

@csrf_exempt
def upload_file(request):
    try:
        if request.method == 'POST' and request.FILES.get('file'):
            uploaded_file = request.FILES['file']
            fs = FileSystemStorage()
            filename = fs.save(uploaded_file.name, uploaded_file)
            file_url = fs.url(filename)

            video = Video(video_file=uploaded_file)
            video.save()

            return JsonResponse({
                'file_url': file_url,
                'video_id': video.id,
                'result': 'Fake',
                'confidence': 87.2,
                'explanation': '테스트용 응답입니다. 프레임 처리는 생략됨.'
            })

        return JsonResponse({'error': 'No file uploaded'}, status=400)

    except Exception as e:
        print(traceback.format_exc())
        return JsonResponse({'error': str(e)}, status=500)


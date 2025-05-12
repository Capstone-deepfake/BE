# views.py
import os
from django.http import JsonResponse
from django.core.files.storage import FileSystemStorage
from django.views.decorators.csrf import csrf_exempt

@csrf_exempt
def upload_file(request):
    if request.method == 'POST' and request.FILES.get('file'):
        # 1. 파일 저장
        uploaded_file = request.FILES['file']
        fs = FileSystemStorage()
        filename = fs.save(uploaded_file.name, uploaded_file)
        file_url = fs.url(filename)  # 예: /media/파일이름.mp4

        # 2. (전처리 및 DB 저장은 생략) ⛔

        # 3. 결과 반환 (임시 고정값)
        return JsonResponse({
            'file_url': file_url,
            'video_id': None,  # DB 없이 placeholder
            'result': "Fake",
            'confidence': 87.2,
            'explanation': "눈 깜빡임이 비정상적이고, 입 모양과 소리가 어긋납니다."
        })

    return JsonResponse({'error': 'No file uploaded'}, status=400)

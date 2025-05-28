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

# 🔹 OpenAI API 설정 (.env에서 안전하게 불러옴)
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
api_key = os.environ.get("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

# GPT 설명 생성 함수
def get_gpt_explanation(image_url: str) -> str:
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "hey what do you see from this picture"
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": image_url,
                                "detail": "high"
                            }
                        }
                    ]
                }
            ],
            max_tokens=300
        )
        return response.choices[0].message.content
    except Exception as e:
        print("OpenAI API error:", e)
        return "설명을 생성하는 데 문제가 발생했습니다."

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

            # 테스트용 Grad-CAM 이미지 URL
            gradcam_url = "https://raw.githubusercontent.com/sercanyesilkoy/test/main/test1.jpg"
            explanation_text = get_gpt_explanation(gradcam_url)

            return JsonResponse({
                'file_url': file_url,
                'video_id': video.id,
                'result': 'Fake',
                'confidence': 87.2,
                'explanation': explanation_text
            })

        return JsonResponse({'error': 'No file uploaded'}, status=400)

    except Exception as e:
        print(traceback.format_exc())
        return JsonResponse({'error': str(e)}, status=500)

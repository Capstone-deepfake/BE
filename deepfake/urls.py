# backend/deepfake/urls.py
from django.urls import path
from .views import upload_file  # 'upload_file' 뷰 임포트

urlpatterns = [
    path('upload/', upload_file, name='upload_file'),  # 'upload_file' 뷰에 대한 경로 정의
]

from django.urls import path
from . import views

urlpatterns = [
    #path('api/detect', views.detect_video, name='detect_video'),
    path('upload/', views.upload_file, name='upload_file'),  # ✅ /api/upload/가 되는 포인트
    path('chat/', views.chat_with_ai, name='chat_with_ai'),
]

from django.contrib import admin
from django.urls import path, include

urlpatterns = [
    path('admin/', admin.site.urls),
    path('api/', include('deepfake.urls')),  # ✅ 이 부분이 핵심
]

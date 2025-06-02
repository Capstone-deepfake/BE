# models.py
from django.db import models

class Video(models.Model):
    video_file = models.FileField(upload_to='videos/')
    uploaded_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.video_file.name

class PreprocessedFrame(models.Model):
    video = models.ForeignKey(Video, on_delete=models.CASCADE)
    frame_image = models.ImageField(upload_to='frames/')
    processed_at = models.DateTimeField(auto_now_add=True)

# backend/deepfake/processing.py

import cv2
import glob
import numpy as np
import face_recognition
from tqdm.autonotebook import tqdm
import os

def average_frame_count(real_fake, language):
    """
    (웹 애플리케이션에서는 사용되지 않음)
    지정된 디렉토리에서 .mp4 파일을 찾아, 150프레임 이상인 파일만 리스트로 반환합니다.
    """
    input_path = f'/Users/jiyeong/Desktop/컴공 캡스톤/PolyGlotFake/Dataset/{real_fake}/{language}/*.mp4'
    video_files = glob.glob(input_path)
    frame_count = []
    video_list = []
    short_frame = []

    for video_file in video_files:
        cap = cv2.VideoCapture(video_file)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        if total_frames < 150:
            short_frame.append(video_file)
            continue
        video_list.append(video_file)
        frame_count.append(total_frames)

    print(f"{real_fake}/{language}")
    print("Total number of videos:", len(frame_count))
    print("Average frame per video:", np.mean(frame_count) if frame_count else 0)
    print("Short frame video:", len(short_frame))
    return video_list

def extract_frames(video_path, fps_sampling=1):
    """
    비디오 파일을 열어, 초당 fps_sampling 프레임마다 (예: fps_sampling=1 → 1초당 1프레임)
    원본 BGR 이미지를 순서대로 yield 합니다.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"비디오를 열 수 없습니다: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    frame_index = 0
    success, frame = cap.read()

    while success:
        # 매 fps_sampling 프레임마다 프레임을 반환
        if frame_index % int(fps) == 0:
            yield frame
        frame_index += 1
        success, frame = cap.read()

    cap.release()

# (웹 애플리케이션에서는 아래 함수들이 사용되지 않습니다.)
def create_face_videos(path_list, out_dir):
    """
    (웹 애플리케이션에서는 사용되지 않음)
    path_list에 있는 각 비디오에 대해 최대 150프레임만 읽어서
    얼굴을 검출 → (112×112)로 크롭 → VideoWriter로 mp4로 저장합니다.
    """
    already_present_count = glob.glob(os.path.join(out_dir, '*.mp4'))
    print("No of videos already present", len(already_present_count))

    for path in tqdm(path_list):
        out_path = os.path.join(out_dir, os.path.basename(path))
        if glob.glob(out_path):
            print("File Already exists:", out_path)
            continue

        out = cv2.VideoWriter(
            out_path,
            cv2.VideoWriter_fourcc('M','J','P','G'),
            30,
            (112, 112)
        )

        frames = []
        for idx, frame in enumerate(extract_frames(path, fps_sampling=1)):
            if idx > 150:
                break
            frames.append(frame)

            # 4프레임씩 묶어서 얼굴 탐지
            if len(frames) == 4:
                faces = face_recognition.batch_face_locations([f[:,:,::-1] for f in frames])
                for i, face in enumerate(faces):
                    if face:
                        top, right, bottom, left = face[0]
                        try:
                            cropped = cv2.resize(frames[i][top:bottom, left:right, :], (112, 112))
                            out.write(cropped)
                        except:
                            pass
                frames = []

        out.release()

from multiprocessing import Process
def run_job(real_fake, language, output_dir):
    """
    (웹 애플리케이션에서는 사용되지 않음)
    average_frame_count → create_face_videos를 순차 또는 병렬 실행합니다.
    """
    video_files = average_frame_count(real_fake, language)
    create_face_videos(video_files, output_dir)

if __name__ == '__main__':
    # (웹 애플리케이션에서는 사용되지 않음)
    output_path = "/Users/jiyeong/Desktop/컴공 캡스톤/PolyGlotFake/Dataset/real_pr/real_pr_8910"
    p1 = Process(target=run_job, args=('real', 'to1', output_path))
    p2 = Process(target=run_job, args=('real', 'to2', output_path))
    p3 = Process(target=run_job, args=('real', 'to3', output_path))
    p1.start(); p2.start(); p3.start()
    p1.join(); p2.join(); p3.join()

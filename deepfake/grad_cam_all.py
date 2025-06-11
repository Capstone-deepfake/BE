import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import os
from django.conf import settings
from .modelNet import Model
import torchvision.transforms as T
import subprocess

# ✅ Grad-CAM 계산 함수
def run_grad_cam(model, input_tensor, target_class=None, device = torch.device("mps")):
    model.eval()
    fmap = None
    grad = None

    # forward hook: 마지막 layer 출력 저장
    def fw_hook(module, inp, out):
        nonlocal fmap
        fmap = out.detach()
        
    # backward hook: 마지막 layer의 gradient 저장
    def bw_hook(module, grad_in, grad_out):
        nonlocal grad
        grad = grad_out[0].detach()

    last_layer = model.model[-1]
    f = last_layer.register_forward_hook(fw_hook)
    b = last_layer.register_backward_hook(bw_hook)

    input_tensor = input_tensor.to(device).unsqueeze(0).unsqueeze(0).requires_grad_(True)
    _, output = model(input_tensor)


    if target_class is None:
        target_class = output.argmax(dim=1).item()

    model.zero_grad()
    output[0, target_class].backward()

    weights = grad.mean(dim=[2, 3], keepdim=True)
    cam = (weights * fmap).sum(dim=1, keepdim=True)
    cam = F.relu(cam)
    cam = cam.squeeze().cpu().numpy()
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    cam = cv2.resize(cam, (input_tensor.shape[-1], input_tensor.shape[-2]))

    f.remove()
    b.remove()

    return cam


def all_apply_gradcam_to_image(model, video_path, save_path):

     # ─── 모델 로드 ────────────────────────────────────────────────────────────────
    # checkpoint_path = os.path.join(settings.BASE_DIR, "model", "checkpoint_1.pt")
    checkpoint_path = os.path.join(settings.BASE_DIR, "deepfake", "checkpoint_1.pt")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"모델 체크포인트를 찾을 수 없습니다: {checkpoint_path}")

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    state_dict = torch.load(checkpoint_path, map_location=device)
    model = Model(num_classes=2, model_name='resnext50_32x4d').to(device)
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()

    # 1) 원본 이미지 영상 불러와서 RGB → PIL → 전처리 → Tensor
    # orig = cv2.imread(img_path)
    # orig_rgb = cv2.cvtColor(orig, cv2.COLOR_BGR2RGB)
    # pil = Image.fromarray(orig_rgb)

    transform = transforms.Compose([
        transforms.ToPILImage(),  # <--- 이 줄을 추가합니다!
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406],
                             std=[0.229,0.224,0.225])
    ])

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)


    # 비디오의 프레임 크기와 FPS 가져오기
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # 비디오 저장을 위한 VideoWriter 초기화
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_video_path = os.path.join(save_path, "grad_cam.mp4")
    out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'MJPG'), 30, (112,112))
    # out = cv2.VideoWriter(save_path, fourcc, fps, (frame_width, frame_height))  # (프레임 너비, 프레임 높이)

    frame_scores = [] # 점수를 기록할 리스트
    frame_images = []  # 프레임 이미지를 저장할 리스트
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        original = frame.copy()
        # img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = transform(frame).to(device)

        cam= run_grad_cam(model, input_tensor=img, device=device)  # Grad-CAM 실행

        # 각 프레임의 활성도 점수 평균을 계산해서 frame_scores에 저장
        score = np.mean(cam)

        heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
        heatmap = cv2.resize(heatmap, (original.shape[1], original.shape[0]))
        overlay = 0.4 * heatmap + 0.6 * original
        overlay = np.clip(overlay, 0, 255).astype(np.uint8)

        frame_images.append((frame_count, overlay))

        frame_scores.append((frame_count, score))

        frame_count += 1

        # Grad-CAM을 비디오로 저장
        out.write(overlay)  # 각 프레임을 비디오에 추가
        print("저장중")
        
        frame_scores.sort(key=lambda x: x[1], reverse=True)  # 점수 기준으로 정렬

    top_1_indices = [idx for idx, score in frame_scores] 
    for rank, idx in enumerate(top_1_indices):
        grad_img = frame_images[idx][1]  
        score = frame_scores[rank][1]  
        top_frame_path = os.path.join(save_path, f"Top_1_frame.jpg")
        cv2.imwrite(top_frame_path, grad_img)

    cap.release()
    out.release()  # 비디오 파일을 저장하고 종료
    print(f">>> DEBUG: Grad-CAM 결과 저장: {save_path}")

    # ffmpeg를 사용하여 변환 수행
    # 이미 저장된 원본 영상 경로
    output_path = os.path.join(save_path, "grad_cam_converted.mp4")

    # ffmpeg를 사용하여 변환 수행
    try:
        subprocess.run([
            'ffmpeg', '-i', output_video_path,
            '-vcodec', 'libx264',
            '-acodec', 'aac',
            '-strict', 'experimental',
            '-y',  # 덮어쓰기
            output_path
        ], check=True)

        print(f"✅ Video converted: {output_path}")
    except subprocess.CalledProcessError as e:
        print(f"Error occurred during conversion: {e}")



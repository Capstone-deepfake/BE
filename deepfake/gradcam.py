# backend/deepfake/gradcam.py

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

def apply_gradcam_to_image(model, img_path, target_layer, save_path):
    """
    1) img_path에서 이미지를 읽고, 224×224 전처리
    2) target_layer에 forward/backward hook 걸기
    3) model(input) → logits → 타깃 클래스(Fake 또는 Real) backward
    4) hook으로 받은 activations, gradients로 CAM 계산
    5) heatmap 합성 후 save_path에 저장
    """

    # 1) 원본 이미지 불러와서 RGB → PIL → 전처리 → Tensor
    orig = cv2.imread(img_path)
    orig_rgb = cv2.cvtColor(orig, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(orig_rgb)

    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406],
                             std=[0.229,0.224,0.225])
    ])
    input_tensor = preprocess(pil).unsqueeze(0).to(next(model.parameters()).device)

    # 2) Hook 준비
    activations = []
    gradients   = []

    def forward_hook(module, input, output):
        activations.append(output)

    def backward_hook(module, grad_in, grad_out):
        gradients.append(grad_out[0])

    # target_layer(conv1)에 hook 연결
    handle_fw = target_layer.register_forward_hook(forward_hook)
    handle_bw = target_layer.register_backward_hook(backward_hook)

    # 3) 모델 추론
    model.zero_grad()
    outputs = model(input_tensor)               # [1, 2]
    pred_class = outputs.argmax(dim=1).item()   # 예: 0=Real, 1=Fake
    score = outputs[0, pred_class]              # 타깃 클래스의 logits

    # 4) backward 호출 → gradients 기록
    score.backward(retain_graph=True)

    # hook 제거
    handle_fw.remove()
    handle_bw.remove()

    # 5) Grad-CAM 계산
    act = activations[0].detach()  # [1, 16, H1, W1]  (여기선 H1=W1=112)
    grad = gradients[0].detach()   # [1, 16, H1, W1]

    # 채널(g)별 평균으로 weight 계산
    weights = grad.mean(dim=(2,3), keepdim=True)  # [1, 16, 1, 1]
    cam = (weights * act).sum(dim=1, keepdim=True) # [1, 1, H1, W1]
    cam = F.relu(cam)                              # ReLU
    cam = cam.squeeze().cpu().numpy()              # [H1, W1]

    # 6) 원본 해상도(720×1280 등)로 CAM 히트맵 리사이즈
    cam_resized = cv2.resize(cam, (orig.shape[1], orig.shape[0]))
    cam_resized -= cam_resized.min()
    cam_resized /= (cam_resized.max() + 1e-8)  # 0~1 정규화
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)

    # 원본 BGR 위에 덧씌우기 (α=0.5)
    overlay = cv2.addWeighted(orig, 0.5, heatmap, 0.5, 0)

    cv2.imwrite(save_path, overlay)
    print(f">>> DEBUG: Grad-CAM 결과 저장: {save_path}")

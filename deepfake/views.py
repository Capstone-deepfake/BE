# backend/deepfake/views.py

import os
import cv2
import uuid
import traceback
import base64
import gc

from django.conf import settings
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt

import torch
from torchvision import transforms
from PIL import Image

from .modelNet import DeepfakeDetector
import face_recognition
from .gradcam import apply_gradcam_to_image

from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])


def get_gpt_explanation_with_data_uri(image_path: str) -> str:
    """
    로컬에 저장된 Grad-CAM 이미지를 Base64 → Data URI로 감싸
    'image_url' 메시지 블록에 넣어 OpenAI로 전송합니다.
    """
    try:
        with open(image_path, "rb") as f:
            img_b64 = base64.b64encode(f.read()).decode("utf-8")
        data_uri = f"data:image/jpeg;base64,{img_b64}"

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "해당 사진이 딥페이크인 근거를 다음과 같이 설명해줘: "
                                "얼굴과 배경 사이 경계선이 부자연스럽게 일그러지고, "
                                "깜빡이는 현상이 발생합니다. 또 얼굴 표정이 말과 어울리지 않으며 "
                                "눈 깜빡임 빈도가 비정상적으로 지나칩니다. 얼굴 이외로 조명과 "
                                "주변 환경으로 그림자의 방향이 이상하고 말 하는 내용과 입술 움직임이 일치하지 않습니다."
                            )
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": data_uri}
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


# ─── 모델 로드 ────────────────────────────────────────────────────────────────

model = DeepfakeDetector()
checkpoint_path = os.path.join(settings.BASE_DIR, "model", "checkpoint_1.pt")
if not os.path.exists(checkpoint_path):
    raise FileNotFoundError(f"모델 체크포인트를 찾을 수 없습니다: {checkpoint_path}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
state_dict = torch.load(checkpoint_path, map_location=device)
model.load_state_dict(state_dict, strict=False)
model.to(device)
model.eval()

preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


def extract_frames(video_path):
    """
    비디오 파일에서 프레임을 한 장씩 반환하는 제너레이터.
    finally절에서 cap.release() 호출.
    """
    cap = cv2.VideoCapture(video_path)
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            yield frame
    finally:
        cap.release()


# ─── 파일 업로드 및 검출 뷰 ────────────────────────────────────────────────────

@csrf_exempt
def upload_file(request):
    try:
        if request.method != "POST" or "file" not in request.FILES:
            return JsonResponse({"error": "잘못된 요청"}, status=400)

        # 1) 업로드된 비디오 저장
        uploaded_file = request.FILES["file"]
        upload_id = str(uuid.uuid4())
        video_dir = os.path.join(settings.MEDIA_ROOT, "uploads", upload_id)
        os.makedirs(video_dir, exist_ok=True)

        video_path = os.path.join(video_dir, uploaded_file.name)
        with open(video_path, "wb") as f:
            for chunk in uploaded_file.chunks():
                f.write(chunk)
        print(f"[DEBUG] video saved: {video_path}")

        # 2) 얼굴 크롭 프레임 추출
        frames_dir = os.path.join(video_dir, "frames")
        os.makedirs(frames_dir, exist_ok=True)

        saved_index = 0
        for idx, frame in enumerate(extract_frames(video_path)):
            if frame is None:
                break

            rgb = frame[:, :, ::-1]  # BGR → RGB
            faces = face_recognition.face_locations(rgb)
            print(f"[DEBUG] frame#{idx}: faces found = {len(faces)}")

            if faces:
                top, right, bottom, left = faces[0]
                face_img = rgb[top:bottom, left:right]
                h, w, _ = face_img.shape
                if h >= 32 and w >= 32:
                    face_bgr = cv2.cvtColor(face_img, cv2.COLOR_RGB2BGR)
                    fname = f"frame_{saved_index:04d}.jpg"
                    out_path = os.path.join(frames_dir, fname)
                    cv2.imwrite(out_path, face_bgr)
                    saved_index += 1
                    print(f"[DEBUG]   saved cropped face: {fname} ({w}×{h})")

            if saved_index >= 150:
                break

        print(f"[DEBUG] total cropped frames saved: {saved_index}")
        frame_files = sorted(os.listdir(frames_dir))
        print(f"[DEBUG] frame_files ({len(frame_files)}): {frame_files[:5]}")

        if not frame_files:
            print("[ERROR] Frame extraction failed: no cropped frames.")
            return JsonResponse(
                {"error": "프레임 추출 실패 (얼굴을 찾을 수 없습니다)"}, status=500
            )

        # 3) 모델 추론 (프레임별 fake 확률 수집)
        predictions = []
        score_list  = []
        for fname in frame_files:
            img_path = os.path.join(frames_dir, fname)
            image = Image.open(img_path).convert("RGB")
            input_tensor = preprocess(image).unsqueeze(0).to(device)
            with torch.no_grad():
                outputs = model(input_tensor)
                probs = torch.softmax(outputs, dim=1)[0].tolist()
                pred  = torch.argmax(outputs, dim=1).item()

            predictions.append(pred)
            score_list.append(probs[1])
            print(f"[DEBUG] frame={fname} → P(fake)={probs[1]:.3f}, pred={pred}")

        # 4) 종료 후 분류: fake_ratio만 0.5 기준
        fake_count   = sum(predictions)
        total_frames = len(predictions)
        fake_ratio   = fake_count / total_frames
        print(f"[DEBUG] fake_ratio={fake_ratio:.3f}, frames={total_frames}")

        if fake_ratio >= 0.5:
            result_label = "Fake"
            confidence   = round(fake_ratio * 100, 1)
        else:
            result_label = "Real"
            confidence   = round((1 - fake_ratio) * 100, 1)

        file_url = settings.MEDIA_URL + f"uploads/{upload_id}/{uploaded_file.name}"

        # 5) Real이면 바로 반환
        if result_label == "Real":
            print(f"[DEBUG] Final result: REAL (confidence {confidence}%)")
            return JsonResponse({
                "file_url":    file_url,
                "result":      result_label,
                "confidence":  confidence,
                "explanation": "이 영상은 Real로 판별되었습니다.",
            })

        # 6) Fake일 때 Grad-CAM + 설명 요청
        target_frame = os.path.join(frames_dir, frame_files[0])
        print(f"[DEBUG] Using {frame_files[0]} for Grad-CAM")
        gradcam_dir  = os.path.join(video_dir, "gradcam")
        os.makedirs(gradcam_dir, exist_ok=True)
        gradcam_path = os.path.join(gradcam_dir, "gradcam_result.jpg")

        apply_gradcam_to_image(model, target_frame, model.conv1, gradcam_path)
        print(f"[DEBUG] Grad-CAM saved: {gradcam_path}")

        # Matplotlib figure 닫기 (gradcam 내부에서 plt 사용 시)
        try:
            import matplotlib.pyplot as plt
            plt.close("all")
        except:
            pass

        explanation = get_gpt_explanation_with_data_uri(gradcam_path)
        print(f"[DEBUG] OpenAI explanation: {explanation[:60]}...")

        # 7) 리소스 정리
        torch.cuda.empty_cache()
        gc.collect()

        return JsonResponse({
            "file_url":    file_url,
            "result":      result_label,
            "confidence":  confidence,
            "gradcam_url": None,
            "explanation": explanation,
        })

    except Exception as e:
        print("[ERROR]", traceback.format_exc())
        # crash 대신 JSON 오류 응답
        return JsonResponse({"error": str(e)}, status=500)


# ─── 채팅 뷰 ─────────────────────────────────────────────────────────────────

@csrf_exempt
def chat_with_ai(request):
    try:
        if request.method != "POST":
            return JsonResponse({"error": "Only POST method is allowed"}, status=405)

        import json
        data = json.loads(request.body)
        user_message = data.get("message")
        video_context = data.get("context", {})

        if not user_message:
            return JsonResponse({"error": "Message is required"}, status=400)

        system_message = """You are DE-fake it's AI expert assistant specialized in deepfake detection. 
        You are knowledgeable about various deepfake detection techniques, their implications, and how to interpret results.
        Always maintain a friendly and professional tone, using emojis appropriately in your responses.
        When discussing confidence scores, explain what they mean in practical terms.
        Help users understand the implications of the detection results and what they should do next."""

        context_message = f"""Analysis Context 🔍:
        - Detection Result: {video_context.get('result','Unknown')} {'🚫' if video_context.get('result')=='Fake' else '✅'}
        - Confidence Score: {video_context.get('confidence',0)}% {'🎯' if video_context.get('confidence',0)>80 else '📊'}
        
        User Question: {user_message}"""

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user",   "content": context_message},
            ],
            max_tokens=500,
            temperature=0.7,
        )

        return JsonResponse({"response": response.choices[0].message.content})

    except Exception as e:
        print("[ERROR]", traceback.format_exc())
        return JsonResponse({"error": str(e)}, status=500)

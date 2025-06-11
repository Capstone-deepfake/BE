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

from .modelNet import Model
import face_recognition
from .gradcam import apply_gradcam_to_image
from .grad_cam_all import all_apply_gradcam_to_image

from openai import OpenAI
from dotenv import load_dotenv
from torchvision import transforms
import numpy as np
import subprocess

import logging
# Set up logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Create handlers
c_handler = logging.StreamHandler()
f_handler = logging.FileHandler('deepfake_debug.log')
c_handler.setLevel(logging.DEBUG)
f_handler.setLevel(logging.DEBUG)

# Create formatters and add it to handlers
log_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
c_handler.setFormatter(log_format)
f_handler.setFormatter(log_format)

# Add handlers to the logger
logger.addHandler(c_handler)
logger.addHandler(f_handler)

load_dotenv()
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

def get_gpt_explanation_with_data_uri(image_path: str, prediction: str, confidence: float) -> str:
    """
    GradCAM 이미지를 분석하여 딥페이크 탐지 결과를 설명합니다.
    이미지는 Base64로 인코딩되어 OpenAI API로 전송됩니다.
    """
    try:
        # Convert image to base64
        with open(image_path, "rb") as f:
            base64_image = base64.b64encode(f.read()).decode("utf-8")

        # Initialize OpenAI client
        # Send request to GPT-4 Vision
        response = client.chat.completions.create(
            model="gpt-4o",
            # model="ft:gpt-4o-2024-08-06:sercanyesilkoy::BbQqY0y0",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f"""모델 분석 결과:
- Verdict: {prediction}
- Confidence: {confidence:.1f}%

Please analyze the GradCAM heatmap in detail:
1. Which specific facial features (eyes, mouth, nose, etc.) show the strongest activation patterns?
2. What visual anomalies or manipulation patterns are highlighted in these regions?
3. How do the intensity and distribution of highlighted areas support the model's {prediction} verdict with {confidence:.1f}% confidence?
4. Are there any inconsistencies or artifacts in facial textures, lighting, or edges that the model detected?
5. Based on the heatmap analysis, which regions were most crucial in determining authenticity?"""
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            max_tokens=500
        )
        
        return response.choices[0].message.content

    except Exception as e:
        logger.error(f"OpenAI API error: {str(e)}")
        logger.error(traceback.format_exc())
        return f"GradCAM 분석 중 오류가 발생했습니다: {str(e)}"


def get_initial_analysis(gradcam_path: str, result: str, confidence: float) -> str:
    """
    Get initial analysis of Grad-CAM image from GPT-4 Vision when first showing results.
    """
    try:
        with open(gradcam_path, "rb") as f:
            base64_image = base64.b64encode(f.read()).decode("utf-8")
        
        logger.info("📊 Initial analysis image stats:")
        logger.info(f"  - Base64 preview: {base64_image[:100]}...")

        response = client.chat.completions.create(
            model="gpt-4o",
            # model="ft:gpt-4o-2024-08-06:sercanyesilkoy::BbQqY0y0",
            messages=[
                {
                    "role": "system",
                    "content": """You are a deepfake detection expert analyzing Grad-CAM visualizations.
In this image, I can see areas that our AI model considers suspicious for manipulation:
- Red/yellow highlights indicate regions the model finds manipulated
- Brighter colors mean stronger evidence of manipulation
- Focus on facial features, textures, and unnatural patterns
- Look for inconsistencies in lighting, skin texture, and facial boundaries"""
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f"""For this specific deepfake analysis:
- Model Verdict: {result}
- Confidence Score: {confidence}%

Based on the Grad-CAM visualization:
1. Looking at the highlighted regions, which specific areas appear to be manipulated?
2. What particular facial features or textures show signs of manipulation?
3. Are there any unnatural patterns or inconsistencies in the highlighted areas?
4. How do these suspicious regions support the {confidence}% confidence score?"""
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            max_tokens=500
        )
        
        logger.info("✅ Successfully received initial GPT-4 Vision analysis")
        return response.choices[0].message.content

    except Exception as e:
        logger.error(f"Error getting initial analysis: {str(e)}")
        logger.error(traceback.format_exc())
        return "Error occurred during Grad-CAM analysis"




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
        # VideoWriter 설정
        output_video_path = os.path.join(frames_dir, "preprocessed_video.mp4")
        out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'MJPG'), 30, (112,112))

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
                    # 잘라낸 얼굴 이미지를 (112, 112) 크기로 조절
                    resized_face = cv2.resize(face_bgr, (112, 112))
                    out.write(resized_face)

                    saved_index += 1
                    print(f"[DEBUG]   saved cropped face: {fname} ({w}×{h})")

            # # 현재 150 프레임만 자르는 상태
            # if saved_index >= 150:
            #     break

        print(f"[DEBUG] total cropped frames saved: {saved_index}")
        frame_files = sorted(os.listdir(frames_dir))
        print(f"[DEBUG] frame_files ({len(frame_files)}): {frame_files[:5]}")

        if not frame_files:
            print("[ERROR] Frame extraction failed: no cropped frames.")
            return JsonResponse(
                {"error": "프레임 추출 실패 (얼굴을 찾을 수 없습니다)"}, status=500
            )
        out.release()

        # 파일 변환
        converted_path = os.path.join(video_dir, "converted_video.mp4")

        # 3. ffmpeg 변환 수행
        try:
            subprocess.run([
                'ffmpeg', '-i', output_video_path,
                '-vcodec', 'libx264',
                '-acodec', 'aac',
                '-strict', 'experimental',
                '-y',  # 덮어쓰기
                converted_path
            ], check=True)

            print(f"✅ Video converted: {converted_path}")

        except subprocess.CalledProcessError as e:
            return JsonResponse({
                "error": "ffmpeg conversion failed",
                "detail": str(e)
            }, status=500)

        finally:
            file_url = settings.MEDIA_URL + f"uploads/{upload_id}/converted_video.mp4"
            # # 원본은 삭제
            # if os.path.exists(output_video_path):
            #     os.remove(output_video_path)

        # 3) 모델 추론 (프레임별 fake 확률 수집)

        # ─── 모델 로드 ────────────────────────────────────────────────────────────────

        checkpoint_path = os.path.join(settings.BASE_DIR, "deepfake", "checkpoint_1.pt")
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"모델 체크포인트를 찾을 수 없습니다: {checkpoint_path}")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        state_dict = torch.load(checkpoint_path, map_location=device)
        model = Model(num_classes=2, model_name='resnext50_32x4d').to(device)
        model.load_state_dict(state_dict, strict=False)
        model.to(device)
        model.eval()

        transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((112, 112)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                            [0.229, 0.224, 0.225])
        ])


        # 결과 저장 리스트
        results = []
        label_list = []
        folder_path_list=[]
        frame_probs = []

        with torch.no_grad():
            cap = cv2.VideoCapture(video_path)
            frame_preds = []
            frame_idx = 0

            success, frame = cap.read()

            while success:
                frame_idx += 1
                if frame_idx % 5 == 0:  # 매 5번째 프레임만 뽑아서 예측 (속도 + 대표성)
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    input_tensor = transform(frame)
                    input_tensor = input_tensor.unsqueeze(0).unsqueeze(0)  # (batch=1, seq_len=1, c=3, h, w)
                    input_tensor = input_tensor.to(device).float()

                    fmap, outputs = model(input_tensor)
                    probs = torch.softmax(outputs, dim=1)
                    frame_probs.append(probs[0].cpu().numpy())  # [fake_prob, real_prob]
                    _, predicted = torch.max(outputs, 1)


                    frame_preds.append(predicted.item())

                success, frame = cap.read()

            cap.release()

            # 비디오 하나에 대한 최종 예측
            frame_probs = np.array(frame_probs)
            if len(frame_preds) == 0:
                final_prediction = 'Unknown'
                final_probability = 0.0
            else:
                avg_probs = np.mean(frame_probs, axis=0)  # [mean_fake, mean_real]
                majority = round(sum(frame_preds) / len(frame_preds))  # 다수결
                print(sum(frame_preds),"here!!!!!",len(frame_preds))
                final_prediction = 'REAL' if majority == 1 else 'FAKE'
                final_probability = avg_probs[1] if final_prediction == 'REAL' else avg_probs[0]

        # predictions = []
        # score_list  = []
        # for fname in frame_files:
        #     img_path = os.path.join(frames_dir, fname)
        #     image = Image.open(img_path).convert("RGB")
        #     input_tensor = preprocess(image).unsqueeze(0).to(device)
        #     with torch.no_grad():
        #         outputs = model(input_tensor)
        #         probs = torch.softmax(outputs, dim=1)[0].tolist()
        #         pred  = torch.argmax(outputs, dim=1).item()

        #     predictions.append(pred)
        #     score_list.append(probs[1])
        #     print(f"[DEBUG] frame={fname} → P(fake)={probs[1]:.3f}, pred={pred}")

        # # 4) 종료 후 분류: fake_ratio만 0.5 기준
        # fake_count   = sum(predictions)
        # total_frames = len(predictions)
        # fake_ratio   = fake_count / total_frames
        # print(f"[DEBUG] fake_ratio={fake_ratio:.3f}, frames={total_frames}")

        # if fake_ratio >= 0.5:
        #     result_label = "Fake"
        #     confidence   = round(fake_ratio * 100, 1)
        # else:
        #     result_label = "Real"
        #     confidence   = round((1 - fake_ratio) * 100, 1)

        print(final_prediction)
        print(f"{final_probability * 100:.2f}")
        # 5) Real이면 바로 반환
        if final_prediction == "REAL":
            print(f"[DEBUG] Final result: REAL (confidence {final_probability * 100:.2f}%)")

            # return JsonResponse({
            #     "file_url":    file_url,
            #     "result":      result_label,
            #     "confidence":  confidence,
            #     "explanation": "이 영상은 Real로 판별되었습니다.",
            # })
            return JsonResponse({
                "file_url":    file_url,
                "result":      final_prediction,
                "confidence":  f"{final_probability * 100:.2f}",
                "explanation": "이 영상은 Real로 판별되었습니다.",
            })

        # 6) Fake일 때 Grad-CAM + 설명 요청
        print(f"[DEBUG] Using {frame_files[0]} for Grad-CAM")
        gradcam_dir  = os.path.join(video_dir, "gradcam")
        os.makedirs(gradcam_dir, exist_ok=True)
        # gradcam_path = os.path.join(gradcam_dir, "gradcam_result.jpg")

       # apply_gradcam_to_image(model, target_frame, model.conv1, gradcam_path)
        all_apply_gradcam_to_image(model,output_video_path , gradcam_dir) 
        print(f"[DEBUG] Grad-CAM saved: {gradcam_dir}")

        # # Matplotlib figure 닫기 (gradcam 내부에서 plt 사용 시)
        # try:
        #     import matplotlib.pyplot as plt
        #     plt.close("all")
        # except:
        #     pass
        
        top_frame_path = os.path.join(gradcam_dir, "Top_1_frame.jpg")
        # Get initial analysis from GPT using the aggregated GradCAM frame
        explanation = get_initial_analysis(top_frame_path, final_prediction, f"{final_probability * 100:.2f}")
        # explanation = get_gpt_explanation_with_data_uri(top_frame_path)
        print(f"[DEBUG] OpenAI explanation: {explanation[:60]}...")
        gradcam_url = settings.MEDIA_URL + f"uploads/{upload_id}/gradcam/grad_cam_converted.mp4"

        # 7) 리소스 정리
        torch.cuda.empty_cache()
        gc.collect()

        return JsonResponse({
            "file_url":    file_url,
            "result":      final_prediction,
            "confidence":  f"{final_probability * 100:.2f}",
            "gradcam_url": gradcam_url,
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
        
        logger.info("🔍 Received chat request:")
        logger.info(f"Message: {user_message}")
        logger.info(f"Context: {json.dumps(video_context, indent=2)}")

        if not user_message:
            return JsonResponse({"error": "Message is required"}, status=400)

        # Get upload ID from either video_url or file_url
        url = video_context.get("file_url") or video_context.get("video_url", "")
        logger.info(f"📁 URL from context: {url}")
        
        if url:
            parts = url.split('/media/uploads/')
            if len(parts) > 1:
                upload_path = parts[1].split('/')
                if len(upload_path) >= 1:
                    upload_id = upload_path[0]
                    logger.info(f"📂 Extracted upload ID: {upload_id}")
                    
                    media_root = settings.MEDIA_ROOT
                    gradcam_frame_path = os.path.join(media_root, "uploads", upload_id, "gradcam","Top_1_frame.jpg")
                    logger.info(f"🖼️ Looking for GradCAM frame at: {gradcam_frame_path}")
                    
                    if os.path.exists(gradcam_frame_path):
                        try:
                            with open(gradcam_frame_path, "rb") as f:
                                base64_image = base64.b64encode(f.read()).decode("utf-8")
                            
                            logger.info("📊 Chat image stats:")
                            logger.info(f"  - Base64 preview: {base64_image[:100]}...")

                            response = client.chat.completions.create(
                                model="gpt-4o",
                                messages=[
                                    {
                                        "role": "system",
                                        "content": """You are a deepfake detection expert analyzing Grad-CAM results.
The highlighted areas in this image indicate potential manipulation:
- Red/yellow regions show where the model detects suspicious features
- Brighter colors indicate stronger evidence of manipulation
- Pay special attention to facial features, skin textures, and boundaries
Please explain what specific aspects suggest this might be manipulated."""
                                    },
                                    {
                                        "role": "user",
                                        "content": [
                                            {
                                                "type": "text",
                                                "text": f"""Analysis Context:
- Detection Result: {video_context.get('result','Unknown')} {'🚫' if video_context.get('result')=='FAKE' else '✅'}
- Confidence Level: {video_context.get('confidence',0)}%

User Question: {user_message}

Looking at the Grad-CAM visualization:
1. Which specific areas show signs of potential manipulation?
2. What unusual patterns or artifacts do you notice in the highlighted regions?
3. How do these highlighted areas explain the {video_context.get('confidence',0)}% confidence score?"""
                                            },
                                            {
                                                "type": "image_url",
                                                "image_url": {
                                                    "url": f"data:image/jpeg;base64,{base64_image}"
                                                }
                                            }
                                        ]
                                    }
                                ],
                                max_tokens=500
                            )
                            
                            logger.info("✅ Successfully sent image to GPT-4 Vision and received response")
                            return JsonResponse({
                                "response": response.choices[0].message.content,
                                "image_processed": True
                            })
                            
                        except Exception as e:
                            logger.error(f"❌ Error processing GradCAM frame: {str(e)}")
                            logger.error(traceback.format_exc())

        # Fall back to text-only response
        logger.info("⚠️ Falling back to text-only response")
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": """You are a deepfake detection expert.
You analyze and explain how our AI system identifies manipulated content.
Please provide detailed insights about potential manipulation indicators."""
                },
                {
                    "role": "user",
                    "content": f"""Analysis Context:
- Detection Result: {video_context.get('result','Unknown')} {'🚫' if video_context.get('result')=='FAKE' else '✅'}
- Confidence Level: {video_context.get('confidence',0)}%

User Question: {user_message}"""
                }
            ],
            max_tokens=500
        )

        return JsonResponse({
            "response": response.choices[0].message.content,
            "image_processed": False
        })

    except Exception as e:
        logger.error(f"❌ Chat error: {str(e)}")
        logger.error(traceback.format_exc())
        return JsonResponse({
            "error": str(e),
            "details": traceback.format_exc()
        }, status=500)

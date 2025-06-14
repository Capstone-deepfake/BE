# backend/deepfake/views.py

import os
import cv2
import uuid
import traceback
import base64
import gc
import logging

from django.conf import settings
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile

import torch
from torchvision import transforms
from PIL import Image

from .modelNet import DeepfakeDetector
from .mesonet_gradcam import EnhancedMesoNet, EnhancedMesoNetLSTM, analyze_video_enhanced, create_gradcam_video
import face_recognition
from .gradcam import apply_gradcam_to_image

from openai import OpenAI
from dotenv import load_dotenv

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



def get_gpt_explanation_with_data_uri(image_path: str, prediction: str, confidence: float) -> str:
    """
    GradCAM 이미지를 분석하여 딥페이크 탐지 결과를 설명합니다.
    이미지는 Base64로 인코딩되어 OpenAI API로 전송됩니다.
    """
    try:
        # Convert image to base64
        with open(image_path, "rb") as f:
            base64_image = base64.b64encode(f.read()).decode("utf-8")



        # Send request to GPT-4 Vision
        response = client.chat.completions.create(
            model="gpt-4o",
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
            messages=[
                {
                    "role": "system",
                    "content": """You are a deepfake detection expert. Provide a very concise analysis in 5-6 sentences maximum that explains:
1. The strongest evidence of manipulation in the Grad-CAM visualization
2. How this evidence supports the model's confidence score"""
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f"""The model detected this as {result} with {confidence}% confidence.

Looking at the Grad-CAM visualization, explain in 5-6 sentences:
1. What are the most significant manipulation indicators?
2. How do these findings justify the {confidence}% confidence score?"""
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
            max_tokens=300
        )
        
        logger.info("✅ Successfully received initial GPT-4 Vision analysis")
        return response.choices[0].message.content

    except Exception as e:
        logger.error(f"Error getting initial analysis: {str(e)}")
        logger.error(traceback.format_exc())
        return "Error occurred during Grad-CAM analysis"


# ─── 모델 로드 ────────────────────────────────────────────────────────────────

try:
    logger.info("Loading model...")
    model = EnhancedMesoNetLSTM()
    checkpoint_path = os.path.join(settings.BASE_DIR, "model", "best_model_base_config (1).pth")
    if not os.path.exists(checkpoint_path):
        logger.error(f"Model checkpoint not found at: {checkpoint_path}")
        raise FileNotFoundError(f"모델 체크포인트를 찾을 수 없습니다: {checkpoint_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
        state_dict = {k.replace('mesonet.', ''): state_dict[k] for k in state_dict.keys()}
        logger.info("Successfully loaded model state dict")
    else:
        state_dict = checkpoint
        logger.info("Using checkpoint directly as state dict")
    
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()
    logger.info("Model loaded successfully")
except Exception as e:
    logger.error(f"Error loading model: {str(e)}")
    logger.error(traceback.format_exc())
    raise

preprocess = transforms.Compose([
    transforms.Resize((128, 128)),  # Changed from 224 to match the model
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
        logger.info("Received file upload request")
        
        if request.method != "POST":
            logger.warning("Invalid request method")
            return JsonResponse({"error": "POST 요청만 허용됩니다"}, status=405)
        
        if "file" not in request.FILES:
            logger.warning("No file in request")
            return JsonResponse({"error": "파일이 업로드되지 않았습니다"}, status=400)

        # Save uploaded video
        uploaded_file = request.FILES["file"]
        logger.info(f"Processing uploaded file: {uploaded_file.name}")
        
        upload_id = str(uuid.uuid4())
        video_dir = os.path.join(settings.MEDIA_ROOT, "uploads", upload_id)
        os.makedirs(video_dir, exist_ok=True)

        video_path = os.path.join(video_dir, uploaded_file.name)
        logger.info(f"Saving video to: {video_path}")
        
        with open(video_path, "wb") as f:
            for chunk in uploaded_file.chunks():
                f.write(chunk)
        logger.info("Video file saved successfully")

        # Extract face frames
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

        # Model inference (collect fake probabilities per frame)
        predictions = []
        score_list = []
        
        # Load model once
        checkpoint_path = os.path.join(settings.BASE_DIR, "model", "best_model_base_config (1).pth")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model = EnhancedMesoNetLSTM()
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()

        # Create GradCAM video and aggregated frame
        try:
            # Path for aggregated GradCAM visualization
            aggregated_gradcam_path = os.path.join(video_dir, "aggregated_gradcam.jpg")
            gradcam_video_path = os.path.join(video_dir, "gradcam_" + uploaded_file.name)
            
            logger.info(f"Creating GradCAM visualizations...")
            
            # Create both video and aggregated frame
            create_gradcam_video(
                frames_dir=frames_dir,
                output_path=gradcam_video_path,
                model=model,
                frame_files=frame_files,
                aggregated_output_path=aggregated_gradcam_path
            )
            
            if not os.path.exists(gradcam_video_path) or not os.path.exists(aggregated_gradcam_path):
                logger.error("GradCAM files were not created")
                return JsonResponse({"error": "GradCAM 생성에 실패했습니다."}, status=500)
                
            logger.info("Successfully created GradCAM visualizations")

            # Analyze frames
            for fname in frame_files:
                img_path = os.path.join(frames_dir, fname)
                try:
                    result = analyze_video_enhanced(img_path, checkpoint_path)
                    if result is None:
                        logger.error(f"Failed to analyze frame {fname}")
                        continue
                    
                    pred = 1 if result['prediction'] == 'FAKE' else 0
                    confidence = result['confidence'] / 100.0  # Convert back to 0-1 range
                    
                    predictions.append(pred)
                    score_list.append(confidence)
                    logger.info(f"[DEBUG] frame={fname} → P(fake)={confidence:.3f}, pred={pred}")
                except Exception as e:
                    logger.error(f"Error analyzing frame {fname}: {str(e)}")
                    logger.error(traceback.format_exc())
                    continue

            if not predictions:
                return JsonResponse({
                    "error": "영상 분석 실패: 얼굴을 찾을 수 없거나 분석에 실패했습니다."
                }, status=500)

            # Get final classification
            fake_count = sum(predictions)
            total_frames = len(predictions)
            fake_ratio = fake_count / total_frames if total_frames > 0 else 0.5
            logger.info(f"[DEBUG] fake_ratio={fake_ratio:.3f}, frames={total_frames}")

            if fake_ratio >= 0.5:
                result_label = "Fake"
                confidence = round(fake_ratio * 100, 1)
            else:
                result_label = "Real"
                confidence = round((1 - fake_ratio) * 100, 1)

            # Get initial analysis from GPT using the aggregated GradCAM frame
            initial_analysis = get_initial_analysis(aggregated_gradcam_path, result_label, confidence)

            # Return URLs and analysis
            file_url = settings.MEDIA_URL + f"uploads/{upload_id}/{uploaded_file.name}"
            gradcam_url = settings.MEDIA_URL + f"uploads/{upload_id}/gradcam_{uploaded_file.name}"
            aggregated_url = settings.MEDIA_URL + f"uploads/{upload_id}/aggregated_gradcam.jpg"

            return JsonResponse({
                "file_url": file_url,
                "gradcam_url": gradcam_url,
                "aggregated_url": aggregated_url,
                "result": result_label,
                "confidence": confidence,
                "explanation": initial_analysis,
                "playback_speed": 0.25,  # Set video playback speed to 0.25x
                "explanation_speed": "fast"  # Speed up explanation text appearance
            })

        except Exception as e:
            logger.error(f"Error creating GradCAM: {str(e)}")
            logger.error(traceback.format_exc())
            return JsonResponse({"error": f"GradCAM 생성 중 오류 발생: {str(e)}"}, status=500)

    except Exception as e:
        logger.error(f"Error processing upload: {str(e)}")
        logger.error(traceback.format_exc())
        return JsonResponse({
            "error": f"처리 중 오류가 발생했습니다: {str(e)}",
            "details": traceback.format_exc()
        }, status=500)


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
        is_first_question = video_context.get("is_first_question", False)
        
        logger.info("🔍 Received chat request:")
        logger.info(f"Message: {user_message}")
        logger.info(f"Context: {json.dumps(video_context, indent=2)}")

        if not user_message:
            return JsonResponse({"error": "Message is required"}, status=400)

        # Only process image for the first question
        if is_first_question:
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
                        gradcam_frame_path = os.path.join(media_root, "uploads", upload_id, "aggregated_gradcam.jpg")
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
- Detection Result: {video_context.get('result','Unknown')} {'🚫' if video_context.get('result')=='Fake' else '✅'}
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

        # Fall back to text-only response for non-first questions
        logger.info("⚠️ Using text-only response")
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
- Detection Result: {video_context.get('result','Unknown')} {'🚫' if video_context.get('result')=='Fake' else '✅'}
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

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import google.generativeai as genai
import os
import json
import base64
from typing import List, Dict, Any
from pydantic import BaseModel
from dotenv import load_dotenv
import fitz  # PyMuPDF
from PIL import Image, ImageOps
import io
import logging
import uuid

import numpy as np

# 환경 변수 로드
load_dotenv()

# .env 파일에 GEMINI_API_KEY가 있는지 확인하고 로드
if os.getenv("GEMINI_API_KEY") is None:
    raise ValueError("GEMINI_API_KEY가 .env 파일에 설정되지 않았습니다.")

app = FastAPI(title="행정문서 OCR & QA 챗봇")

# uploads 디렉토리 생성
os.makedirs("static/uploads", exist_ok=True)

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Gemini 클라이언트 설정
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# docTR OCR 모델 초기화
try:
    from doctr.io import DocumentFile
    from doctr.models import ocr_predictor
    ocr_model = ocr_predictor('db_resnet50', 'crnn_vgg16_bn', pretrained=True)
    print("docTR OCR 모델이 성공적으로 로드되었습니다.")
except Exception as e:
    print(f"docTR OCR 모델 로드 실패: {e}")
    ocr_model = None

# 데이터 모델
class OCRResult(BaseModel):
    id: str
    text: str
    bbox: List[float]

class OCRResponse(BaseModel):
    ocr_data: List[OCRResult]
    image_url: str
    image_width: int
    image_height: int

class QARequest(BaseModel):
    user_question: str
    ocr_result: List[OCRResult]

class QAResponse(BaseModel):
    answer: str
    sources: List[Dict[str, Any]]

# 전역 변수로 OCR 결과 저장
current_ocr_result = []
current_image_dimensions = {"width": 0, "height": 0}

def convert_doctr_to_json(doctr_result, img_w: int, img_h: int):
    """docTR 결과를 JSON 형식으로 변환"""
    ocr_data = []
    
    for page_idx, page in enumerate(doctr_result.pages):
        for block_idx, block in enumerate(page.blocks):
            for line_idx, line in enumerate(block.lines):
                # 라인의 모든 단어를 합쳐서 텍스트 생성
                text = " ".join([word.value for word in line.words])
                
                # bbox 좌표 추출 (상대 좌표를 절대 좌표로 변환)
                geometry = line.geometry
                x1, y1, x2, y2 = geometry[0][0], geometry[0][1], geometry[1][0], geometry[1][1]
                
                # 절대 좌표로 변환
                abs_bbox = [
                    x1 * img_w,
                    y1 * img_h,
                    x2 * img_w,
                    y2 * img_h
                ]
                
                ocr_data.append({
                    "id": f"block_{page_idx}_{block_idx}_{line_idx}",
                    "text": text,
                    "bbox": abs_bbox
                })
    
    return ocr_data

def qa_search(question: str, ocr_data: List[Dict]):
    """Gemini API를 사용한 지능적 QA 함수"""
    try:
        # OCR 데이터를 텍스트로 변환
        document_text = ""
        text_blocks = []
        
        for i, block in enumerate(ocr_data):
            block_text = f"[블록 {i+1}] {block['text']}"
            document_text += block_text + "\n"
            text_blocks.append({
                "index": i+1,
                "id": block['id'],
                "text": block['text'],
                "bbox": block['bbox']
            })
        
        if not document_text.strip():
            return {
                "answer": "문서에서 텍스트를 찾을 수 없습니다.",
                "sources": []
            }
        
        # Gemini API 호출
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        prompt = f"""다음은 문서에서 추출된 텍스트입니다:

{document_text}

질문: {question}

위 문서 내용을 바탕으로 질문에 대한 정확하고 구체적인 답변을 제공해주세요. 
답변할 때는 다음 규칙을 따라주세요:
1. 문서에 있는 정보만을 바탕으로 답변하세요
2. 답변의 근거가 되는 블록 번호를 명시하세요 (예: [블록 1], [블록 3] 참조)
3. 문서에서 해당 정보를 찾을 수 없다면 "문서에서 해당 정보를 찾을 수 없습니다"라고 답변하세요
4. 간결하고 명확하게 답변하세요"""

        response = model.generate_content(prompt)
        answer = response.text
        
        # 답변에서 참조된 블록 찾기
        sources = []
        for block in text_blocks:
            if f"블록 {block['index']}" in answer or block['text'] in question.lower():
                sources.append({
                    "id": block['id'],
                    "bbox": block['bbox'],
                    "text": block['text']
                })
        
        # 소스가 없으면 관련성이 높은 블록을 찾아서 추가
        if not sources:
            q_keywords = question.lower().split()
            for block in text_blocks:
                score = sum([1 for kw in q_keywords if kw in block['text'].lower()])
                if score > 0:
                    sources.append({
                        "id": block['id'],
                        "bbox": block['bbox'],
                        "text": block['text']
                    })
                    break
        
        return {
            "answer": answer,
            "sources": sources[:3]  # 최대 3개의 소스만 반환
        }
        
    except Exception as e:
        logging.error(f"QA 검색 오류: {e}")
        return {
            "answer": f"답변 생성 중 오류가 발생했습니다: {str(e)}",
            "sources": []
        }

def _normalize_bbox(bbox, img_w: int, img_h: int):
    """BBOX 정규화 - 항상 [x1, y1, x2, y2] 픽셀 좌표로 반환
    - 모델이 [y1, x1, y2, x2]처럼 축이 뒤바뀐 값을 주는 경우를 자동 교정
    - [x, y, w, h] 형식도 처리
    """
    try:
        a, b, c, d = [float(x) for x in bbox[:4]]
    except Exception:
        return None
    
    def to_xyxy(a, b, c, d, within01: bool, swapped: bool, xywh: bool):
        if within01:
            # 정규화 좌표 (0~1)
            if not swapped:
                x1, y1, x2, y2 = a * img_w, b * img_h, c * img_w, d * img_h
            else:
                x1, y1, x2, y2 = b * img_w, a * img_h, d * img_w, c * img_h
        else:
            # 픽셀 좌표
            if xywh:
                if not swapped:
                    x1, y1, x2, y2 = a, b, a + c, b + d
                else:
                    x1, y1, x2, y2 = b, a, b + d, a + c
            else:
                if not swapped:
                    x1, y1, x2, y2 = a, b, c, d
                else:
                    x1, y1, x2, y2 = b, a, d, c
        return x1, y1, x2, y2
    
    def clamp(v, lo, hi):
        return max(lo, min(hi, v))
    
    def sanitize(x1, y1, x2, y2):
        # 좌표 순서 보정 및 경계 내 클램프
        x1, y1, x2, y2 = float(x1), float(y1), float(x2), float(y2)
        if x2 < x1:
            x1, x2 = x2, x1
        if y2 < y1:
            y1, y2 = y2, y1
        x1 = clamp(x1, 0, img_w)
        y1 = clamp(y1, 0, img_h)
        x2 = clamp(x2, 0, img_w)
        y2 = clamp(y2, 0, img_h)
        return x1, y1, x2, y2
    
    def score_bbox(x1, y1, x2, y2):
        w = max(0.0, x2 - x1)
        h = max(0.0, y2 - y1)
        if w < 1 or h < 1:
            return -1.0
        # 경계 위반 패널티(클램프 이전 유효성은 대략적으로 반영)
        out = 0.0
        out += 1.0 if x1 <= 0 or y1 <= 0 else 0.0
        out += 1.0 if x2 >= img_w or y2 >= img_h else 0.0
        ar = w / h if h > 0 else 999.0
        skinny_penalty = 1.0 if ar < 0.08 else 0.0  # 비정상적으로 가는 세로줄 방지
        return (min(1.0, w / img_w) + min(1.0, h / img_h)) - out - skinny_penalty
    
    within01 = 0.0 <= a <= 1.0 and 0.0 <= b <= 1.0 and 0.0 <= c <= 1.0 and 0.0 <= d <= 1.0
    xywh = False
    if not within01:
        # [x1,y1,x2,y2]인지 [x,y,w,h]인지 대략 판별
        if not (c > a and d > b):
            xywh = True
    
    # 후보 1: 정상 해석
    o_x1, o_y1, o_x2, o_y2 = to_xyxy(a, b, c, d, within01, swapped=False, xywh=xywh)
    o_x1, o_y1, o_x2, o_y2 = sanitize(o_x1, o_y1, o_x2, o_y2)
    o_score = score_bbox(o_x1, o_y1, o_x2, o_y2)
    
    # 후보 2: X/Y 스왑 해석
    s_x1, s_y1, s_x2, s_y2 = to_xyxy(a, b, c, d, within01, swapped=True, xywh=xywh)
    s_x1, s_y1, s_x2, s_y2 = sanitize(s_x1, s_y1, s_x2, s_y2)
    s_score = score_bbox(s_x1, s_y1, s_x2, s_y2)
    
    # 더 그럴듯한 후보 선택 (스왑 점수가 확실히 좋으면 교체)
    x1, y1, x2, y2 = (o_x1, o_y1, o_x2, o_y2)
    if s_score > o_score + 0.25:
        x1, y1, x2, y2 = (s_x1, s_y1, s_x2, s_y2)
    
    if (x2 - x1) < 1 or (y2 - y1) < 1:
        return None
    
    return [int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))]

@app.post("/api/ocr", response_model=OCRResponse)
async def extract_ocr(file: UploadFile = File(...)):
    """업로드된 이미지 또는 PDF에서 OCR을 수행하여 JSON 형식과 이미지 URL을 반환"""
    global current_ocr_result, current_image_dimensions
    
    try:
        # 파일 읽기
        contents = await file.read()
        
        # 고유 파일명 생성 및 저장 경로 설정
        file_extension = os.path.splitext(file.filename)[1]
        if not file_extension or file.content_type == "application/pdf":
            file_extension = '.png'
        unique_filename = f"{uuid.uuid4()}{file_extension}"
        save_path = os.path.join("static", "uploads", unique_filename)
        image_url = f"/static/uploads/{unique_filename}"
        
        # 원본 이미지 크기
        img_w, img_h = 0, 0
        
        # 파일 타입 확인
        if file.content_type == "application/pdf":
            # PDF 처리
            pdf_document = fitz.open(stream=contents, filetype="pdf")
            
            # 첫 페이지만 이미지로 변환 (고해상도)
            page = pdf_document.load_page(0)
            # 해상도를 높여서 더 정확한 좌표 추출
            mat = fitz.Matrix(2.0, 2.0)  # 2배 확대
            pix = page.get_pixmap(matrix=mat)
            img_data = pix.tobytes("png")
            img_w, img_h = pix.width, pix.height
            
            # 이미지 저장 및 base64 인코딩
            with open(save_path, "wb") as f:
                f.write(img_data)
            base64_image = base64.b64encode(img_data).decode('utf-8')
            
            pdf_document.close()
                
        elif file.content_type.startswith('image/'):
            # 이미지 처리 (EXIF Orientation 보정 후 저장)
            try:
                pil = Image.open(io.BytesIO(contents))
                # EXIF 기반 실제 회전 적용 (가로/세로 스왑 문제 예방)
                pil = ImageOps.exif_transpose(pil)
                img_w, img_h = pil.size

                # PNG로 저장하여 일관된 표시/전송 (EXIF 제거 효과 포함)
                file_extension = '.png'
                unique_filename = f"{uuid.uuid4()}{file_extension}"
                save_path = os.path.join("static", "uploads", unique_filename)
                image_url = f"/static/uploads/{unique_filename}"

                buf = io.BytesIO()
                pil.save(buf, format="PNG")
                img_data = buf.getvalue()

                with open(save_path, "wb") as f:
                    f.write(img_data)
                base64_image = base64.b64encode(img_data).decode('utf-8')
            except Exception as ex:
                # 실패 시 원본 그대로 저장 및 크기 추출 시도
                with open(save_path, "wb") as f:
                    f.write(contents)
                base64_image = base64.b64encode(contents).decode('utf-8')
                try:
                    pil = Image.open(io.BytesIO(contents))
                    img_w, img_h = pil.size
                except Exception:
                    img_w, img_h = 0, 0
        else:
            raise HTTPException(status_code=400, detail="지원하지 않는 파일 형식입니다. 이미지 또는 PDF 파일만 업로드 가능합니다.")
        
        # 이미지 차웬 저장
        current_image_dimensions = {"width": img_w, "height": img_h}
        
        # docTR OCR 처리
        if ocr_model is not None:
            try:
                # 이미지를 docTR 형식으로 변환
                doc = DocumentFile.from_images([save_path])
                result = ocr_model(doc)
                
                # docTR 결과를 JSON으로 변환
                ocr_data = convert_doctr_to_json(result, img_w, img_h)
                
                # OCRResult 객체로 변환
                normalized = []
                for item in ocr_data:
                    normalized.append(OCRResult(
                        id=item['id'],
                        text=item['text'],
                        bbox=item['bbox']
                    ))
                
                current_ocr_result = normalized
                return OCRResponse(
                    ocr_data=current_ocr_result, 
                    image_url=image_url,
                    image_width=img_w,
                    image_height=img_h
                )
                
            except Exception as doctr_err:
                logging.error(f"docTR OCR 처리 실패: {doctr_err}")
                # Gemini로 폴백
                pass
        
        # Gemini API 폴백 호출
        try:
            model = genai.GenerativeModel('gemini-1.5-flash')
            
            prompt = f"""이미지를 분석해서 문서의 모든 텍스트를 추출하고 각각의 정확한 위치를 찾아주세요.
이미지 크기는 {img_w} x {img_h} 픽셀입니다.

다음 JSON 형식으로 응답해주세요:
[
  {{
    "id": "block_001",
    "text": "추출된 텍스트",
    "bbox": [x1, y1, x2, y2]
  }}
]

중요 사항:
1. bbox는 반드시 [x1, y1, x2, y2] 형식의 픽셀 좌표입니다
2. x1,y1은 텍스트 블록의 왼쪽 위 모서리
3. x2,y2는 텍스트 블록의 오른쪽 아래 모서리
4. 좌표는 0부터 시작하며 이미지 경계를 벗어나면 안 됩니다
5. 제목, 본문, 표, 목록 등을 모두 포함해서 추출해주세요
6. 텍스트가 여러 줄인 경우 전체 영역을 포함하는 bbox를 만들어주세요

반드시 유효한 JSON 배열만 응답하고 다른 텍스트는 포함하지 마세요."""

            image_part = {
                "mime_type": "image/png",
                "data": base64_image
            }

            response = model.generate_content([prompt, image_part])
            content = response.text

            # JSON 파싱
            try:
                if content.strip().startswith("```json"):
                    content = content.strip()[7:-3].strip()
                elif content.strip().startswith("```"):
                    # 일반적인 코드 블록 제거
                    lines = content.strip().split('\n')
                    content = '\n'.join(lines[1:-1])
                
                ocr_data = json.loads(content)
                
                # BBOX 정규화 및 검증
                normalized = []
                for i, item in enumerate(ocr_data):
                    try:
                        bbox = item.get('bbox', [])
                        if len(bbox) != 4:
                            continue
                            
                        normalized_bbox = _normalize_bbox(bbox, img_w, img_h)
                        if normalized_bbox:
                            normalized.append(OCRResult(
                                id=item.get('id', f'block_{i:03d}'),
                                text=item.get('text', ''),
                                bbox=normalized_bbox
                            ))
                    except Exception as e:
                        logging.warning(f"BBOX 정규화 실패: {e}")
                        continue
                
                current_ocr_result = normalized
                return OCRResponse(
                    ocr_data=current_ocr_result, 
                    image_url=image_url,
                    image_width=img_w,
                    image_height=img_h
                )
                
            except json.JSONDecodeError as e:
                logging.error(f"JSON 파싱 실패: {e}, 응답: {content[:500]}")
                # JSON 추출 재시도
                import re
                json_match = re.search(r'\[.*\]', content, re.DOTALL)
                if json_match:
                    try:
                        json_data = json.loads(json_match.group())
                        normalized = []
                        for i, item in enumerate(json_data):
                            bbox = item.get('bbox', [])
                            if len(bbox) == 4:
                                normalized_bbox = _normalize_bbox(bbox, img_w, img_h)
                                if normalized_bbox:
                                    normalized.append(OCRResult(
                                        id=item.get('id', f'block_{i:03d}'),
                                        text=item.get('text', ''),
                                        bbox=normalized_bbox
                                    ))
                        current_ocr_result = normalized
                        return OCRResponse(
                            ocr_data=current_ocr_result, 
                            image_url=image_url,
                            image_width=img_w,
                            image_height=img_h
                        )
                    except Exception as e2:
                        logging.error(f"재시도 JSON 파싱도 실패: {e2}")
                
                raise HTTPException(status_code=500, detail="OCR 결과를 파싱할 수 없습니다.")

        except Exception as api_err:
            logging.error(f"Gemini API 오류: {api_err}")
            raise HTTPException(status_code=500, detail=f"OCR 처리 중 오류: {str(api_err)}")
        
    except Exception as e:
        logging.error("OCR 처리 중 예외 발생", exc_info=True)
        raise HTTPException(status_code=500, detail=f"OCR 처리 중 오류가 발생했습니다: {str(e)}")

@app.post("/api/qa", response_model=QAResponse)
async def answer_question(request: QARequest):
    """OCR 결과와 사용자 질문을 바탕으로 답변 생성"""
    try:
        # OCR 결과를 딕셔너리 형태로 변환
        ocr_blocks = []
        for item in request.ocr_result:
            ocr_blocks.append({
                "id": item.id,
                "text": item.text,
                "bbox": item.bbox
            })
        
        # docTR 기반 검색 QA 사용
        qa_result = qa_search(request.user_question, ocr_blocks)
        
        return QAResponse(
            answer=qa_result["answer"],
            sources=qa_result["sources"]
        )
        
    except Exception as e:
        logging.error(f"QA 처리 오류: {e}")
        raise HTTPException(status_code=500, detail=f"QA 처리 중 오류가 발생했습니다: {str(e)}")

@app.get("/api/ocr-result")
async def get_current_ocr_result():
    """현재 저장된 OCR 결과 반환"""
    return {
        "ocr_data": current_ocr_result,
        "image_dimensions": current_image_dimensions
    }

# 정적 파일 서빙
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", response_class=HTMLResponse)
async def read_root():
    """메인 페이지"""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>행정문서 OCR & QA 챗봇</title>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <style>
            body { font-family: Arial, sans-serif; margin: 0; padding: 20px; background-color: #f5f5f5; }
            .container { max-width: 1920px; margin: 0 auto; }
            .header { text-align: center; margin-bottom: 30px; }
            .main-content { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
            .panel { background: white; border-radius: 8px; padding: 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
            .upload-area { border: 2px dashed #ddd; border-radius: 8px; padding: 40px; text-align: center; margin-bottom: 20px; }
            .upload-area.dragover { border-color: #007bff; background-color: #f8f9fa; }
            .chat-container { height: 400px; overflow-y: auto; border: 1px solid #ddd; border-radius: 8px; padding: 15px; margin-bottom: 15px; }
            .message { margin-bottom: 15px; }
            .user-message { text-align: right; }
            .bot-message { text-align: left; }
            .message-content { display: inline-block; padding: 10px 15px; border-radius: 18px; max-width: 70%; }
            .user-message .message-content { background-color: #007bff; color: white; }
            .bot-message .message-content { background-color: #e9ecef; color: #333; }
            .input-group { display: flex; gap: 10px; }
            .input-group input { flex: 1; padding: 10px; border: 1px solid #ddd; border-radius: 4px; }
            .btn { padding: 10px 20px; border: none; border-radius: 4px; cursor: pointer; }
            .btn-primary { background-color: #007bff; color: white; }
            .btn-secondary { background-color: #6c757d; color: white; }
            .document-viewer { height: 700px; overflow: auto; border: 1px solid #ddd; border-radius: 8px; position: relative; }
            .document-content { position: relative; display: inline-block; }
            .document-viewer img { max-width: 100%; height: auto; display: block; }
            .text-block { 
                position: absolute; 
                border: 2px solid rgba(255, 0, 0, 0.5); 
                cursor: pointer; 
                pointer-events: none;
                box-sizing: border-box;
            }
            .text-block:hover { background-color: rgba(255, 0, 0, 0.2); }
            .text-block.highlighted { 
                background-color: rgba(255, 255, 0, 0.4); 
                border-color: #ffc107; 
                border-width: 3px;
            }
            .loading { text-align: center; padding: 20px; }
            .error { color: #dc3545; padding: 10px; background-color: #f8d7da; border-radius: 4px; margin: 10px 0; }
            .debug-info { font-size: 12px; color: #666; margin-top: 10px; }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🏛️ 행정문서 OCR & QA 챗봇</h1>
                <p>행정문서를 업로드하고 궁금한 내용을 질문해보세요!</p>
            </div>
            
            <div class="main-content">
                <div class="panel">
                    <h3>📄 문서 업로드 & 채팅</h3>
                    
                    <div class="upload-area" id="uploadArea">
                        <p>📁 여기에 행정문서 이미지 또는 PDF를 드래그하거나 클릭하여 업로드하세요</p>
                        <input type="file" id="fileInput" accept="image/*,application/pdf" style="display: none;">
                        <button class="btn btn-secondary" onclick="document.getElementById('fileInput').click()">파일 선택</button>
                    </div>
                    
                    <div id="uploadStatus"></div>
                    
                    <div class="chat-container" id="chatContainer">
                        <div class="message bot-message">
                            <div class="message-content">
                                안녕하세요! 행정문서 이미지 또는 PDF를 업로드하시면 문서 내용에 대해 질문하실 수 있습니다.
                            </div>
                        </div>
                    </div>
                    
                    <div class="input-group">
                        <input type="text" id="questionInput" placeholder="문서에 대해 질문해보세요..." disabled>
                        <button class="btn btn-primary" id="sendBtn" onclick="sendQuestion()" disabled>전송</button>
                    </div>
                </div>
                
                <div class="panel">
                    <h3>📋 문서 뷰어</h3>
                    <div class="document-viewer" id="documentViewer">
                        <div class="document-content">
                            <p style="text-align: center; color: #6c757d; margin-top: 100px;">
                                이미지 또는 PDF 문서를 업로드하면 여기에 내용이 표시됩니다.
                            </p>
                        </div>
                    </div>
                    <div id="debugInfo" class="debug-info"></div>
                </div>
            </div>
        </div>

        <script>
            let currentOcrResult = [];
            let imageDimensions = {width: 0, height: 0};
            
            // 파일 업로드 관련
            const uploadArea = document.getElementById('uploadArea');
            const fileInput = document.getElementById('fileInput');
            const uploadStatus = document.getElementById('uploadStatus');
            const questionInput = document.getElementById('questionInput');
            const sendBtn = document.getElementById('sendBtn');
            const chatContainer = document.getElementById('chatContainer');
            const documentViewer = document.getElementById('documentViewer');
            const debugInfo = document.getElementById('debugInfo');
            
            // 드래그 앤 드롭 이벤트
            uploadArea.addEventListener('dragover', (e) => {
                e.preventDefault();
                uploadArea.classList.add('dragover');
            });
            
            uploadArea.addEventListener('dragleave', () => {
                uploadArea.classList.remove('dragover');
            });
            
            uploadArea.addEventListener('drop', (e) => {
                e.preventDefault();
                uploadArea.classList.remove('dragover');
                const files = e.dataTransfer.files;
                if (files.length > 0) {
                    handleFileUpload(files[0]);
                }
            });
            
            uploadArea.addEventListener('click', () => {
                fileInput.click();
            });
            
            fileInput.addEventListener('change', (e) => {
                if (e.target.files.length > 0) {
                    handleFileUpload(e.target.files[0]);
                }
            });
            
            // 엔터 키로 질문 전송
            questionInput.addEventListener('keypress', (e) => {
                if (e.key === 'Enter' && !questionInput.disabled) {
                    sendQuestion();
                }
            });
            
            async function handleFileUpload(file) {
                if (!file.type.startsWith('image/') && file.type !== 'application/pdf') {
                    showError('이미지 파일 또는 PDF 파일만 업로드 가능합니다.');
                    return;
                }
                
                uploadStatus.innerHTML = '<div class="loading">📤 문서를 분석 중입니다...</div>';
                
                const formData = new FormData();
                formData.append('file', file);
                
                try {
                    const response = await fetch('/api/ocr', {
                        method: 'POST',
                        body: formData
                    });
                    
                    if (!response.ok) {
                        const errorData = await response.json();
                        throw new Error(errorData.detail || 'OCR 처리 실패');
                    }
                    
                    const data = await response.json();
                    currentOcrResult = data.ocr_data;
                    imageDimensions = {width: data.image_width, height: data.image_height};
                    
                    uploadStatus.innerHTML = '<div style="color: #28a745; padding: 10px; background-color: #d4edda; border-radius: 4px;">✅ 문서 분석 완료! 이제 질문하실 수 있습니다.</div>';
                    
                    // 채팅 입력 활성화
                    questionInput.disabled = false;
                    sendBtn.disabled = false;
                    questionInput.focus();
                    
                    // 문서 뷰어 업데이트
                    updateDocumentViewer(data.image_url, currentOcrResult);
                    
                } catch (error) {
                    showError('문서 분석 중 오류가 발생했습니다: ' + error.message);
                }
            }
            
            function updateDocumentViewer(imageUrl, ocrData) {
                documentViewer.innerHTML = '';

                const container = document.createElement('div');
                container.className = 'document-content';
                documentViewer.appendChild(container);

                const img = new Image();
                img.src = imageUrl;
                container.appendChild(img);

                const drawBoundingBoxes = () => {
                    if (!img.complete || img.naturalWidth === 0) {
                        setTimeout(drawBoundingBoxes, 100);
                        return;
                    }

                    // 기존 박스 제거
                    container.querySelectorAll('.text-block').forEach(el => el.remove());

                    // 원본 이미지 크기 및 현재 표시 크기
                    const naturalWidth = imageDimensions.width || img.naturalWidth;
                    const naturalHeight = imageDimensions.height || img.naturalHeight;
                    const displayedWidth = img.clientWidth;
                    const displayedHeight = img.clientHeight;

                    debugInfo.innerHTML = `이미지 정보: 원본(${naturalWidth}x${naturalHeight}) → 표시(${displayedWidth}x${displayedHeight}) 원본 비율 기준으로 bbox 표시`;

                    // OCR 데이터로 바운딩 박스 그리기 (퍼센트 기반: 원본 기준으로 위치/크기 지정)
                    ocrData.forEach((block) => {
                        const bbox = block.bbox;
                        if (!Array.isArray(bbox) || bbox.length !== 4) return;

                        const [x1, y1, x2, y2] = bbox.map(Number);

                        const leftPct = (x1 / naturalWidth) * 100;
                        const topPct = (y1 / naturalHeight) * 100;
                        const widthPct = ((x2 - x1) / naturalWidth) * 100;
                        const heightPct = ((y2 - y1) / naturalHeight) * 100;

                        if (widthPct <= 0 || heightPct <= 0) return;

                        const blockDiv = document.createElement('div');
                        blockDiv.className = 'text-block';
                        blockDiv.dataset.id = block.id;
                        blockDiv.title = `[${block.id}] ${block.text}`;
                        blockDiv.style.left = `${leftPct}%`;
                        blockDiv.style.top = `${topPct}%`;
                        blockDiv.style.width = `${widthPct}%`;
                        blockDiv.style.height = `${heightPct}%`;
                        
                        container.appendChild(blockDiv);
                    });
                };

                img.onload = () => {
                    setTimeout(drawBoundingBoxes, 100);
                };

                img.onerror = () => {
                    showError('이미지를 불러오는 데 실패했습니다.');
                };

                // 리사이즈 시에도 퍼센트 기반이라 자동 스케일됨. 필요 시 재도 그리기
                let resizeTimeout;
                window.addEventListener('resize', () => {
                    clearTimeout(resizeTimeout);
                    resizeTimeout = setTimeout(drawBoundingBoxes, 200);
                });
            }
            
            async function sendQuestion() {
                const question = questionInput.value.trim();
                if (!question || !currentOcrResult.length) return;
                
                // 사용자 메시지 추가
                addMessage(question, 'user');
                questionInput.value = '';
                
                // 로딩 메시지 추가
                const loadingId = addMessage('🤔 답변을 생성 중입니다...', 'bot');
                
                try {
                    const response = await fetch('/api/qa', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json'
                        },
                        body: JSON.stringify({
                            user_question: question,
                            ocr_result: currentOcrResult
                        })
                    });
                    
                    if (!response.ok) {
                        const errorData = await response.json();
                        throw new Error(errorData.detail || '답변 생성 실패');
                    }
                    
                    const result = await response.json();
                    
                    // 로딩 메시지 제거
                    document.getElementById(loadingId).remove();
                    
                    // 답변 메시지 추가
                    addMessage(result.answer, 'bot');
                    
                    // 출처 하이라이트
                    highlightSources(result.sources);
                    
                } catch (error) {
                    // 로딩 메시지 제거
                    document.getElementById(loadingId).remove();
                    addMessage('❌ 답변 생성 중 오류가 발생했습니다: ' + error.message, 'bot');
                }
            }
            
            function addMessage(content, type) {
                const messageId = 'msg_' + Date.now();
                const messageDiv = document.createElement('div');
                messageDiv.id = messageId;
                messageDiv.className = `message ${type}-message`;
                messageDiv.innerHTML = `<div class="message-content">${content}</div>`;
                
                chatContainer.appendChild(messageDiv);
                chatContainer.scrollTop = chatContainer.scrollHeight;
                
                return messageId;
            }
            
            function showError(message) {
                uploadStatus.innerHTML = `<div class="error">❌ ${message}</div>`;
            }
             
            // 결과 출처 id 기반 하이라이트 함수 (QA 응답에 사용)
            function highlightSources(sources) {
                // 기존 하이라이트 제거
                document.querySelectorAll('.text-block.highlighted').forEach(el => {
                    el.classList.remove('highlighted');
                });
                
                console.log('하이라이트할 소스:', sources);
                
                // 새 하이라이트 적용
                if (Array.isArray(sources) && sources.length > 0) {
                    sources.forEach((source, index) => {
                        console.log(`소스 ${index}:`, source);
                        const block = document.querySelector(`[data-id="${source.id}"]`);
                        if (block) {
                            block.classList.add('highlighted');
                            console.log(`블록 ${source.id} 하이라이트 적용됨`);
                            // 첫 번째 하이라이트된 블록으로 스크롤
                            if (index === 0) {
                                setTimeout(() => {
                                    block.scrollIntoView({ 
                                        behavior: 'smooth', 
                                        block: 'center',
                                        inline: 'center'
                                    });
                                }, 100);
                            }
                        } else {
                            console.log(`블록 ${source.id}을 찾을 수 없음`);
                        }
                    });
                } else {
                    console.log('하이라이트할 소스가 없음');
                }
            }
        </script>
    </body>
    </html>
    """

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
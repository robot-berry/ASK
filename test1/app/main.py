from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool

from .processor import QuestionImageProcessor

BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / "output"
MAX_UPLOAD_BYTES = 20 * 1024 * 1024
processor = QuestionImageProcessor(OUTPUT_DIR)

app = FastAPI(title="题图预处理机", version="1.0.0")
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")
app.mount("/output", StaticFiles(directory=OUTPUT_DIR), name="output")


@app.get("/", include_in_schema=False)
def home():
    return FileResponse(
        BASE_DIR / "static" / "index.html",
        headers={"Cache-Control": "no-store, max-age=0"},
    )


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/api/process")
async def process_image(
    image: UploadFile = File(...),
    mode: str = Form("binary"),
    enable_ocr: bool = Form(False),
    handwriting_mode: str = Form("none"),
    rotation: str = Form("auto"),
    quality: str = Form("fast"),
    extract_questions: bool = Form(True),
    deskew_mode: str = Form("safe"),
):
    if image.content_type and not image.content_type.startswith("image/"):
        raise HTTPException(415, "仅支持图片文件")
    data = await image.read(MAX_UPLOAD_BYTES + 1)
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "图片不能超过 20 MB")
    try:
        result = await run_in_threadpool(
            processor.process,
            data,
            image.filename or "camera.jpg",
            mode,
            enable_ocr,
            handwriting_mode,
            rotation,
            quality,
            extract_questions,
            deskew_mode,
        )
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    payload = result.to_dict()
    payload["urls"] = {key: f"/output/{payload[key]}" for key in ("png", "pdf", "markdown")}
    for question in payload.get("questions") or []:
        question["url"] = f"/output/{question['png']}"
    if payload.get("questions_pdf"):
        payload["urls"]["questions_pdf"] = f"/output/{payload['questions_pdf']}"
    if payload.get("questions_markdown"):
        payload["urls"]["questions_markdown"] = f"/output/{payload['questions_markdown']}"
    return payload

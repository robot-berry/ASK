"""Owner-only prompt editor routes; device credentials cannot change prompts."""
import json
from pathlib import Path

from fastapi import HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse

import prompt_store


def install(app, owner_check):
    def authorize(request):
        if not owner_check(request):
            raise HTTPException(403, "请使用主管理令牌登录；设备令牌不能管理提示词")

    def response(data):
        return JSONResponse(data, headers={"Cache-Control": "no-store"})

    @app.get("/prompt-settings", include_in_schema=False)
    def page():
        return FileResponse(Path(__file__).parent / "portal/source/static/prompt-settings.html", headers={
            "Cache-Control": "no-store", "X-Frame-Options": "DENY",
            "Content-Security-Policy": "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; frame-ancestors 'none'; base-uri 'none'",
        })

    @app.get("/api/v1/prompt-settings/{subject}", include_in_schema=False)
    def read(subject: str, request: Request):
        authorize(request)
        try:
            return response({**prompt_store.current(subject), "default_text": prompt_store.default_prompt(subject), "history": prompt_store.history(subject)})
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from None

    @app.put("/api/v1/prompt-settings/{subject}", include_in_schema=False)
    async def write(subject: str, request: Request):
        authorize(request)
        # A custom header plus JSON prevents cross-origin form submission.
        if request.headers.get("x-prompt-editor") != "1" or request.headers.get("content-type", "").split(";")[0] != "application/json":
            raise HTTPException(415, "请通过提示词管理页面保存")
        raw = bytearray()
        async for chunk in request.stream():
            raw.extend(chunk)
            if len(raw) > 200000:
                raise HTTPException(413, "提示词过长")
        try:
            body = json.loads(raw)
            if not isinstance(body, dict):
                raise ValueError("请求格式不正确")
            result = prompt_store.save(subject, body.get("text"), body.get("revision"))
            return response(result)
        except prompt_store.Conflict as exc:
            raise HTTPException(409, str(exc)) from None
        except (ValueError, UnicodeDecodeError) as exc:
            raise HTTPException(400, str(exc)) from None

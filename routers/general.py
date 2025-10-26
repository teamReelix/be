from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates

from config import STATIC_DIR
from progress_state import progress_data, progress_lock

router = APIRouter()

templates = Jinja2Templates(directory=STATIC_DIR)

@router.get("/", response_class=HTMLResponse)
async def read_index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@router.get("/progress")
async def get_progress():
    with progress_lock:
        return JSONResponse(progress_data)
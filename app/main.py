import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, File, Header, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import rag, vectorstore
from .config import BASE_DIR, SUPPORTED_EXTENSIONS, settings

logging.basicConfig(level=logging.INFO)


@asynccontextmanager
async def lifespan(app: FastAPI):
    rag.ingest_knowledge_dir()
    yield


app = FastAPI(title="PlacementPrep AI", version="1.0.0", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")


def require_admin(x_admin_key: str | None = Header(default=None)):
    """Upload/delete are protected when ADMIN_KEY is set (use it in production)."""
    if settings.admin_key and x_admin_key != settings.admin_key:
        raise HTTPException(status_code=401, detail="Invalid or missing admin key")


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=settings.max_question_chars)
    top_k: int = Field(default=settings.top_k, ge=1, le=10)


@app.get("/")
def index():
    return FileResponse(BASE_DIR / "static" / "index.html")


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "llm_configured": bool(settings.gemini_api_key),
        "documents": len(vectorstore.list_documents()),
    }


@app.get("/api/documents")
def documents():
    return vectorstore.list_documents()


@app.post("/api/documents", dependencies=[Depends(require_admin)])
def upload(file: UploadFile = File(...)):
    name = Path(file.filename or "").name
    if Path(name).suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise HTTPException(400, "Only PDF, TXT and MD files are supported")
    data = file.file.read(settings.max_upload_mb * 1024 * 1024 + 1)
    if len(data) > settings.max_upload_mb * 1024 * 1024:
        raise HTTPException(413, f"File exceeds {settings.max_upload_mb} MB")
    try:
        chunks = rag.ingest_file(name, data)
    except ValueError as exc:
        raise HTTPException(422, str(exc))
    except RuntimeError as exc:
        raise HTTPException(503, str(exc))
    return {"name": name, "chunks": chunks}


@app.delete("/api/documents/{name}", dependencies=[Depends(require_admin)])
def delete(name: str):
    vectorstore.delete_source(Path(name).name)
    return {"deleted": name}


@app.post("/api/ask")
def ask(req: AskRequest):
    try:
        return rag.answer(req.question.strip(), req.top_k)
    except RuntimeError as exc:
        raise HTTPException(503, str(exc))
    except Exception:
        logging.exception("ask failed")
        raise HTTPException(502, "The AI service failed. Please try again.")

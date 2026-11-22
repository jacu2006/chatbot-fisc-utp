import csv
from datetime import datetime
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from rag_engine import FISCRagEngine

LOGS_DIR = Path("logs")
LOGS_DIR.mkdir(exist_ok=True)
QUERIES_CSV = LOGS_DIR / "consultas.csv"
HANDOFF_CSV = LOGS_DIR / "derivaciones.csv"

app = FastAPI(title="Chatbot Académico FISC-UTP (PLN + RAG)")
app.mount("/static", StaticFiles(directory="static"), name="static")

rag_engine: FISCRagEngine = None


class ChatRequest(BaseModel):
    question: str
    category: str = "Prueba Piloto SUS"


class HandoffRequest(BaseModel):
    student_name: str
    student_email: str
    question: str
    notes: str = "Prueba Piloto SUS"


def _append_csv(path: Path, headers: list, row: list):
    file_exists = path.exists()
    with open(path, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(headers)
        writer.writerow(row)


@app.on_event("startup")
def startup_event():
    global rag_engine
    rag_engine = FISCRagEngine()


@app.get("/")
def serve_home():
    return FileResponse("static/index.html")


@app.post("/api/chat")
def chat_endpoint(req: ChatRequest):
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="La consulta no puede estar vacía.")

    try:
        result = rag_engine.ask(req.question.strip())
    except Exception as exc:
        err_msg = str(exc)
        if "429" in err_msg or "RESOURCE_EXHAUSTED" in err_msg:
            raise HTTPException(status_code=429, detail=err_msg)
        raise HTTPException(status_code=500, detail=err_msg)

    # RF5: Registro estructurado del historial de consultas y métricas de latencia
    _append_csv(
        QUERIES_CSV,
        ["timestamp", "categoria", "pregunta", "confianza", "latencia_s", "derivado", "fuentes"],
        [
            datetime.now().isoformat(timespec="seconds"),
            req.category,
            req.question.strip(),
            result["confidence"],
            result["latency_s"],
            "Sí" if result["derived"] else "No",
            " | ".join(result["sources"])
        ]
    )
    return result


@app.post("/api/derivar")
def handoff_endpoint(req: HandoffRequest):
    ticket_id = f"FISC-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    _append_csv(
        HANDOFF_CSV,
        ["ticket_id", "timestamp", "nombre", "correo_utp", "consulta", "observaciones"],
        [
            ticket_id,
            datetime.now().isoformat(timespec="seconds"),
            req.student_name.strip(),
            req.student_email.strip(),
            req.question.strip(),
            req.notes.strip()
        ]
    )
    return {
        "status": "ok",
        "ticket_id": ticket_id,
        "message": f"Tu consulta fue derivada a la Secretaría Académica con el código {ticket_id}."
    }
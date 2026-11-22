import os
import re
import json
import time
import unicodedata
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Tuple
from dotenv import load_dotenv
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI

load_dotenv()

DOCS_DIR = Path("docs")
CACHE_VECTORS = Path("vector_cache.npz")
CACHE_META = Path("vector_meta.json")

CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.70"))
LLM_MODEL = os.getenv("LLM_MODEL", "gemini-3.5-flash-lite")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "models/gemini-embedding-001")


class FISCRagEngine:
    def __init__(self):
        self.api_key = os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError("Falta GOOGLE_API_KEY en el archivo .env")

        self.embeddings_model = GoogleGenerativeAIEmbeddings(
            model=EMBEDDING_MODEL,
            google_api_key=self.api_key
        )
        self.llm = ChatGoogleGenerativeAI(
            model=LLM_MODEL,
            google_api_key=self.api_key,
            temperature=0.1  # Baja temperatura para máxima fidelidad documental
        )
        self.chunks: List[Dict[str, Any]] = []
        self.vectors: np.ndarray = np.array([])
        self._load_or_build_index()

    def _extract_documents(self) -> List[Dict[str, Any]]:
        """Extrae texto de todos los archivos PDF y TXT en la carpeta docs/."""
        if not DOCS_DIR.exists():
            DOCS_DIR.mkdir(parents=True, exist_ok=True)

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=900,
            chunk_overlap=180,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        extracted_chunks = []

        for file_path in sorted(DOCS_DIR.iterdir()):
            if file_path.suffix.lower() == ".pdf":
                reader = PdfReader(str(file_path))
                for page_num, page in enumerate(reader.pages, start=1):
                    text = (page.extract_text() or "").strip()
                    if not text:
                        continue
                    for piece in splitter.split_text(text):
                        extracted_chunks.append({
                            "text": piece,
                            "source": file_path.name,
                            "page": page_num
                        })
            elif file_path.suffix.lower() == ".txt":
                text = file_path.read_text(encoding="utf-8", errors="ignore").strip()
                if text:
                    for piece in splitter.split_text(text):
                        extracted_chunks.append({
                            "text": piece,
                            "source": file_path.name,
                            "page": 1
                        })

        if not extracted_chunks:
            raise RuntimeError(
                "La carpeta 'docs/' está vacía. Coloca al menos un PDF o TXT oficial de la FISC."
            )
        return extracted_chunks

    def _load_or_build_index(self):
        """Carga el índice desde caché o lo genera desde los documentos en docs/."""
        current_files = sorted([f.name for f in DOCS_DIR.iterdir() if f.suffix.lower() in {".pdf", ".txt"}]) if DOCS_DIR.exists() else []

        if CACHE_VECTORS.exists() and CACHE_META.exists():
            meta = json.loads(CACHE_META.read_text(encoding="utf-8"))
            if meta.get("files") == current_files:
                self.chunks = meta["chunks"]
                self.vectors = np.load(CACHE_VECTORS)["vectors"]
                print(f"[RAG] Índice cargado desde caché: {len(self.chunks)} fragmentos de {len(current_files)} documentos.")
                return

        print("[RAG] Indexando documentos oficiales de la FISC...")
        self.chunks = self._extract_documents()
        texts = [c["text"] for c in self.chunks]

        # Generar embeddings en lotes con control de cuota (Free Tier: ~100 req/min)
        all_vecs = []
        batch_size = 50
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            vecs = self._embed_batch_with_retry(batch)
            all_vecs.extend(vecs)
            print(f"[RAG] Embeddings procesados: {min(i + batch_size, len(texts))}/{len(texts)}")
            if i + batch_size < len(texts):
                time.sleep(32)  # Pausa preventiva para no exceder 100 solicitudes/min

        matrix = np.array(all_vecs, dtype=np.float32)
        # Normalizar L2 para que el producto punto sea exactamente la similitud coseno
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self.vectors = matrix / norms

        np.savez_compressed(CACHE_VECTORS, vectors=self.vectors)
        CACHE_META.write_text(
            json.dumps({"files": current_files, "chunks": self.chunks}, ensure_ascii=False),
            encoding="utf-8"
        )
        print(f"[RAG] Indexación completa: {len(self.chunks)} fragmentos guardados en caché.")

    def _embed_batch_with_retry(self, batch: List[str], max_retries: int = 5) -> List[List[float]]:
        """Ejecuta embed_documents con reintento exponencial ante límites de cuota (429)."""
        wait_s = 20
        for attempt in range(max_retries):
            try:
                return self.embeddings_model.embed_documents(batch)
            except Exception as exc:
                err_msg = str(exc)
                if ("429" in err_msg or "RESOURCE_EXHAUSTED" in err_msg) and attempt < max_retries - 1:
                    print(f"[RAG] Límite de cuota (429) al indexar lote; reintentando en {wait_s}s...")
                    time.sleep(wait_s)
                    wait_s = min(wait_s * 2, 90)
                else:
                    raise

    def _retrieve(self, query: str, top_k: int = 4) -> Tuple[List[Dict[str, Any]], float]:
        """Recupera los fragmentos más similares y devuelve el puntaje máximo de similitud."""
        q_vec = np.array(self.embeddings_model.embed_query(query), dtype=np.float32)
        q_norm = np.linalg.norm(q_vec)
        if q_norm > 0:
            q_vec = q_vec / q_norm

        similarities = np.dot(self.vectors, q_vec)
        top_indices = np.argsort(similarities)[::-1][:top_k]

        results = []
        for idx in top_indices:
            results.append({
                "text": self.chunks[idx]["text"],
                "source": self.chunks[idx]["source"],
                "page": self.chunks[idx]["page"],
                "score": float(similarities[idx])
            })

        max_score = results[0]["score"] if results else 0.0
        return results, max_score

    def _match_conversational_intent(self, question: str) -> str | None:
        """Responde saludos, agradecimientos y consultas sobre el alcance del asistente."""
        norm = unicodedata.normalize("NFKD", question.lower())
        norm = "".join(ch for ch in norm if not unicodedata.combining(ch))
        norm = re.sub(r"[^a-z0-9\s]", " ", norm)
        norm = re.sub(r"\s+", " ", norm).strip()

        greetings = {
            "hola", "buenos dias", "buenas tardes", "buenas noches", "buenas",
            "buen dia", "saludos", "hey", "hi", "hello", "hola buenas", "hola buen dia"
        }
        thanks = {
            "gracias", "muchas gracias", "mil gracias", "ok gracias", "listo gracias"
        }
        help_phrases = {
            "ayuda", "help", "quien eres", "que eres", "para que sirves",
            "como funcionas", "que haces", "que puedes hacer",
            "en que me puedes ayudar", "en que puedes ayudarme", "en que puedes ayudar",
            "como me puedes ayudar", "como puedes ayudarme",
            "sobre que te puedo preguntar", "que te puedo preguntar",
            "que temas cubres", "que temas manejas", "que tramites cubres",
            "que tramites manejas", "que sabes hacer", "cuales son tus funciones",
            "que informacion tienes", "de que me puedes informar"
        }

        if norm in thanks:
            return (
                "¡Con mucho gusto! Si tienes otra consulta sobre matrícula, graduación, "
                "convalidaciones o el calendario académico de la FISC-UTP, escríbeme cuando gustes."
            )

        is_greeting = norm in greetings
        is_help = norm in help_phrases or any(
            p in norm for p in (
                "en que me puedes ayudar",
                "en que puedes ayudar",
                "como me puedes ayudar",
                "que puedes hacer",
                "que te puedo preguntar",
                "sobre que puedo preguntar",
                "que tramites cubres",
                "que temas cubres",
            )
        )
        if is_greeting or is_help:
            return (
                "¡Hola! Soy el asistente virtual de la Secretaría Académica de la **FISC-UTP**. "
                "Puedo ayudarte con consultas basadas en los reglamentos oficiales sobre:\n"
                "• **Matrícula y créditos:** requisitos, fechas de pago ordinario/recargo y créditos adicionales.\n"
                "• **Retiro e inclusión de asignaturas:** plazos y procedimientos.\n"
                "• **Trabajo de Graduación:** opciones (Tesis Teórica, Teórico-Práctica, Práctica Profesional, Postgrado, Cursos en el Extranjero y Certificación Internacional), índices mínimos y requisitos.\n"
                "• **Convalidaciones de créditos:** internas (cambio de carrera) y externas.\n"
                "• **Calendario Académico 2026–2027:** inicio/fin de clases, exámenes finales y verano.\n"
                "• **Otros servicios FISC:** examen de rehabilitación, examen extraordinario y copia de examen semestral.\n\n"
                "Si necesitas revisar notas o trámites personales, puedes usar el botón **Contactar Secretaría**."
            )
        return None

    def ask(self, question: str) -> Dict[str, Any]:
        """Procesa la pregunta, aplica umbral de derivación < 0.70 y mide la latencia."""
        start_time = time.perf_counter()

        conversational_reply = self._match_conversational_intent(question)
        if conversational_reply is not None:
            latency = round(time.perf_counter() - start_time, 3)
            return {
                "answer": conversational_reply,
                "sources": ["Base Documental Oficial FISC-UTP"],
                "confidence": 1.0,
                "latency_s": latency,
                "derived": False
            }

        retrieved, max_score = self._retrieve(question, top_k=4)

        # RF4: Derivación automática cuando la confianza/similitud es menor al umbral
        if max_score < CONFIDENCE_THRESHOLD:
            latency = round(time.perf_counter() - start_time, 3)
            return {
                "answer": (
                    "No encontré respaldo suficiente en los reglamentos oficiales de la FISC-UTP "
                    "para responder esta consulta con certeza (o corresponde a un trámite que requiere "
                    "revisión de datos personales). Te recomiendo derivar el caso directamente a un "
                    "funcionario de la Secretaría Académica usando el botón inferior."
                ),
                "sources": [],
                "confidence": round(max_score, 3),
                "latency_s": latency,
                "derived": True
            }

        # Construir contexto estrictamente acotado
        context_blocks = []
        unique_sources = []
        for item in retrieved:
            ref = f"{item['source']} (pág. {item['page']})"
            if ref not in unique_sources:
                unique_sources.append(ref)
            context_blocks.append(f"[Fuente: {ref}]\n{item['text']}")

        context_str = "\n\n---\n\n".join(context_blocks)

        prompt = (
            "Eres el asistente virtual oficial de la Secretaría Académica de la Facultad de "
            "Ingeniería de Sistemas Computacionales (FISC) de la Universidad Tecnológica de Panamá (UTP).\n"
            "REGLAS ESTRICTAS:\n"
            "1. Responde ÚNICAMENTE utilizando la información del CONTEXTO OFICIAL proporcionado.\n"
            "2. No inventes fechas, requisitos, montos ni artículos que no aparezcan en el contexto.\n"
            "3. Si el usuario pide datos personales (notas, estado de cuenta, historial), indica que por "
            "privacidad debe consultar el portal institucional o contactar a la Secretaría.\n"
            "4. Al final de tu respuesta, cita brevemente el documento oficial del cual extrajiste el dato.\n"
            "5. Sé claro, conciso y amable (máximo 150 palabras).\n\n"
            f"CONTEXTO OFICIAL:\n{context_str}\n\n"
            f"CONSULTA DEL ESTUDIANTE:\n{question}\n\n"
            "RESPUESTA:"
        )

        response = self.llm.invoke(prompt)
        latency = round(time.perf_counter() - start_time, 3)

        content = response.content
        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, str):
                    parts.append(item)
                elif isinstance(item, dict) and "text" in item:
                    parts.append(str(item["text"]))
            answer_text = "\n".join(parts).strip()
        else:
            answer_text = str(content or "").strip()

        return {
            "answer": answer_text,
            "sources": unique_sources,
            "confidence": round(max_score, 3),
            "latency_s": latency,
            "derived": False
        }
import time
import statistics
import requests
from collections import defaultdict

API_URL = "http://127.0.0.1:8000/api/chat"
REPETITIONS = 3

# 25 consultas organizadas exactamente como en la Tabla 4 del artículo
TEST_SUITE = {
    "Matrícula": [
        "¿Cuáles son los requisitos generales para matricularse en la UTP?",
        "¿Cuál es la fecha límite de matrícula ordinaria en el calendario académico?",
        "¿Qué ocurre si realizo el pago de matrícula después de la fecha ordinaria?",
        "¿Cuántos créditos máximos puede matricular un estudiante regular por semestre?",
        "¿Cómo se solicita autorización para matricular créditos adicionales?"
    ],
    "Retiro e inclusión": [
        "¿Cuál es el periodo establecido para realizar retiro e inclusión de asignaturas?",
        "¿Qué procedimiento debe seguir un estudiante para retirar una materia?",
        "¿El retiro de asignaturas dentro del plazo afecta el índice académico?",
        "¿Dónde se tramita la inclusión de una asignatura con cupo especial?"
    ],
    "Requisitos de graduación": [
        "¿Cuáles son las opciones de trabajo de graduación disponibles en la FISC?",
        "¿Cuál es el índice académico mínimo requerido para inscribir trabajo de graduación?",
        "¿Por cuánto tiempo tiene vigencia o exclusividad un tema de trabajo de graduación?",
        "¿Cuántos estudiantes pueden participar en un mismo trabajo de graduación?"
    ],
    "Créditos y convalidaciones": [
        "¿Cuáles son los requisitos para tramitar una convalidación externa de créditos?",
        "¿Cuál es el costo del trámite de convalidación de créditos en la UTP?",
        "¿Cómo se solicita una convalidación interna por cambio de carrera en la FISC?",
        "¿Qué calificación mínima debe tener una asignatura para ser convalidada?"
    ],
    "Calendario académico": [
        "¿Cuándo inician las clases del segundo semestre según el calendario académico?",
        "¿En qué fechas se realizan los exámenes semestrales del periodo académico?",
        "¿Cuándo inicia y finaliza el periodo de verano académico?",
        "¿Cuál es el último día de clases regulares del semestre?"
    ],
    "Consultas fuera de alcance (< 0.70)": [
        "¿Cuál es mi nota final en la materia de Ingeniería de Software II?",
        "¿Cuánto dinero debo actualmente en mi estado de cuenta estudiantil?",
        "¿Qué ruta de Metrobus pasa frente a la Universidad Tecnológica de Panamá?",
        "¿Cuál es el menú de almuerzo de hoy en la cafetería de la UTP?"
    ]
}

def main():
    stats = defaultdict(lambda: {"latencies": [], "derived_count": 0, "total": 0})
    all_latencies = []

    print("Iniciando batería de pruebas de repetibilidad (25 consultas × 3 = 75 ejecuciones)...\n")

    for category, questions in TEST_SUITE.items():
        for q in questions:
            for rep in range(1, REPETITIONS + 1):
                wait_s = 10
                data = None
                for attempt in range(6):
                    res = requests.post(API_URL, json={"question": q, "category": category}, timeout=60)
                    if res.status_code == 200:
                        data = res.json()
                        break
                    if res.status_code == 429:
                        detail = res.text
                        if "PerDay" in detail or "per day" in detail.lower():
                            raise RuntimeError(f"Cuota diaria agotada (429): {detail}")
                        print(f"[{category} | Rep {rep}] 429 RPM, reintentando en {wait_s}s...")
                        time.sleep(wait_s)
                        wait_s = min(wait_s * 2, 120)
                    else:
                        raise RuntimeError(f"Error HTTP {res.status_code}: {res.text}")
                if data is None:
                    raise RuntimeError(f"Se agotaron los reintentos por 429 en: {q}")

                lat = data["latency_s"]
                derived = data["derived"]

                stats[category]["latencies"].append(lat)
                stats[category]["total"] += 1
                if derived:
                    stats[category]["derived_count"] += 1
                all_latencies.append(lat)

                print(f"[{category} | Rep {rep}] {lat:.3f}s | Conf: {data['confidence']:.3f} | Derivado: {derived}")
                time.sleep(2.0)  # Pausa preventiva para respetar RPM de la capa gratuita

    print("\n" + "=" * 85)
    print(f"{'Categoría académica':<34} | {'n×3':<5} | {'Latencia (s) ± DE':<20} | {'Derivaciones (%)'}")
    print("-" * 85)

    for category, d in stats.items():
        mean_lat = statistics.mean(d["latencies"])
        sd_lat = statistics.stdev(d["latencies"]) if len(d["latencies"]) > 1 else 0.0
        deriv_pct = (d["derived_count"] / d["total"]) * 100
        print(f"{category:<34} | {d['total']:<5} | {mean_lat:.2f} ± {sd_lat:.2f}          | {deriv_pct:.1f} %")

    global_mean = statistics.mean(all_latencies)
    global_sd = statistics.stdev(all_latencies)
    print("-" * 85)
    print(f"{'TOTAL / PROMEDIO GLOBAL':<34} | {len(all_latencies):<5} | {global_mean:.2f} ± {global_sd:.2f}")
    print("=" * 85)
    print("Registro completo guardado en logs/consultas.csv")

if __name__ == "__main__":
    main()
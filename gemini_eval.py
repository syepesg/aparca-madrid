"""Pregunta a Gemini si cada hueco candidato es una plaza real y está libre.
uso: gemini_eval.py <set_dir> [modelo]   (clave en ~/.config/aparca/gemini_key)"""
import base64, json, sys, time, urllib.request, urllib.error
from pathlib import Path
D = Path(sys.argv[1]); MODEL = sys.argv[2] if len(sys.argv) > 2 else "gemini-3.1-flash-lite"
import os
KEY = os.environ.get("GEMINI_API_KEY") or (Path.home() / ".config/aparca/gemini_key").read_text().strip()
URL = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL}:generateContent"
PROMPT = """Eres un verificador de huecos de aparcamiento para una app que usa cámaras de tráfico de Madrid.
Imagen 1: vista completa de la cámara; el rectángulo magenta marca el sitio candidato.
Imagen 2: el mismo sitio ampliado (rectángulo magenta).
Responde sobre lo que hay DENTRO del rectángulo magenta:
- is_parking_spot: true solo si es un sitio donde se aparca un coche (banda de aparcamiento junto al bordillo, plaza pintada, parking o explanada usada para aparcar). false si es un carril de circulación, glorieta, paso de peatones, acera, jardín, mobiliario, contenedores, texto sobreimpreso de la cámara u otra cosa.
- is_free: true solo si dentro del rectángulo no hay ningún vehículo (coche, moto, furgoneta, autobús) y cabría un coche.
- confidence: 0 a 1, lo seguro que estás de ambas respuestas (baja si la imagen es borrosa, oscura o muy lejana).
- reason: máximo 12 palabras."""
SCHEMA = {"type": "OBJECT", "properties": {
    "is_parking_spot": {"type": "BOOLEAN"}, "is_free": {"type": "BOOLEAN"},
    "confidence": {"type": "NUMBER"}, "reason": {"type": "STRING"}},
    "required": ["is_parking_spot", "is_free", "confidence", "reason"]}
b64 = lambda p: base64.b64encode(p.read_bytes()).decode()
def ask(key):
    body = {"contents": [{"parts": [{"text": PROMPT},
        {"inline_data": {"mime_type": "image/jpeg", "data": b64(D / f"{key}_ctx.jpg")}},
        {"inline_data": {"mime_type": "image/jpeg", "data": b64(D / f"{key}_crop.jpg")}}]}],
        "generationConfig": {"temperature": 0, "responseMimeType": "application/json", "responseSchema": SCHEMA}}
    req = urllib.request.Request(URL, json.dumps(body).encode(), {"Content-Type": "application/json", "x-goog-api-key": KEY})
    for attempt in range(5):
        try:
            r = json.loads(urllib.request.urlopen(req, timeout=60).read())
            txt = r["candidates"][0]["content"]["parts"][0]["text"]
            return json.loads(txt), r.get("usageMetadata", {})
        except urllib.error.HTTPError as e:
            msg = e.read().decode()[:300]
            if e.code in (429, 500, 503): print("  reintento", e.code, msg[:120]); time.sleep(20 * (attempt + 1)); continue
            raise SystemExit(f"HTTP {e.code}: {msg}")
    raise SystemExit("demasiados reintentos")
items = json.load(open(D / "items.json"))
out_p = D / f"gemini_{MODEL}.json"
out = json.loads(out_p.read_text()) if out_p.exists() else {}
for it in items:
    if it["key"] in out: continue
    t = time.time(); ans, usage = ask(it["key"]); ans["latency"] = round(time.time() - t, 2); ans["usage"] = usage
    out[it["key"]] = ans; out_p.write_text(json.dumps(out, indent=1, ensure_ascii=False))
    print(it["key"], ans["is_parking_spot"], ans["is_free"], ans["confidence"], ans["reason"])
    time.sleep(4.5)   # plan gratuito: pocas consultas por minuto
print("hecho", len(out))

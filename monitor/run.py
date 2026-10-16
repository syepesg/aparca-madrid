"""Monitor de plazas de aparcamiento sobre las cámaras de tráfico de Madrid.

Cada ejecución:
  1. Descarga la lista de cámaras (datos abiertos del Ayuntamiento) y sus imágenes.
  2. Detecta vehículos con YOLO.
  3. Reconoce el encuadre de cada cámara (muchas son motorizadas) y actualiza sus plazas
     con la lógica de slots.py: coche quieto = misma caja y mismo aspecto; libre = sin
     detección y con la imagen de la plaza cambiada.
  4. Escribe state/cams/<camara>.json (memoria entre ejecuciones) y site/data/status.json (web).
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import email.utils
import hashlib
import io
import json
import re
import time
import urllib.request
from pathlib import Path

from PIL import Image

import slots as sl
from detector import VehicleDetector, merge_dets

KML_URL = "https://datos.madrid.es/egob/catalogo/202088-0-trafico-camaras.kml"
UA = {"User-Agent": "madrid-aparca/1.0 (+https://github.com)"}
MIN_BRIGHTNESS = 12              # imagen negra / caída
STALE_FREE = 4 * 3600            # libre tanto tiempo: dudoso, no se cuenta
STATE_VERSION = 2


def fetch(url, timeout=25):
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read(), dict(r.headers)


def load_cameras(cache: Path):
    try:
        raw, _ = fetch(KML_URL, 40)
        text = raw.decode("utf-8-sig", "replace")
    except Exception as e:  # el portal de datos abiertos a veces falla: usar caché
        print("KML no disponible, uso caché:", e)
        return json.loads(cache.read_text())
    cams = []
    for pm in re.findall(r"<Placemark>(.*?)</Placemark>", text, re.S):
        img = re.search(r"src=(https?://[^\s&]+\.jpg)", pm)
        num = re.search(r'name="Numero">\s*<Value>([^<]+)', pm)
        name = re.search(r'name="Nombre">\s*<Value>([^<]+)', pm)
        coord = re.search(r"<coordinates>\s*([-\d.]+),([-\d.]+)", pm)
        if not (img and num and coord):
            continue
        cams.append({
            "id": num.group(1).strip(),
            "name": (name.group(1).strip() if name else num.group(1)).title(),
            "lat": round(float(coord.group(2)), 6),
            "lon": round(float(coord.group(1)), 6),
            "img": img.group(1).replace("http://", "https://"),
        })
    if cams:
        cache.write_text(json.dumps(cams, ensure_ascii=False))
    return cams


def slot_rois(slots):
    """Zona que engloba las plazas, para analizarla ampliada."""
    boxes = [s["box"] for s in slots]
    if not boxes:
        return []
    x0, y0 = min(b[0] for b in boxes), min(b[1] for b in boxes)
    x1, y1 = max(b[2] for b in boxes), max(b[3] for b in boxes)
    mx, my = max(0.04, (x1 - x0) * 0.12), max(0.04, (y1 - y0) * 0.12)
    roi = [max(0, x0 - mx), max(0, y0 - my), min(1, x1 + mx), min(1, y1 + my)]
    if (roi[2] - roi[0]) * (roi[3] - roi[1]) > 0.6:
        return []  # casi toda la imagen: la pasada normal ya basta
    return [roi]


def analyze(detector, st: dict, img: Image.Image, ts: float):
    """Detección + encuadre + plazas para una captura. Devuelve (detecciones, eventos)."""
    dets, w, h, bright = detector.detect_image(img)
    if bright < MIN_BRIGHTNESS:
        st["dark"] = True
        return [], []
    st["dark"] = False
    view = sl.assign_view(st, dets, ts, img)
    rois = slot_rois([s for s in view["slots"] if sl.is_real_slot(s)])
    if rois:
        dets = merge_dets(dets, detector.detect_image(img, rois, full=False)[0])
    return dets, sl.update_view(view, dets, ts, img, clear=st.get("view_sim", 1) >= sl.CLEAR_VIEW_SIM)


def public_camera(cam, st, now):
    """Resumen para el mapa y la lista (status.json) y detalle para la ficha (cams/<id>.json)."""
    views = []
    for v in st.get("views", []):
        real = [(s, sl.slot_public(s, now, STALE_FREE)) for s in v["slots"] if sl.is_real_slot(s)]
        if real:
            views.append({"id": v["id"], "thumb": v["thumb"],
                          "slots": [dict(p, ref=s.get("ref_occ")) for s, p in real]})
    cur = next((v for v in views if v["id"] == st.get("view")), None)
    slots = cur["slots"] if cur else []
    free = [s for s in slots if s["state"] == "free" and s["sure"]]
    summary = {
        **cam,
        "ts": int(st.get("img_ts", 0)),
        "dark": st.get("dark", False),
        "slots": len(slots),
        "free": len(free),
        "free_since": max((s["since"] for s in free), default=None),
        "views": len(views),
        "learning": sum(1 for v in st.get("views", []) for s in v["slots"] if s["obs"] < sl.CONFIRM_OBS),
    }
    detail = {"id": cam["id"], "ts": summary["ts"], "view": st.get("view"), "views": views,
              "dets": [d["box"] for d in st.get("last_dets", [])]}
    return summary, detail


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/yolo11s-960.onnx")
    ap.add_argument("--state", default="state")
    ap.add_argument("--out", default="site/data")
    ap.add_argument("--limit", type=int, default=0, help="procesar solo N cámaras (pruebas)")
    ap.add_argument("--budget", type=int, default=0, help="segundos máximos de proceso")
    args = ap.parse_args()

    t0 = time.time()
    state_dir, out_dir = Path(args.state), Path(args.out)
    (state_dir / "cams").mkdir(parents=True, exist_ok=True)
    (out_dir / "cams").mkdir(parents=True, exist_ok=True)
    cams = load_cameras(state_dir / "cameras.json")
    if args.limit:
        cams = cams[: args.limit]
    print(f"{len(cams)} cámaras")

    detector = VehicleDetector(args.model, conf=0.25)

    def download(cam):
        try:
            data, hdr = fetch(cam["img"])
            lm = hdr.get("Last-Modified") or hdr.get("last-modified")
            img_ts = email.utils.parsedate_to_datetime(lm).timestamp() if lm else time.time()
            return cam, data, img_ts
        except Exception as e:
            return cam, None, str(e)

    events_path = state_dir / "events.json"
    try:
        events = json.loads(events_path.read_text())
    except Exception:
        events = []

    public, processed, skipped, failed = [], 0, 0, 0
    now = time.time()
    with cf.ThreadPoolExecutor(12) as pool:
        for cam, data, img_ts in pool.map(download, cams):
            sp = state_dir / "cams" / f"{cam['id']}.json"
            try:
                st = json.loads(sp.read_text())
            except Exception:
                st = {}
            if st.get("version") != STATE_VERSION:
                st = {"version": STATE_VERSION}   # memoria de la versión anterior: se reaprende
            over_budget = args.budget and time.time() - t0 > args.budget
            if data is None:
                failed += 1
            elif over_budget:
                skipped += 1
            else:
                digest = hashlib.md5(data).hexdigest()
                if digest == st.get("digest"):
                    skipped += 1  # imagen sin cambios desde la última vez
                else:
                    img = Image.open(io.BytesIO(data)).convert("RGB")
                    dets, evs = analyze(detector, st, img, img_ts)
                    st["digest"], st["img_ts"], st["last_dets"] = digest, img_ts, dets
                    events += [{"cam": cam["id"], **ev} for ev in evs]
                    processed += 1
                    sp.write_text(json.dumps(st, separators=(",", ":")))
            summary, detail = public_camera(cam, st, now)
            public.append(summary)
            (out_dir / "cams" / f"{cam['id']}.json").write_text(json.dumps(detail, separators=(",", ":")))

    events = [e for e in events if now - e["ts"] < 6 * 3600][-500:]
    events_path.write_text(json.dumps(events))
    status = {
        "generated": int(now),
        "stats": {"processed": processed, "unchanged": skipped, "failed": failed,
                  "seconds": round(time.time() - t0, 1)},
        "cameras": public,
        "events": events,
    }
    (out_dir / "status.json").write_text(json.dumps(status, separators=(",", ":"), ensure_ascii=False))
    print(json.dumps(status["stats"]),
          "plazas:", sum(c["slots"] for c in public),
          "libres:", sum(c["free"] for c in public))


if __name__ == "__main__":
    main()

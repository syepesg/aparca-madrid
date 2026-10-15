"""Monitor de plazas de aparcamiento sobre las cámaras de tráfico de Madrid.

Cada ejecución:
  1. Descarga la lista de cámaras (datos abiertos del Ayuntamiento) y sus imágenes.
  2. Detecta vehículos con YOLO.
  3. Aprende "plazas": zonas donde un vehículo permanece quieto entre capturas
     separadas varios minutos (un coche en circulación nunca repite la misma caja).
  4. Marca cada plaza como ocupada / libre y registra cuándo se ha liberado.
  5. Escribe state/<camara>.json (memoria entre ejecuciones) y site/data/status.json (web).
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import email.utils
import hashlib
import json
import os
import re
import time
import urllib.request
from datetime import datetime
from zoneinfo import ZoneInfo
from pathlib import Path

from detector import VehicleDetector

KML_URL = "https://datos.madrid.es/egob/catalogo/202088-0-trafico-camaras.kml"
UA = {"User-Agent": "madrid-aparca/1.0 (+https://github.com)"}

# Parámetros del aprendizaje de plazas
MIN_STATIONARY_GAP = 4 * 60      # s entre capturas para considerar un coche "quieto"
MAX_STATIONARY_GAP = 45 * 60     # más separación no prueba nada (podría ser otro coche)
STATIONARY_IOU = 0.72            # misma caja en ambas capturas
SLOT_MATCH_IOU = 0.35            # detección quieta que refuerza una plaza existente
CONFIRM_OBS = 2                  # veces visto aparcado para confirmar una plaza
OCC_IOU = 0.25                   # una detección ocupa la plaza
OCC_COVER = 0.45                 # ... o cubre esta fracción de la plaza (oclusión)
CANDIDATE_TTL = 3 * 86400        # candidatos no confirmados caducan
SLOT_TTL = 10 * 86400            # plazas que no vuelven a ocuparse caducan
STALE_FREE = 18 * 3600           # libre tanto tiempo: probablemente la cámara se movió
MIN_BRIGHTNESS = 12              # imagen negra / caída
MADRID = ZoneInfo("Europe/Madrid")
PARKABLE = {"car", "truck", "motorcycle"}


def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def cover(slot, det):
    """Fracción del área de la plaza tapada por la detección."""
    ix = max(0.0, min(slot[2], det[2]) - max(slot[0], det[0]))
    iy = max(0.0, min(slot[3], det[3]) - max(slot[1], det[1]))
    area = (slot[2] - slot[0]) * (slot[3] - slot[1])
    return ix * iy / area if area > 0 else 0.0


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


def update_camera(state: dict, dets: list, ts: float):
    """Actualiza las plazas aprendidas de una cámara con una nueva captura."""
    slots = state.setdefault("slots", [])
    prev = state.get("prev")  # {"ts":..., "dets":[...]}
    events = []

    # 1) vehículos quietos respecto a la captura anterior
    stationary = []
    if prev and MIN_STATIONARY_GAP <= ts - prev["ts"] <= MAX_STATIONARY_GAP:
        for d in dets:
            if d["cls"] not in PARKABLE:
                continue
            if any(iou(d["box"], p) >= STATIONARY_IOU for p in prev["dets"]):
                stationary.append(d["box"])

    # 2) reforzar o crear plazas
    for box in stationary:
        best, best_iou = None, 0.0
        for s in slots:
            v = iou(box, s["box"])
            if v > best_iou:
                best, best_iou = s, v
        if best and best_iou >= SLOT_MATCH_IOU:
            n = best["obs"]
            w = 1.0 / min(n + 1, 8)  # media móvil de la caja
            best["box"] = [round(a * (1 - w) + b * w, 4) for a, b in zip(best["box"], box)]
            best["obs"] = n + 1
            best["last_parked"] = ts
        else:
            slots.append({
                "id": hashlib.md5(f"{box}{ts}".encode()).hexdigest()[:8],
                "box": box, "obs": 1, "first": ts, "last_parked": ts,
                "state": "occupied", "since": ts, "free_n": 0, "hours": {},
            })

    # 3) estado de cada plaza
    all_boxes = [d["box"] for d in dets]
    hour = datetime.fromtimestamp(ts, MADRID).strftime("%H")
    for s in slots:
        occupied = any(iou(s["box"], b) >= OCC_IOU or cover(s["box"], b) >= OCC_COVER for b in all_boxes)
        if occupied:
            s["last_occupied"] = ts
            if s["state"] == "free":
                s["state"], s["since"] = "occupied", ts
            s["free_n"] = 0
        else:
            s["free_n"] = s.get("free_n", 0) + 1
            if s["state"] == "occupied":
                s["state"], s["since"] = "free", ts
                if s["obs"] >= CONFIRM_OBS:
                    events.append({"slot": s["id"], "ts": ts})
        if s["obs"] >= CONFIRM_OBS:
            h = s.setdefault("hours", {}).setdefault(hour, [0, 0])
            h[0] += int(occupied)
            h[1] += 1

    # 4) limpieza
    keep = []
    for s in slots:
        last = max(s.get("last_occupied", 0), s["last_parked"])
        if s["obs"] < CONFIRM_OBS and ts - s["last_parked"] > CANDIDATE_TTL:
            continue
        if ts - last > SLOT_TTL:
            continue
        if any(iou(s["box"], k["box"]) > 0.55 for k in keep):  # fusionar duplicados
            k = max((k for k in keep if iou(s["box"], k["box"]) > 0.55), key=lambda k: k["obs"])
            k["obs"] = max(k["obs"], s["obs"])
            continue
        keep.append(s)
    state["slots"] = keep
    state["prev"] = {"ts": ts, "dets": [d["box"] for d in dets if d["cls"] in PARKABLE]}
    return events


def slot_rois(slots):
    """Zona que engloba las plazas confirmadas, para analizarla ampliada."""
    boxes = [s["box"] for s in slots if s["obs"] >= CONFIRM_OBS]
    if not boxes:
        return []
    x0, y0 = min(b[0] for b in boxes), min(b[1] for b in boxes)
    x1, y1 = max(b[2] for b in boxes), max(b[3] for b in boxes)
    mx, my = max(0.04, (x1 - x0) * 0.12), max(0.04, (y1 - y0) * 0.12)
    roi = [max(0, x0 - mx), max(0, y0 - my), min(1, x1 + mx), min(1, y1 + my)]
    if (roi[2] - roi[0]) * (roi[3] - roi[1]) > 0.6:
        return []  # casi toda la imagen: la pasada normal ya basta
    return [roi]


def slot_public(s, ts):
    hrs = s.get("hours", {})
    occ = sum(v[0] for v in hrs.values())
    tot = sum(v[1] for v in hrs.values())
    state = s["state"]
    if state == "free" and ts - s["since"] > STALE_FREE:
        state = "stale"
    return {
        "id": s["id"], "box": s["box"], "state": state, "since": int(s["since"]),
        "sure": s.get("free_n", 0) >= 2 if s["state"] == "free" else True,
        "obs": s["obs"], "occ_rate": round(occ / tot, 2) if tot >= 6 else None,
        "hours": {h: round(v[0] / v[1], 2) for h, v in hrs.items() if v[1] >= 2},
    }


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
    out_dir.mkdir(parents=True, exist_ok=True)
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
            dets = st.get("last_dets", [])
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
                    dets, w, h, bright = detector.detect(data, slot_rois(st.get("slots", [])))
                    st["digest"], st["img_ts"] = digest, img_ts
                    if bright < MIN_BRIGHTNESS:
                        st["dark"] = True
                        dets = []
                    else:
                        st["dark"] = False
                        for ev in update_camera(st, dets, img_ts):
                            events.append({"cam": cam["id"], **ev})
                        st["last_dets"] = dets
                    processed += 1
                    sp.write_text(json.dumps(st, separators=(",", ":")))
            slots = [slot_public(s, now) for s in st.get("slots", []) if s["obs"] >= CONFIRM_OBS]
            public.append({
                **cam,
                "ts": int(st.get("img_ts", 0)),
                "dark": st.get("dark", False),
                "dets": [d["box"] for d in dets],
                "slots": slots,
                "candidates": sum(1 for s in st.get("slots", []) if s["obs"] < CONFIRM_OBS),
                "free": sum(1 for s in slots if s["state"] == "free"),
            })

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
          "plazas:", sum(len(c["slots"]) for c in public),
          "libres:", sum(c["free"] for c in public))


if __name__ == "__main__":
    main()

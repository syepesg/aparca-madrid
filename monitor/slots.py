"""Aprendizaje de plazas y decisión libre/ocupada para una cámara.

Además de las cajas de YOLO se usa la apariencia de la imagen (pequeños parches de color):
  * "aparcado" = misma caja Y mismo coche (parche parecido) en capturas separadas minutos.
    En una vía con tráfico denso la caja se repite, pero el coche es otro.
  * "libre" = ninguna detección encima Y el parche de la plaza ha cambiado respecto a cuando
    había un coche. Si YOLO simplemente no ve el coche (noche, lluvia, lejos) o lo "aparcado"
    era un objeto fijo (monumento, marquesina), el parche no cambia y no se da por libre.
  * Muchas cámaras son motorizadas y alternan entre varios encuadres (o hacen zoom).
    Cada cámara guarda varios "encuadres" con sus propias plazas; cada captura se asigna
    al encuadre que más se le parece (ignorando las zonas con vehículos) o crea uno nuevo.
"""
from __future__ import annotations

import base64
import hashlib
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
from PIL import Image

MADRID = ZoneInfo("Europe/Madrid")
PARKABLE = {"car", "truck", "motorcycle"}

MIN_STATIONARY_GAP = 4 * 60      # s entre capturas para considerar un coche "quieto"
MAX_STATIONARY_GAP = 45 * 60
STATIONARY_IOU = 0.70            # misma caja...
SAME_CAR_SIM = 0.70              # ...y mismo aspecto
MIN_BOX_W, MIN_BOX_H = 0.014, 0.016   # cajas más pequeñas (≈18x12 px) no son fiables
SLOT_MATCH_IOU = 0.35
CONFIRM_OBS = 3                  # veces visto el mismo coche quieto para confirmar la plaza
OCC_IOU = 0.20                   # una detección ocupa la plaza...
OCC_COVER = 0.30                 # ...o tapa esta fracción de ella
STILL_THERE_SIM = 0.55           # parche aún parecido al coche que había: no está libre
MERGE_IOU, MERGE_COVER = 0.30, 0.60
MIN_OCC_SAMPLES, MIN_OCC_RATE = 12, 0.30  # sitio casi siempre vacío: no es una plaza
MIN_DWELL = 40 * 60              # con un solo coche visto, debe haber estado al menos esto
                                 # (una furgoneta en doble fila no se queda tanto)
CANDIDATE_TTL = 2 * 86400
SLOT_TTL = 10 * 86400
VIEW_SIM = 0.75                  # parecido mínimo para considerar que es el mismo encuadre
MAX_VIEWS = 6                    # muchas cámaras rotan entre posiciones fijas o hacen zoom
VIEW_TTL = 2 * 86400             # encuadres sin plazas que no se repiten caducan

PATCH_W, PATCH_H = 12, 8


# ---------- geometría ----------
def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def cover(a, b):
    """Fracción del área de a tapada por b."""
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    area = (a[2] - a[0]) * (a[3] - a[1])
    return ix * iy / area if area > 0 else 0.0


def big_enough(box):
    return box[2] - box[0] >= MIN_BOX_W and box[3] - box[1] >= MIN_BOX_H


# ---------- apariencia ----------
def patch(img: Image.Image, box) -> np.ndarray:
    """Parche de color 12x8 del interior de la caja (recortado un 12 % para no coger vecinos)."""
    W, H = img.size
    x0, y0, x1, y1 = box
    mx, my = (x1 - x0) * 0.12, (y1 - y0) * 0.12
    crop = img.crop((int((x0 + mx) * W), int((y0 + my) * H),
                     max(int((x1 - mx) * W), int((x0 + mx) * W) + 2),
                     max(int((y1 - my) * H), int((y0 + my) * H) + 2)))
    return np.asarray(crop.resize((PATCH_W, PATCH_H), Image.BILINEAR), dtype=np.uint8)


def enc(p: np.ndarray) -> str:
    return base64.b64encode(p.tobytes()).decode()


def dec(s: str) -> np.ndarray:
    return np.frombuffer(base64.b64decode(s), dtype=np.uint8).reshape(PATCH_H, PATCH_W, 3)


def similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Parecido 0..1 entre dos parches: estructura (correlación) y color, robusto a cambios de luz."""
    a = a.astype(np.float32)
    b = b.astype(np.float32)
    ga, gb = a.mean(2), b.mean(2)
    sa, sb = ga.std(), gb.std()
    if sa < 6 and sb < 6:          # ambos lisos (asfalto vacío con asfalto vacío)
        struct = 1.0
    elif sa < 6 or sb < 6:         # uno liso y el otro con textura
        struct = 0.0
    else:
        struct = float(((ga - ga.mean()) * (gb - gb.mean())).mean() / (sa * sb))
    # cromaticidad media (independiente del brillo)
    ca = a.reshape(-1, 3).mean(0); ca = ca / (ca.sum() + 1e-6)
    cb = b.reshape(-1, 3).mean(0); cb = cb / (cb.sum() + 1e-6)
    chroma = float(np.abs(ca - cb).sum())          # 0 igual … ~0.3 muy distinto
    # brillo relativo: coche oscuro sobre asfalto claro, etc.
    la, lb = ga.mean(), gb.mean()
    light = abs(la - lb) / (max(la, lb) + 1e-6)
    return max(0.0, min(1.0, 0.65 * max(struct, 0) + 0.35 * (1 - min(1, chroma * 6)) - 0.5 * max(0, light - 0.35)))


TW, TH = 48, 27   # miniatura para reconocer el encuadre


def scene_thumb(img: Image.Image) -> np.ndarray:
    """Miniatura en gris 48x27 (media por áreas) para reconocer el encuadre."""
    return np.asarray(img.convert("L").resize((TW, TH), Image.BOX), dtype=np.float32)


def box_mask(boxes) -> np.ndarray:
    """True en las celdas de la miniatura libres de vehículos (los coches cambian, el fondo no)."""
    m = np.ones((TH, TW), bool)
    for x0, y0, x1, y1 in boxes:
        m[int(y0 * TH):int(np.ceil(y1 * TH)), int(x0 * TW):int(np.ceil(x1 * TW))] = False
    return m


def _ncc(a, b, m):
    a, b = a[m], b[m]
    if a.size < 50:
        return 0.0
    a = a - a.mean(); b = b - b.mean()
    d = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / d) if d > 0 else 0.0


def scene_similarity(a: np.ndarray, b: np.ndarray, mask=None) -> float:
    """Parecido de encuadre: correlación de gris y de bordes fuera de los vehículos."""
    m = np.ones(a.shape, bool) if mask is None else mask
    ga = np.hypot(*np.gradient(a)); gb = np.hypot(*np.gradient(b))
    return 0.5 * _ncc(a, b, m) + 0.5 * _ncc(ga, gb, m)


# ---------- actualización ----------
def enc_thumb(t: np.ndarray) -> str:
    return base64.b64encode(np.clip(np.round(t), 0, 255).astype(np.uint8).tobytes()).decode()


def dec_thumb(s: str) -> np.ndarray:
    return np.frombuffer(base64.b64decode(s), dtype=np.uint8).reshape(TH, TW).astype(np.float32)


def match_view(state: dict, thumb: np.ndarray, mask: np.ndarray, ts: float):
    """Devuelve (encuadre, parecido) para esta captura; crea uno nuevo si ninguno encaja."""
    views = state.setdefault("views", [])
    best, best_sim = None, -1.0
    for v in views:
        sim = scene_similarity(thumb, dec_thumb(v["thumb"]), mask & box_mask(v.get("mask_boxes", [])))
        if sim > best_sim:
            best, best_sim = v, sim
    if best is not None and best_sim >= VIEW_SIM:
        # seguir despacio los cambios de luz del día
        best["thumb"] = enc_thumb(0.7 * dec_thumb(best["thumb"]) + 0.3 * thumb)
        best["last"], best["n"] = ts, best.get("n", 0) + 1
        return best, best_sim
    v = {"id": hashlib.md5(f"{ts}{len(views)}".encode()).hexdigest()[:6], "thumb": enc_thumb(thumb),
         "first": ts, "last": ts, "n": 1, "slots": [], "prev": None}
    views.append(v)
    if len(views) > MAX_VIEWS:  # descartar el encuadre menos útil
        views.sort(key=lambda x: (sum(1 for s in x["slots"] if is_real_slot(s)), x["last"]))
        views.pop(0)
    return v, best_sim


def update_camera(state: dict, dets: list, ts: float, img: Image.Image):
    """Asigna la captura a un encuadre y actualiza sus plazas. Devuelve eventos de liberación."""
    return update_view(assign_view(state, dets, ts, img), dets, ts, img)


def assign_view(state: dict, dets: list, ts: float, img: Image.Image):
    boxes = [d["box"] for d in dets]
    thumb = scene_thumb(img)
    view, sim = match_view(state, thumb, box_mask(boxes), ts)
    view["mask_boxes"] = [[round(x, 3) for x in b] for b in boxes]
    state["view"], state["view_sim"] = view["id"], round(sim, 3)
    state["views"] = [v for v in state["views"] if v is view or ts - v["last"] < VIEW_TTL
                      or any(is_real_slot(s) for s in v["slots"])]
    return view


def update_view(view: dict, dets: list, ts: float, img: Image.Image):
    slots = view.setdefault("slots", [])
    events = []
    parked = [d for d in dets if d["cls"] in PARKABLE and big_enough(d["box"])]
    cur = [(d["box"], patch(img, d["box"])) for d in parked]

    # 1) mismo coche quieto respecto a la captura anterior de este encuadre
    stationary = []
    prev = view.get("prev")
    if prev and MIN_STATIONARY_GAP <= ts - prev["ts"] <= MAX_STATIONARY_GAP:
        prev_items = [(b, dec(p)) for b, p in prev["items"]]
        for box, p in cur:
            for pb, pp in prev_items:
                if iou(box, pb) >= STATIONARY_IOU and similarity(p, pp) >= SAME_CAR_SIM:
                    stationary.append((box, p))
                    break

    # 2) reforzar o crear plazas
    for box, p in stationary:
        best, best_iou = None, 0.0
        for s in slots:
            v = iou(box, s["box"])
            if v > best_iou:
                best, best_iou = s, v
        if best and best_iou >= SLOT_MATCH_IOU:
            n = best["obs"]
            w = 1.0 / min(n + 1, 8)
            best["box"] = [round(a * (1 - w) + b * w, 4) for a, b in zip(best["box"], box)]
            best["obs"] = n + 1
            if best["state"] == "free" or ts - best["last_parked"] > MAX_STATIONARY_GAP:
                best["episodes"] = best.get("episodes", 1) + 1   # otro coche ha aparcado aquí
                best["ep_start"] = prev["ts"]
            best["last_parked"] = ts
            best["dwell"] = max(best.get("dwell", 0), ts - best.get("ep_start", best["first"]))
        else:
            slots.append({
                "id": hashlib.md5(f"{box}{ts}".encode()).hexdigest()[:8],
                "box": box, "obs": 1, "first": ts, "last_parked": ts,
                "episodes": 1, "ep_start": prev["ts"], "dwell": ts - prev["ts"],
                "state": "occupied", "since": ts, "free_n": 0, "hours": {},
            })

    # 3) estado de cada plaza
    all_boxes = [d["box"] for d in dets]
    hour = datetime.fromtimestamp(ts, MADRID).strftime("%H")
    for s in slots:
        sp = patch(img, s["box"])
        hit = any(iou(s["box"], b) >= OCC_IOU or cover(s["box"], b) >= OCC_COVER for b in all_boxes)
        if hit:
            occupied = True
            # referencia de "coche aquí" solo cuando la detección encaja bien con la plaza
            if any(iou(s["box"], b) >= 0.5 for b in all_boxes):
                s["ref_occ"] = enc(sp)
        elif "ref_occ" in s and similarity(sp, dec(s["ref_occ"])) >= STILL_THERE_SIM:
            occupied = True    # YOLO no lo ve, pero la imagen no ha cambiado
            s["miss_n"] = s.get("miss_n", 0) + 1
        elif "ref_occ" not in s:
            occupied = True    # sin referencia no podemos asegurar que esté libre
        else:
            occupied = False

        if occupied:
            s["last_occupied"] = ts
            if s["state"] == "free":
                s["state"], s["since"] = "occupied", ts
            s["free_n"] = 0
        else:
            s["free_n"] = s.get("free_n", 0) + 1
            if s["state"] == "occupied":
                s["state"], s["since"] = "free", ts
            if s["free_n"] == 2 and is_real_slot(s):
                events.append({"slot": s["id"], "ts": s["since"]})
        if s["obs"] >= CONFIRM_OBS:   # estadística por hora (también decide si es plaza real)
            h = s.setdefault("hours", {}).setdefault(hour, [0, 0])
            h[0] += int(occupied)
            h[1] += 1

    # 4) limpieza y fusión de duplicados (se conserva la más observada)
    keep = []
    for s in sorted(slots, key=lambda s: -s["obs"]):
        last = max(s.get("last_occupied", 0), s["last_parked"])
        if s["obs"] < CONFIRM_OBS and ts - s["last_parked"] > CANDIDATE_TTL:
            continue
        if ts - last > SLOT_TTL:
            continue
        dup = next((k for k in keep if iou(s["box"], k["box"]) > MERGE_IOU
                    or cover(s["box"], k["box"]) > MERGE_COVER
                    or cover(k["box"], s["box"]) > MERGE_COVER), None)
        if dup:
            continue
        keep.append(s)
    view["slots"] = keep
    view["prev"] = {"ts": ts, "items": [(b, enc(p)) for b, p in cur]}
    return events


def current_slots(state: dict):
    """Plazas del encuadre que muestra ahora la cámara."""
    v = next((v for v in state.get("views", []) if v["id"] == state.get("view")), None)
    return v["slots"] if v else []


def occupancy(s):
    hrs = s.get("hours", {})
    occ = sum(v[0] for v in hrs.values())
    tot = sum(v[1] for v in hrs.values())
    return (occ / tot if tot else None), tot


def is_real_slot(s):
    """Plaza confirmada que se comporta como un aparcamiento (suele estar ocupada)."""
    if s["obs"] < CONFIRM_OBS:
        return False
    if s.get("episodes", 1) < 2 and s.get("dwell", 0) < MIN_DWELL:
        return False
    rate, n = occupancy(s)
    return not (n >= MIN_OCC_SAMPLES and rate < MIN_OCC_RATE)


def slot_public(s, ts, stale_free):
    rate, n = occupancy(s)
    state = s["state"]
    if state == "free" and ts - s["since"] > stale_free:
        state = "stale"
    return {
        "id": s["id"], "box": s["box"], "state": state, "since": int(s["since"]),
        "sure": s.get("free_n", 0) >= 2 if s["state"] == "free" else True,
        "obs": s["obs"], "occ_rate": round(rate, 2) if n >= 6 else None,
        "hours": {h: round(v[0] / v[1], 2) for h, v in s.get("hours", {}).items() if v[1] >= 2},
    }

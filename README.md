# Aparca Madrid

Detecta huecos de aparcamiento libres en las **357 cámaras públicas de tráfico del Ayuntamiento de Madrid** usando **YOLO11**, sin servidores y gratis:

- **GitHub Actions** (cron cada ~10 min) descarga todas las imágenes de `informo.madrid.es`, detecta vehículos con YOLO11s (ONNX, 960 px) y aprende dónde se aparca.
- **GitHub Pages** sirve la web: mapa, lista de huecos libres por cercanía y ficha de cada cámara con la imagen en directo y las plazas superpuestas.
- En la web, **"Analizar ahora"** ejecuta YOLO11n en tu propio móvil/navegador (onnxruntime-web) sobre la imagen más reciente, y **"Vigilar y avisarme"** repite el análisis cada minuto y te avisa (sonido, vibración, notificación) cuando se libera un hueco.

## Cómo aprende las plazas

Las cámaras no traen plazas marcadas, así que el sistema las descubre solo:

1. Un vehículo cuya caja es prácticamente idéntica (IoU ≥ 0,72) en dos capturas separadas entre 4 y 45 minutos **está aparcado** (el tráfico nunca repite posición).
2. Cada sitio donde eso ocurre es una plaza candidata; al verlo aparcado **2 veces** pasa a ser plaza vigilada.
3. En cada pasada, una plaza vigilada sin ningún vehículo encima pasa a **libre** (amarilla con una sola captura, verde a partir de dos). La zona de plazas se analiza además ampliada para no perder coches lejanos.
4. Se guarda la ocupación por hora del día para mostrar la probabilidad de encontrar hueco.
5. Las plazas que no vuelven a ocuparse en 10 días caducan (p. ej. si la cámara cambia de encuadre) y las libres más de 18 h se ocultan.

La memoria entre ejecuciones vive en la rama `state` (un único commit que se sobrescribe), así que el historial de `main` no crece.

## Limitaciones

- Las cámaras enfocan sobre todo calzadas: solo una parte ve aparcamiento en superficie.
- Una "plaza" es donde se ha visto un coche quieto; puede ser zona SER, carga y descarga, vado o doble fila.
- GitHub no garantiza la puntualidad del cron (a veces se retrasa 5–20 min) y las imágenes de las cámaras se renuevan cada ~5 min.

## Estructura

```
monitor/detector.py   YOLO11 en ONNX con numpy (sin torch)
monitor/run.py        descarga, detección, aprendizaje de plazas, status.json
models/               yolo11s-960.onnx (servidor), yolo11n.onnx (navegador)
site/index.html       web (Leaflet + onnxruntime-web)
.github/workflows/    cron + despliegue en Pages
```

Ejecución local:

```bash
pip install -r requirements.txt
python monitor/run.py --model models/yolo11s-960.onnx --state state --out site/data
cp models/yolo11n.onnx site/models/ && python -m http.server -d site
```

Datos: cámaras de tráfico del [Portal de datos abiertos del Ayuntamiento de Madrid](https://datos.madrid.es). Modelos: [Ultralytics YOLO11](https://github.com/ultralytics/ultralytics) (AGPL-3.0).

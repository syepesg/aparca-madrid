# Aparca Madrid

Detecta huecos de aparcamiento libres en las **357 cámaras públicas de tráfico del Ayuntamiento de Madrid** usando **YOLO11**, sin servidores y gratis:

- **GitHub Actions** (cron cada ~10 min) descarga todas las imágenes de `informo.madrid.es`, detecta vehículos con YOLO11s (ONNX, 960 px) y aprende dónde se aparca.
- **GitHub Pages** sirve la web: mapa, lista de huecos libres por cercanía y ficha de cada cámara con la imagen en directo y las plazas superpuestas.
- En la web, **"Analizar ahora"** ejecuta YOLO11n en tu propio móvil/navegador (onnxruntime-web) sobre la imagen más reciente, y **"Vigilar y avisarme"** repite el análisis cada minuto y te avisa (sonido, vibración, notificación) cuando se libera un hueco.

## Cómo aprende las plazas

Las cámaras no traen plazas marcadas, así que el sistema las descubre solo (`monitor/slots.py`):

1. **Encuadres.** Muchas cámaras son motorizadas: rotan entre posiciones o hacen zoom cada pocos minutos. Cada cámara guarda hasta 6 encuadres, cada uno con sus propias plazas. Cada captura se asigna al encuadre más parecido (miniatura 48×27 en gris y bordes, ignorando las zonas con vehículos; umbral 0,75) o crea uno nuevo. Las plazas de un encuadre solo se evalúan cuando la cámara lo está mostrando.
2. **Coche aparcado = misma caja y mismo coche.** Un vehículo cuya caja coincide (IoU ≥ 0,70) en dos capturas separadas de 4 a 45 minutos y cuyo aspecto (parche de color 12×8) también coincide (similitud ≥ 0,70) está quieto. En una vía con tráfico denso la caja se repite, pero el coche es otro, así que no cuenta. Las cajas diminutas (< 18×12 px) se ignoran.
3. **Plaza real.** Hace falta ver el coche quieto 3 veces y, además, que hayan aparcado ahí dos coches distintos o que uno haya estado al menos 40 minutos (una furgoneta en doble fila no se queda tanto). Los sitios que casi nunca están ocupados (menos del 30 % de las capturas) se descartan. Los duplicados se fusionan.
4. **Libre = nada encima y la imagen ha cambiado.** Una plaza solo se da por libre si YOLO no detecta nada encima **y** su parche ya no se parece al del coche que había (similitud < 0,55). Así, si YOLO simplemente no ve el coche (noche, lluvia, lejos) o el "coche" era un objeto fijo (monumento, marquesina, el texto de la hora), no hay falso hueco. Se exige en **dos capturas seguidas** antes de mostrarlo, y el hueco debe verse igual en ambas (si cambia, hay ramas, peatones u otro coche maniobrando). No se declaran huecos en zonas demasiado oscuras ni cuando la imagen encaja de forma dudosa con su encuadre (reflejos del sol, neblina, gotas). La referencia "coche aquí" se va actualizando mientras el coche sigue, para seguir los cambios de luz.
5. La zona de plazas se analiza también ampliada para no perder coches lejanos, y se guarda la ocupación por hora del día.

La web aplica las mismas reglas en tu móvil (reconoce el encuadre de la imagen en directo y compara los parches) cuando pulsas "Analizar ahora" o "Vigilar".

La memoria entre ejecuciones vive en la rama `state` (un único commit que se sobrescribe), así que el historial de `main` no crece.

## Limitaciones

- Las cámaras enfocan sobre todo calzadas: solo una parte ve aparcamiento en superficie.
- Una "plaza" es donde se ha visto un coche quieto; puede ser zona SER, carga y descarga, vado o doble fila.
- GitHub no garantiza la puntualidad del cron (a veces se retrasa 5–20 min) y las imágenes de las cámaras se renuevan cada ~5 min.

## Estructura

```
monitor/detector.py   YOLO11 en ONNX con numpy (sin torch)
monitor/run.py        descarga, detección y salida para la web
monitor/slots.py      encuadres, aprendizaje de plazas y decisión libre/ocupada
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

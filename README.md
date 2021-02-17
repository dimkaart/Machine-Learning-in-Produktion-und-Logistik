# Machine Learning in Produktion und Logistik

<img src="/data/helpers/KennzeichenErkennung.png"  width="80%" >

# Ziel
Ziel dieses Projektes im Rahmen des Fachlabors "Machine Learning in Produktion und Logistik" an der TU Dortmund war es, eine Pipeline zur Kennzeichenerkennung zu entwickeln.
Dabei sollten verschiedene Methoden getestet werden und am Ende die beste ausgewählt werden.
- Die Ideen wurden in der Gruppe erarbeitet und den Kursteilnehmern präsentiert
- Vorworfen wurden beispielsweise
  - Reine Edge Detection (mittels OpenCV)
  - versch. Aufbereitungsmethoden
  - etc.
- finale Pipeline:
  - YOLOv4 zur Kennzeichenerkennung
  - YOLOv4 zur Buchstabenerkennung und segmentierung
  - PyTesseract für Buchstabenübersetzung 
  - Image PreProcessing in **allen** Schritten
  

## Die Idee hinter der Pipeline
![](/data/helpers/ideepipeline.png)
![](/data/helpers/finaleansaetze.png)

## Pseudocode
- Initialiseren
  - Ergebnisdatei öffnen
  - YOLOv4 Modelle laden
- Für jedes Bild in Input Ordner
  - Suche Kennzeichen mittels YOLOv4 Modell
  - Markiere Kennzeichen (iou Schwellwert=0.5)
  - Schneide das Bild auf Kennzeichen zu und lege entsprechend ab
- Für jedes Bild in "Zugeschnittenen Kennzeichen" Ordner
  - prüfe ob Ausrichtung des Kennzeichens (drehen) notwendig
    - wenn ja, durchführen. Sonst Bild beibehalten
  - Suche nach Klasse: "Buchstabe"
  - Markiere Buchstaben und vergleiche Proportionen
  - Schneide auf Buchstaben zu und lege diese je Kennzeichen in einen Ordner
- Für jedes Bild in "Buchstaben eines Kennzeichen" Ordner
  - **PreProcessing**
  - Durchlaufe OCR von PyTesseract
    - falls nichts gefunden wurden wende OCR aufs ganze Kennzeichen an
  - Füge String von links nach rechts zusammen
- Ausgabe des String in Ergebnis.txt

# Vorbereitung
## Conda (Empfohlen)

```bash
# Tensorflow CPU
conda env create -f conda-cpu.yml
conda activate yolov4-cpu

# Tensorflow GPU
conda env create -f conda-gpu.yml
conda activate yolov4-gpu
```
### Pip (Alternativ)
```bash
# TensorFlow CPU
pip install -r requirements.txt

# TensorFlow GPU
pip install -r requirements-gpu.txt
```

### Nvidia Driver (Für GPU, wenn du nicht Conda Environment nutzt und CUDA installiert hast)
Make sure to use CUDA Toolkit version 10.1 as it is the proper version for the TensorFlow version used in this repository.
https://developer.nvidia.com/cuda-10.1-download-archive-update2

## Custom Weights Kennzeichen implementieren
Wir haben für das Modell verschiedene YOLOv4 Modelle trainiert und mittels 10-Fold CrossValidation vergleichen.
Ein Auszug aus den Ergebnissen nachfolgend:
![](/data/helpers/CrossValid_final.png)

#### Zusätzlich galt es den Loss zu analysieren.
<img src="/data/helpers/Test%201%200-1500.jpg"  width="50%" >

Zu sehen ist, dass der iou Loss immer weiter abnimmt. Der Unterschied zwischen "Ground Truth" und Prediticion immer geringer wird.
Das Training muss rechtzeitig gestoppt werden, um ein Übertraining zu verhindern. Wir nehmen die Weights nach 1.500 Iterationen, da diese sowohl im Testdatensatz innerhalb des Trainings, als auch manueller Kontrolle die besten Ergebnisse hervorrufen.


<img src="/data/helpers/iou%20loss.png"  width="50%" >

Aufgrund von begrenztem Speicher im GitHub sind die von uns trainierten Weights extern zu beziehen.

## Custom Weights Character implementieren

Iterationen | mAP [%] | IoU [%]
--- | --- | ---
1000 | 99.69 | 76.79
1100 | 99.60 | 78.82
1200 | 99.70 | 80.29 
1300 | 99.46 | 81.87
1400 | 95.75 | 64.67
1500 | 99.23 | 76.21
1600 | 99.76 | 80.38
1700 | 99.71 | 81.78
1800 | 99.89 | 82.81
1900 | 99.80 | 83.90
2000 | 99.92 | 83.27
2100 | 99.78 | 83.06
2200 | 99.64 | 82.71
2300 | 99.67 | 83.04
2400 | 99.71 | 83.34
2500 | 99.54 | 79.82

## Ergebnisse des Tunings

oem | psm | pixel | score | Ø-Levenshtein-Distanz
--- |--- |---|---|---
13 | 1 | 2 | 0.5 | 2.648
13 | 1 | 2 | 0.6 | 2.648
13 | 1 | 2 | 0.7 | 2.653
13 | 1 | 1 | 0.6 | 2.775
13 | 1 | 1 | 0.7 | 2.780
13 | 1 | 1 | 0.5 | 2.780
13 | 1 | 3 | 0.6 | 2.784
13 | 1 | 3 | 0.7 | 2.784
13 | 1 | 3 | 0.5 | 2.784
10 | 1 | 2 | 0.5 | 2.872
10 | 1 | 2 | 0.6 | 2.872
10 | 1 | 2 | 0.7 | 2.874
10 | 1 | 1 | 0.6 | 3.013
10 | 1 | 1 | 0.5 | 3.015
10 | 1 | 1 | 0.7 | 3.016
10 | 1 | 3 | 0.5 | 3.058
10 | 1 | 3 | 0.6 | 3.058
10 | 1 | 3 | 0.7 | 3.060
13 | 1 | 0 | 0.6 | 3.463
13 | 1 | 0 | 0.5 | 3.465
13 | 1 | 0 | 0.7 | 3.465
10 | 1 | 0 | 0.6 | 3.584
10 | 1 | 0 | 0.5 | 3.585
10 | 1 | 0 | 0.7 | 3.588

## Zum Download der Weights bitte die entsprechenden Depotlinks nutzen.
### Kennzeichen Erkennung 
- **https://depot.tu-dortmund.de/e8faf** 
- Die Datei bitte im ./data/ Ordner als "custom.weights" -Datei ablegen.

### Character Segmentation:
- **https://depot.tu-dortmund.de/hvw7v**
- Die Datei in ./data/ als "char.weights" ablegen.

## YOLOv4 Tensorflow (tf, .pb model) einstellen
To implement YOLOv4 using TensorFlow, first we convert the .weights into the corresponding TensorFlow model files.
```bash
# Convert darknet weights to tensorflow

# Kennzeichen
python save_model.py --weights ./data/custom.weights --output ./checkpoints/custom-416 --input_size 416 --model yolov4 

# Character Segmentation
python save_model.py --weights ./data/char.weights --output ./checkpoints/char-416 --input_size 416 --model yolov4 --nolicense
```

## Tesseract Pfad anpassen
In functions.py in Zeile 14 den Pfad zu "tesseract.exe" anpassen.
![](/data/helpers/TesseractPfad.png)

# Pipeline anwenden
## Bilder in Input Ordner legen
Die zu erkennenden Bilder von Fahrzeugen mit Kennzeichen bitte in den ./Input/ Ordner legen.
![](/data/helpers/Input.png)

## Einstellungsmöglichkeiten Pipeline
In Zeile 194 & 194 können die Parameter final & tune eingestellt werden. Dabei wird mit tune = True das Tuning durchgeführt und final = True berechnet keine Levenshtein Distanz.

## Pipeline ausführen
```
# cmd öffnen
# prüfen ob conda env yolov4-cpu/gpu aktiv ist (s. oben)

# Pipeline starten:
python Pipeline_detect.py
```

## Die Ergebnisse befinden sich im ./Output/ Ordner
Innerhalb des Ordners befinden sich:
- Crop, die zugeschnittenen Fotos auf das Kennzeichen
- Erkennung, Bounding Box und Genauigkeit auf Fahrzeug
- Textdatei mit Ergebnissen
![](/data/helpers/Output.png)

Es wird zusätzlich eine Ergebnis-Datei erstellt, in der Bildname und erkanntes Kennzeichen bzw. die Buchstaben ausgegeben werden.
![](/data/helpers/Ergebnis.png)

## Übersicht der Preprocessingschritte:
![](/data/helpers/preprocessingschritte.png)
- **Resizing:** Bildausschnitt wird vergrößert, da das Nummernschilder auf den meisten Bildern nur einen sehr kleinen Teilbereich darstellt. Zudem wird bei der OCR, eine bestimmte Pixelhöhe für die Extraktion benötigt
- **Grayscaling:** Da für das Auslesen des Nummernschildes Farben vernachlässigt werden können, wird als nächstes das Bild in Graustufen konvertiert. Somit lässt sich zusätzlich Rechenleistung einsparen
- **Gaussian-Blur:** Um des weiteren zusätzlich Rauschen und irrelevante Informationen zu entfernen, wird die Gaußche-Unschärfetechnik verwendet. Wichtig ist dabei zu beachten, dass je höher der Einstellungswert, desto weniger Rauschen entsteht, jedoch auch gleichzeitig mehr Bildinformation verloren wird
- **Thresholding Otsu-Methode:** Damit die Zeichen auf dem Nummernschild herausstechen, muss des weiteren mit Hilfe des Schwellenwertverfahrens nach der Otsu-Methode der Vordergrund vom Hintergrund getrennt werden, indem der Hintergrund umgekehrt wird

## Rotation (falls nötig und möglich):
![](/data/helpers/rotation.png)

# Verworfene Ideen auf dem Weg zur finalen Pipeline
![](/data/helpers/pipelineansaetze.png)



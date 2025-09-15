# Segmenting Hydroxylapatit

Dieses Repository dokumentiert die Umsetzung meiner Bachelorarbeit zur automatisierten Analyse von Schichtdicken von Hydroxylapatit-Beschichtungen auf Titanimplantaten.  
Die Arbeit umfasst die Datenaufbereitung, das Training und die Evaluation von Segmentierungsmodellen mit dem **Segment Anything Model (SAM)** als zentraler Architektur.

---

## Struktur des Repositories

- **notebooks/**  
  Enthält die im Rahmen der Arbeit entwickelten Google-Colab-Notebooks.  
  Sie dokumentieren Datenexploration, Modelltraining sowie Evaluationsschritte.  

- **Predict_HA/SAM_models**  
  Enthält die trainierten SAM-Varianten und Finetuning-Gewichte.  
  Große Dateien werden über **Git LFS** verwaltet.  

- **input_files/**  
  Verwendete Datensätze. Aufgeteilt in images, masks und TPS_layer

  - colab_image_preprocessing.py beinhaltet das Beschriebene Verfahren der Datenvorverarbeitung
  - patched_images enhält die unterschiedlich vorbereiteten Datensätze

- **threshold_method/**  
  Verzeichnis für den Otsu Thresholding Ansatz

- **outpu_files/**
  Beinhaltet die im Ergebnisteil Evaluierten Methoden Otsu_Threshold, SAM und, HA_SAM

---



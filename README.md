# Segmenting Hydroxylapatit

Dieses Repository dokumentiert die Umsetzung meiner Bachelorarbeit zur automatisierten Analyse von Schichtdicken von Hydroxylapatit-Beschichtungen auf Titanimplantaten.  
Die Arbeit umfasst die Datenaufbereitung, das Training und die Evaluation von Segmentierungsmodellen mit dem **Segment Anything Model (SAM)** als zentraler Architektur.

---

## Struktur des Repositories

- **Notebooks/**  
  Enthält die im Rahmen der Arbeit entwickelten Google-Colab-Notebooks.  
  Sie dokumentieren Datenexploration, Modelltraining sowie Evaluationsschritte.  

- **Modelle/**  
  Enthält die trainierten SAM-Varianten und Finetuning-Gewichte.  
  Große Dateien werden über **Git LFS** verwaltet.  

- **src/** (optional, falls vorhanden)  
  Python-Skripte zur Vorverarbeitung, Trainingssteuerung und Evaluierung.  

- **data/** (nicht versioniert)  
  Verzeichnis für Roh- und verarbeitete Daten. Aufgrund der Dateigröße und Datenschutzrichtlinien werden die Daten nicht im Repository gespeichert.  

---

## Installation

Das Projekt setzt eine Python-Umgebung voraus (empfohlen: Python ≥3.9).  
Alle notwendigen Pakete können über die bereitgestellte `requirements.txt` installiert werden:

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

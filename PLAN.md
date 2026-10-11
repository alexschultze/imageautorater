# Plan: ImageAutoRater mit PyTorch verteilen (Installation über uv)

## Ziel
Das bestehende Python-Skript **ImageAutoRater** (nutzt PyTorch, lädt Pakete und Modelle nach) wird mit einer GUI an Endanwender verteilt.
Die Anwender haben unterschiedliche Hardware: nur CPU, NVIDIA, AMD oder Intel. GPU-Unterstützung soll automatisch erkannt und später änderbar sein.

## Entscheidungen
- **Verteilung als installierbares Python-Paket über uv.** Es gibt keinen Exe-Build und keine eigene Signatur (Exe-Weg siehe Plan B).
- **Hardware-Erkennung übernimmt uv:** `uv pip install ... --torch-backend=auto` erkennt CUDA-Treiber, AMD-GPUs und Intel-GPUs und wählt den passenden PyTorch-Index. Ohne GPU nimmt uv die CPU-Version.
- **Einschränkung:** Laut uv-Dokumentation ist `--torch-backend` derzeit nur in der `uv pip`-Schnittstelle verfügbar, nicht bei `uv tool install` oder `uvx`. Deshalb wird in einer eigenen Umgebung installiert.
- **AMD unter Windows:** wird nicht berücksichtigt; diese Anwender bekommen die CPU-Version (PyTorch veröffentlicht ROCm-Builds nur für Linux).
- **Python 3.12** (`requires-python = ">=3.12"`). uv besorgt die Version beim Anwender selbst. Das Skript nutzt außer PyPI-Paketen nichts, ein späterer Wechsel auf 3.13 oder 3.14 ist möglich, sobald sich alle Abhängigkeiten dafür installieren lassen.
- **Intel (XPU)** und **NVIDIA (CUDA)** werden unter Windows und Linux unterstützt.
- **Voraussetzung beim Anwender:** uv ist installiert (z. B. `winget install astral-sh.uv`). Installation und Modell-Download setzen Internet voraus; eine Offline-Installation ist nicht vorgesehen.
- **Modelle werden zur Laufzeit geladen** (z. B. Hugging Face) in einen Cache-Ordner unter `%LOCALAPPDATA%`, nicht ins Paket gepackt.
- **GPU wechseln:** Ein Menüpunkt "GPU-Unterstützung installieren" ruft denselben Installationsbefehl noch einmal auf, mit `cu130`, `xpu` oder `cpu` statt `auto`.
- **CI/CD (klein):** GitHub Actions führt Tests aus, baut bei einem Tag wie `v1.0.0` mit `uv build` ein Wheel und hängt es ans Release. Bei einem privaten Repo wird das Wheel aus dem Release installiert, weil `git+https` dort eine Anmeldung braucht.

## Repo-Aufbau
```
imageautorater/
├── pyproject.toml
├── uv.lock
├── src/imageautorater/
│   ├── __init__.py
│   ├── gui.py        # CustomTkinter-Oberfläche
│   ├── runner.py     # startet das Skript als Unterprozess
│   └── pipeline.py   # bisheriges Skript (Modelle durchlaufen)
└── .github/workflows/
```
- `pyproject.toml` mit Build-Backend (z. B. hatchling) und `[project.gui-scripts]` (Start ohne Konsolenfenster), zusätzlich `[project.scripts]` für die Kommandozeile.
- Wichtige Versionen werden direkt in `dependencies` festgelegt, weil `uv.lock` und `[tool.uv.sources]` bei einer Installation per `uv pip` nicht greifen.

## Installation beim Anwender
```
uv venv
uv pip install git+https://github.com/DEIN-NAME/imageautorater --torch-backend=auto
.venv\Scripts\imageautorater
```
Alternativ als `install.bat` oder beim ersten Start durch die GUI. Aktualisieren: derselbe `uv pip install`-Befehl mit `--upgrade`.

## Hardware-Varianten
| Variante | Umsetzung | Zu prüfen |
|---|---|---|
| CPU | Standard und Rückfall | – |
| NVIDIA | `--torch-backend=auto` oder `cuXXX` | Passende CUDA-Version zum Treiber |
| Intel (Arc, integrierte Xe) | `--torch-backend=auto` oder `xpu` | Funktioniert die Erkennung unter Windows zuverlässig? |
| AMD unter Windows | nicht berücksichtigt, CPU-Version | – |
| AMD unter Linux | ROCm über `--torch-backend=auto` | Nur falls Linux-Anwender erwartet werden |

## GUI und Ablauf
Die Oberfläche wählt einen Ordner aus und startet dort das Skript, das nacheinander mehrere Modelle durchläuft (Laufzeit wahrscheinlich lang).

- **Toolkit:** CustomTkinter (klein, modern genug). PySide6 nur, falls die Oberfläche später deutlich anspruchsvoller wird.
- **Skript als eigener Unterprozess** (`subprocess`), nie im GUI-Thread. Die Oberfläche bleibt bedienbar und überlebt Abstürze oder Speichermangel von PyTorch.
- **Fortschritt über stdout:** Das Skript schreibt pro Schritt eine JSON-Zeile (z. B. Modell, Schritt, Gesamtzahl), die GUI liest mit. Das Skript bleibt auch ohne GUI per Kommandozeile nutzbar.
- **Zwei Fortschrittsebenen:** gesamt über alle Modelle und aktuelles Modell, dazu geschätzte Restzeit aus der Dauer der bisherigen Modelle.
- **Abbrechen-Knopf**, der den Unterprozess sauber beendet.
- **Ergebnisse pro Modell sofort in den gewählten Ordner schreiben**, damit bei Abbruch nichts verloren geht.
- **Fortsetzen:** Modelle mit vorhandenem Ergebnis im Ordner werden übersprungen.
- **Logfenster und Logdatei** für die Fehlersuche bei fremden Anwendern.

## Aufgabenliste
- [ ] **Test zuerst:** Funktioniert `uv pip install <Paket aus Git oder Wheel> --torch-backend=auto` auch für `torch` als Abhängigkeit des eigenen Pakets? Auf einem Rechner mit und einem ohne GPU prüfen.
- [ ] Skript und Abhängigkeiten erfassen, `pyproject.toml` im src-Layout anlegen, Einstiegspunkte (`gui-scripts`, `scripts`) definieren.
- [ ] Skript so umbauen, dass es Fortschritt als JSON-Zeilen auf stdout meldet und vorhandene Ergebnisse überspringt (Fortsetzen).
- [ ] GUI mit CustomTkinter: Ordnerauswahl, Start und Abbrechen, Gesamt- und Modell-Fortschritt, Restzeitschätzung, Logfenster.
- [ ] Unterprozess-Steuerung in der GUI: Ausgabe lesen, sauberer Abbruch, Fehleranzeige bei Absturz.
- [ ] Modell-Download mit Cache-Ordner, Fortschrittsanzeige und Wiederaufnahme bei Abbruch.
- [ ] Installation und GPU-Wechsel: `install.bat` bzw. GUI-Menüpunkt, der `uv pip install ... --torch-backend=<auto|cuXXX|xpu|cpu>` aufruft.
- [ ] GitHub-Actions-Workflow: Tests, `uv build`, Wheel an Release hängen.
- [ ] Tests auf Rechnern ohne GPU und mit je einer GPU pro Hersteller, soweit verfügbar.

## Plan B: eigenes Windows-Programm (nur falls uv beim Anwender nicht zumutbar ist)
- Kleiner Launcher als Exe (PyInstaller `--onedir`), der beim ersten Start mit gebündeltem `uv.exe` die Umgebung einrichtet.
- Build in GitHub Actions auf `windows-latest`, `uv.exe` in fest verdrahteter Version ins Release-Paket legen.
- Optional ein Installer (z. B. Inno Setup). Das Ergebnis ist unsigniert, solange kein eigenes Code-Signing-Zertifikat verwendet wird; unsignierte Dateien lösen unter Windows eine SmartScreen-Warnung aus.

## Offene Fragen
- Ist das Repo öffentlich oder privat (entscheidet über Installation aus Git oder aus dem Release)?

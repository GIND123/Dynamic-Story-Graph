# DSG Labelling Application

A dedicated Flask-based web application to label the audit sheets for the Dynamic Story Graph project:
1. **Canon-Violation Detector Audit** (`audit/detector/`)
2. **Revision Calculus Audit** (`audit/calculus/`)

---

## 1. Quick Start

Ensure you have `flask` installed:
```bash
pip install flask
```

Run the application from the repository root:
```bash
python -m dsg.annotate --me annotator_01 --initials <your_name/initials>
```
*(Replace `annotator_01` with your assigned file, and `<your_name/initials>` with your initials).*

Then open your browser at:
👉 **`http://127.0.0.1:8502`**

---

## 2. Command Options

| Argument | Default | Description |
|---|---|---|
| `--me` | *(Required)* | Your assigned file name (`annotator_01` or `annotator_02`). |
| `--initials` | *(Required)* | Your initials, automatically saved to the `labeller` column. |
| `--port` | `8502` | Port to bind the local server. |
| `--no-sync` | `False` | Disables git auto-pull and auto-push (useful for offline work). |
| `--interval` | `300` | Seconds between background git pushes (default: 5 minutes). |
| `--branch` | `main` | Git branch to sync with. |
| `--root` | `.` | Repository root directory. |

### Running Offline (Without Git Auto-Push)
If you don't have GitHub write credentials configured in your shell or want to work offline:
```bash
python -m dsg.annotate --me annotator_01 --initials SK --no-sync
```

---

## 3. How Git Auto-Sync Works

Unless `--no-sync` is passed, the app:
1. **Pulls `main` on startup** to ensure you have the latest state.
2. **Auto-commits & pushes** only your own files (`audit/*/annotators/<me>.csv`) every 5 minutes while there are new saves.
3. **Pushes once more on shutdown** when you stop the app (`Ctrl-C`).
4. Commits carry `[skip ci]` so frequent pushes do not trigger GitHub Actions workflows.

---

## 4. Keyboard Shortcuts

### In Canon Detector Audit:
* <kbd>Y</kbd> — Select **Yes**
* <kbd>N</kbd> — Select **No**
* <kbd>U</kbd> — Select **Unsure** (automatically focuses the notes box)
* <kbd>Enter</kbd> — **Save and next row**

*(On `unflagged_mention` rows, which ask both violation and restatement, click a question or use Tab to switch between them).*

### In Revision Calculus Audit:
* Label choices:
  * <kbd>S</kbd> — `SUPERSEDE`
  * <kbd>R</kbd> — `REVISE`
  * <kbd>E</kbd> — `ELABORATE`
  * <kbd>X</kbd> — `EXTRACTION_ERROR`
  * <kbd>U</kbd> — `UNSURE`
* Which value is true now:
  * <kbd>1</kbd> — `new`
  * <kbd>2</kbd> — `old`
  * <kbd>3</kbd> — `both`
  * <kbd>4</kbd> — `neither`
  * <kbd>5</kbd> — `unsure`
* <kbd>Enter</kbd> — **Save and next row**

---

## 5. Verification Check

At any time, you can verify your labeled CSVs from the command line:
```bash
python -m dsg.eval.labels check --sheet audit/detector --partial
python -m dsg.eval.labels check --sheet audit/calculus --partial
```

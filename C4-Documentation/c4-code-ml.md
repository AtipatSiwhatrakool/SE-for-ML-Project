# C4 Code Diagram — ML Inference Pipeline

Two complementary views of the ML code at the lowest level:

| File | View | What it shows |
|---|---|---|
| [`c4-code-ml-inference.puml`](./c4-code-ml-inference.puml) | **Class / structure diagram** | Modules, functions, globals, and external-library boundaries |
| [`c4-code-ml-predict-sequence.puml`](./c4-code-ml-predict-sequence.puml) | **Sequence diagram** | Control flow for a single `POST /api/v1/predict` call |

Both written in standard PlantUML (compatible with the latest PlantUML releases — no external `!include` required for the sequence diagram).

## How to render

**Online (fastest):** paste either `.puml` file into <https://www.plantuml.com/plantuml/uml/> and export PNG.

**CLI:**
```bash
docker run --rm -v "$PWD/C4-Documentation:/work" plantuml/plantuml \
  -tpng /work/c4-code-ml-inference.puml /work/c4-code-ml-predict-sequence.puml
```

**VS Code:** install the "PlantUML" extension (jebbs.plantuml), open a `.puml` file, `Alt+D` to preview, right-click → Export.

## Scope

Both diagrams focus on the **inference pipeline** (the slice the rubric labels "ML component") — specifically `api/main.py`, `api/monitoring.py`, and `api/auth.py` plus their external dependencies.

## Structure diagram — what's on it

### Modules (in `api/` package)

- **`main.py`** — FastAPI app, endpoints, lifespan-scoped model loading, three-stage inference in `predict()`.
- **`monitoring.py`** — image-quality metrics, Postgres I/O for `prediction_logs`, review CRUD.
- **`auth.py`** — bcrypt verification, session helpers, `require_role()` FastAPI dependency.

### External libraries

- **torchvision** — `efficientnet_v2_s` pretrained, classifier swapped to `nn.Identity` → 1280-dim features.
- **sklearn** — `PCA` (1280 → 256) and `LogisticRegression`.
- **psycopg2** — Postgres driver.
- **bcrypt / pgcrypto** — password hashing (seeded at DB init).
- **itsdangerous** — signs the session cookie via Starlette's `SessionMiddleware`.

### Artifacts / data stores

- **`current_model.json`** — the model-promotion pointer the training DAG writes.
- **Pickled PCA and LogReg `.pkl`** — artifact files referenced by the pointer.
- **`prediction_logs` (Postgres)** — the 16-column contract table.
- **`users` (Postgres)** — seeded by `predictions_init.sql`.

## Sequence diagram — what it shows

The flow of a single `POST /api/v1/predict`:

1. `require_role("user")` runs auth, verifies the signed cookie, returns the user dict.
2. File extension and content are validated; bytes are decoded to a PIL RGB image.
3. `pil_to_rgb_array` + `compute_brightness` + `compute_blur_score` extract drift-monitoring features.
4. Image → transform → backbone → **1280-dim feature vector**.
5. PCA.transform → **256 dims**.
6. LogReg `predict` + `predict_proba` → class index + softmax probabilities.
7. `requires_manual_review = confidence < 0.60`.
8. `append_prediction_log` writes the row (including raw image bytes) to Postgres.
9. 200 OK JSON returned.

## Why these two diagrams together

- The **class diagram** answers "what is the code made of and what does it depend on?"
- The **sequence diagram** answers "what actually happens when a user calls `/predict`?"

For the slide deck, the sequence diagram usually reads faster to a mixed audience. The class diagram is better for the written record / appendix.

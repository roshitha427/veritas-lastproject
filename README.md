# Veritas — Flask and MySQL

The SR user and admin interfaces, analysis workflows, history, source management,
and downloadable reports are retained. The backend uses Flask with MySQL 8+
through SQLAlchemy/PyMySQL. Existing SR model artifacts are packaged separately
from source; training never runs during web startup or deployment.

## Run locally

Use Python 3.12, Node.js 22, and MySQL 8+. In PowerShell, from this folder:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-inference.txt
npm.cmd --prefix frontend ci
npm.cmd --prefix frontend run build
Copy-Item backend/.env.example backend/.env
```

Complete the frontend/model setup first. When ready to connect your database,
create a dedicated MySQL database and account in MySQL Workbench, then fill in
`MYSQL_HOST`, `MYSQL_PORT`, `MYSQL_DATABASE`, `MYSQL_USER`, and `MYSQL_PASSWORD`
in `backend/.env`. The account needs permissions to create and update tables in
that database. The application creates its tables automatically. Alternatively,
review and run [database/schema.sql](database/schema.sql) in Workbench. Your existing
SR accounts, reports, and credentials are not copied.

```powershell
.\.venv\Scripts\python.exe -m scripts.check_setup
.\.venv\Scripts\python.exe -m scripts.check_database
.\.venv\Scripts\python.exe -m scripts.check_system
.\.venv\Scripts\python.exe -m flask --app backend.app:create_app create-admin
.\.venv\Scripts\python.exe -m scripts.serve
```

Open `http://127.0.0.1:5000`. Administrator creation prompts for a private password;
there is no shared default admin account. For frontend development, run
`npm --prefix frontend run dev` alongside the Flask server.

Docker is an alternative local setup: `docker compose up --build` starts the app
and a separate MySQL 8.4 database. `docker compose up -d mysql` starts just MySQL
for native Python development; its development database/user are both `veritas`
and its default password is `local-development-password`. Override these defaults
using environment variables before using shared infrastructure. Create an admin
with `docker compose exec web python -m flask --app backend.app:create_app create-admin`.

## Models and training

Keep trained weights, user files, and datasets outside Git. Export only trusted
inference artifacts and their model cards from SR:

```powershell
.\.venv\Scripts\python.exe -m scripts.export_models --source ../SR/models --output artifacts/sr-models.zip --profile full
.\.venv\Scripts\python.exe -m scripts.install_models --archive artifacts/sr-models.zip --sha256 DIGEST_PRINTED_BY_EXPORTER
```

The exporter maps SR's active continued text model, artwork-origin model, and
retrained temporal model to the application's canonical paths. `--text`,
`--origin`, and `--temporal` override source directory names. The `core` profile
includes text, image, AI-writing, image-origin, and temporal inference. `full`
also includes local evidence comparison and Whisper speech transcription.
Training checkpoints, optimizer state, cached features, private reports,
training manifests, and local database files are excluded. The installer checks
the archive digest, every file digest, and extraction paths before installation.
Stop the app before replacing installed models with `--replace`. The active SR
models are already installed in this checkout; exporting is needed for a new
release or a fresh clone. Runtime versions are pinned to the verified setup.

Install `requirements-training.txt` only in a separate training environment.
Supported training entry points include:

```text
python -m ml.train_text --train data/train.csv --validation data/validation.csv --test data/test.csv --output models/text-new
python -m ml.train_ai_text --data data/hc3 --output models/ai-text-new
python -m ml.finetune_image --archive PATH_TO_IMAGE_ARCHIVE --manifest data/image-manifest.json --output models/image-new
python -m ml.train_temporal --manifest data/deepfaketimit/manifest.json --backbone models/image-v2/backbone.pt --output models/temporal-new
```

For image-origin training, use `python -m ml.train_diverse_origin --manifest
PATH_TO_MANIFEST --backbone PATH_TO_BACKBONE --output models/origin-new`.
The manifest contains entries with `path` or `archive`/`member`, integer
`label` (0 real, 1 AI), `split`, `sha256`, `dataset`, and `source_group`.
Personal adaptation accepts explicit `--real-image` and `--ai-image` arguments
through `ml.train_personal_origin`; it never scans private folders.

Use new output directories, licensed data, disjoint training/validation/test
splits, and reviewed model cards. Evaluate a trained model before configuring
its path in `backend/.env` and exporting a release. The retained source models
have experimental limits: model confidence and historical accuracy do not
establish that a current claim is true. Missing optional models are reported as
unavailable rather than receiving fabricated scores.

`EVIDENCE_PROVIDER=public` enables public evidence search without an API key.
Optional Gemini media reading/report summaries and Tavily search require your
own backend keys. SMTP settings enable email password recovery. Leave their
settings empty when those integrations are not configured.

## GitHub and Render

Push this source to your own GitHub repository, then create a Render Blueprint
from `render.yaml`. The multi-stage Docker image builds React and installs CPU
inference dependencies. Render starts one Waitress process and checks
`/api/health`. GitHub Actions runs backend checks against a disposable MySQL
service and builds the frontend.
It also builds the complete CPU Docker image and smoke tests production
startup against MySQL with a fresh mounted data directory. This image check
does not install your separately hosted model bundle.

Provide the actual `MYSQL_*` connection values and the final HTTPS
`PUBLIC_BASE_URL` in Render. A `DATABASE_URL` using the `mysql+pymysql` driver
can replace the individual connection fields; URL-encode credentials in that
form. For a remote TLS database, upload its provider's CA certificate as a
Render secret file and set `MYSQL_SSL_CA` to that file path. Hostname and
certificate verification remain enabled. Keep credentials in runtime settings,
never in source or Docker build arguments.

The Blueprint asks for `TAVILY_API_KEY` and `GEMINI_API_KEY` to preserve the
configured search and explanation features. Enter keys in Render, not in
GitHub files. For operation without these integrations, change
`EVIDENCE_PROVIDER` to `public` and leave the keys empty. SMTP remains optional.

Render does not provide managed MySQL. Connect an external MySQL service, or
run a private MySQL container with its own persistent disk mounted at
`/var/lib/mysql`. The supplied Blueprint provisions only the application. See
[Render's MySQL guide](https://render.com/docs/deploy-mysql).

For model deployment, host a trusted exported ZIP at a public HTTPS artifact
URL and set `MODEL_BUNDLE_URL` plus its exact `MODEL_BUNDLE_SHA256` in Render,
then rebuild. For the bundle already in `artifacts/sr-active-models.zip`,
PowerShell can print its checksum with `Get-FileHash
artifacts/sr-active-models.zip -Algorithm SHA256`. Render exposes these values as Docker build arguments; do not
use credential-bearing or signed URLs. The Docker build verifies and installs
the bundle. Without a bundle, the UI and evidence workflows run while absent
local model modules report unavailable. `INSTALL_INFERENCE=false` builds a
smaller image for operation without neural models.
The SR-compatible PyTorch and Transformers pins retain upstream advisories,
including vulnerabilities in model-loading and training APIs. Install only
your trusted exported model bundle; a matching checksum proves integrity,
not that third-party weights are safe. Revalidate the trained models when
upgrading these runtimes. Production loads local artifacts with network model
downloads disabled and does not accept model uploads.

The Blueprint requests paid compute with 2 GB RAM and a persistent `/var/data`
disk for private report videos and application runtime files. Larger model
combinations may need more memory. Keep one service instance and one process:
the analysis queue and request limits are process-local. Back up MySQL with
database-aware tools and back up retained media separately. Ordinary Render
filesystems are ephemeral; see [persistent disk behavior](https://render.com/docs/disks).
The Docker entrypoint initializes the mounted data directory, then drops to
the dedicated application user before starting Python. Docker `DATA_DIR`
must remain at `/var/data` or a directory inside that mount.

Render's Blueprint explicitly trusts its ingress forwarding headers so login
limits use the client's IP instead of grouping everyone under a load balancer.
Wildcard trust is accepted only when Render sets `RENDER=true`. Keep the web
port behind Render ingress; do not expose it directly or send untrusted
private-network traffic to it. `WAITRESS_TRUSTED_PROXY_COUNT=1` trusts the
rightmost forwarded address and ignores client-supplied prefixes. Verify the
forwarded chain during deployment if adding another proxy, and adjust this
bounded count only for known trusted hops. Outside Render, leave proxy trust
unset or set `WAITRESS_TRUSTED_PROXY` to the single actual proxy IP.

GitHub holds source and runs CI; GitHub Pages cannot execute the Flask/MySQL
backend. Render can redeploy from pushes to the linked GitHub branch. See
[Render's Git integration](https://render.com/docs/git-provider) and
[Docker deployment guide](https://render.com/docs/docker).

## Verify changes

```text
python -m pip install -r requirements-dev.txt
python -m pytest -q
npm --prefix frontend run build
```

Database tests require `TEST_DATABASE_URL` pointing to a disposable MySQL
database whose name ends in `_test`; those tests clear its application tables.
Without that variable, database-dependent tests are skipped. Model integration
checks require installed artifacts and inference dependencies. Browser tests
under `frontend/e2e` use the frontend's Playwright configuration. For backend
browser checks, set `TEST_DATABASE_URL` to a disposable database ending in
`_test`, run `python -m scripts.qa_server`, then run the following from another
terminal with the Python environment active:

```powershell
$env:BASE_URL="http://127.0.0.1:5001"
npm.cmd --prefix frontend run test:e2e -- --grep-invert '@live'
```

The QA helper seeds an administrator only in that test database. Install
Playwright's Chromium first, or set `PLAYWRIGHT_EXECUTABLE_PATH` to an installed
Chromium-based browser. Live provider checks are opt-in. Docker and Render
configuration are prepared; verify the Docker image on a Docker-capable host
before production deployment.

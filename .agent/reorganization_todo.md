# Project Reorganization Plan

This document outlines the steps to declutter the project root and organize assets, infrastructure, and source code.

## 1. Consolidate Assets and Documentation

- [ ] Move `images/` to `static/images/` (for the UI) or `docs/images/`.
- [ ] Move `presentation/` to `docs/presentation/`.
- [ ] Remove `README.md.bak`.

## 2. Organize Infrastructure (Deployment)

- [ ] Create `deployment/` directory.
- [ ] Move `Dockerfile` to `deployment/api.Dockerfile`.
- [ ] Move `Dockerfile.streamlit` to `deployment/ui.Dockerfile`.
- [ ] Move `api/Dockerfile` (if redundant) or consolidate it into `deployment/`.
- [ ] Update `docker-compose.yml` to point to new Dockerfile paths.

## 3. Streamline Python Source (`src/`)

- [ ] Move `main.py` to `src/atrium_csd_ner/api.py`.
- [ ] Move `ui.py` to `src/atrium_csd_ner/ui.py`.
- [ ] Ensure all internal imports are updated to use the package structure.

## 4. Clean up Legacy/Demo Scripts

- [ ] Remove `demo.py` (legacy).
- [ ] Finalize production entry points in `pyproject.toml` or thin root wrappers.

## Target Structure Goal

```text
.
├── .agent/              # AI instructions, logs, & tasks
├── src/                 # Core logic, API, & UI
├── scripts/             # Training & Data pipelines
├── docs/                # Manuals, images, & presentations
├── data/                # Local cache & trained models
├── deployment/          # All Docker & orchestration files
├── static/              # Static assets for the Web UI
├── tests/               # Unit and regression tests
├── pyproject.toml       # Central PEP-621 config
└── README.md            # Project entry point
```

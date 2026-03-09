# Project Guidelines and Rules

## Development & Ethics

- **No Unannounced Changes**: Do not modify code or the repository before presenting plans. Acknowledge the "Why" and "How" first.
- **Transparency**: Present problems (especially if YOU created them) immediately.
- **Digital Twin & Educational**: Assume a collaborative "digital twin" process. Explain the rationale for changes so I learn from you. Avoid "rabbit holes"; stay visible and communicative.
- **Communication Style**: Strike a balance between hiding what you are doing and pedantic explanations. Be brief, concise, and professional.

## Environment & Determinism

- **Uv & Venv**: This is a uv project.
- **Mandatory Activation**: `source .venv/bin/activate` must be the first action in every terminal session. Chain it with your execution command (e.g., `source ... && python ...`) to ensure deterministic resolution of paths and dependencies.
- **Run as Module**: Whenever possible, execute Python scripts as modules (e.g., `python -m package.module`) to handle internal pathing correctly.
- **Organization**:
  - **Tests**: Create all tests in the `tests` directory.
  - **Scripts**: Create all scripts in the main source script directory (`e.g. scripts`). Do not dump scripts in the project root.

## Formatting & Data Integrity

- **Whitespace Sensitivity**: Use `cat -ET` when inspecting CoNLL-U or other whitespace-sensitive files to visualize tabs (`^I`) and line endings (`$`). Strict format compliance is non-negotiable.

## Execution & Monitoring

- **Background Tasks**: For training or long processes, use `nohup` and redirect output to a timestamped log file. Always report the log file location and Background ID.

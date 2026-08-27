# Working agreement for this project

Personal project, being revived to become an agentic pipeline for semantic
search over scientific papers (fetch → OCR → embed → rerank), backed by
Postgres + Pinecone. See `docs/mvp-plan.md` for the current plan.

## Workflow rules

- Commit incrementally. Messages are short, plain English, present tense
  (e.g. `worker: add retry on OCR failure`).
- Every subtask gets a GitHub issue first: short description + implementation
  plan, before writing code. Don't pre-create issues for future subtasks —
  open one only when starting work on it.
- Issues stay snappy — no essays. Body has just enough to start; keep it short.
- Each issue has at most one comment. When work is done, update that single
  comment (create it if missing) with what was actually done. Don't leave a
  trail of comments — edit/replace the existing one.
- When a subtask is done: merge the commits and close the issue.
- When blocked (needs a decision, credentials, external account, etc.): stop,
  tell the user what's needed, and label the issue `blocked`.

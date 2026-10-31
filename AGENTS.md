
# Project: Tomato CNN Deployment Engineering

## Purpose
This repository contains an existing TensorFlow/Keras tomato leaf
classifier. The current goal is to learn software deployment,
CI/CD, infrastructure, reliability, and microservices.

The trained model is treated as an existing artifact.
Do not retrain, redesign, or improve the CNN unless explicitly asked.

## Existing project
- Preserve existing training, evaluation, and data preparation code.
- Inspect the actual repository before proposing architecture changes.
- Reuse existing code and conventions where practical.
- Do not assume that files described in the README actually exist.
- Keep training dependencies separate from inference dependencies
  when doing so is useful and justified.

## Engineering principles
- Prefer the simplest maintainable solution.
- Use clear names, small focused functions, and explicit dependencies.
- Avoid unnecessary abstraction and premature generalization.
- Use type hints for new Python application code.
- Handle expected errors explicitly.
- Keep configuration separate from application logic.
- Avoid duplicated preprocessing, configuration, and business logic.
- Add comments to explain non-obvious decisions, not obvious syntax.
- Never silently swallow exceptions.
- Never log uploaded image contents, credentials, or secrets.

## Scope control
- Implement only the currently approved checkpoint.
- Before editing, inspect relevant existing files.
- Propose the exact files to create or modify.
- Wait for approval of the plan and file list.
- Never create additional files without explicit approval.
- Never rename, move, delete, or restructure existing files
  without explicit approval.
- Prefer modifying existing appropriate files over creating new ones.
- Do not generate placeholder modules, boilerplate directories,
  unrelated utilities, sample apps, or speculative infrastructure.
- Do not introduce a new library unless necessary.
- Do not add Kubernetes, Redis, Celery, Kafka, or other infrastructure
  unless the active checkpoint explicitly requires it.
- Treat uncommitted user changes as protected.

## Testing
- Add focused tests for new behavior.
- Unit tests should not require retraining or dataset downloads.
- Use mocks where appropriate, but test real-model integration
  separately when the model is available.
- Run relevant tests and checks after implementing a checkpoint.
- Report commands actually executed and their results.
- Never claim that unexecuted tests passed.
- Do not silently modify tests just to make failures disappear.

## Security and operations
- Never commit credentials, tokens, or real .env files.
- Keep trained model artifacts versioned and reproducible.
- Use bounded image uploads and safe input validation.
- Keep deployment and release actions behind explicit approval.
- Do not access external services or publish artifacts without approval.

## Checkpoint completion report
At the end of every implementation, report:
1. What was implemented.
2. Files created, modified, or deleted.
3. Why each file was necessary.
4. Tests and validation commands with actual results.
5. How to run and manually verify the deliverable.
6. Known limitations or unresolved issues.
7. Any changes that would require approval.

Stop after the current checkpoint.
Do not automatically begin the next checkpoint.
Do not commit, push, or deploy without explicit permission.

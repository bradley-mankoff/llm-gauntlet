---
name: grill-me
description: User-invoked wrapper for a one-question-at-a-time grilling session before planning or implementation. Use when the user explicitly asks for grill-me or wants a stronger model to interrogate an idea before `to-prd`, `to-issues`, or `precise-plan`.
disable-model-invocation: true
---

# Grill Me

Run a `grilling` session.

For this stack, the likely next step is `orchestrate-change`: the orchestration
loop can begin with this grilling intake, then write the Brief/PRD, slice
issues, plan, execute, and review. Ask only questions that materially affect
scope, invariants, validation, or issue boundaries. If a question can be
answered by repo inspection, the orchestrator should dispatch `scout` rather
than ask the user.

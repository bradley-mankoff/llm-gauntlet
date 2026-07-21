---
name: grilling
description: Interview the user one question at a time to sharpen a plan, refactor, tweak, design, or implementation idea before it becomes a brief, issues, or a precise plan. Use when the user asks to be grilled, stress-tested, interrogated, aligned before planning, or uses any "grill" trigger phrase.
---

# Grilling

Interview the user until the change is sharp enough to route into the next workflow. Ask one question at a time and wait for the answer before continuing.

## Default Posture

Most of this stack is for tweaks, refactors, and codebase improvements, not greenfield product work. Bias questions toward decisions that change implementation scope:

- intended behavior and observable outcome
- non-goals and things to preserve
- domain terms, ownership, and invariants
- risk areas and rollback/deferral boundaries
- validation expectations
- likely issue boundaries for small executor agents

If a fact can be found by inspecting the repo, inspect the repo instead of asking; in OMP-native orchestration mode, prefer parent-dispatched `scout` evidence for repo inspection. The decisions, though, belong to the user: put each decision to them and wait for their answer. If the repo has `CONTEXT.md`, `CONTEXT-MAP.md`, or `docs/adr/`, use that vocabulary and respect those decisions.

## Question Discipline

Ask the single most useful unresolved question. Include your recommended answer so the user can accept, correct, or refine it quickly.

Good question shape:

```text
Question: [one decision]
My recommended answer: [specific default]
Why it matters: [what this changes in scope, risk, or validation]
```

Stop grilling when the remaining unknowns would not change the brief, issue boundaries, or validation strategy. Do not keep asking interesting-but-nonblocking questions.
Do not enact the plan or continue into planning until the user confirms the shared understanding is sufficient.

## Optional Domain Pass

If the uncertainty is mostly vocabulary, ownership, or invariants, use `domain-modeling` before producing a brief. Examples:

- two names may refer to the same concept
- a refactor might move behavior across a domain boundary
- an invariant needs a stable name before code changes
- an ADR-worthy decision is being made

Keep the result brief: resolved terms, ownership, invariants, and any ADR/context update needed.

## Handoff

When the answers are enough, summarize:

- desired change
- non-goals
- preserved behavior/invariants
- risk areas
- validation expectations
- whether to go next to `to-prd`, `to-issues`, or directly to `precise-plan`

Also write the summary as `plans/<feature-slug>/GRILL_HANDOFF.md` when the user wants to continue in a fresh orchestration session. Keep it compact enough to seed `to-prd`/Change Brief writing without replaying the full exploratory conversation.

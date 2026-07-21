---
name: orchestrate-change
description: Orchestrate a code change with a lean main-agent and sidekick loop. The main agent resolves ambiguity and reviews results; scout gathers code context; plan creates one proportionate plan when useful; worker writes code and tests, fixes lint and bugs, and validates. Generic and ponytail review skills are never invoked automatically.
---

# Orchestrate Change

Use a lean delegation loop:

```text
user → main agent
         ├─ scout: code exploration
         ├─ plan: one proportionate plan, when useful
         └─ worker: code + tests + lint + bug fixes
main agent → review result → worker fixes → final code
```

This follows the sidekick principle: the main agent takes minimal actions, delegates routine exploration and implementation, and keeps the significant decisions—interpretation, planning judgment, integration, and final review.

## Roles

- **Main agent:** owns the user conversation, scope, ambiguity, delegation, integration, ordinary code review, and completion judgment.
- **Scout:** read-only code exploration. Returns relevant files, symbols, conventions, relationships, and validation seams. Use it frequently inside and outside this skill.
- **Plan:** one planning agent and skill. Turns the request and scout evidence into one proportionate plan. It may write `plans/<change-slug>/PLAN.md`, but does not generate a mandatory PRD → issues → precise-plan chain.
- **Worker:** normal implementation sidekick. Accepts a direct request, bug report, plan, or coherent work slice; writes code and tests; fixes lint and bugs; runs focused validation; adapts to small repository mismatches.
- **Researcher:** optional external documentation and web evidence.

`code-review` and `ponytail-review` remain explicit user-invoked skills. This orchestration skill MUST NOT invoke them, their reviewer agents, or a review responder. The main agent reviews worker output itself and sends concrete defects back to `worker`.

## When to invoke

Use this skill when the user explicitly asks to orchestrate a change or wants an end-to-end delegated implementation loop. Do not require it for normal delegation: the main agent should use `scout` and `worker` whenever they improve context efficiency or isolate coherent work.

Do not create planning artifacts merely because orchestration is active. Small changes should go directly from scoped request to worker.

## Loop

### 1. Frame the request

Extract the observable goal, constraints, non-goals, and success check from the conversation. Inspect tools and repository context before asking questions. Ask only when the answer materially changes behavior, architecture, destructive operations, or validation. Use `grilling` only when the request is genuinely unclear or the user asked for it.

### 2. Gather code context

Dispatch `scout` when file locations, conventions, call paths, or validation seams are not already clear. Run independent scouts in parallel when that reduces latency. Ask for bounded findings, not pasted files.

The main agent should read only the exact snippets needed to make decisions or review edits.

### 3. Decide whether planning helps

Skip a separate planner when the change is small and the worker assignment can state the outcome and success check directly.

Dispatch `plan` once when the change is multi-file, risky, architecture-sensitive, resumable, or benefits from ordered work slices. Give it the request, constraints, supplied artifacts, and scout evidence. Accept one plan; do not follow it with issue slicing, a plan gate, or per-issue precise plans.

Legacy PRDs, issue files, ledgers, and precise plans may be read as input. They do not force the old workflow.

### 4. Delegate implementation

Dispatch `worker` with either:

- the direct outcome, constraints, likely area, and success check; or
- the plan path plus the coherent slice to implement.

Prefer one worker for a coherent code/test surface. Parallelize only genuinely independent work with disjoint ownership. Tell every worker it is not alone in the repository and must preserve concurrent edits.

The worker owns local exploration, source edits, test edits, lint fixes, ordinary debugging, and focused validation. Do not dispatch separate implementer, unblocker, or test-writing stages for the same coherent change.

### 5. Main-agent review

When a worker returns:

1. Inspect the reported files and relevant diff from the user's perspective.
2. Check the requested behavior, invariants, scope, tests, error paths, and integration points.
3. Run or require the tightest independent verification that covers the change.
4. Send concrete defects or failed checks back to `worker` as a fix assignment.
5. Repeat until the behavior works and verification passes.

This is ordinary engineering review by the main agent, not invocation of a review skill.

### 6. Finish

Update a saved `PLAN.md` only when it is serving as a useful resume artifact; a short status and key decisions are enough. Do not create `PROGRESS.md`, issue capsules, review logs, completed-plan logs, state-prefixed filenames, or other ledgers unless the repository or user explicitly requires them.

Run cleanup only after the behavior works: remove debug scaffolding, keep tests/docs aligned where affected, and perform the required daily response log step for file-changing turns.

## Blockers

The worker should handle stale paths, renamed symbols, small adjacent edits, test failures, and lint failures. The main agent decides larger deviations.

Escalate to the user only for a missing product decision, destructive action, unavailable credential or service, conflicting ownership, or evidence that the requested scope must change. State the exact evidence and prerequisite.

## Final response

Report:

- behavior delivered;
- important files changed;
- verification run and results;
- any unresolved risk or blocker.

Do not report administrative stage counts, agent contract markers, gate passes, issue waves, or review-log statistics.

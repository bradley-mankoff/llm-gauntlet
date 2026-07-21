---
name: code-review
description: >
  Review a branch, PR, work-in-progress diff, or completed implementation against
  a fixed point. Use for generic correctness/quality review, requests like
  "review this", "review since main", "request code review", "PR review",
  "before merge", or "is this to spec?". Runs Standards and Spec review axes in
  parallel subagents, then aggregates Critical/Important/Minor findings and a
  merge-readiness verdict. Do not use for over-engineering-only review; use
  ponytail-review for that.
---

# Code Review

Review the diff between `HEAD` and a fixed point along two isolated axes:

- **Standards** — does the changed code follow documented repo standards and avoid obvious code smells?
- **Spec** — does the changed code faithfully implement the originating issue / PRD / plan / request?

The axes run as **parallel read-only subagents** so standards judgment does not contaminate spec judgment, and spec judgment does not excuse poor code. The parent agent aggregates their reports, triages severity, and owns any follow-up fixes.

This skill is opt-in only. Invoke it when the user explicitly asks for review; `orchestrate-change` and routine worker delegation MUST NOT invoke it automatically. Keep `ponytail-review` separate: ponytail is deletion/over-engineering-only.

## Process

### 1. Pin the fixed point

Use the fixed point the user supplied: commit SHA, branch name, tag, `main`, `origin/main`, `HEAD~5`, or equivalent. Do not reinterpret it.

If the user did not specify a fixed point, ask exactly one question before reviewing:

> Review against what fixed point — a branch, a commit, or `main`?

Once known, use three-dot diff semantics so review compares against the merge base:

```bash
git diff <fixed-point>...HEAD
git log <fixed-point>..HEAD --oneline
```

If a caller already provides a bounded diff command, base/head SHA, or cumulative diff scope, use that exact scope and record it in the review packet.

### 2. Identify the spec source

Look for the originating spec in this order:

1. Explicit path/content supplied by the user or parent agent.
2. Feature request, PRD, Change Brief, issue, or plan associated with the reviewed work.
3. Issue references in commit messages (`#123`, `Closes #45`, GitLab `!67`) and the configured issue-tracker workflow when available.
4. A PRD/spec/plan under `plans/`, `docs/`, `specs/`, or `.scratch/` matching the branch name or feature slug.
5. If none is found, ask where the spec is. If the user says there is no spec, skip the Spec axis and report `no spec available`.

The Spec axis may use a plan, task text, PRD, Change Brief, issue body, or concise user requirements. It must cite the specific requirement when reporting a finding.

### 3. Identify the standards source

Look for documented standards in this order:

1. `docs/standards/`, `standards/`, `CODE_STANDARDS.md`, `.github/CODE_STANDARDS.md`, `CONTRIBUTING.md`.
2. Repo-local agent guidance already loaded for the current repo.
3. A standards path supplied by the user or parent agent.
4. If none exists, use the built-in baseline in `references/standards-reviewer.md` and say `no documented standards found; used baseline code-smell review`.

Documented project standards override the baseline. Skip style rules enforced by formatter/linter unless the diff makes the tool fail.

### 4. Dispatch parallel read-only review axes

Dispatch both axes in one parallel batch when both are available. Do not serialize them unless the harness requires it.

Use the packets in:

- `references/standards-reviewer.md`
- `references/spec-reviewer.md`

Each subagent must receive:

- fixed point / diff command / commit list
- relevant changed-file list when available
- relevant standards or spec source
- read-only constraints
- output requirements

If an axis lacks required source material and the user confirmed it is unavailable, skip only that axis and record why.

### 5. Aggregate findings

Aggregate with `references/aggregate-review.md` or dispatch/read `code-reviewer` as the read-only aggregator in OMP.

The aggregate report must preserve axis separation and then assign overall severity:

- **Critical** — data loss, security vulnerability, broken core behavior, or spec-mandated behavior absent/incorrect.
- **Important** — missing non-core requirement, significant edge-case bug, integration issue, weak error handling, meaningful test gap, or maintainability issue likely to cause defects.
- **Minor** — localized polish, naming, documentation, small clarity/perf improvements that do not block merge.

A finding is actionable only if it includes file:line when available, the violated requirement/rule/smell, why it matters, and a concrete fix direction.

### 6. Act on feedback

Reviewers are read-only. The parent agent decides what to do with findings:

- Fix Critical immediately.
- Fix Important before declaring completion.
- Note Minor unless the fix is tiny and clearly in scope.
- Push back on wrong findings with technical evidence.
- For non-tiny accepted findings, dispatch a concrete fix assignment to `worker`; reviewers do not edit.

## Output Format

```markdown
## Review Verdict

Ready to merge: Yes | With fixes | No
Reasoning: [1-2 sentence technical assessment]

## Critical

[Findings or `None found.`]

## Important

[Findings or `None found.`]

## Minor

[Findings or `None found.`]

## Standards Axis

[Summary of Standards subagent result, including skip reason if skipped.]

## Spec Axis

[Summary of Spec subagent result, including skip reason if skipped.]

## Cross-Axis Notes

[Duplicate findings merged, disagreements preserved, false-positive risks.]

## Evidence Reviewed

- Fixed point / diff command: [...]
- Commits: [...]
- Spec source: [...]
- Standards source: [...]
```

## Critical Rules

- Never edit files during review.
- Never move HEAD, reset, checkout, rebase, commit, or mutate git state.
- Never say `looks good` without inspected evidence.
- Never hide that an axis skipped.
- Never let clean code quality excuse missing requirements.
- Never let spec compliance excuse fragile, insecure, or unmaintainable code.
- Keep ponytail-only deletion advice out of this review unless the complexity creates a correctness or maintainability risk.

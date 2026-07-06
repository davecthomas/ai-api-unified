# 2026-07-06 summary

## Snapshot

- Captured 2 memory events.
- Main work: The policy formalizes conventions already visible in repo history (version-bump-in-shipping-PR per PRs #17 and #19) and anchors them to existing enforcement artifacts rather than introducing new mechanisms.
- Top decision: Release versioning in this library had been an unenforced convention, and it kept failing in the same way: bumps updated the package metadata but missed the human-facing README title, so published releases advertised a stale version. As agent-driven PRs accelerated the release cadence, relying on contributors to remember every location stopped scaling. This checkpoint marks the shift from tribal knowledge to codified, test-enforced policy: the version lives in exactly three sanctioned places, they are always bumped together in the PR that ships the change, and the test suite makes drift impossible to merge silently. Future agents should treat the policy document and the sync test as the source of truth for release mechanics. ([2026-07-06 15:03:54 UTC by 2355287-davecthomas](events/2026-07-06T15-05-35Z--2355287-davecthomas--thread_9ca2f51a-9a16-4dde-bc71-13294144cd1c--turn_e98e51e2f0.md))
- Blockers: None.

| Metric | Value |
|---|---|
| Memory events captured | 2 |
| Repo files changed | 2 |
| Decision candidates | 1 |
| Active blockers | 0 |

## Major work completed

- The policy formalizes conventions already visible in repo history (version-bump-in-shipping-PR per PRs #17 and #19) and anchors them to existing enforcement artifacts rather than introducing new mechanisms.
- The version was synchronized to 2.7.0 across the package metadata, the runtime version attribute, and the README title, covering the minor bump for capability-gated streaming completions. A new sync test fails the mocked suite whenever the three locations disagree, and a new agent-instructions document codifies the semver bump rules, the three-location invariant, and the release flow: bump in the shipping PR, tag on main after merge, explicit publish as a separate step. An immediately preceding turn repaired the lagging README title as the first step of the same effort.

## Why this mattered

- Agents working in ai_api_unified must repeatedly make version-bump decisions, and the version literal lives in exactly two files that drift if bumped independently. Codifying the semver rules (patch/minor/major criteria, docs/test/CI exemptions), the release flow (bump lands in the shipping PR, tags cut on main, PyPI publish is a separate explicit step), and working conventions (no direct commits to main, mocked-suite pytest before commit, ruff/black) turns previously implicit repo practice into durable, discoverable instructions.
- Release versioning in this library had been an unenforced convention, and it kept failing in the same way: bumps updated the package metadata but missed the human-facing README title, so published releases advertised a stale version. As agent-driven PRs accelerated the release cadence, relying on contributors to remember every location stopped scaling. This checkpoint marks the shift from tribal knowledge to codified, test-enforced policy: the version lives in exactly three sanctioned places, they are always bumped together in the PR that ships the change, and the test suite makes drift impossible to merge silently. Future agents should treat the policy document and the sync test as the source of truth for release mechanics.

## Active blockers

- None

## Decision candidates

- Release versioning in this library had been an unenforced convention, and it kept failing in the same way: bumps updated the package metadata but missed the human-facing README title, so published releases advertised a stale version. As agent-driven PRs accelerated the release cadence, relying on contributors to remember every location stopped scaling. This checkpoint marks the shift from tribal knowledge to codified, test-enforced policy: the version lives in exactly three sanctioned places, they are always bumped together in the PR that ships the change, and the test suite makes drift impossible to merge silently. Future agents should treat the policy document and the sync test as the source of truth for release mechanics. ([2026-07-06 15:03:54 UTC by 2355287-davecthomas](events/2026-07-06T15-05-35Z--2355287-davecthomas--thread_9ca2f51a-9a16-4dde-bc71-13294144cd1c--turn_e98e51e2f0.md))

## Next likely steps

- Land CLAUDE.md via a branch and PR rather than a direct commit to main, per the repo convention the file itself establishes.
- Commit the sync work, then cut the v2.7.0 tag on main; watch that publish.sh keeps reading the version from pyproject.toml and that no new version literals appear outside the three sanctioned locations.

## Relevant event shards

- [2026-07-06 15:02:16 UTC by 2355287-davecthomas](events/2026-07-06T15-02-16Z--2355287-davecthomas--thread_451c3cb5-1bf1-4f06-911e-bb274923e3b1--turn_7ccc8998f5.md)
- [2026-07-06 15:03:54 UTC by 2355287-davecthomas](events/2026-07-06T15-05-35Z--2355287-davecthomas--thread_9ca2f51a-9a16-4dde-bc71-13294144cd1c--turn_e98e51e2f0.md)

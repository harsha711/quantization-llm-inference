# E052 working record (draft; final result goes to docs/EXPERIMENTS.md)

- 2026-11-18: predeclared, commit a969f04 (EXPERIMENTS E052 entry, AGENTS section, scripts/e052/).
- Prerequisite job 343334 submitted pinned to c10 (c7 held by another user; c8 running E051 343333). Pending: nodes reserved for a turin-partition 16-node job (343247/343248, 15:14-18:15); expected start 18:15.
- Prerequisite 343334 ran on c10 (UUID GPU-38a74fc5-9f81-89d1-4ad5-3f5796d39bf2, driver 610.57.04) 13:51-14:11 EST, exit 0, inspected locally:
  E017 runtime, vLLM and ORT venvs present on c10 (INSTALLED markers from 2026-11-10); llama.cpp venv ABSENT on c10.
  Library hash a24a78fa... OK, statically linked cudart, exports e050_tc/e050_tc_ladder.
  opcheck: 65 cases (5 shapes incl. head 1536x151936; M 1,2,16,63,64,65,128,2048,16384; head M 1,64), 0 bad elements, rel L2 <= 2.13e-4, repeat and graph replay bitwise equal, misaligned A refused.
  memcheck (all kernels, quick subset, 22 cases) 0 errors; racecheck (tc kernels) 0 hazards.
  Gate at 128: eager E021 form ACCEPTED both passages (rel L2 0.00208/0.00209; held-out first token difference at step 47, control margin 0.0156, near-tie rule; compared 48 steps);
  graph E026 form ACCEPTED (0.00210/0.00195, tokens equal F64). Marlin control ACCEPTED (0.00210/0.00200).
  Smoke cells (not results): heldout 128 s1 GTA 315.0 ms, GQA 326.4 ms.
- Submitted on c10: core 343363 (afterok prereq), long 343364 (afterok core), head 343365 (afterok core, afterany long), ext 343366 (afterok core, afterany head).

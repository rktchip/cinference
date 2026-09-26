# S1 frozen prompts (2026-09-25)

Each name maps to exactly one text. `s1_client.py` asserts these SHA-1
hashes at startup and stamps them into every S-line. To change a prompt,
add a NEW name — never edit text under an existing name.

- `fox`: high-accept prose regime.
  `Write a short passage about a fox crossing a river at dawn. Describe what it sees and hears in three sentences.`
- `paris`: LOW-ACCEPT regime (original PARIS from all oracle gates and
  the wide sweep; basis of the ship bar, the adaptive-gate requirement,
  and PARIS-500 drift).
  `Paris is the capital of France. Explain why the Eiffel Tower was built, who designed it, and when the construction finished.`
- `paris-city`: medium-accept prose regime (added for S1 arm A; kept for
  continuity).
  `Describe the city of Paris in three sentences: one about its history, one about its architecture, and one about its food.`

Screen rule: the winner of a ~10-step screen is NOT frozen until a
rerun confirms it (minimum-of-noisy picks for luck). Screen runs
thinking-OFF (preambles flatten); the frozen lowaccept regime is the
confirmed minimum-mean-accept candidate.

LOWACCEPT (frozen 2026-09-25): `fox`. Thinking-off mean-a 0.95, most
partial accepts in the set, already hashed, runs in every arm. Stands
unless rand-nums reads clearly lower at 100 steps (screen4). Ship bar
reads "lowaccept no worse than off"; drift check runs on FOX.

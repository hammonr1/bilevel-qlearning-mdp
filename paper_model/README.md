# paper_model — rebuild of the 30-asset / 8-salvo RESS model

Independent of the 2-asset prototype in the repository root. Build stages:

- (a) `instance.py`, `check_instance.py` — Sec. 5.1.1 instance; gate on Table 1 coverage. **Done.**
- (b) environment + learner, 50-episode smoke run with per-salvo trace.
- (c) Tables 1–2.
- (d) refinement trace, Tables 3–7, Figures 2–4.

Run `python paper_model/check_instance.py` first; it exits non-zero if the instance does not
reproduce Table 1. See `DECISIONS.md` for every modelling choice.

"""Stage (a) gate: regenerate the instance and verify Table 1 coverage and the Section 5.4 static rules.
Exit status is non-zero on any mismatch, so nothing downstream can run on a wrong instance."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paper_model.instance import (CANDIDATES, REFINED_PLAN, TABLE1_COVERED, coverage_count_rule,
                                  generate_instance, greedy_set_cover, value_weighted_rule,
                                  write_instance_csv)

inst = generate_instance(seed=7)
ok = True
print(f"Instance: n={inst.n}, |I_H|={len(inst.high)}, |I_L|={len(inst.low)}, R={inst.R}")
print(f"High-importance assets I_H = {list(inst.high)}")
print(f"Value range  H: {inst.v[list(inst.high)].min():.2f}-{inst.v[list(inst.high)].max():.2f}   "
      f"L: {inst.v[list(inst.low)].min():.2f}-{inst.v[list(inst.low)].max():.2f}   total value = {inst.v.sum():.2f}")
print(f"Damage prob  H: {inst.p[list(inst.high)].min():.3f}-{inst.p[list(inst.high)].max():.3f}   "
      f"L: {inst.p[list(inst.low)].min():.3f}-{inst.p[list(inst.low)].max():.3f}")

print("\nTable 1 'Initially covered assets' check:")
print(f"  {'Candidate':10s} {'SAM-allocation locations':28s} {'paper':>5s} {'ours':>5s}")
for key, sites in list(CANDIDATES.items()) + [("refined", REFINED_PLAN)]:
    got, want = inst.covered_count(sites), TABLE1_COVERED[key]
    ok &= got == want
    print(f"  {str(key):10s} {', '.join(map(str, sites)):28s} {want:5d} {got:5d}  {'OK' if got == want else 'MISMATCH'}")

print("\nSection 5.4 static allocation rules:")
for name, fn, want in [("greedy incremental set cover", greedy_set_cover, CANDIDATES[0]),
                       ("static value-weighted coverage", value_weighted_rule, CANDIDATES[1]),
                       ("static coverage-count", coverage_count_rule, CANDIDATES[2])]:
    got = fn(inst)
    ok &= got == want
    print(f"  {name:32s} -> {got}  paper: {want}  {'OK' if got == want else 'MISMATCH'}")

out = Path(__file__).resolve().parent / "results" / "instance.csv"
write_instance_csv(inst, out)
print(f"\nInstance written to {out} and {out.with_name('coverage_matrix.csv')}")
print("\nRESULT:", "ALL CHECKS PASSED" if ok else "CHECKS FAILED")
sys.exit(0 if ok else 1)

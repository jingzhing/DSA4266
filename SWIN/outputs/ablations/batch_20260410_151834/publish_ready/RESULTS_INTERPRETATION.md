# Results Interpretation (Tightened Claims)

## Claim Discipline
- Uses observed practical effects; does not claim formal statistical significance.
- Uses constrained model-selection rule, not best single metric.
- Selection rule (frozen): (1) fake_recall >= 0.95, (2) maximize real_recall, (3) minimize ECE, (4) maximize AUC, (5) minimize FP.

## Champion / Runner-up / Fallback (A6)
- Champion: `20260410_172149_swin_full_swin_gen_v2` (random, seed=23)
- Runner-up: `20260410_192140_swin_full_swin_gen_v2` (random, seed=42)
- Fallback: `20260410_152158_swin_full_swin_gen_v2` (random, seed=17)

## Practical Interpretation
- Chronological progression shows substantial movement from threshold-fragile behavior toward constrained-operationally useful points.
- AUC alone is insufficient: multiple runs with high AUC still fail operational constraints via low real recall or high ECE.
- Protocol comparison in this batch should be interpreted as observed batch effect, not universal superiority.
- Seed variance remains material and should be part of reporting and decision confidence.

## Table Outputs
- Table 1: `/home/ubuntu/work/DSA4266/outputs/ablations/batch_20260410_151834/publish_ready/table1_chronological_progression.csv`
- Table 2: `/home/ubuntu/work/DSA4266/outputs/ablations/batch_20260410_151834/publish_ready/table2_a6_grouped_by_protocol.csv`
- Table 3: `/home/ubuntu/work/DSA4266/outputs/ablations/batch_20260410_151834/publish_ready/table3_a1a2_policy_calibration.csv`

## Immediate Next Steps
- Run confirmatory repeats on champion setup with new seeds (>=3).
- Keep threshold+calibration as part of the deployable model definition.
- Report random vs source_aware as optimistic vs harder protocol estimate, respectively.

"""Calibration of the served AI validator against human majorities, from label_ai_assessment + label_validation.

Usage: python analyze_calib.py ai_assessments.csv > calib.txt   (input from pull_ai_assessments.sh; keep both outside
the repo). Any columns after h_unsure are treated as individual expert validators' votes (section 6).
"""
import sys, numpy as np, pandas as pd
df = pd.read_csv(sys.argv[1], low_memory=False)
df = df[df.schema != "sidewalk_validation"]  # internal test deployment
df["city"] = df.schema.str.replace("sidewalk_", "", regex=False)
df["h_n"] = df.h_agree + df.h_dis
df["h_maj"] = np.where(df.h_agree > df.h_dis, "Agree", np.where(df.h_dis > df.h_agree, "Disagree", None))
df["gate"] = df.ai_accuracy >= 0.92
df["right"] = df.ai_result == df.h_maj
TYPES = ["CurbRamp","NoCurbRamp","Obstacle","SurfaceProblem","Crosswalk"]
BASE = ["schema","label_id","label_type","deleted","severity","ai_result","ai_accuracy","ai_confidence","ai_date","model_id",
        "h_agree","h_dis","h_unsure","city","h_n","h_maj","gate","right"]
print(f"assessed labels: {len(df):,} in {df.city.nunique()} cities; assessment dates {df.ai_date.min()} .. {df.ai_date.max()}")
print(f"deleted labels among them: {df.deleted.eq(True).mean():.1%} (kept in analysis; votes still count)")

# 1. gate pass rate and verdict mix per type
print("\n== Gate (served accuracy >= 0.92): pass rate and verdict mix per type ==")
g = df.groupby("label_type").apply(lambda s: pd.Series({"n": len(s), "pass%": s.gate.mean(),
    "raw Agree%": (s.ai_result=="Agree").mean(), "Agree pass%": s[s.ai_result=="Agree"].gate.mean(), "Disagree pass%": s[s.ai_result=="Disagree"].gate.mean()}))
print(g.loc[TYPES].to_string(float_format=lambda x: f"{x:.3f}"))

# 2. served vs observed precision, by verdict and gate status, >=2 human votes
m = df[(df.h_n >= 2) & df.h_maj.notna()].copy()
print(f"\n== Served accuracy vs observed human-majority agreement (>=2 humans, n={len(m):,}) ==")
rows = []
for t in TYPES:
    for v in ["Agree","Disagree"]:
        for gate in [True, False]:
            s = m[(m.label_type==t)&(m.ai_result==v)&(m.gate==gate)]
            if len(s) < 30: continue
            rows.append((t, v, "voted" if gate else "abstained", len(s), s.ai_accuracy.mean(), s.right.mean()))
print(pd.DataFrame(rows, columns=["type","raw verdict","gate","n","served acc (mean)","observed"]).to_string(index=False, float_format=lambda x: f"{x:.3f}"))

# 3. reliability by served-accuracy bin, per verdict (all types pooled and per type)
print("\n== Reliability: served accuracy bin -> observed agreement (>=2 humans) ==")
m["acc_bin"] = pd.cut(m.ai_accuracy, [0,0.8,0.85,0.9,0.92,0.94,0.96,0.98,1.0001], right=False)
for v in ["Agree","Disagree"]:
    print(f"-- raw verdict {v} --")
    t = m[m.ai_result==v].groupby("acc_bin", observed=True).agg(n=("right","size"), served=("ai_accuracy","mean"), observed=("right","mean"))
    print(t.to_string(float_format=lambda x: f"{x:.3f}"))
# ECE over gated votes
gv = m[m.gate]
ece = (gv.groupby("acc_bin", observed=True).apply(lambda s: len(s)/len(gv)*abs(s.ai_accuracy.mean()-s.right.mean()))).sum()
print(f"\nECE of served accuracy on gated votes: {ece:.3f}; mean served {gv.ai_accuracy.mean():.3f} vs observed {gv.right.mean():.3f}")

# 4. raw confidence -> observed precision per type/verdict; what confidence gives observed >= 0.92 and at what coverage
print("\n== Empirical gate from production: min raw confidence where cumulative observed precision >= 0.92 (>=2 humans) ==")
rows = []
for t in TYPES:
    for v in ["Agree","Disagree"]:
        s = m[(m.label_type==t)&(m.ai_result==v)].sort_values("ai_confidence", ascending=False)
        if len(s) < 100: continue
        cum = s.right.expanding().mean().values; conf = s.ai_confidence.values
        ok = np.where(cum >= 0.92)[0]
        # last index where cumulative precision still >= 0.92 (largest coverage)
        idx = ok.max() if len(ok) else None
        allv = df[(df.label_type==t)&(df.ai_result==v)]
        rows.append((t, v, len(s), s.right.mean(), conf[idx] if idx is not None else np.nan, (idx+1)/len(s) if idx is not None else 0.0,
                     (allv.ai_confidence >= conf[idx]).mean() if idx is not None else 0.0, allv.gate.mean()))
print(pd.DataFrame(rows, columns=["type","verdict","n(>=2h)","precision@all","conf for 0.92","coverage of verdict","coverage all assessed","today's gate coverage"]).to_string(index=False, float_format=lambda x: f"{x:.3f}"))

# 5. per city: observed precision of gated votes, Agree and Disagree
print("\n== Per city (>=2 humans, gated votes): observed precision ==")
c = m[m.gate].groupby("city").apply(lambda s: pd.Series({"n": len(s), "Agree n": (s.ai_result=="Agree").sum(), "Agree obs": s[s.ai_result=="Agree"].right.mean(),
      "Disagree n": (s.ai_result=="Disagree").sum(), "Disagree obs": s[s.ai_result=="Disagree"].right.mean(), "served mean": s.ai_accuracy.mean()}))
print(c[c.n>=300].sort_values("Disagree obs").to_string(float_format=lambda x: f"{x:.3f}"))

# 6. experts vs AI by confidence
print("\n== Experts: where the AI contradicts an expert validator, what was its confidence? ==")
for k in [c for c in df.columns if c not in BASE]:
    s = df[df[k].isin(["Agree","Disagree"])]
    dis = s[s[k] != s.ai_result]
    print(f"  {k}: AI assessed {len(s):,} labels {k} voted on; contradicts {k} on {len(dis):,} ({len(dis)/len(s):.1%}); of those, gated (actually voted) {dis.gate.mean():.1%}; median conf {dis.ai_confidence.median():.3f} vs {s[s[k]==s.ai_result].ai_confidence.median():.3f} when agreeing")
    print(pd.crosstab(dis.label_type, [dis.gate, dis.ai_result]).to_string())

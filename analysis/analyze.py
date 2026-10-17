"""AI validator vs human majority and vs two expert validators, from rawLabels exports.

Per label we keep every vote from the `validations` JSON. The AI's own vote is identified by
validator_type == "AI"; the experts (env EXPERTS) by user_id. The "crowd" for any comparison is always the
set of human votes EXCLUDING the person being compared, so nobody is scored against themselves.
"""
import glob, json, os, sys, collections
import numpy as np, pandas as pd
from sklearn.metrics import cohen_kappa_score

CACHE = sys.argv[1]
# Expert validators to compare against, as EXPERTS="name1=<user_id>,name2=<user_id>" (kept out of the repo on purpose).
EXPERTS = dict(kv.split("=", 1) for kv in os.environ.get("EXPERTS", "").split(",") if "=" in kv)
TYPES = ["CurbRamp", "NoCurbRamp", "Obstacle", "SurfaceProblem", "Crosswalk"]

rows = []
for f in sorted(glob.glob(os.path.join(CACHE, "*.csv"))):
    city = os.path.basename(f)[:-4]
    df = pd.read_csv(f, low_memory=False)
    df["city"] = city
    rows.append(df)
df = pd.concat(rows, ignore_index=True)
df = df[df.label_type.isin(TYPES)].copy()
df["label_uid"] = df.city + ":" + df.label_id.astype(str)
assert df.label_uid.is_unique

def parse(v):
    if not isinstance(v, str) or not v: return []
    return json.loads(v)
df["votes"] = df.validations.apply(parse)

def vote_of(votes, uid):
    for v in votes:
        if v["user_id"] == uid: return v["validation"]
    return None
def ai_vote(votes):
    for v in votes:
        if v.get("validator_type") == "AI": return v["validation"]
    return None
def human_counts(votes, exclude=()):
    a = d = u = 0
    for v in votes:
        if v.get("validator_type") == "AI" or v["user_id"] in exclude: continue
        if v["validation"] == "Agree": a += 1
        elif v["validation"] == "Disagree": d += 1
        else: u += 1
    return a, d, u

df["ai"] = df.votes.apply(ai_vote)
for k, uid in EXPERTS.items():
    df[k] = df.votes.apply(lambda v, uid=uid: vote_of(v, uid))
ac = df.votes.apply(human_counts)
df["h_agree"] = ac.apply(lambda t: t[0]); df["h_dis"] = ac.apply(lambda t: t[1]); df["h_unsure"] = ac.apply(lambda t: t[2])
df["h_n"] = df.h_agree + df.h_dis
df["h_maj"] = np.where(df.h_agree > df.h_dis, "Agree", np.where(df.h_dis > df.h_agree, "Disagree", None))
# crowd excluding each expert
for k, uid in EXPERTS.items():
    c = df.votes.apply(lambda v, uid=uid: human_counts(v, exclude=(uid,)))
    df[f"o{k}_agree"] = c.apply(lambda t: t[0]); df[f"o{k}_dis"] = c.apply(lambda t: t[1])
    df[f"o{k}_n"] = df[f"o{k}_agree"] + df[f"o{k}_dis"]
    df[f"o{k}_maj"] = np.where(df[f"o{k}_agree"] > df[f"o{k}_dis"], "Agree", np.where(df[f"o{k}_dis"] > df[f"o{k}_agree"], "Disagree", None))

def agree_stats(a, b, name):
    m = a.notna() & b.notna() & a.isin(["Agree","Disagree"]) & b.isin(["Agree","Disagree"])
    if m.sum() < 20: return f"{name}: n={m.sum()} (too few)"
    acc = (a[m] == b[m]).mean(); k = cohen_kappa_score(a[m], b[m])
    return f"{name}: n={m.sum():,} agreement={acc:.1%} kappa={k:.2f}"

def tbl(d, a, b, label):
    print(f"\n== {label} ==")
    print("  " + agree_stats(d[a], d[b], "all types"))
    for t in TYPES:
        print("  " + agree_stats(d.loc[d.label_type==t, a], d.loc[d.label_type==t, b], t))

print(f"labels (5 validator types): {len(df):,} across {df.city.nunique()} cities")
print(f"labels with an AI vote: {df.ai.notna().sum():,}  (Agree {(df.ai=='Agree').sum():,} / Disagree {(df.ai=='Disagree').sum():,} / Unsure {(df.ai=='Unsure').sum():,})")
print("labels with a vote from " + "; ".join(f"{k}: {df[k].notna().sum():,}" for k in EXPERTS))

# 1. AI vs human majority, min 1 human and min 2 humans
d1 = df[df.h_n >= 1]; d2 = df[df.h_n >= 2]
tbl(d1, "ai", "h_maj", "AI vs human majority (>=1 human vote, AI excluded)")
tbl(d2, "ai", "h_maj", "AI vs human majority (>=2 human votes)")

# verdict confirmation rates
print("\n== How often humans confirm each AI verdict (>=1 human, majority defined) ==")
m = df.ai.isin(["Agree","Disagree"]) & df.h_maj.notna()
for t in ["ALL"] + TYPES:
    s = df[m & ((df.label_type==t) if t!="ALL" else True)]
    for v in ["Agree","Disagree"]:
        ss = s[s.ai==v]
        if len(ss): print(f"  {t:15s} AI={v:8s} n={len(ss):7,}  humans confirm {((ss.h_maj==v).mean()):.1%}")

# 2. Experts vs crowd-of-others; AI vs same crowd; on identical label sets
for k in EXPERTS:
    tbl(df[df[f"o{k}_n"]>=2], k, f"o{k}_maj", f"{k} vs majority of OTHER humans (>=2 others)")
    # same labels for AI
    same = df[(df[f"o{k}_n"]>=2) & df[k].isin(["Agree","Disagree"]) & df.ai.isin(["Agree","Disagree"])]
    print(f"\n== Same label set (both {k} and AI voted, >=2 other humans): n={len(same):,} ==")
    print("  " + agree_stats(same[k], same[f"o{k}_maj"], f"{k} vs others"))
    print("  " + agree_stats(same["ai"], same[f"o{k}_maj"], "AI vs others"))
    for t in TYPES:
        st = same[same.label_type==t]
        print(f"   {t:15s} " + agree_stats(st[k], st[f"o{k}_maj"], k) + " | " + agree_stats(st["ai"], st[f"o{k}_maj"], "AI"))
    tbl(df, "ai", k, f"AI vs {k} directly")
    # when AI and expert disagree, whom does crowd side with
    dd = df[df.ai.isin(["Agree","Disagree"]) & df[k].isin(["Agree","Disagree"]) & (df.ai != df[k]) & (df[f"o{k}_n"]>=2) & df[f"o{k}_maj"].notna()]
    if len(dd): print(f"  AI and {k} disagree, >=2 others voted: n={len(dd):,}; crowd sides with {k} {(dd[k]==dd[f'o{k}_maj']).mean():.1%}")
    dd = df[df.ai.isin(["Agree","Disagree"]) & df[k].isin(["Agree","Disagree"]) & (df.ai != df[k])]
    print(f"  AI/{k} disagreements by direction: " + ", ".join(f"AI={a}/{k}={b}: {((dd.ai==a)&(dd[k]==b)).sum():,}" for a,b in [("Agree","Disagree"),("Disagree","Agree")]))

# 3. Error analysis: AI vs majority of >=2 humans
e = df[(df.h_n>=2) & df.h_maj.notna() & df.ai.isin(["Agree","Disagree"])].copy()
e["err"] = e.ai != e.h_maj
e["kind"] = np.where(~e.err, "ok", np.where(e.ai=="Disagree", "false_disagree", "false_agree"))
print(f"\n== Error analysis base: n={len(e):,}, error rate {e.err.mean():.1%}; false_disagree {(e.kind=='false_disagree').sum():,}, false_agree {(e.kind=='false_agree').sum():,} ==")

def rate_by(col, minn=200, top=None):
    g = e.groupby(col).agg(n=("err","size"), err=("err","mean"), fd=("kind", lambda s: (s=="false_disagree").mean()), fa=("kind", lambda s: (s=="false_agree").mean()))
    g = g[g.n>=minn].sort_values("err", ascending=False)
    if top: g = g.head(top)
    print(f"\n-- error rate by {col} (n>={minn}) --"); print(g.to_string(float_format=lambda x: f"{x:.1%}"))

rate_by("label_type")
rate_by("city", minn=300)
e["severity_b"] = e.severity.fillna(-1).astype(int)
rate_by(["label_type","severity_b"], minn=200)
rate_by("high_quality_user")
rate_by("pano_source")
e["img_age_y"] = ((pd.to_datetime(e.time_created, utc=True, errors="coerce") - pd.to_datetime(e.image_capture_date, errors="coerce").dt.tz_localize("UTC")).dt.days/365.25).round()
e["img_age_bin"] = pd.cut(e.img_age_y, [-1,1,3,5,8,30], labels=["<=1y","1-3y","3-5y","5-8y",">8y"])
rate_by("img_age_bin")
e["rel_y"] = e.pano_y / e.pano_height
e["y_bin"] = pd.cut(e.rel_y, [0,0.5,0.55,0.6,0.65,0.7,0.75,1.0])
rate_by("y_bin")
e["h_n_bin"] = pd.cut(e.h_n, [1,2,3,5,10,1000], labels=["2","3","4-5","6-10",">10"])
rate_by("h_n_bin")
e["h_margin"] = (e.h_agree - e.h_dis).abs() / e.h_n
e["unanimous"] = e.h_margin == 1
rate_by("unanimous")

# tags enriched in errors
print("\n-- tags: error rate when tag present (n>=150), per label type --")
def tags(s):
    try: return [t.strip().strip("'\"") for t in json.loads(s)] if isinstance(s,str) and s.startswith("[") else []
    except Exception: return [t.strip() for t in s.strip("[]").split(",") if t.strip()]
e["taglist"] = e.tags.apply(tags)
for t in TYPES:
    et = e[e.label_type==t]; base = et.err.mean()
    cnt = collections.Counter(); errc = collections.Counter()
    for tl, er in zip(et.taglist, et.err):
        for tg in set(tl): cnt[tg]+=1; errc[tg]+=er
    out = sorted(((tg, cnt[tg], errc[tg]/cnt[tg]) for tg in cnt if cnt[tg]>=150), key=lambda x:-x[2])
    print(f"  {t} (base {base:.1%}): " + "; ".join(f"{tg} n={n} err={r:.1%}" for tg,n,r in out[:8]))
    print(f"     no tags: n={(et.taglist.apply(len)==0).sum()} err={et[et.taglist.apply(len)==0].err.mean():.1%}")

# save the error set for hard-example mining
e[["label_uid","city","label_id","label_type","severity","tags","ai","h_agree","h_dis","h_unsure",*EXPERTS,"kind","pano_id","high_quality_user","image_capture_date"]].to_csv(os.path.join(CACHE,"..","ai_vs_humans.csv"), index=False)
print("\nwrote ai_vs_humans.csv")

# 4. AI vote distribution per label type (does any model only ever cast one verdict?)
print("\n== AI vote distribution per label type (all labels with an AI vote) ==")
print(pd.crosstab(df.loc[df.ai.notna(),"label_type"], df.loc[df.ai.notna(),"ai"]).to_string())

# 5. per-city table: AI vs >=2-human majority, with Agree/Disagree confirmation
print("\n== Per city: AI vs majority of >=2 humans ==")
g = e.groupby("city").apply(lambda s: pd.Series({"n": len(s), "agree%": (~s.err).mean(), "kappa": cohen_kappa_score(s.ai, s.h_maj) if s.ai.nunique()>1 and s.h_maj.nunique()>1 else np.nan,
    "AI=Agree confirmed": (s[s.ai=="Agree"].h_maj=="Agree").mean(), "AI=Disagree confirmed": (s[s.ai=="Disagree"].h_maj=="Disagree").mean(), "n_disagree": (s.ai=="Disagree").sum()}))
g = g[g.n>=200].sort_values("agree%")
print(g.to_string(float_format=lambda x: f"{x:.2f}"))

# 6. hard-example pool: confident human consensus that contradicts the AI
hp = df[df.ai.isin(["Agree","Disagree"]) & (df.h_n>=3) & df.h_maj.notna() & ((df.h_agree==0)|(df.h_dis==0))]
hp = hp[hp.ai != hp.h_maj]
print(f"\n== Hard-example pool: unanimous >=3 humans contradict the AI: n={len(hp):,} ==")
print(pd.crosstab(hp.label_type, hp.ai).to_string())
if EXPERTS:
    against = np.zeros(len(df), dtype=bool)
    for k in EXPERTS:
        against |= (df[k].isin(["Agree","Disagree"]) & (df[k] != df.ai)).values
    ex = df[df.ai.isin(["Agree","Disagree"]).values & against]
    print(f"Expert-contradicted pool (an expert voted against the AI): n={len(ex):,}")
    print(pd.crosstab(ex.label_type, ex.ai).to_string())
# 7. expert gold: labels where the first two experts both voted
if len(EXPERTS) >= 2:
    k1, k2 = list(EXPERTS)[:2]
    both = df[df[k1].isin(["Agree","Disagree"]) & df[k2].isin(["Agree","Disagree"])]
    print(f"\n{k1} & {k2} both voted: n={len(both):,}, they agree {((both[k1]==both[k2]).mean()):.1%}; " + agree_stats(both[k1], both[k2], "kappa"))

import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from sklearn.metrics.pairwise import cosine_similarity

def _norm(text):
    if pd.isna(text): return ""
    return re.sub(r"\s+", " ", str(text).strip().lower())

# Adjust to your exact file paths
orig_df = pd.read_csv("put original dataset path here.csv")
orig_df = orig_df.rename(columns={"ID": "Review ID", "Pos/Neg": "Sentiment"})
orig_df = orig_df[["Review ID", "Topic", "Selected Content", "Sentiment"]].copy()
orig_df = orig_df[orig_df["Topic"].notna() & (orig_df["Topic"].str.strip() != "Off")]
orig_df = orig_df[orig_df["Sentiment"].notna() & (orig_df["Sentiment"].str.strip() != "")]
orig_df["norm"] = orig_df["Selected Content"].apply(_norm)
orig_df = orig_df[orig_df["norm"] != ""].reset_index(drop=True)

out_df = pd.read_csv("put LLM output path here.csv")
out_df = out_df[["Review ID", "Topic", "Selected Content", "Sentiment", "Valid"]].copy()
out_df = out_df[out_df["Topic"] != "(parse failed)"]
out_df = out_df[out_df["Selected Content"].notna() & (out_df["Selected Content"].str.strip() != "")]
out_df["norm"] = out_df["Selected Content"].apply(_norm)
out_df = out_df[out_df["norm"] != ""].reset_index(drop=True)

workings = []

workings.append("============================================================")
workings.append("TOPIC & SENTIMENT (ROW-BY-ROW ALIGNMENT WORKINGS)")
workings.append("============================================================")

def match_substring(orig_norms, out_norms):
    used_orig, used_out, mo, mu = set(), set(), [], []
    for i, on in enumerate(out_norms):
        for j, rn in enumerate(orig_norms):
            if j not in used_orig and on == rn:
                mo.append(j); mu.append(i); used_orig.add(j); used_out.add(i)
                break
    for i, on in enumerate(out_norms):
        if i in used_out: continue
        for j, rn in enumerate(orig_norms):
            if j not in used_orig and (on in rn or rn in on):
                mo.append(j); mu.append(i); used_orig.add(j); used_out.add(i)
                break
    return mo, mu

rids = sorted(set(orig_df["Review ID"].unique()) | set(out_df["Review ID"].unique()))
global_orig_idx, global_out_idx = [], []

for rid in rids:
    orig_g = orig_df[orig_df["Review ID"] == rid].reset_index(drop=True)
    out_g = out_df[out_df["Review ID"] == rid].reset_index(drop=True)
    if orig_g.empty and out_g.empty: continue
    mo, mu = match_substring(orig_g["norm"].tolist(), out_g["norm"].tolist())
    orig_rows = orig_df[orig_df["Review ID"] == rid].index.tolist()
    out_rows = out_df[out_df["Review ID"] == rid].index.tolist()
    
    for oi, ui in zip(mo, mu):
        global_orig_idx.append(orig_rows[oi])
        global_out_idx.append(out_rows[ui])
        o_row = orig_g.iloc[oi]
        u_row = out_g.iloc[ui]
        workings.append(f"\n[Review ID: {rid}]")
        workings.append(f"  Orig Content : {o_row['Selected Content']}")
        workings.append(f"  LLM Content  : {u_row['Selected Content']}")
        workings.append(f"  Topic Check  : Orig='{o_row['Topic']}' vs LLM='{u_row['Topic']}'")
        workings.append(f"  Sent Check   : Orig='{o_row['Sentiment']}' vs LLM='{u_row['Sentiment']}'")

orig_m = orig_df.loc[global_orig_idx].reset_index(drop=True)
out_m = out_df.loc[global_out_idx].reset_index(drop=True)

if not orig_m.empty:
    t_true = orig_m["Topic"].str.strip().tolist()
    t_pred = out_m["Topic"].str.strip().tolist()
    s_true = orig_m["Sentiment"].str.strip().tolist()
    s_pred = out_m["Sentiment"].str.strip().tolist()
    
    def _m(yt, yp):
        p, r, f, _ = precision_recall_fscore_support(yt, yp, average="macro", zero_division=0)
        return accuracy_score(yt, yp), p, r, f
    
    t_acc, t_p, t_r, t_f = _m(t_true, t_pred)
    s_acc, s_p, s_r, s_f = _m(s_true, s_pred)
else:
    t_acc = t_p = t_r = t_f = 0
    s_acc = s_p = s_r = s_f = 0

workings.append("\n============================================================")
workings.append("SELECTED CONTENT (EXACT & COSINE WORKINGS)")
workings.append("============================================================")

tot, p_m, c_m = 0, 0, 0
threshold = 0.3

for rid in out_df["Review ID"].unique():
    orig_g = orig_df[orig_df["Review ID"] == rid]
    out_g = out_df[out_df["Review ID"] == rid]
    if orig_g.empty:
        for out_t in out_g["norm"].tolist():
            tot += 1
            workings.append(f"\n[Review ID: {rid}]")
            workings.append(f"  LLM Content: '{out_t}'")
            workings.append("  Status: NO MATCH (Original row is empty or filtered out)")
        continue
        
    orig_texts = orig_g["norm"].tolist()
    for _, out_row in out_g.iterrows():
        tot += 1
        out_t = out_row["norm"]
        workings.append(f"\n[Review ID: {rid}]")
        workings.append(f"  LLM Content: '{out_row['Selected Content']}'")
        
        is_parity = out_t in orig_texts
        if is_parity:
            p_m += 1
            c_m += 1
            workings.append("  Parity Match : YES")
            workings.append("  Cosine Match : YES (Implied by exact match)")
            continue
            
        workings.append("  Parity Match : NO")
        try:
            vec = TfidfVectorizer(min_df=1)
            tfidf = vec.fit_transform([out_t] + orig_texts)
            sims = cosine_similarity(tfidf[0:1], tfidf[1:])[0]
            best_idx = sims.argmax()
            best_score = sims[best_idx]
            
            if best_score >= threshold:
                c_m += 1
                workings.append(f"  Cosine Match : YES (Score: {best_score:.4f})")
                workings.append(f"  Matched With : '{orig_g.iloc[best_idx]['Selected Content']}'")
            else:
                workings.append(f"  Cosine Match : NO  (Best score: {best_score:.4f} < {threshold})")
                workings.append(f"  Closest To   : '{orig_g.iloc[best_idx]['Selected Content']}'")
        except Exception:
            workings.append("  Cosine Match : NO (TF-IDF Calculation Failed)")

res = []
res.append("============================================================")
res.append("Topic and Sentiment Classification")
res.append("============================================================")
res.append(f"\nBased on {len(orig_m)} aligned pairs (using content inclusion match):")
res.append("  Topic")
res.append(f"    Accuracy  : {t_acc:.4f}")
res.append(f"    Precision : {t_p:.4f}")
res.append(f"    Recall    : {t_r:.4f}")
res.append(f"    F1        : {t_f:.4f}")
res.append("\n  Sentiment")
res.append(f"    Accuracy  : {s_acc:.4f}")
res.append(f"    Precision : {s_p:.4f}")
res.append(f"    Recall    : {s_r:.4f}")
res.append(f"    F1        : {s_f:.4f}")

res.append("\n============================================================")
res.append("Selected Context Output Match (Accuracy of Generation)")
res.append("============================================================")
res.append(f"Total outputs checked from LLM: {tot}")
res.append("\nMethod: Text Parity (Exact Sentence Match)")
res.append(f"  Matches  : {p_m} / {tot}")
res.append(f"  Accuracy : {(p_m/tot*100) if tot else 0:.2f}%")
res.append(f"\nMethod: Cosine Similarity (threshold >= {threshold})")
res.append(f"  Matches  : {c_m} / {tot}")
res.append(f"  Accuracy : {(c_m/tot*100) if tot else 0:.2f}%")

print("\n".join(res))

# Save outputs
with open("workings.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(workings))

with open("results.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(res))
    
print("Successfully generated workings.txt and results.txt")
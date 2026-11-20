import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO

st.set_page_config(page_title="Carrefour Nutri-Score Analysis", page_icon="🛒", layout="wide")

st.title("🛒 Carrefour Nutri-Score Analysis")
st.caption("Compare Carrefour own-label products with other listed brands in the French Open Food Facts dataset.")

st.info(
    "Upload the cleaned CSV exported by the Carrefour analysis notebook. "
    "This app analyses the uploaded data; it does not invent or estimate missing results."
)

uploaded = st.file_uploader("Upload your cleaned Carrefour CSV", type=["csv"])

if uploaded is None:
    st.markdown("### How to use")
    st.markdown(
        "1. Run `Carrefour_Complete_Analysis.ipynb` in Google Colab.  \n"
        "2. Download the cleaned CSV produced by the notebook.  \n"
        "3. Upload that CSV here to display the dashboard."
    )
    st.stop()

try:
    df = pd.read_csv(uploaded)
except Exception as e:
    st.error(f"I couldn't read this CSV: {e}")
    st.stop()

df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]

def find_col(options):
    for c in options:
        if c in df.columns:
            return c
    return None

grade_col = find_col(["nutriscore_grade", "nutrition_grade_fr", "nutri_score", "grade"])
brand_group_col = find_col(["brand_group", "brand_type", "comparison_group", "group"])
brand_col = find_col(["brands", "brand", "brands_tags"])
sugar_col = find_col(["sugars_100g", "sugars"])
salt_col = find_col(["salt_100g", "salt"])
satfat_col = find_col(["saturated_fat_100g", "saturated_fat"])

if grade_col is None:
    st.error("The CSV needs a Nutri-Score column (for example: `nutriscore_grade`). Please upload the cleaned CSV from the notebook.")
    st.write("Columns found:", list(df.columns))
    st.stop()

df[grade_col] = df[grade_col].astype(str).str.strip().str.upper()
df = df[df[grade_col].isin(list("ABCDE"))].copy()

if brand_group_col is None:
    if brand_col is None:
        st.error("I couldn't find a brand or comparison-group column. Please use the cleaned CSV exported by the notebook.")
        st.write("Columns found:", list(df.columns))
        st.stop()
    # Conservative fallback: classify only explicit Carrefour brand mentions.
    brand_text = df[brand_col].fillna("").astype(str).str.lower()
    df["comparison_group"] = brand_text.apply(
        lambda x: "Carrefour own-label" if "carrefour" in x else "Other listed brands"
    )
    brand_group_col = "comparison_group"

df[brand_group_col] = df[brand_group_col].astype(str).str.strip()
df = df[df[brand_group_col].isin(["Carrefour own-label", "Other listed brands"])].copy()

if df.empty:
    st.warning("No valid records remain after filtering. Check that the uploaded CSV is the cleaned export from the notebook.")
    st.stop()

st.success(f"Loaded {len(df):,} valid product records.")

groups = ["Carrefour own-label", "Other listed brands"]
available_groups = [g for g in groups if g in set(df[brand_group_col])]
if len(available_groups) < 2:
    st.warning("The uploaded file contains only one comparison group. Upload the complete cleaned dataset to compare both groups.")

c1, c2, c3 = st.columns(3)
c1.metric("Products analysed", f"{len(df):,}")
c2.metric("Carrefour own-label", f"{(df[brand_group_col] == groups[0]).sum():,}")
c3.metric("Other listed brands", f"{(df[brand_group_col] == groups[1]).sum():,}")

st.subheader("1. Nutri-Score distribution")
counts = pd.crosstab(df[brand_group_col], df[grade_col], normalize="index") * 100
counts = counts.reindex(index=available_groups, columns=list("ABCDE"), fill_value=0)
fig, ax = plt.subplots(figsize=(9, 4.5))
counts.plot(kind="bar", stacked=True, ax=ax)
ax.set_ylabel("Share of products (%)")
ax.set_xlabel("")
ax.set_title("Nutri-Score grades by brand group")
ax.legend(title="Nutri-Score", bbox_to_anchor=(1.02, 1), loc="upper left")
plt.xticks(rotation=0)
st.pyplot(fig, use_container_width=True)
plt.close(fig)

st.subheader("2. Share of products with A/B and E scores")
share = df.assign(
    ab=df[grade_col].isin(["A", "B"]),
    e=df[grade_col].eq("E")
).groupby(brand_group_col)[["ab", "e"]].mean() * 100
share = share.reindex(available_groups)
fig, ax = plt.subplots(figsize=(8, 4))
share.rename(columns={"ab": "Nutri-Score A/B", "e": "Nutri-Score E"}).plot(kind="bar", ax=ax)
ax.set_ylabel("Share of products (%)")
ax.set_xlabel("")
ax.set_title("A/B and E shares by brand group")
plt.xticks(rotation=0)
st.pyplot(fig, use_container_width=True)
plt.close(fig)

st.subheader("3. Nutrient comparison")
nutrient_options = {
    "Sugars (g per 100 g)": sugar_col,
    "Salt (g per 100 g)": salt_col,
    "Saturated fat (g per 100 g)": satfat_col,
}
available_nutrients = {k:v for k,v in nutrient_options.items() if v is not None}
if available_nutrients:
    selected_label = st.selectbox("Choose a nutrient", list(available_nutrients))
    selected_col = available_nutrients[selected_label]
    df[selected_col] = pd.to_numeric(df[selected_col], errors="coerce")
    plot_df = df[[brand_group_col, selected_col]].dropna()
    fig, ax = plt.subplots(figsize=(8, 4))
    data = [plot_df.loc[plot_df[brand_group_col] == g, selected_col].values for g in available_groups]
    ax.boxplot(data, labels=available_groups)
    ax.set_ylabel(selected_label)
    ax.set_title(f"{selected_label}: distribution by brand group")
    plt.xticks(rotation=0)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)
else:
    st.warning("No sugar, salt, or saturated-fat column was found. The nutrient chart is unavailable.")

st.subheader("Key descriptive results")
rows = []
for g in available_groups:
    subset = df[df[brand_group_col] == g]
    row = {"Group": g, "Products": len(subset),
           "A/B share (%)": round(subset[grade_col].isin(["A","B"]).mean()*100, 1),
           "E share (%)": round(subset[grade_col].eq("E").mean()*100, 1)}
    for label, col in [("Median sugars (g/100g)", sugar_col),
                       ("Median salt (g/100g)", salt_col),
                       ("Median saturated fat (g/100g)", satfat_col)]:
        if col:
            vals = pd.to_numeric(subset[col], errors="coerce")
            row[label] = round(vals.median(), 2) if vals.notna().any() else None
    rows.append(row)
summary = pd.DataFrame(rows)
st.dataframe(summary, use_container_width=True, hide_index=True)

st.subheader("Interpretation prompts")
st.markdown(
    "- **Finding:** Describe the largest visible difference between the two groups.  \n"
    "- **Context:** Mention the number of products and the period/source represented by the dataset.  \n"
    "- **So what?:** Explain what the difference could mean for product positioning or consumer choice, without claiming causation."
)

st.subheader("Exactly three caveats to report")
st.markdown(
    "1. Open Food Facts is a crowdsourced database, so product coverage and data quality may vary.  \n"
    "2. Nutri-Score and nutrient information may be missing, outdated, or entered inconsistently.  \n"
    "3. This is a descriptive comparison of listed products, not a representative sample of every product sold in France; it does not prove that the brand caused the observed differences."
)

st.subheader("AI-use log")
st.markdown(
    "AI assistance was used to help structure and debug Python code, suggest visualisations, and improve explanations. "
    "The dataset source, filtering decisions, outputs, and interpretation should be checked by the student."
)

csv_bytes = summary.to_csv(index=False).encode("utf-8")
st.download_button("Download summary table (CSV)", csv_bytes, "carrefour_summary.csv", "text/csv")
st.caption("Source: user-uploaded cleaned data exported from the Open Food Facts analysis notebook.")

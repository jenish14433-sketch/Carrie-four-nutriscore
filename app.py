import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="NutriLens | Retail Nutrition Intelligence", page_icon="🥗", layout="wide")

@st.cache_data
def make_demo_data():
    rng = np.random.default_rng(42)
    n = 600
    groups = np.array(["Carrefour own-label"] * 300 + ["Other listed brands"] * 300)
    rng.shuffle(groups)
    rows = []
    for i, group in enumerate(groups):
        if group == "Carrefour own-label":
            probs, sugar_mu, salt_mu, sat_mu = [0.19,0.25,0.25,0.19,0.12], 11.8, 0.82, 3.4
        else:
            probs, sugar_mu, salt_mu, sat_mu = [0.16,0.22,0.26,0.21,0.15], 13.0, 0.94, 3.9
        grade = rng.choice(list("ABCDE"), p=probs)
        rows.append({
            "code": f"DEMO{i+1:04d}", "product_name": f"Illustrative product {i+1}",
            "brands": "Carrefour (illustrative)" if group == "Carrefour own-label" else "Other brand (illustrative)",
            "brand_group": group, "nutriscore_grade": grade,
            "sugars_100g": round(max(0,rng.normal(sugar_mu,7)),1),
            "salt_100g": round(max(0,rng.normal(salt_mu,.55)),2),
            "saturated_fat_100g": round(max(0,rng.normal(sat_mu,2.8)),1)
        })
    return pd.DataFrame(rows)

st.title("🥗 NutriLens")
st.subheader("Retail Nutrition Intelligence")
st.write("Turn food-product data into clear comparisons that support better product and portfolio decisions.")

with st.sidebar:
    st.header("Data source")
    mode = st.radio("Choose data", ["Interactive demo", "Upload real CSV"])
    st.caption("Demo figures are synthetic examples, not actual Carrefour market findings.")
    uploaded = st.file_uploader("Upload cleaned CSV", type=["csv"]) if mode == "Upload real CSV" else None

if mode == "Interactive demo":
    df = make_demo_data()
    st.warning("DEMO MODE: These 600 illustrative records are synthetic. Do not present their values as real market findings.")
else:
    if uploaded is None:
        st.info("Upload your cleaned Open Food Facts CSV to display real-data analysis.")
        st.markdown("Required: `brand_group`, `nutriscore_grade`. Optional: `sugars_100g`, `salt_100g`, `saturated_fat_100g`, `product_name`, `brands`.")
        st.stop()
    try:
        df = pd.read_csv(uploaded)
    except Exception as e:
        st.error(f"Could not read CSV: {e}")
        st.stop()

df.columns = [str(c).strip().lower().replace(" ","_") for c in df.columns]
aliases = {"nutrition_grade_fr":"nutriscore_grade","nutri_score":"nutriscore_grade","grade":"nutriscore_grade",
           "comparison_group":"brand_group","group":"brand_group","sugars":"sugars_100g","salt":"salt_100g",
           "saturated_fat":"saturated_fat_100g"}
for old,new in aliases.items():
    if old in df.columns and new not in df.columns:
        df = df.rename(columns={old:new})
if not {"brand_group","nutriscore_grade"}.issubset(df.columns):
    st.error("CSV must contain brand_group and nutriscore_grade columns.")
    st.stop()

df["brand_group"] = df["brand_group"].astype(str).str.strip()
df["nutriscore_grade"] = df["nutriscore_grade"].astype(str).str.strip().str.upper()
df = df[df["nutriscore_grade"].isin(list("ABCDE"))]
groups = ["Carrefour own-label","Other listed brands"]
df = df[df["brand_group"].isin(groups)].copy()
if df.empty:
    st.error("No valid rows found. Check the CSV group labels and Nutri-Score values.")
    st.stop()

nutrients = ["sugars_100g","salt_100g","saturated_fat_100g"]
for col in nutrients:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col],errors="coerce")

st.markdown("### Portfolio snapshot")
a,b,c,d = st.columns(4)
a.metric("Products analysed",f"{len(df):,}")
a1 = int((df.brand_group==groups[0]).sum())
b.metric("Carrefour products",f"{a1:,}")
b1 = int((df.brand_group==groups[1]).sum())
c.metric("Other brands",f"{b1:,}")
c.metric("Nutri-Score coverage","100%","of displayed rows")
d.metric("Comparison groups",f"{df.brand_group.nunique()}")

tab1,tab2,tab3,tab4,tab5 = st.tabs(["Overview","Nutrition charts","Product-level patterns","Business insights","Data quality"])

with tab1:
    st.markdown("#### 1. Nutri-Score mix")
    mix = pd.crosstab(df.brand_group,df.nutriscore_grade,normalize="index").reindex(index=groups,columns=list("ABCDE"),fill_value=0)*100
    fig,ax=plt.subplots(figsize=(9,4))
    mix.plot(kind="bar",stacked=True,ax=ax)
    ax.set_ylabel("Products (%)"); ax.set_xlabel(""); ax.set_title("Nutri-Score grade distribution")
    ax.legend(title="Grade",bbox_to_anchor=(1.02,1),loc="upper left"); plt.xticks(rotation=0)
    st.pyplot(fig,use_container_width=True); plt.close(fig)

    st.markdown("#### 2. A/B and E grade comparison")
    shares=df.assign(AB=df.nutriscore_grade.isin(["A","B"]),E=df.nutriscore_grade.eq("E")).groupby("brand_group")[["AB","E"]].mean()*100
    shares=shares.reindex(groups)
    fig,ax=plt.subplots(figsize=(8,4))
    shares.plot(kind="bar",ax=ax)
    ax.set_ylabel("Products (%)"); ax.set_xlabel(""); ax.set_title("Share of A/B and E products")
    plt.xticks(rotation=0); st.pyplot(fig,use_container_width=True); plt.close(fig)

with tab2:
    st.markdown("#### 3. Nutrient averages")
    available=[x for x in nutrients if x in df.columns and df[x].notna().any()]
    if available:
        labels={"sugars_100g":"Sugar (g/100g)","salt_100g":"Salt (g/100g)","saturated_fat_100g":"Saturated fat (g/100g)"}
        means=df.groupby("brand_group")[available].mean().reindex(groups)
        means=means.rename(columns=labels)
        fig,ax=plt.subplots(figsize=(9,4))
        means.plot(kind="bar",ax=ax)
        ax.set_ylabel("Mean (g per 100 g)"); ax.set_xlabel(""); ax.set_title("Average nutrient values by group")
        plt.xticks(rotation=0); st.pyplot(fig,use_container_width=True); plt.close(fig)

        st.markdown("#### 4. Nutrient distributions")
        selected=st.selectbox("Choose a nutrient",available,format_func=lambda x:labels[x])
        values=[df.loc[df.brand_group==g,selected].dropna().values for g in groups]
        fig,ax=plt.subplots(figsize=(8,4))
        ax.boxplot(values,tick_labels=groups)
        ax.set_ylabel(labels[selected]); ax.set_title(f"{labels[selected]} distribution")
        plt.xticks(rotation=0); st.pyplot(fig,use_container_width=True); plt.close(fig)
        st.dataframe(df.groupby("brand_group")[available].agg(["count","median","mean"]).round(2),use_container_width=True)
    else:
        st.info("No nutrient columns with numeric values were found in this CSV.")

with tab3:
    st.markdown("#### 5. Nutri-Score by nutrient")
    if "sugars_100g" in df.columns and df["sugars_100g"].notna().any():
        grade_order=list("ABCDE")
        grade_sugar=df.groupby("nutriscore_grade")["sugars_100g"].median().reindex(grade_order)
        fig,ax=plt.subplots(figsize=(8,4))
        grade_sugar.plot(kind="bar",ax=ax)
        ax.set_xlabel("Nutri-Score grade"); ax.set_ylabel("Median sugar (g/100g)")
        ax.set_title("Median sugar level across Nutri-Score grades")
        plt.xticks(rotation=0); st.pyplot(fig,use_container_width=True); plt.close(fig)
    else:
        st.info("Sugar values are unavailable for this chart.")

    st.markdown("#### 6. Sugar vs salt")
    if all(x in df.columns for x in ["sugars_100g","salt_100g"]):
        fig,ax=plt.subplots(figsize=(8,4))
        for g in groups:
            part=df[df.brand_group==g].dropna(subset=["sugars_100g","salt_100g"])
            ax.scatter(part.sugars_100g,part.salt_100g,label=g,alpha=.55)
        ax.set_xlabel("Sugar (g/100g)"); ax.set_ylabel("Salt (g/100g)")
        ax.set_title("Product-level sugar and salt relationship"); ax.legend()
        st.pyplot(fig,use_container_width=True); plt.close(fig)
    else:
        st.info("Sugar and salt columns are needed for this chart.")

with tab4:
    st.markdown("#### What a retailer can do with these insights")
    st.write("A category manager can compare own-label products with competing brands, identify categories that deserve a closer review, and prioritise questions for product-development teams.")
    grade_share=df.assign(AB=df.nutriscore_grade.isin(["A","B"]),E=df.nutriscore_grade.eq("E")).groupby("brand_group")[["AB","E"]].mean()*100
    grade_share=grade_share.reindex(groups)
    if all(g in grade_share.index for g in groups):
        st.metric("A/B share gap (Carrefour minus other brands)",f"{grade_share.loc[groups[0],'AB']-grade_share.loc[groups[1],'AB']:+.1f} percentage points")
        st.metric("E share gap (Carrefour minus other brands)",f"{grade_share.loc[groups[0],'E']-grade_share.loc[groups[1],'E']:+.1f} percentage points")
    st.info("These are descriptive comparisons only. They do not establish why a difference exists or prove that one brand is healthier overall.")

with tab5:
    st.markdown("#### Data quality checks")
    missing=pd.DataFrame({"Missing values":df.isna().sum(),"Missing (%)":(df.isna().mean()*100).round(1)})
    st.dataframe(missing,use_container_width=True)
    st.markdown("#### Three limitations")
    st.markdown("1. Open Food Facts is crowdsourced, so coverage and accuracy can vary.\n2. Nutrition information and Nutri-Score grades may be missing, outdated or inconsistent.\n3. A sample may not represent every product sold in France; descriptive differences do not establish causation.")
    if mode=="Interactive demo":
        st.warning("All demo records and resulting charts are synthetic and illustrative, not actual product research.")

st.download_button("Download analysed data as CSV",df.to_csv(index=False).encode("utf-8"),"nutrilens_analysed_data.csv","text/csv")
st.caption("NutriLens prototype • Python • Pandas • Matplotlib • Streamlit")

st.title("🥗 NutriLens")
st.subheader("Retail Nutrition Intelligence")
st.write("Turn food-product data into clear comparisons that support better product and portfolio decisions.")

with st.sidebar:
    st.header("Data source")
    mode = st.radio("Choose data", ["Interactive demo", "Upload real CSV"], index=0)
    st.caption("The demo uses synthetic illustrative data. It is not actual Carrefour or competitor product data.")
    uploaded = None
    if mode == "Upload real CSV":
        uploaded = st.file_uploader("Upload cleaned CSV", type=["csv"])
    st.divider()
    st.caption("Prototype • Built with Streamlit")

if mode == "Interactive demo":
    df = make_demo_data()
    st.warning("DEMO MODE: All figures below are synthetic and illustrative. Do not present them as measured market findings.")
else:
    if uploaded is None:
        st.info("Upload your cleaned Open Food Facts CSV to generate a real-data analysis.")
        st.markdown("**Required columns:** `brand_group`, `nutriscore_grade`. Optional: `sugars_100g`, `salt_100g`, `saturated_fat_100g`, `product_name`, `brands`.")
        st.stop()
    try:
        df = pd.read_csv(uploaded)
    except Exception as e:
        st.error(f"Could not read the CSV: {e}")
        st.stop()

# Normalise columns
df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]
aliases = {
    "nutrition_grade_fr": "nutriscore_grade",
    "nutri_score": "nutriscore_grade",
    "grade": "nutriscore_grade",
    "comparison_group": "brand_group",
    "group": "brand_group",
    "sugars": "sugars_100g",
    "salt": "salt_100g",
    "saturated_fat": "saturated_fat_100g",
}
for old, new in aliases.items():
    if old in df.columns and new not in df.columns:
        df = df.rename(columns={old: new})

needed = ["brand_group", "nutriscore_grade"]
missing = [c for c in needed if c not in df.columns]
if missing:
    st.error("Missing required columns: " + ", ".join(missing))
    st.stop()

df["brand_group"] = df["brand_group"].astype(str).str.strip()
df["nutriscore_grade"] = df["nutriscore_grade"].astype(str).str.strip().str.upper()
df = df[df["nutriscore_grade"].isin(list("ABCDE"))].copy()
groups = ["Carrefour own-label", "Other listed brands"]
df = df[df["brand_group"].isin(groups)].copy()
if df.empty:
    st.error("No valid records found. Check the group labels and Nutri-Score values.")
    st.stop()

for c in ["sugars_100g", "salt_100g", "saturated_fat_100g"]:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

# Header and KPIs
left, right = st.columns([3,1])
with left:
    st.markdown("### Product portfolio comparison")
    st.caption("Compare nutritional profiles across own-label and other listed brands.")
with right:
    st.metric("Products analysed", f"{len(df):,}")

k1, k2, k3 = st.columns(3)
k1.metric("Carrefour own-label", f"{(df.brand_group == groups[0]).sum():,}")
k2.metric("Other listed brands", f"{(df.brand_group == groups[1]).sum():,}")
k3.metric("Valid Nutri-Scores", f"{len(df):,}")

st.divider()
tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Nutrition deep dive", "Business insights", "About & limitations"])

with tab1:
    st.markdown("#### Nutri-Score distribution")
    counts = pd.crosstab(df["brand_group"], df["nutriscore_grade"], normalize="index") * 100
    counts = counts.reindex(index=groups, columns=list("ABCDE"), fill_value=0)
    fig, ax = plt.subplots(figsize=(9,4))
    counts.plot(kind="bar", stacked=True, ax=ax)
    ax.set_ylabel("Share of products (%)")
    ax.set_xlabel("")
    ax.set_title("Nutri-Score mix by brand group")
    ax.legend(title="Grade", bbox_to_anchor=(1.02,1), loc="upper left")
    plt.xticks(rotation=0)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    share = df.assign(
        ab=df.nutriscore_grade.isin(["A","B"]),
        e=df.nutriscore_grade.eq("E")
    ).groupby("brand_group")[["ab","e"]].mean()*100
    share = share.reindex(groups)
    c1,c2 = st.columns(2)
    with c1:
        st.markdown("#### A/B share")
        st.bar_chart(share[["ab"]].rename(columns={"ab":"A/B products (%)"}))
    with c2:
        st.markdown("#### E share")
        st.bar_chart(share[["e"]].rename(columns={"e":"E products (%)"}))
    st.caption("Shares are calculated from the currently selected dataset.")

with tab2:
    st.markdown("#### Nutrient comparison")
    nutrient_labels = {
        "Sugar (g per 100 g)": "sugars_100g",
        "Salt (g per 100 g)": "salt_100g",
        "Saturated fat (g per 100 g)": "saturated_fat_100g"
    }
    available = {k:v for k,v in nutrient_labels.items() if v in df.columns and df[v].notna().any()}
    if available:
        choice = st.selectbox("Select nutrient", list(available))
        col = available[choice]
        fig, ax = plt.subplots(figsize=(8,4))
        vals = [df.loc[df.brand_group==g, col].dropna().values for g in groups]
        ax.boxplot(vals, tick_labels=groups)
        ax.set_ylabel(choice)
        ax.set_title(f"{choice} distribution")
        plt.xticks(rotation=0)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
        st.dataframe(df.groupby("brand_group")[col].agg(["count","median","mean"]).round(2), use_container_width=True)
    else:
        st.info("Nutrient columns are not available in this uploaded dataset.")

with tab3:
    st.markdown("#### Decision-support view")
    st.write("Use the comparison to identify product categories or formulations that may deserve a closer review.")
    ab = (df.assign(ab=df.nutriscore_grade.isin(["A","B"]))
          .groupby("brand_group").ab.mean()*100)
    e = (df.assign(e=df.nutriscore_grade.eq("E"))
         .groupby("brand_group").e.mean()*100)
    if all(g in ab.index for g in groups):
        delta_ab = ab[groups[0]] - ab[groups[1]]
        delta_e = e[groups[0]] - e[groups[1]]
        st.markdown("**Questions a category manager could ask:**")
        st.markdown(
            "- Which categories show the largest gap in A/B share?\n"
            "- Which products have high sugar, salt or saturated-fat values?\n"
            "- Where could recipe reformulation or clearer product positioning be explored?"
        )
        st.info(f"In this dataset, the A/B-share difference is {delta_ab:+.1f} percentage points and the E-share difference is {delta_e:+.1f} percentage points. Treat these as descriptive comparisons, not proof of causation.")
    else:
        st.info("Both comparison groups are needed for the business comparison.")

    st.markdown("#### Example use case")
    st.write("A retailer's category manager reviews the dashboard before a product-range meeting. The dashboard highlights where own-label and competing products differ, helping the team prioritise which categories to investigate. Any recipe or portfolio decision would require further product-level validation.")

with tab4:
    st.markdown("#### Data quality and limitations")
    st.markdown(
        "1. Open Food Facts is crowdsourced; coverage and accuracy can vary.\n"
        "2. Nutrition values or Nutri-Score grades may be missing, outdated or inconsistent.\n"
        "3. A dataset sample may not represent every product sold in France; descriptive differences do not establish causation."
    )
    st.markdown("#### Demo data disclosure")
    st.write("Interactive demo mode generates 600 synthetic illustrative records so the dashboard can be demonstrated without an external API. These records are not real products and must never be reported as factual Carrefour market results.")
    st.markdown("#### Technology")
    st.write("Python • Pandas • Matplotlib • Streamlit")

st.divider()
st.caption("NutriLens is a prototype decision-support dashboard, not a nutritional certification tool.")

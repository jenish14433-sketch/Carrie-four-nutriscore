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
    mode = st.radio("Choose data", ["Interactive demo", "Upload real CSV"], key="data_source_mode")
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

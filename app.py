import streamlit as st
import pandas as pd
import requests
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timezone

st.set_page_config(page_title="Carrefour | Nutri-Score Intelligence", page_icon="🛒", layout="wide")

# ---------- Styling ----------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@500;600;700;800&display=swap');
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.stApp { background: #F4F7FB; }
.block-container { padding-top: 1.5rem; max-width: 1450px; }
.hero { background: linear-gradient(115deg,#062B68 0%,#0756A8 62%,#0B75C9 100%); padding: 28px 32px; border-radius: 20px; color: white; box-shadow: 0 12px 30px rgba(7,53,112,.16); margin-bottom: 22px; }
.hero h1 { font-family: Manrope,sans-serif; font-size: 30px; line-height:1.15; margin: 0 0 8px 0; color:white; font-weight:800; }
.hero p { margin:0; color:#DCEBFF; font-size:14px; }
.eyebrow { letter-spacing:1.8px; text-transform:uppercase; font-size:10px; font-weight:700; color:#B9D8FF; margin-bottom:10px; }
.metric-card { background:white; border:1px solid #E6ECF4; border-radius:16px; padding:18px 20px; min-height:112px; box-shadow:0 4px 14px rgba(16,40,80,.035); }
.metric-label { color:#718096; font-size:12px; font-weight:600; margin-bottom:9px; }
.metric-value { color:#102A50; font-family:Manrope,sans-serif; font-size:27px; font-weight:800; line-height:1.1; }
.metric-note { color:#8A98AA; font-size:11px; margin-top:7px; }
.section-title { color:#102A50; font-family:Manrope,sans-serif; font-size:19px; font-weight:800; margin: 18px 0 3px 0; }
.section-sub { color:#7A8798; font-size:12px; margin-bottom:12px; }
.panel { background:white; border:1px solid #E6ECF4; border-radius:16px; padding:18px 20px; }
.pill { display:inline-block; border-radius:30px; padding:5px 10px; font-size:10px; font-weight:700; background:#EAF3FF; color:#0756A8; }
.small-muted { color:#8190A3; font-size:11px; }
div[data-testid="stMetric"] { background:white; border:1px solid #E6ECF4; padding:14px 16px; border-radius:14px; }
.stButton>button { border-radius:10px; font-weight:700; min-height:42px; }
footer {visibility:hidden;}
</style>
""", unsafe_allow_html=True)

GRADE_COLORS = {"a":"#168B45", "b":"#78B82A", "c":"#F2C230", "d":"#EE8A24", "e":"#D94343"}
GRADE_LABELS = {"a":"A · Best", "b":"B", "c":"C", "d":"D", "e":"E · Lowest"}
DEFAULT_KEYWORDS = "carrefour, carrefour bio, carrefour classic, carrefour extra, simpl, reflets de france"

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_off_products(limit: int, country_tag: str, keywords: str):
    """Fetch paginated product records from OFF v2 search. No fabricated fallback data."""
    url = "https://world.openfoodfacts.org/api/v2/search"
    headers = {"User-Agent": "CarrefourNutriScoreStudentResearch/1.0 (Streamlit app)"}
    fields = "code,product_name,brands,nutriscore_grade,nutrition_grades,categories,countries_tags,last_modified_t"
    page_size = 50
    pages = (limit + page_size - 1) // page_size
    all_rows = []
    for page in range(1, pages + 1):
        params = {
            "countries_tags_en": country_tag,
            "nutrition_grades_tags": "a|b|c|d|e",  # pipe means OR; commas mean AND in Open Food Facts
            "fields": fields,
            "page_size": page_size,
            "page": page,
            "sort_by": "popularity_key",
        }
        last_error = None
        payload = None
        for attempt in range(3):
            try:
                r = requests.get(url, params=params, headers=headers, timeout=35)
                if r.status_code in (429, 500, 502, 503, 504):
                    last_error = f"Open Food Facts returned HTTP {r.status_code}"
                    import time
                    time.sleep(2 ** attempt * 2)
                    continue
                r.raise_for_status()
                payload = r.json()
                break
            except requests.RequestException as e:
                last_error = str(e)
                import time
                time.sleep(2 ** attempt * 2)
        if payload is None:
            raise RuntimeError(last_error or "The Open Food Facts API did not respond.")
        products = payload.get("products", [])
        if not products:
            break
        all_rows.extend(products)
        if len(all_rows) >= limit or len(products) < page_size:
            break
    return all_rows[:limit]


def clean_products(products, keywords):
    if not products:
        return pd.DataFrame()
    df = pd.DataFrame(products)
    for c in ["code", "product_name", "brands", "nutriscore_grade", "nutrition_grades", "categories"]:
        if c not in df.columns:
            df[c] = ""
    # OFF records can expose the grade under either field name depending on the record/API schema.
    df["nutriscore_grade"] = df["nutriscore_grade"].replace("", pd.NA).fillna(df["nutrition_grades"])
    df = df[["code", "product_name", "brands", "nutriscore_grade", "categories"]].copy()
    df["brands"] = df["brands"].fillna("").astype(str).str.strip()
    df["product_name"] = df["product_name"].fillna("").astype(str).str.strip()
    df["nutriscore_grade"] = df["nutriscore_grade"].fillna("").astype(str).str.lower().str.strip()
    df = df[df["brands"].ne("") & df["nutriscore_grade"].isin(list("abcde"))].copy()
    df = df.drop_duplicates(subset=["code"], keep="first")
    terms = [x.strip().lower() for x in keywords.split(",") if x.strip()]
    def classify(value):
        b = value.lower()
        return "Carrefour own-label" if any(term in b for term in terms) else "Other listed brands"
    df["group"] = df["brands"].apply(classify)
    df["grade_label"] = df["nutriscore_grade"].map(GRADE_LABELS)
    return df.reset_index(drop=True)

# ---------- Sidebar ----------
st.sidebar.markdown("## 🛒 Carrefour Intelligence")
st.sidebar.caption("NUTRITION & ASSORTMENT ANALYTICS")
st.sidebar.markdown("---")
st.sidebar.markdown("**Analysis settings**")
country = st.sidebar.selectbox("Market", ["France", "Belgium", "Germany", "Spain", "Italy"], index=0)
country_tag = country.lower()
limit = st.sidebar.select_slider("Products to retrieve", options=[100, 200, 300, 500], value=200)
keywords = st.sidebar.text_area("Carrefour brand keywords", value=DEFAULT_KEYWORDS, height=110, help="Comma-separated. Review these brand rules before reporting.")
refresh = st.sidebar.button("↻  Refresh live data", use_container_width=True, type="primary")
st.sidebar.markdown("---")
st.sidebar.caption("Data source: Open Food Facts public product database")
st.sidebar.caption("This dashboard uses a popularity-sorted API sample, not a census.")

# ---------- Header ----------
st.markdown(f"""
<div class="hero">
  <div class="eyebrow">Retail intelligence · {country} market</div>
  <h1>Nutri-Score Intelligence</h1>
  <p>Understanding the nutrition profile of Carrefour own-label products versus other listed brands.</p>
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="pill">PROJECT CLIENT · CARREFOUR</div>', unsafe_allow_html=True)
st.markdown('<div class="section-title">Executive overview</div>', unsafe_allow_html=True)
st.markdown('<div class="section-sub">A live, sample-based view of product nutrition grades.</div>', unsafe_allow_html=True)

if refresh:
    st.cache_data.clear()

if "products" not in st.session_state or refresh:
    with st.spinner("Connecting to Open Food Facts and retrieving product records…"):
        try:
            st.session_state.products = fetch_off_products(limit, country_tag, keywords)
            st.session_state.fetch_error = None
            st.session_state.fetched_at = datetime.now(timezone.utc).strftime("%d %b %Y, %H:%M UTC")
        except Exception as e:
            st.session_state.fetch_error = str(e)
            st.session_state.products = []

if st.session_state.get("fetch_error"):
    st.error("Live data could not be retrieved")
    st.markdown(f"**Connection detail:** {st.session_state.fetch_error}")
    st.info("Open Food Facts may be temporarily unavailable or rate-limiting requests. Try Refresh live data later. The app deliberately does not insert demo or fabricated product data.")
    st.markdown("[Open Food Facts data portal](https://world.openfoodfacts.org/data) · [API documentation](https://openfoodfacts.github.io/documentation/)")
    st.stop()

df = clean_products(st.session_state.products, keywords)
if df.empty:
    st.warning("The source returned no products with both a brand and valid Nutri-Score. Try refreshing later or adjust the market.")
    st.stop()

n_total = len(df)
n_carrefour = int((df.group == "Carrefour own-label").sum())
n_other = int((df.group == "Other listed brands").sum())
ab_share = df[df.group == "Carrefour own-label"].nutriscore_grade.isin(["a", "b"]).mean() * 100 if n_carrefour else None
other_ab = df[df.group == "Other listed brands"].nutriscore_grade.isin(["a", "b"]).mean() * 100 if n_other else None

def metric_card(label, value, note):
    st.markdown(f'<div class="metric-card"><div class="metric-label">{label}</div><div class="metric-value">{value}</div><div class="metric-note">{note}</div></div>', unsafe_allow_html=True)

m1, m2, m3, m4 = st.columns(4)
with m1: metric_card("USABLE PRODUCTS", f"{n_total:,}", f"From up to {limit:,} API records")
with m2: metric_card("CARREFOUR PRODUCTS", f"{n_carrefour:,}", "Keyword-classified own-label")
with m3: metric_card("OTHER LISTED BRANDS", f"{n_other:,}", "Non-Carrefour brands in sample")
with m4: metric_card("CARREFOUR A/B SHARE", f"{ab_share:.1f}%" if ab_share is not None else "—", "Higher Nutri-Score grades" if ab_share is not None else "No Carrefour matches")

st.caption(f"Last retrieved: {st.session_state.get('fetched_at','—')} · Source: Open Food Facts · Market filter: {country}")

# Filters
st.markdown('<div class="section-title">Explore the product sample</div>', unsafe_allow_html=True)
colf1, colf2 = st.columns([1, 2])
with colf1:
    group_filter = st.selectbox("Product group", ["All groups", "Carrefour own-label", "Other listed brands"])
with colf2:
    search = st.text_input("Search product or brand", placeholder="e.g. biscuits, yoghurt, Carrefour Bio…")
view = df.copy()
if group_filter != "All groups": view = view[view.group == group_filter]
if search.strip():
    mask = view["product_name"].str.contains(search, case=False, na=False) | view["brands"].str.contains(search, case=False, na=False)
    view = view[mask]

left, right = st.columns([1.15, 1])
with left:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown("**Nutri-Score distribution**")
    dist = pd.crosstab(df["group"], df["nutriscore_grade"]).reindex(columns=list("abcde"), fill_value=0)
    pct = dist.div(dist.sum(axis=1).replace(0, pd.NA), axis=0) * 100
    chart_df = pct.reset_index().melt(id_vars="group", var_name="Grade", value_name="Share (%)")
    chart_df["Grade"] = chart_df["Grade"].str.upper()
    fig = px.bar(chart_df, x="Grade", y="Share (%)", color="Grade", facet_col="group", barmode="group",
                 color_discrete_map={k.upper():v for k,v in GRADE_COLORS.items()}, height=340)
    fig.update_layout(margin=dict(l=10,r=10,t=35,b=10), showlegend=False, plot_bgcolor="white", paper_bgcolor="white", font_color="#31445E")
    fig.update_yaxes(gridcolor="#EDF1F6", title=None)
    fig.update_xaxes(title=None)
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
with right:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown("**Higher vs lower grades**")
    comp_rows = []
    for g, part in df.groupby("group"):
        comp_rows.append({"Product group":g, "A / B":round(part.nutriscore_grade.isin(["a","b"]).mean()*100,1), "D / E":round(part.nutriscore_grade.isin(["d","e"]).mean()*100,1)})
    comp = pd.DataFrame(comp_rows)
    if not comp.empty:
        cfig = go.Figure()
        for col, color in [("A / B", "#1B9A59"), ("D / E", "#E36B55")]:
            cfig.add_trace(go.Bar(name=col, x=comp["Product group"], y=comp[col], marker_color=color))
        cfig.update_layout(barmode="group", height=340, margin=dict(l=5,r=5,t=25,b=5), plot_bgcolor="white", paper_bgcolor="white", yaxis_title="Share of products (%)", legend_title="Grades")
        cfig.update_yaxes(gridcolor="#EDF1F6", range=[0,100])
        st.plotly_chart(cfig, use_container_width=True)
    else:
        st.info("Not enough data to compare groups.")
    st.markdown('</div>', unsafe_allow_html=True)

# Findings
st.markdown('<div class="section-title">Key observations</div>', unsafe_allow_html=True)
st.markdown('<div class="section-sub">Automatically calculated from the retrieved sample. Verify before using in a report.</div>', unsafe_allow_html=True)
if n_carrefour and n_other:
    delta = ab_share - other_ab
    f1, f2 = st.columns(2)
    with f1:
        st.metric("Carrefour A/B share", f"{ab_share:.1f}%", help=f"Based on {n_carrefour} keyword-classified Carrefour products.")
    with f2:
        st.metric("Difference vs other listed brands", f"{delta:+.1f} percentage points", help=f"Other listed brands A/B share: {other_ab:.1f}% ({n_other} products).")
    direction = "higher" if delta > 0 else "lower" if delta < 0 else "the same as"
    st.info(f"In this retrieved sample, the share of A/B Nutri-Scores among Carrefour-classified products is {abs(delta):.1f} percentage points {direction} the share among other listed brands.")
else:
    st.warning("Both groups are needed to calculate the comparison. Review the keyword list or retrieve a larger sample.")

# Product table
st.markdown('<div class="section-title">Product explorer</div>', unsafe_allow_html=True)
st.markdown('<div class="section-sub">Filtered products from the retrieved sample.</div>', unsafe_allow_html=True)
table = view[["product_name","brands","nutriscore_grade","group","categories"]].copy()
table.columns = ["Product", "Brand", "Nutri-Score", "Group", "Categories"]
st.dataframe(table, use_container_width=True, hide_index=True, height=360)

csv = df.to_csv(index=False).encode("utf-8")
st.download_button("⬇ Download analysed sample (CSV)", data=csv, file_name="carrefour_nutriscore_sample.csv", mime="text/csv")

# Methodology & caveats
with st.expander("Methodology, data quality & limitations", expanded=False):
    st.markdown(f"""
    **Data source:** Open Food Facts public product database, accessed through its v2 search API.  
    **Retrieval:** Up to {limit} popularity-sorted records for {country}; the API may return fewer.  
    **Cleaning:** Products must have a non-empty brand and Nutri-Score grade A–E; duplicate product codes are removed.  
    **Classification:** A brand is labelled Carrefour own-label if its text contains one of the sidebar keywords. All other non-empty brands are labelled “Other listed brands”.

    **Important limitations**
    - Popularity-sorted results are not a random or representative sample of the market.
    - The “Other listed brands” group may include other retailers' private labels; it is not necessarily a pure manufacturer-brand group.
    - Keyword matching can produce false positives and false negatives. Inspect the product explorer and validate brands.
    - Open Food Facts is community-contributed; missing and inaccurate product information may exist.
    - Nutri-Score comparisons are descriptive and do not prove causation or overall product healthfulness.
    - Record the retrieval date and sample size in your coursework report.
    """)

st.markdown("<hr style='border:0;border-top:1px solid #E5EBF3;margin-top:32px'>", unsafe_allow_html=True)
st.markdown("<div class='small-muted'>CARREFOUR NUTRI-SCORE INTELLIGENCE · Student research prototype · Data provided by Open Food Facts</div>", unsafe_allow_html=True)

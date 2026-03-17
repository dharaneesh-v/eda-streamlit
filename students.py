# app.py
import os
import re
import time
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import google.generativeai as genai

# =========================
# Page Config
# =========================
st.set_page_config(page_title="Student Data Quality & Insights", layout="wide")
st.title("🎓 Student Data Quality • Visual Insights • AI Assistant")

# =========================
# Helpers & Caching
# =========================
@st.cache_data(ttl=600, show_spinner=False)
def load_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)

@st.cache_data(ttl=600, show_spinner=False)
def build_compact_summary(df: pd.DataFrame, exclude_cols=None, top_n_cats=5) -> str:
    """
    Build a SMALL, structured summary for LLM (schema, nulls, numerics, categories, correlations).
    Keeps token size small for speed & accuracy.
    """
    if exclude_cols is None:
        exclude_cols = []

    df_ = df.drop(columns=[c for c in exclude_cols if c in df.columns], errors="ignore").copy()

    # Basic info
    rows, cols = df_.shape
    schema = {c: str(df_[c].dtype) for c in df_.columns}

    # Nulls (only non-zero)
    nulls = df_.isna().sum()
    nulls = {k: int(v) for k, v in nulls.items() if v > 0}

    # Numeric stats
    num_cols = df_.select_dtypes(include=[np.number]).columns.tolist()
    num_stats = {}
    if num_cols:
        desc = df_[num_cols].describe().T
        for c in num_cols:
            s = desc.loc[c]
            num_stats[c] = {
                "count": int(s["count"]),
                "mean": round(float(s["mean"]), 3) if "mean" in s else None,
                "std": round(float(s["std"]), 3) if "std" in s else None,
                "min": round(float(s["min"]), 3) if "min" in s else None,
                "p25": round(float(s["25%"]), 3) if "25%" in s else None,
                "median": round(float(s["50%"]), 3) if "50%" in s else None,
                "p75": round(float(s["75%"]), 3) if "75%" in s else None,
                "max": round(float(s["max"]), 3) if "max" in s else None,
            }

    # Categorical top-k
    cat_cols = df_.select_dtypes(exclude=[np.number]).columns.tolist()
    cat_tops = {}
    for c in cat_cols:
        vc = df_[c].astype(str).value_counts().head(top_n_cats)
        cat_tops[c] = [{"value": idx, "count": int(cnt)} for idx, cnt in vc.items()]

    # Lightweight correlations focused on target-like columns
    corr_focus = {}
    target_cols = [c for c in ["gpa", "attendance_percentage"] if c in df_.columns]
    # Convert placement to binary if present
    if "placement_status" in df_.columns:
        tmp = df_[["placement_status"]].copy()
        tmp["placement_binary"] = (df_["placement_status"].astype(str).str.title() == "Placed").astype(float)
        df_["placement_binary"] = tmp["placement_binary"]
        target_cols.append("placement_binary")

    if len(target_cols) >= 2:
        corr = df_[target_cols].corr(numeric_only=True)
        for r in corr.index:
            for c in corr.columns:
                if r != c and pd.notna(corr.loc[r, c]):
                    corr_focus[f"{r}~{c}"] = round(float(corr.loc[r, c]), 3)

    # Cap length to avoid long prompts
    summary = {
        "shape": {"rows": rows, "cols": cols},
        "schema": schema,
        "non_null_issues": nulls,
        "numeric_stats": num_stats,
        "categorical_tops": cat_tops,
        "corr_focus": corr_focus,
    }
    import json
    text = json.dumps(summary, ensure_ascii=False)
    if len(text) > 8000:
        # truncate safely if extremely large
        text = text[:8000] + "...(truncated)"
    return text

def normalize_phone(phone):
    """Return '##### #####' if 10 digits can be extracted, else NaN."""
    if pd.isna(phone):
        return np.nan
    digits = re.sub(r"\D", "", str(phone))
    if len(digits) == 10:
        return f"{digits[:5]} {digits[5:]}"
    return np.nan

def clean_skills_column(df: pd.DataFrame) -> pd.DataFrame:
    if "skills" not in df.columns:
        return df
    # Split into up to 5 skill columns
    parts = df["skills"].fillna("").astype(str).apply(
        lambda s: [p.strip().title() for p in s.split(",") if p.strip()]
    )
    skill_df = pd.DataFrame(parts.tolist(), index=df.index)
    # Expand to 5 columns
    for i in range(5):
        df[f"skill_{i+1}"] = skill_df[i] if i in skill_df.columns else np.nan
    return df

def boxplot_by_category(df, cat_col, y_col, title):
    fig, ax = plt.subplots(figsize=(6, 4))
    groups = list(df.groupby(cat_col))
    if not groups:
        return fig
    data = [grp[y_col].dropna() for _, grp in groups]
    labels = [str(k) for k, _ in groups]
    ax.boxplot(data, labels=labels, patch_artist=True)
    ax.set_title(title)
    ax.set_xlabel(cat_col)
    ax.set_ylabel(y_col)
    ax.grid(alpha=0.2)
    return fig

def corr_heatmap(df, cols, title):
    corr = df[cols].corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(5, 4))
    cax = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(cols)))
    ax.set_yticks(range(len(cols)))
    ax.set_xticklabels(cols, rotation=45, ha="right")
    ax.set_yticklabels(cols)
    fig.colorbar(cax, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title(title)
    return fig

def extract_student_skills_row(record: pd.Series) -> set:
    """Collect normalized (title-case) skills from skill_1..skill_5 columns."""
    skills = set()
    for i in range(1, 6):
        col = f"skill_{i}"
        if col in record.index and pd.notna(record[col]) and str(record[col]).strip():
            skills.add(str(record[col]).strip().title())
    return skills

def normalize_skill_list(skills_in):
    """Normalize a list of skills (case/spacing)."""
    return [s.strip().title() for s in skills_in if isinstance(s, str) and s.strip()]

# =========================
# Load Data
# =========================
df = load_csv("student_data.csv")

# =========================
# Tabs
# =========================
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs(
    [
        "Initial Data",
        "Data Cleaning",
        "Cleaned Data",
        "Visualization",
        "AI Insights",
        "Student Profile",
        "At-Risk Detector",
        "Skill Gap Analyzer",
    ]
)

# =========================
# Tab 1: Initial Data
# =========================
with tab1:
    st.subheader("Raw Dataset Preview")
    st.dataframe(df, use_container_width=True)
    st.caption(f"Rows: {len(df):,} • Columns: {len(df.columns)}")

# =========================
# Tab 2: Data Cleaning
# =========================
with tab2:
    st.subheader("Cleaning Steps")
    initial_null_df = df.isna().sum()

    # —— Remove duplicates
    df.drop_duplicates(keep="first", inplace=True)

    if "email" in df.columns:
        with st.expander("Email"):
            col1, col2 = st.columns(2)
            with col1:
                st.write("Invalid emails (sample, before):")
                invalid_email_before = df[
                    ~df["email"].astype(str).str.match(
                        r"^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$", na=False
                    )
                ]
                st.write(invalid_email_before["email"].head(5))

            email_pattern = r"^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$"
            df["email"] = df["email"].apply(
                lambda x: x if isinstance(x, str) and re.match(email_pattern, x) else np.nan
            )
            # Fill NA emails with college ID
            if "register_number" in df.columns:
                clg_email = df["register_number"].astype(str) + "@skcet.ac.in"
                df["email"] = df["email"].fillna(clg_email)

            with col2:
                st.write("Email column after cleaning (sample rows):")
                st.write(df["email"].head(10))

    if "department" in df.columns:
        with st.expander("Department"):
            col1, col2 = st.columns(2)
            with col1:
                st.write("Before:")
                st.write(df["department"].value_counts())

            df["department"] = df["department"].replace(
                {
                    "IT": "Information Technology",
                    "It": "Information Technology",
                    "it": "Information Technology",
                    "CSE": "Computer Science",
                    "cs": "Computer Science",
                    "Meck": "Mechanical",
                    "MCT": "Mechatronics",
                    "mct": "Mechatronics",
                }
            )

            with col2:
                st.write("After:")
                st.write(df["department"].value_counts())

    if "year" in df.columns:
        with st.expander("Year"):
            col1, col2 = st.columns(2)
            with col1:
                st.write("Before:")
                st.write(df["year"].value_counts())

            df["year"] = df["year"].replace(
                {
                    "First Year": 1,
                    "Second Year": 2,
                    "Third Year": 3,
                    "Fourth Year": 4,
                }
            )
            df["year"] = pd.to_numeric(df["year"], errors="coerce")
            df["year"] = df["year"].where(df["year"].between(1, 4))

            with col2:
                st.write("After:")
                st.write(df["year"].value_counts(dropna=False))

    if "gender" in df.columns:
        with st.expander("Gender"):
            col1, col2 = st.columns(2)
            with col1:
                st.write("Before:")
                st.write(df["gender"].value_counts())

            df["gender"] = df["gender"].replace(
                {
                    "M": "Male",
                    "MALE": "Male",
                    "male": "Male",
                    "F": "Female",
                    "FEMALE": "Female",
                    "female": "Female",
                    "O": "Other",
                    "OTHER": "Other",
                    "other": "Other",
                    "BN": "Non-Binary",
                    "NON-BINARY": "Non-Binary",
                    "non-binary": "Non-Binary",
                }
            )
            df["gender"] = df["gender"].where(
                df["gender"].isin(["Male", "Female", "Other", "Non-Binary"])
            )

            with col2:
                st.write("After:")
                st.write(df["gender"].value_counts(dropna=False))

    if "phone" in df.columns:
        with st.expander("Phone Number"):
            col1, col2 = st.columns(2)

            invalid_before = df[
                ~df["phone"].astype(str).str.match(r"^\d{5}\s\d{5}$", na=False)
            ]
            with col1:
                st.write("Invalid & inconsistent formats (sample):")
                st.write(invalid_before["phone"].head(10))

            df["phone"] = df["phone"].apply(normalize_phone)

            with col2:
                st.write("Normalized samples:")
                st.write(df["phone"].head(10))

    if "gpa" in df.columns:
        with st.expander("GPA"):
            col1, col2 = st.columns(2)
            # Identify invalid
            gpa_numeric = pd.to_numeric(df["gpa"], errors="coerce")
            invalid_gpa = df[
                (df["gpa"].astype(str).str.match(r"^\-\d\.\d$", na=False))
                | (df["gpa"].astype(str).str.match(r"^\d\,\d$", na=False))
                | (~gpa_numeric.between(0, 10))
            ]
            with col1:
                st.write("Invalid (sample):")
                st.write(invalid_gpa["gpa"].head(10))

            df["gpa"] = (
                df["gpa"]
                .astype(str)
                .str.replace(",", ".", regex=False)
                .str.extract(r"([+]?\d*\.?\d+)", expand=False)
            )
            df["gpa"] = pd.to_numeric(df["gpa"], errors="coerce")
            df.loc[~df["gpa"].between(0, 10), "gpa"] = np.nan
            df["gpa"] = df["gpa"].round(2)

            with col2:
                st.write("Sanitized (sample):")
                st.write(df["gpa"].head(10))

    if "attendance_percentage" in df.columns:
        with st.expander("Attendance"):
            col1, col2 = st.columns(2)
            invalid_attendance = df[
                ~pd.to_numeric(df["attendance_percentage"], errors="coerce").between(1, 100)
            ]
            with col1:
                st.write("Invalid (sample):")
                st.write(invalid_attendance["attendance_percentage"].head(5))

            df["attendance_percentage"] = pd.to_numeric(
                df["attendance_percentage"], errors="coerce"
            )
            df["attendance_percentage"] = df["attendance_percentage"].where(
                df["attendance_percentage"].between(1, 100), np.nan
            )
            with col2:
                st.write("After (sample):")
                st.write(df["attendance_percentage"].head(10))

    if "placement_status" in df.columns:
        with st.expander("Placement Status"):
            col1, col2 = st.columns(2)
            with col1:
                st.write("Before:")
                st.write(df["placement_status"].value_counts(dropna=False))
            df["placement_status"] = df["placement_status"].astype(str).str.title()
            with col2:
                st.write("After:")
                st.write(df["placement_status"].value_counts(dropna=False))

    with st.expander("Skills"):
        if "skills" in df.columns:
            col1, col2 = st.columns([1, 2])
            with col1:
                st.write("Raw skills (sample):")
                st.write(df["skills"].head(10))
            df = clean_skills_column(df)
            with col2:
                st.write("Split skills (sample):")
                skill_cols = [f"skill_{i}" for i in range(1, 6)]
                st.write(df[skill_cols].head(10))
        else:
            st.info("No 'skills' column found. Skipping skill split.")

    with st.expander("Null Values"):
        col1, col2 = st.columns(2)
        with col1:
            st.write("Before:")
            st.write(initial_null_df)
        with col2:
            st.write("After:")
            st.write(df.isna().sum())

# =========================
# Tab 3: Cleaned Data + Download
# =========================
with tab3:
    st.subheader("Cleaned Dataset")
    st.dataframe(df.drop(columns=["skills"], errors="ignore"), use_container_width=True)
    st.download_button(
        "⬇️ Download Cleaned CSV",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="student_data_cleaned.csv",
        mime="text/csv",
        use_container_width=True,
    )

# =========================
# Tab 4: Visualization (with Filters)
# =========================
with tab4:
    st.subheader("Interactive Filters")
    filt_col1, filt_col2, filt_col3, filt_col4 = st.columns(4)

    dept_sel = filt_col1.multiselect(
        "Department",
        sorted(df["department"].dropna().unique().tolist()) if "department" in df.columns else [],
        key="viz_dept"
    )
    year_sel = filt_col2.multiselect(
        "Year",
        sorted(df["year"].dropna().unique().tolist()) if "year" in df.columns else [],
        key="viz_year"
    )
    gender_sel = filt_col3.multiselect(
        "Gender",
        sorted(df["gender"].dropna().unique().tolist()) if "gender" in df.columns else [],
        key="viz_gender"
    )
    place_sel = filt_col4.multiselect(
        "Placement Status",
        sorted(df["placement_status"].dropna().unique().tolist()) if "placement_status" in df.columns else [],
        key="viz_place"
    )

    dfv = df.copy()
    if dept_sel and "department" in dfv.columns:
        dfv = dfv[dfv["department"].isin(dept_sel)]
    if year_sel and "year" in dfv.columns:
        dfv = dfv[dfv["year"].isin(year_sel)]
    if gender_sel and "gender" in dfv.columns:
        dfv = dfv[dfv["gender"].isin(gender_sel)]
    if place_sel and "placement_status" in dfv.columns:
        dfv = dfv[dfv["placement_status"].isin(place_sel)]

    st.caption(f"Filtered rows: {len(dfv):,}")

    # Attendance vs GPA (scatter)
    if set(["attendance_percentage", "gpa"]).issubset(dfv.columns):
        st.markdown("### Attendance vs GPA")
        fig1, ax1 = plt.subplots(figsize=(6, 4))
        ax1.scatter(dfv["attendance_percentage"], dfv["gpa"], alpha=0.6)
        ax1.set_xlabel("Attendance (%)")
        ax1.set_ylabel("GPA")
        ax1.grid(alpha=0.2)
        st.pyplot(fig1, use_container_width=False)

    # Placement vs GPA (boxplot)
    if set(["placement_status", "gpa"]).issubset(dfv.columns):
        st.markdown("### GPA by Placement Status (Boxplot)")
        fig2 = boxplot_by_category(dfv.dropna(subset=["gpa"]), "placement_status", "gpa", "GPA by Placement")
        st.pyplot(fig2, use_container_width=False)

    # Gender counts
    if "gender" in dfv.columns:
        st.markdown("### Gender Distribution (Count)")
        st.bar_chart(dfv["gender"].value_counts())

    # Department counts
    if "department" in dfv.columns:
        st.markdown("### Department Distribution (Count)")
        st.bar_chart(dfv["department"].value_counts())

    # Department-wise placement rate
    if set(["department", "placement_status"]).issubset(dfv.columns):
        st.markdown("### Department-wise Placement Rate (%)")
        placement_rate = (
            dfv.groupby("department")["placement_status"]
            .apply(lambda x: (x == "Placed").mean() * 100)
            .reset_index(name="placement_rate")
            .sort_values("placement_rate", ascending=False)
        )
        if not placement_rate.empty:
            st.bar_chart(placement_rate.set_index("department")["placement_rate"])

    # Gender-wise placement rate
    if set(["gender", "placement_status"]).issubset(dfv.columns):
        st.markdown("### Gender-wise Placement Rate (%)")
        placement_rate_gender = (
            dfv.groupby("gender")["placement_status"]
            .apply(lambda x: (x == "Placed").mean() * 100)
            .reset_index(name="placement_rate")
            .sort_values("placement_rate", ascending=False)
        )
        if not placement_rate_gender.empty:
            st.bar_chart(placement_rate_gender.set_index("gender")["placement_rate"])

    # Correlation heatmap (if possible)
    num_cols = [c for c in ["gpa", "attendance_percentage"] if c in dfv.columns]
    if "placement_status" in dfv.columns:
        dfv = dfv.copy()
        dfv["placement_binary"] = (dfv["placement_status"] == "Placed").astype(float)
        num_cols.append("placement_binary")
    if len(num_cols) >= 2:
        st.markdown("### Correlation Heatmap")
        fig_corr = corr_heatmap(dfv.dropna(subset=num_cols), num_cols, "Correlation (key metrics)")
        st.pyplot(fig_corr)

    # Top Skills
    st.markdown("### Top Skills")
    skill_cols = [c for c in dfv.columns if c.startswith("skill_")]
    if skill_cols:
        skills_series = (
            pd.Series(
                [
                    s
                    for s in dfv[skill_cols].values.ravel()
                    if isinstance(s, str) and s.strip()
                ]
            )
            .value_counts()
            .head(15)
        )
        if not skills_series.empty:
            st.bar_chart(skills_series)

# =========================
# Tab 5: AI Insights (Simplified & Fast)
# =========================
with tab5:
    st.subheader("🤖 AI Insights")

    # API Key
    api_key = st.secrets.get("GEMINI_API_KEY_", "") or os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        st.error("🔐 Missing API key. Add GEMINI_API_KEY_ in Streamlit secrets or set env var GEMINI_API_KEY.")
        st.stop()

    genai.configure(api_key=api_key)

    # Build compact summary (cached)
    exclude_cols = ["register_number", "first_name", "last_name", "email", "phone", "skills"]
    compact_summary = build_compact_summary(df, exclude_cols=exclude_cols)

    # Input
    user_question = st.text_input(
        "Ask anything about the dataset",
        placeholder="Example: Which department has the best placement rate?",
        key="ai_question_input"
    )

    if st.button("Generate Insights", use_container_width=True, key="ai_generate_btn"):

        if not user_question.strip():
            st.warning("Please type a question.")
            st.stop()

        with st.spinner("Analyzing dataset…"):

            prompt = f"""
You are a senior data analyst.
Provide the final answer directly.

DATASET SUMMARY:
{compact_summary}

USER QUESTION:
{user_question}

Respond in this fixed format:

### 📌 KEY INSIGHTS
- Bullet points only.
- Highlight patterns, comparisons, or anomalies.

### 📊 VISUAL ANALYSIS
Provide ONE ASCII visual (bar chart, table, distribution blocks, or correlation matrix).

### 🎯 RECOMMENDATIONS
- Short, actionable recommendations based on insights.
"""

            try:
                model = genai.GenerativeModel("gemini-2.0-flash")
                response = model.generate_content(prompt)
                st.markdown(response.text)
            except Exception as e:
                st.error(f"AI Error: {e}")

# =========================
# Tab 6: Student Profile Search
# =========================
with tab6:
    st.subheader("🔍 Student Profile Search")

    reg_input = st.text_input(
        "Enter Register Number",
        placeholder="Example: 22CS081",
        key="profile_reg_input"
    )

    if st.button("Search Student", use_container_width=True, key="profile_search_btn"):
        if not reg_input.strip():
            st.warning("Please enter a valid register number.")
            st.stop()

        if "register_number" not in df.columns:
            st.error("❌ 'register_number' column not found in the dataset.")
            st.stop()

        # Convert both to string for safety
        df["register_number"] = df["register_number"].astype(str)
        reg_str = str(reg_input).strip()

        # Filter record
        student = df[df["register_number"] == reg_str]

        if student.empty:
            st.error("❌ No student found with this register number.")
            st.stop()

        st.success("✅ Student Found")

        record = student.iloc[0]

        # Student Profile Display
        st.markdown("### 🧑‍🎓 Student Details")
        col1, col2 = st.columns(2)

        def safe_get(col):
            return record[col] if col in student.columns else "—"

        with col1:
            st.write(f"**Name:** {safe_get('first_name')} {safe_get('last_name')}")
            st.write(f"**Register Number:** {safe_get('register_number')}")
            st.write(f"**Department:** {safe_get('department')}")
            st.write(f"**Year:** {safe_get('year')}")

        with col2:
            st.write(f"**Gender:** {safe_get('gender')}")
            st.write(f"**Placement Status:** {safe_get('placement_status')}")
            st.write(f"**GPA:** {safe_get('gpa')}")
            st.write(f"**Attendance:** {safe_get('attendance_percentage')}%")

        # Skills
        st.markdown("### 🧩 Skills")
        skill_cols = [c for c in df.columns if c.startswith("skill_")]
        skills = [record[c] for c in skill_cols if (c in student.columns) and pd.notna(record[c]) and str(record[c]).strip()]
        if skills:
            st.write(", ".join(map(str, skills)))
        else:
            st.write("_No skills listed_")

        # Visual Summary
        st.markdown("### 📊 Student Visual Summary")

        b1, b2 = st.columns(2)
        try:
            gpa_val = float(record["gpa"]) if pd.notna(record["gpa"]) else np.nan
        except Exception:
            gpa_val = np.nan
        try:
            att_val = float(record["attendance_percentage"]) if pd.notna(record["attendance_percentage"]) else np.nan
        except Exception:
            att_val = np.nan

        with b1:
            st.metric("📘 GPA", f"{gpa_val if pd.notna(gpa_val) else '—'}")
        with b2:
            st.metric("📅 Attendance (%)", f"{att_val if pd.notna(att_val) else '—'}")

        st.markdown("#### GPA vs Attendance (ASCII Bars)")
        gpa_bar = "█" * int(((gpa_val or 0) / 10) * 20) if pd.notna(gpa_val) else ""
        att_bar = "█" * int(((att_val or 0) / 100) * 20) if pd.notna(att_val) else ""
        st.text(f"GPA        | {gpa_bar}")
        st.text(f"Attendance | {att_bar}")

        # Optional: AI Mini Insight for the student
        st.markdown("### 🤖 AI Mini‑Insight (Student Specific)")
        try:
            api_key2 = st.secrets.get("GEMINI_API_KEY_", "") or os.environ.get("GEMINI_API_KEY", "")
            if not api_key2:
                st.info("Set GEMINI_API_KEY_/GEMINI_API_KEY to enable AI mini‑insights.")
            else:
                genai.configure(api_key=api_key2)
                model = genai.GenerativeModel("gemini-2.0-flash")

                mini_prompt = f"""
Analyze this student's performance and give 3 short insights.

STUDENT DATA:
GPA: {gpa_val}
Attendance: {att_val}
Department: {safe_get('department')}
Placement Status: {safe_get('placement_status')}
Skills: {', '.join(map(str, skills)) if skills else 'None'}

Respond ONLY in bullet points.
"""
                ai_note = model.generate_content(mini_prompt)
                st.markdown(ai_note.text)
        except Exception as e:
            st.error(f"AI system issue: {e}")

# =========================
# Tab 7: At-Risk Student Detector
# =========================
with tab7:
    st.subheader("🚩 At‑Risk Student Detector")

    if not set(["gpa", "attendance_percentage"]).issubset(df.columns):
        st.warning("Missing required columns: 'gpa' and/or 'attendance_percentage'.")
    else:
        col_a, col_b, col_c, col_d = st.columns(4)
        gpa_thresh = col_a.slider("GPA threshold (below → risk)", 0.0, 10.0, 7.0, 0.1, key="risk_gpa_thresh")
        att_thresh = col_b.slider("Attendance threshold % (below → risk)", 0, 100, 75, 1, key="risk_att_thresh")
        min_skills_needed = col_c.slider("Min skills (below → risk)", 0, 5, 2, 1, key="risk_min_skills")
        include_placed = col_d.checkbox("Also flag 'Placed' students", value=False, key="risk_include_placed")

        # Optional filters for detector
        f1, f2, f3 = st.columns(3)
        dept_filter = f1.multiselect(
            "Filter by Department (optional)",
            sorted(df["department"].dropna().unique().tolist()) if "department" in df.columns else [],
            key="risk_dept"
        )
        year_filter = f2.multiselect(
            "Filter by Year (optional)",
            sorted(df["year"].dropna().unique().tolist()) if "year" in df.columns else [],
            key="risk_year"
        )
        gender_filter = f3.multiselect(
            "Filter by Gender (optional)",
            sorted(df["gender"].dropna().unique().tolist()) if "gender" in df.columns else [],
            key="risk_gender"
        )

        dfr = df.copy()

        # Apply filters
        if dept_filter and "department" in dfr.columns:
            dfr = dfr[dfr["department"].isin(dept_filter)]
        if year_filter and "year" in dfr.columns:
            dfr = dfr[dfr["year"].isin(year_filter)]
        if gender_filter and "gender" in dfr.columns:
            dfr = dfr[dfr["gender"].isin(gender_filter)]

        # Prepare skills count per student
        skill_cols = [c for c in dfr.columns if c.startswith("skill_")]
        if skill_cols:
            dfr["skill_count"] = dfr[skill_cols].apply(
                lambda row: sum(1 for x in row if isinstance(x, str) and x.strip()), axis=1
            )
        else:
            dfr["skill_count"] = np.nan

        # Risk rules
        dfr["risk_score"] = 0
        dfr["reason_gpa"] = dfr["gpa"] < gpa_thresh
        dfr["reason_att"] = dfr["attendance_percentage"] < att_thresh
        dfr["reason_skill"] = dfr["skill_count"] < min_skills_needed if not dfr["skill_count"].isna().all() else False

        dfr["risk_score"] += dfr["reason_gpa"].fillna(False).astype(int)
        dfr["risk_score"] += dfr["reason_att"].fillna(False).astype(int)
        if not dfr["skill_count"].isna().all():
            dfr["risk_score"] += dfr["reason_skill"].fillna(False).astype(int)

        if "placement_status" in dfr.columns:
            not_placed = dfr["placement_status"].astype(str).str.title() != "Placed"
            dfr["reason_place"] = not_placed
            if include_placed:
                dfr["risk_score"] += dfr["reason_place"].fillna(False).astype(int)
        else:
            dfr["reason_place"] = False

        # Risk label
        def risk_label(score):
            if pd.isna(score):
                return "Unknown"
            if score >= 3:
                return "High"
            elif score == 2:
                return "Medium"
            elif score <= 1:
                return "Low"
            return "Unknown"

        dfr["risk_level"] = dfr["risk_score"].apply(risk_label)

        # Reasons text
        def build_reasons(row):
            reasons = []
            if row.get("reason_gpa", False): reasons.append(f"GPA<{gpa_thresh}")
            if row.get("reason_att", False): reasons.append(f"Attendance<{att_thresh}%")
            if not pd.isna(row.get("skill_count", np.nan)) and row.get("reason_skill", False):
                reasons.append(f"Skills<{min_skills_needed}")
            if row.get("reason_place", False) and (include_placed or (row.get("placement_status", "") != "Placed")):
                reasons.append("Not placed")
            return ", ".join(reasons) if reasons else "—"

        dfr["risk_reasons"] = dfr.apply(build_reasons, axis=1)

        # Summary
        st.markdown("### Overview")
        counts = dfr["risk_level"].value_counts().reindex(["High", "Medium", "Low"], fill_value=0)
        st.bar_chart(counts)

        # Show table of High & Medium risk
        st.markdown("### Flagged Students")
        show_cols = [c for c in ["register_number", "first_name", "last_name", "department", "year", "gender",
                                 "gpa", "attendance_percentage", "placement_status", "skill_count",
                                 "risk_score", "risk_level", "risk_reasons"] if c in dfr.columns]
        flagged = dfr[dfr["risk_level"].isin(["High", "Medium"])][show_cols].sort_values(
            ["risk_level", "risk_score"], ascending=[True, False]
        )
        st.dataframe(flagged, use_container_width=True)

        # Download
        st.download_button(
            "⬇️ Download Flagged Students CSV",
            data=flagged.to_csv(index=False).encode("utf-8"),
            file_name="students_at_risk.csv",
            mime="text/csv",
            use_container_width=True,
        )

        # Optional AI summary button
        if st.button("Generate AI Summary for At‑Risk Cohort", use_container_width=True, key="risk_ai_btn"):
            api_key = st.secrets.get("GEMINI_API_KEY_", "") or os.environ.get("GEMINI_API_KEY", "")
            if not api_key:
                st.info("Set GEMINI_API_KEY_/GEMINI_API_KEY to enable AI summary.")
            else:
                genai.configure(api_key=api_key)
                # Compact cohort summary
                cohort_summary = {
                    "total": int(len(dfr)),
                    "counts": counts.to_dict(),
                    "gpa_threshold": gpa_thresh,
                    "attendance_threshold": att_thresh,
                    "min_skills": min_skills_needed,
                    "dept": dept_filter,
                    "year": year_filter,
                    "gender": gender_filter,
                }
                import json
                prompt = f"""
You are an academic advisor. Summarize the at-risk cohort and give 5 quick actions.

COHORT SUMMARY (JSON):
{json.dumps(cohort_summary, ensure_ascii=False)}

Respond with:
- 3 bullets: what stands out
- 1 small ASCII bar of High/Medium/Low counts
- 5 actionable recommendations
"""
                try:
                    model = genai.GenerativeModel("gemini-2.0-flash")
                    resp = model.generate_content(prompt)
                    st.markdown(resp.text)
                except Exception as e:
                    st.error(f"AI Error: {e}")

# =========================
# Tab 8: Skill Gap Analyzer
# =========================
with tab8:
    st.subheader("🧠 Skill Gap Analyzer")

    # Preset roles → skills (title-case)
    PRESET_ROLES = {
        "Data Analyst": ["Sql", "Excel", "Power Bi", "Python", "Statistics"],
        "Software Engineer": ["Data Structures", "Algorithms", "Java", "Python", "Oop"],
        "Web Developer": ["Html", "Css", "Javascript", "React", "Node"],
        "Ml Engineer": ["Python", "Numpy", "Pandas", "Scikit-Learn", "Machine Learning"],
        "Devops Engineer": ["Linux", "Docker", "Kubernetes", "Ci/Cd", "Aws"],
        "Embedded Developer": ["C", "C++", "Embedded Systems", "Microcontrollers", "Iot"],
        "Cybersecurity Analyst": ["Networking", "Linux", "Python", "Vulnerability Assessment", "Siem"],
    }

    c1, c2 = st.columns([1, 1])
    role_choice = c1.selectbox("Choose a role (preset)", ["-- Select --"] + list(PRESET_ROLES.keys()), key="gap_role_choice")
    custom_role = c2.text_input("Or enter a custom role (optional)", key="gap_custom_role")
    custom_skills = st.text_input(
        "Custom required skills (comma-separated, overrides preset if filled)",
        placeholder="e.g., Python, SQL, Power BI",
        key="gap_custom_skills"
    )

    # Filters
    f1, f2, f3, f4 = st.columns(4)
    dept_sel = f1.multiselect(
        "Department",
        sorted(df["department"].dropna().unique().tolist()) if "department" in df.columns else [],
        key="gap_dept"
    )
    year_sel = f2.multiselect(
        "Year",
        sorted(df["year"].dropna().unique().tolist()) if "year" in df.columns else [],
        key="gap_year"
    )
    gender_sel = f3.multiselect(
        "Gender",
        sorted(df["gender"].dropna().unique().tolist()) if "gender" in df.columns else [],
        key="gap_gender"
    )
    place_sel = f4.multiselect(
        "Placement Status",
        sorted(df["placement_status"].dropna().unique().tolist()) if "placement_status" in df.columns else [],
        key="gap_place"
    )

    dfs = df.copy()
    if dept_sel and "department" in dfs.columns:
        dfs = dfs[dfs["department"].isin(dept_sel)]
    if year_sel and "year" in dfs.columns:
        dfs = dfs[dfs["year"].isin(year_sel)]
    if gender_sel and "gender" in dfs.columns:
        dfs = dfs[dfs["gender"].isin(gender_sel)]
    if place_sel and "placement_status" in dfs.columns:
        dfs = dfs[dfs["placement_status"].isin(place_sel)]

    # Determine required skills
    if custom_skills.strip():
        req_skills = normalize_skill_list([s for s in custom_skills.split(",")])
        role_used = custom_role if custom_role.strip() else role_choice if role_choice != "-- Select --" else "Custom Role"
    elif role_choice != "-- Select --":
        req_skills = normalize_skill_list(PRESET_ROLES[role_choice])
        role_used = role_choice
    else:
        req_skills = []
        role_used = None

    st.markdown("### Required Skills")
    if req_skills:
        st.write(", ".join(req_skills))
    else:
        st.info("Select a preset role or provide custom required skills.")

    min_match = st.slider(
        "Minimum matched skills to include",
        0, 10, max(1, min(3, len(req_skills))) if req_skills else 1, 1,
        key="gap_min_match"
    )

    if req_skills:
        # Build per-student skills set
        skills_cols = [c for c in dfs.columns if c.startswith("skill_")]
        if not skills_cols:
            st.warning("No parsed skill columns (skill_1..skill_5) found. Ensure 'Skills' step in Cleaning tab was executed.")
        else:
            # Compute matches
            def compute_match(row):
                student_sk = extract_student_skills_row(row)
                match = [s for s in req_skills if s in student_sk]
                missing = [s for s in req_skills if s not in student_sk]
                return pd.Series({"matched_count": len(match), "matched_skills": ", ".join(match), "missing_skills": ", ".join(missing)})

            matches = dfs.apply(compute_match, axis=1)
            result = pd.concat([dfs, matches], axis=1)
            result = result[result["matched_count"] >= min_match].copy()

            # Display top candidates
            st.markdown(f"### Candidates for **{role_used}**")
            show_cols = [c for c in ["register_number", "first_name", "last_name", "department", "year", "gender",
                                     "placement_status", "gpa", "attendance_percentage",
                                     "matched_count", "matched_skills", "missing_skills"] if c in result.columns]
            result_sorted = result.sort_values(["matched_count", "gpa", "attendance_percentage"], ascending=[False, False, False])
            st.dataframe(result_sorted[show_cols], use_container_width=True, height=400)

            # Coverage distribution
            st.markdown("### Coverage Distribution (Matched Skills Count)")
            cov = result_sorted["matched_count"].value_counts().sort_index()
            if not cov.empty:
                st.bar_chart(cov)

            # Top missing skills overall (from all students considered)
            st.markdown("### Most Missing Required Skills (Across Filtered Students)")
            all_matches = dfs.apply(compute_match, axis=1)
            missing_all = pd.Series(
                [s for row in all_matches["missing_skills"].tolist() for s in (row.split(", ") if isinstance(row, str) and row else [])]
            )
            if not missing_all.empty:
                st.bar_chart(missing_all.value_counts().head(15))

            # Download
            st.download_button(
                "⬇️ Download Matching Students CSV",
                data=result_sorted[show_cols].to_csv(index=False).encode("utf-8"),
                file_name=f"skill_matches_{(role_used or 'role').replace(' ', '_').lower()}.csv",
                mime="text/csv",
                use_container_width=True,
            )

            # Optional AI summary
            if st.button("Generate AI Summary for Skill Gap", use_container_width=True, key="gap_ai_btn"):
                api_key = st.secrets.get("GEMINI_API_KEY_", "") or os.environ.get("GEMINI_API_KEY", "")
                if not api_key:
                    st.info("Set GEMINI_API_KEY_/GEMINI_API_KEY to enable AI summary.")
                else:
                    genai.configure(api_key=api_key)
                    # Build tight summary to keep tokens small
                    summary = {
                        "role": role_used,
                        "required_skills": req_skills,
                        "n_candidates": int(len(result_sorted)),
                        "coverage_counts": cov.to_dict() if not cov.empty else {},
                        "top_missing": missing_all.value_counts().head(10).to_dict() if not missing_all.empty else {},
                    }
                    import json
                    prompt = f"""
You are a placement advisor. Given the role and coverage, write:
- 3 bullets: where students are strong
- 3 bullets: key gaps to fix
- 5 bullets: concrete training actions
Add one ASCII bar showing coverage of matched skills (use counts).

SUMMARY (JSON):
{json.dumps(summary, ensure_ascii=False)}
"""
                    try:
                        model = genai.GenerativeModel("gemini-2.0-flash")
                        resp = model.generate_content(prompt)
                        st.markdown(resp.text)
                    except Exception as e:
                        st.error(f"AI Error: {e}")
    else:
        st.info("Pick a role or enter custom skills to run the analyzer.")

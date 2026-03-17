import pandas as pd
import numpy as np
import streamlit as st
import re
import matplotlib.pyplot as plt
import google.generativeai as genai

tab1, tab2, tab3, tab4, tab5, tab6 , tab7 = st.tabs(["Initial Data", "Data Cleaning", "Cleaned Data", "Visualization","Student profile","Filter Students","AI Insights"])
df = pd.read_csv('student_data.csv')

with tab1:
    st.dataframe(df)
    
with tab2:
    
    initial_null_df = df.isna().sum()
    
    df.drop_duplicates(keep='first',inplace=True)
    # the file has 75 duplicate data of students- same data repeated. used drop_duplicates() to remove those data
    
    
    with st.expander("Email"):
        # email
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("Email column before cleaning:")
            invalid_email_before = df[
                ~df["email"].astype(str).str.match(r'^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$', na=False)
            ]
    
            st.write(invalid_email_before['email'].head(5))

        #   checking for invalid emails and replacing ithh null value
        
        email_pattern = r'^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$'
        df['email'] = df['email'].apply(
            lambda x: x if isinstance(x, str) and re.match(email_pattern, x) else np.nan
        )

         #including College mail id for the null mail columns
        clg_email = df['register_number'] + "@skcet.ac.in"
        df.fillna({'email':clg_email}, inplace = True )
        
        with col2:
            st.write("Email column after cleaning:")
            st.write(df.loc[[7,19,25,32,41], "email"])    
    

    with st.expander("Department"):
        # st.subheader("Department")
        col1, col2 = st.columns(2)
        with col1:
            st.write(df['department'].value_counts())

        df['department'] = df['department'].replace({
            "IT":"Information Technology",
            "It":"Information Technology",
            "it":"Information Technology",
            "CSE": "Computer Science",
            "cs":"Computer Science",
            "Meck": "Mechanical",
            "MCT": "Mechatronics",
            "mct": "Mechatronics",
        })
        with col2:
            st.write(df['department'].value_counts())
    
    with st.expander("Year"):
        # st.subheader('Year')
        col1, col2 = st.columns(2)
        with col1:
            st.write(df['year'].value_counts())
        
        df['year'] = df['year'].replace({
        "First Year": 1,
        "Second Year": 2,
        "Third Year": 3,
        "Fourth Year": 4,
        })
        
        df["year"] = pd.to_numeric(df["year"], errors="coerce")
        df["year"]= df["year"].where(df["year"].between(1, 4))
        with col2:
            st.write(df['year'].value_counts())
    
    with st.expander("Gender"):
        # st.subheader("Gender")
        col1, col2 = st.columns(2)
        with col1:
            st.write(df['gender'].value_counts())
        df['gender'] = df['gender'].replace({
            "M": "Male",
            "MALE" : "Male",
            "male" : "Male",
            "F": "Female",
            "FEMALE":"Female",
            "female":"Female",
            "O": "Other",
            "OTHER": "Other",
            "other": "Other",
            "BN": "Non-Binary",
            "NON-BINARY": "Non-Binary",
            "non-binary": "Non-Binary"
        })
            # also can use .title() to convert to Title / Capitalize case format
        
        df["gender"] = df["gender"].where(df["gender"].isin(["Male", "Female", "Other", "Non-Binary"]))
        #where() -> f.where (cond, other=np.nan, inplace=False)
        
        with col2:
            st.write(df['gender'].value_counts())
    
    
    with st.expander("Phone Number"):
        col1, col2 = st.columns(2)
        invalid_phone_before = df[
            ~df["phone"].astype(str).str.match(r'^\d{5}\s\d{5}$', na=False)]
        
        with col1:
            st.write("Sample Invalid & Inconsistent Phone Formats (Before Cleaning):")
            st.write(invalid_phone_before["phone"].iloc[40:50])

        def clean_phone(phone):
        
            if pd.isna(phone):
                return np.nan

            phone = str(phone)

            # to replace format: 469-465-1419 to 46946 51419
            if re.match(r'^\d{3}-\d{3}-\d{4}$', phone):
                digits = phone.replace("-", "")
                return digits[:5] + " " + digits[5:]

            # to replace format: 87998-87116 to 87998 87116
            elif re.match(r'^\d{5}-\d{5}$', phone):
                digits = phone.replace("-", " ")
                return digits

            else:
                return np.nan
        df["phone"] = df["phone"].apply(clean_phone)
        
        with col2:
            st.write("\nSample Cleaned Phone Numbers (After Cleaning):")
            st.write(df["phone"].iloc[40:50])
    
    


    with st.expander("GPA"):
        col1, col2 = st.columns(2)
        gpa_numeric = pd.to_numeric(df["gpa"], errors="coerce")
        invalid_gpa = df[
            (df["gpa"].astype(str).str.match(r'^\-\d\.\d$', na=False)) |
            (df["gpa"].astype(str).str.match(r'^\d\,\d$', na=False)) |
            (~gpa_numeric.between(1,10))
            
        ]
        with col1:
            st.write(invalid_gpa['gpa'].head(10))
        
        df["gpa"] = (
            df["gpa"]
            .astype(str)
            .str.replace(",", ".", regex=False) # "7,8" -> "7.8"
            .str.extract(r"([+]?\d*\.?\d+)", expand=False) # pull numeric token if mixed text
        )
        df["gpa"] = pd.to_numeric(df["gpa"], errors="coerce")

        # Invalidate out-of-range values (keep only 0..10)
        df.loc[~df["gpa"].between (0, 10), "gpa"] = np.nan

        # Optional: round for consistency
        df["gpa"] = df["gpa"].round(2)

        with col2:
            st.write(df['gpa'].iloc[[5,6,7,16,25,36,54,56,66,95]])


    with st.expander("Attendance"):
        col1, col2 = st.columns(2)
        invalid_attendance = df[~df["attendance_percentage"].between(1,100)]
        with col1:
            st.write(invalid_attendance['attendance_percentage'].head(5))
        # st.write(invalid_attendance['attendance_percentage'].tail(5))
        
        df['attendance_percentage'] = pd.to_numeric(df['attendance_percentage'], errors='coerce')
        df['attendance_percentage'] = df['attendance_percentage'].where(df['attendance_percentage'].between (1, 100), other=np.nan)

        with col2:
            st.write(df['attendance_percentage'].iloc[[0,129,130,131,5]])
        # st.write(df['attendance_percentage'].tail(5))

    with st.expander("Placement Status"):
        col1, col2 = st.columns(2)
        #Before
        with col1:
            st.write(df["placement_status"].value_counts())
            
        # Standardizing to common case. Example placed, PLACED -> Placed  
        df["placement_status"] = df["placement_status"].str.title()
        
        #After
        with col2:
            st.write(df["placement_status"].value_counts())

    with st.expander('Skills'):
        # Splitting skill column using .split(",") seperated by comma.
        col1, col2 = st.columns([1,2])
        with col1:
            st.write(df["skills"])
        df[['skill_1','skill_2','skill_3','skill_4','skill_5']] = df['skills'].str.split(",", expand=True)
        with col2:
            st.write(df[["skill_1","skill_2","skill_3","skill_4","skill_5",]])


    print(df)
    
    with st.expander("Null Values"):
        col1, col2 = st.columns(2)
        with col1:
            st.write(initial_null_df)
        with col2:
            st.write(df.isna().sum())
    
with tab3:
    st.write(df.drop("skills",axis=1))
    # del df['skills']
    # st.dataframe(df)
    
    
    
with tab4:
    df_clean = df.dropna()   # drop all null rows, null values affect some visualization ...
    
    
    st.subheader("Attendance vs GPA")
    st.write("To analyze that the students with higher attendance have better GPA than others.")
    
    plt.scatter(df["attendance_percentage"], df["gpa"])
    plt.xlabel("Attendance")
    plt.ylabel("GPA")
    st.pyplot(plt)
    
    st.subheader("Placement Status vs GPA")
    st.write("To analyze that the students with higher GPA have placement.")
    
    plt.scatter(df["placement_status"], df["gpa"])
    plt.xlabel("Placement status")
    plt.ylabel("GPA")
    st.pyplot(plt)
    
    # st.scatter_chart(
    # df,
    # x="placement_status",
    # y="gpa",
    # color="",
    # size="",)

    st.subheader("Gender")
    st.write("Bar chart representing the count of students by gender.")
    st.bar_chart(df["gender"].value_counts())
    
    
    st.subheader("Department")
    st.write("Bar chart representing the count of students by department.")
    st.bar_chart(df["department"].value_counts())
    
    st.subheader("Department wise placement rate")
    st.write("")
    placement_rate = (
        df.groupby("department")['placement_status']
        .apply(lambda x: (x == 'Placed'). mean()*100)
        .reset_index(name= 'placement_rate'))
    
    st.bar_chart(placement_rate.set_index("department"))

    #st.bar_chart(data=df, x="placement_rate", y="department")
    
    st.subheader("Gender wise placement rate")
    st.write("")
    placement_rate_gender = (
        df.groupby("gender")['placement_status']
        .apply(lambda x: (x == 'Placed').mean()*100)
        .reset_index(name= 'placement_rate_gender'))
    
    st.line_chart(placement_rate_gender.set_index("gender"))
    
    
    
with tab7:
    # genai.configure(api_key=GEMINI_API_KEY)
    api_key1 = st.secrets.get("GEMINI_API_KEY_", "") 
    genai.configure(api_key=api_key1)
    model = genai.GenerativeModel("gemini-2.5-flash")
    
    # for m in genai.list_models():
    #     if "generateContent" in m.supported_generation_methods:
    #         print(m.name)
    
    exclude_cols = ["register_number","first_name","last_name","email","phone"]
    df_summary = df.drop(columns=exclude_cols)
    summary = df_summary.to_string()
    # st.write(summary)
    
    user_question = st.text_input("Ask questions about the dataset or request insights")
    
    if user_question:
        prompt = f"""
You are an advanced data analyst.

Your task:
Analyze the dataset summarized below and answer the user's question using clear, visual, and insight‑driven outputs.

Dataset Summary:
{summary}

User Question:
{user_question}

Your Response Must Include:

1. **Key Insights**
   - Present findings in a concise, insight‑oriented format.
   - Prioritize visuals over long theory.
   - Use charts, tables, or bullet points to communicate insights clearly.
   - Examples of visuals you may use:
       - Bar/line charts (ASCII or text‑friendly if needed)
       - Trend tables
       - Correlation matrices
       - Sparkline‑style visualizations

2. **Visual Analysis**
   - Add at least one visual element (chart/table/matrix) to support the insights.
   - Ensure visuals are simple and interpretable in plain text.

3. **Recommendations**
   - Provide data‑driven, actionable recommendations.
   - Keep them concise and tied to observed patterns.

Guidelines:
- Avoid lengthy theoretical explanations.
- Focus on patterns, anomalies, trends, comparisons, and actionable interpretations.
- Maintain a professional, analytical tone.
        """

        # - Skill distribution summary
        # - Skill gaps

        response = model.generate_content(prompt)

        st.subheader("GenAI Insights")
        st.write(response.text)


        # ASCII bars
        st.text(f"GPA: {'█' * int(rec['gpa']*2)}")
        st.text(f"Attendance: {'█' * int(rec['attendance_percentage']/5)}")





with tab5:
    st.subheader("🔍 Student Profile Search")

    reg_input = st.text_input(
        "Enter Register Number",
        placeholder="Example: 22CS081",
        key="profile_reg"
    )

    if st.button("Search Student", use_container_width=True, key="profile_btn"):
        df["register_number"] = df["register_number"].astype(str)

        if not reg_input.strip():
            st.warning("Please enter a valid register number.")
            st.stop()

        student = df[df["register_number"] == reg_input.strip()]

        if student.empty:
            st.error("❌ Student not found.")
            st.stop()

        st.success("✅ Student Found")
        rec = student.iloc[0]

        # Student info
        st.markdown("### 🧑‍🎓 Student Details")
        col1, col2 = st.columns(2)

        with col1:
            st.write(f"**Name:** {rec['first_name']} {rec['last_name']}")
            st.write(f"**Register Number:** {rec['register_number']}")
            st.write(f"**Department:** {rec['department']}")
            st.write(f"**Year:** {rec['year']}")

        with col2:
            st.write(f"**Gender:** {rec['gender']}")
            st.write(f"**Placement Status:** {rec['placement_status']}")
            st.write(f"**GPA:** {rec['gpa']}")
            st.write(f"**Attendance:** {rec['attendance_percentage']}%")

        # Skills
        st.markdown("### 🧩 Skills")
        skill_cols = [c for c in df.columns if c.startswith("skill_")]
        skills = [rec[c] for c in skill_cols if pd.notna(rec[c]) and str(rec[c]).strip()]

        if skills:
            st.success(", ".join(skills))
        else:
            st.info("No skills listed")

        # ASCII Visual Summary
        st.markdown("### 📊 Performance Summary (ASCII Bars)")
        gpa_bar = "█" * int((rec["gpa"] / 10) * 20) if pd.notna(rec["gpa"]) else ""
        att_bar = "█" * int((rec["attendance_percentage"] / 100) * 20) if pd.notna(rec["attendance_percentage"]) else ""

        st.text(f"GPA        | {gpa_bar}")
        st.text(f"Attendance | {att_bar}")




with tab6:
   
    st.subheader("🚩 Filter Students")
    st.write(" Filter students by GPA, Attendance and skills, To analyze their career insights ")
    # Risk thresholds

    # ---- Filters (Department & Year)
    fcol1, fcol2 = st.columns(2)
    dept_options = sorted(df["department"].dropna().unique().tolist()) if "department" in df.columns else []
    year_options = sorted(pd.to_numeric(df["year"], errors="coerce").dropna().unique().astype(int).tolist()) if "year" in df.columns else []

    dept_filter = fcol1.multiselect(
        "Filter by Department (optional)",
        options=dept_options,
        key="risk_dept_filter"
    )
    year_filter = fcol2.multiselect(
        "Filter by Year (optional)",
        options=year_options,
        key="risk_year_filter"
    )

    # Apply filters to a working copy
    df_risk = df.copy()
    if dept_filter and "department" in df_risk.columns:
        df_risk = df_risk[df_risk["department"].isin(dept_filter)]
    if year_filter and "year" in df_risk.columns:
        # coerce to numeric for robust matching
        df_risk["year"] = pd.to_numeric(df_risk["year"], errors="coerce")
        df_risk = df_risk[df_risk["year"].isin(year_filter)]

    st.caption(f"Filtered rows: {len(df_risk):,}")

    # ---- Risk thresholds
    col_a, col_b, col_c = st.columns(3)
    gpa_th = col_a.slider("Flag if GPA is below", 0.0, 10.0, 7.0, key="risk_gpa")
    att_th = col_b.slider("Flag if Attendance (%) is below", 0, 100, 75, key="risk_att")
    min_skill = col_c.slider("Minimum number of skills required", 0, 5, 2, key="risk_skill")

    # ---- Skill count (skill_1..skill_5 if present)
    skill_cols = [c for c in df_risk.columns if c.startswith("skill_")]
    if skill_cols:
        df_risk["skill_count"] = df_risk[skill_cols].apply(
            lambda row: sum(
                1 for c in skill_cols
                if pd.notna(row[c]) and str(row[c]).strip() != ""
            ),
            axis=1
        )
    else:
        # If skills not parsed, assume 0 (so rule can still work)
        df_risk["skill_count"] = 0

    # ---- Ensure numeric types for comparisons
    if "gpa" in df_risk.columns:
        df_risk["gpa"] = pd.to_numeric(df_risk["gpa"], errors="coerce")
    if "attendance_percentage" in df_risk.columns:
        df_risk["attendance_percentage"] = pd.to_numeric(df_risk["attendance_percentage"], errors="coerce")

    # ---- Risk rules
    # Safely handle missing columns
    gpa_cond = df_risk["gpa"] < gpa_th if "gpa" in df_risk.columns else False
    att_cond = df_risk["attendance_percentage"] < att_th if "attendance_percentage" in df_risk.columns else False
    skill_cond = df_risk["skill_count"] < min_skill

    df_risk["risky"] = (gpa_cond.fillna(False)) | (att_cond.fillna(False)) | (skill_cond.fillna(False))

    # ---- Summary & Table
    st.markdown("### Summary")
    if df_risk["risky"].notna().any():
        st.bar_chart(df_risk["risky"].value_counts())
    else:
        st.info("No records to summarize with the current filters.")

    st.markdown("### Students Identified ")
    show_cols = [c for c in [
        "register_number", "first_name", "last_name",
        "department", "year", "gender",
        "gpa", "attendance_percentage", "skill_count"
    ] if c in df_risk.columns]

    at_risk = df_risk[df_risk["risky"] == True][show_cols].copy()

    # Sort for readability (by severity proxies)
    sort_cols = [c for c in ["gpa", "attendance_percentage", "skill_count"] if c in at_risk.columns]
    if sort_cols:
        at_risk = at_risk.sort_values(by=sort_cols, ascending=[True if c != "skill_count" else True for c in sort_cols])

    st.dataframe(at_risk, use_container_width=True)

    st.download_button(
        "⬇ Download filtered Students CSV",
        at_risk.to_csv(index=False).encode("utf-8"),
        file_name="at_risk_students.csv",
        use_container_width=True
    )

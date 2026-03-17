import pandas as pd
import numpy as np
import streamlit as st
import re
import matplotlib.pyplot as plt
import google.generativeai as genai

tab1, tab2, tab3, tab4, tab5 = st.tabs(["Initial Data", "Data Cleaning", "Cleaned Data", "Visualization","AI Insights"])
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
    
    
    
with tab5:
    from dotenv import load_dotenv
    import os
    
    load_dotenv("secrets.env")
    api_key1 = os.getenv("GEMINI_API_KEY")
    # Gen AI Insights using GEMINI AI LLM
    
    # genai.configure(api_key=GEMINI_API_KEY)
    genai.configure(api_key="AIzaSyCwKNpqAxt9dO4U9gvyEmJCQENfaqEMeWI")
    model = genai.GenerativeModel("gemini-2.5-flash-tts")
    
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
        You are a data analyst.
        The response should not more theoratical paras, visualizations are better.
        Here is a dataset summary:
        {summary}
        Based on this dataset answer the user's question and generate insights.
        User Question:
        {user_question}
        Provide:
        - Key insights
        
        - Recommendations"""

        # - Skill distribution summary
        # - Skill gaps

        response = model.generate_content(prompt)

        st.subheader("GenAI Insights")
        st.write(response.text)
    

# app.py
import streamlit as st
import pandas as pd
import io
import zipfile
from engine import parse_ddl_with_regex, generate_table_rows, push_dataframe_to_postgres
from query_engine import run_security_guardrail, generate_sql_query, execute_sql_safely, generate_natural_response, generate_seaborn_chart

st.set_page_config(layout="wide", page_title="Synthetic Engine Studio & Analytics")

if "generated_data" not in st.session_state:
    st.session_state.generated_data = {}
if "blueprint" not in st.session_state:
    st.session_state.blueprint = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

st.sidebar.title("🎛️ Navigation")
app_mode = st.sidebar.radio("Go to:", ["Data Generation", "Talk to your Data"])

# --- DATA GENERATION VIEW ---
if app_mode == "Data Generation":
    st.title("🧬 Local Synthetic Data Generator")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("1. Configuration")
        uploaded_file = st.file_uploader("Upload DDL Schema (.sql, .txt, .ddl)", type=["sql", "txt", "ddl"])
        user_prompt = st.text_area("Extra Generation Instructions")
        
        generation_temp = st.slider("Temperature (Creativity)", min_value=0.0, max_value=1.0, value=0.2, step=0.1)
        row_count = st.number_input("Rows per table", min_value=5, max_value=200, value=15)
        generate_btn = st.button("Generate Data", type="primary")

    with col2:
        st.subheader("2. Previews & Actions")
        
        if generate_btn and uploaded_file is not None:
            ddl_content = uploaded_file.read().decode("utf-8")
            
            with st.spinner("Analyzing schema structures via Regex Parser..."):
                blueprint = parse_ddl_with_regex(ddl_content)
                st.session_state.blueprint = blueprint
            
            generated_dfs = {}
            parent_pk_registry = {}
            status_container = st.container()
            
            for table_name in blueprint["execution_order"]:
                status_container.write(f"⏳ Streaming data records for `{table_name}`...")
                table_rule = blueprint["tables"][table_name]
                
                raw_rows = generate_table_rows(table_rule, parent_pk_registry, batch_size=row_count, user_guidelines=user_prompt)
                df = pd.DataFrame(raw_rows)
                generated_dfs[table_name] = df
                
                pk_col = next((c["name"] for c in table_rule["columns"] if c["is_primary_key"]), None)
                if pk_col and pk_col in df.columns:
                    parent_pk_registry[table_name] = df[pk_col].tolist()
                    
                # PERSISTENCE REQUIREMENT: Automatically write rows right down into live Postgres instance
                push_dataframe_to_postgres(table_name, df)
                        
            st.session_state.generated_data = generated_dfs
            st.success("🎉 All dataset table structures successfully created and persisted to database!")

        if st.session_state.generated_data:
            tabs = st.tabs(list(st.session_state.generated_data.keys()))
            for tab, t_name in zip(tabs, list(st.session_state.generated_data.keys())):
                with tab:
                    st.dataframe(st.session_state.generated_data[t_name], use_container_width=True)

# --- TALK TO YOUR DATA VIEW ---
elif app_mode == "Talk to your Data":
    st.title("💬 Talk to your Data (Conversational AI)")
    
    if not st.session_state.blueprint:
        st.warning("⚠️ Please configure and run data generation first to seed the context blueprints.")
    else:
        # Display existing chat log arrays
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.write(message["content"])
                if "sql" in message:
                    st.code(message["sql"], language="sql")
                if "data" in message:
                    st.dataframe(message["data"], use_container_width=True)
                if "image" in message and message["image"] is not None:
                    st.image(message["image"])

        if user_query := st.chat_input("Ask a question about your schema data:"):
            with st.chat_message("user"):
                st.write(user_query)
            st.session_state.chat_history.append({"role": "user", "content": user_query})
            
            with st.chat_message("assistant"):
                # GUARDRAIL REQUIREMENT: Validate context safety boundaries before running generation
                if not run_security_guardrail(user_query):
                    security_warning = "⚠️ Safety Alert: The engine has intercepted a potential prompt injection or out-of-scope query request."
                    st.error(security_warning)
                    st.session_state.chat_history.append({"role": "assistant", "content": security_warning})
                else:
                    with st.spinner("Processing text to operational SQL queries..."):
                        try:
                            compiled_sql = generate_sql_query(user_query, st.session_state.blueprint["tables"])
                            st.code(compiled_sql, language="sql")
                            
                            df_result = execute_sql_safely(compiled_sql)
                            st.dataframe(df_result, use_container_width=True)
                            
                            ai_summary = generate_natural_response(user_query, df_result, compiled_sql)
                            st.write(ai_summary)
                            
                            # SEABORN REQUIREMENT: Automatically plot chart figures if visual words exist
                            img_stream = None
                            if any(w in user_query.lower() for w in ["chart", "plot", "graph", "bar"]):
                                img_stream = generate_seaborn_chart(df_result, user_query)
                                if img_stream:
                                    st.image(img_stream)
                            
                            st.session_state.chat_history.append({
                                "role": "assistant", "content": ai_summary, "sql": compiled_sql, "data": df_result, "image": img_stream
                            })
                        except Exception as e:
                            st.error(f"Execution handling failure: {str(e)}")
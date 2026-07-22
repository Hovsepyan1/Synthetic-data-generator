# engine.py
import re
import json
from typing import List, Dict, Any
from google import genai
from google.genai import types
from sqlalchemy import create_engine
from langfuse import observe 
from config import settings


client = genai.Client(
    vertexai=True,
    project='gd-gcp-gridu-genai'
)

db_engine = create_engine(settings.db_url)

def clean_and_split_ddl(inside_brackets: str) -> List[str]:
    balance = 0
    definitions = []
    current_part = []
    for char in inside_brackets:
        if char == '(': balance += 1
        elif char == ')': balance -= 1
        if char == ',' and balance == 0:
            definitions.append("".join(current_part).strip())
            current_part = []
        else:
            current_part.append(char)
    if current_part:
        definitions.append("".join(current_part).strip())
    return [d for d in definitions if d]

def parse_ddl_with_regex(ddl_text: str) -> Dict[str, Any]:
    schema_map = {}
    clean_ddl = re.sub(r'--.*?\n', '\n', ddl_text)
    table_blocks = re.findall(r'CREATE\s+TABLE\s+(\w+)\s*\((.*?)\)\s*;\s*', clean_ddl, re.IGNORECASE | re.DOTALL)
    
    for table_name, inside_brackets in table_blocks:
        definitions = clean_and_split_ddl(inside_brackets)
        columns = []
        dependencies = []
        
        for item in definitions:
            item = " ".join(item.split())
            if not item: continue
                
            fk_match = re.search(r'FOREIGN\s+KEY\s*\((.*?)\)\s*REFERENCES\s*(\w+)\s*\((.*?)\)', item, re.IGNORECASE)
            if fk_match:
                fk_col = fk_match.group(1).strip().replace('"', '').replace('`', '').replace(' ', '')
                ref_table = fk_match.group(2).strip()
                if ref_table != table_name and ref_table not in dependencies:
                    dependencies.append(ref_table)
                for col in columns:
                    if col["name"] == fk_col:
                        col["is_foreign_key"] = True
                        col["reference_table"] = ref_table
                continue

            col_parts = item.split()
            if len(col_parts) >= 2:
                col_name = col_parts[0].replace('"', '').replace('`', '')
                data_type = col_parts[1]
                if col_name.upper() in ["CONSTRAINT", "PRIMARY", "FOREIGN", "KEY"]: continue
                is_pk = "PRIMARY" in item.upper() and "KEY" in item.upper()
                columns.append({
                    "name": col_name, "data_type": data_type, "is_primary_key": is_pk, "is_foreign_key": False, "reference_table": None
                })
                
        schema_map[table_name] = {"table_name": table_name, "dependencies": dependencies, "columns": columns}
        
    alter_matches = re.findall(r'ALTER\s+TABLE\s+(\w+)\s+ADD\s+(?:CONSTRAINT\s+\w+\s+)?FOREIGN\s+KEY\s*\((.*?)\)\s*REFERENCES\s*(\w+)', clean_ddl, re.IGNORECASE)
    for target_table, fk_col, ref_table in alter_matches:
        target_table, ref_table, fk_col = target_table.strip(), ref_table.strip(), fk_col.strip().replace(' ', '')
        if target_table in schema_map and ref_table != target_table:
            if ref_table not in schema_map[target_table]["dependencies"]:
                schema_map[target_table]["dependencies"].append(ref_table)
            for col in schema_map[target_table]["columns"]:
                if col["name"] == fk_col:
                    col["is_foreign_key"] = True
                    col["reference_table"] = ref_table

    return {"execution_order": compute_topological_sort(schema_map), "tables": schema_map}

def compute_topological_sort(schema_map: Dict[str, Any]) -> List[str]:
    order = []
    visited, temp_visited = set(), set()
    def visit(node):
        if node in temp_visited: return
        if node not in visited:
            temp_visited.add(node)
            if node in schema_map:
                for dep in schema_map[node]["dependencies"]:
                    if dep in schema_map: visit(dep)
            temp_visited.remove(node)
            visited.add(node)
            order.insert(0, node) # Top sort correction: Parents land first
    for table in schema_map:
        if table not in visited: visit(table)
    return order[::-1]

@observe(name="Phase1_Data_Generation")
def generate_table_rows(table_rule: Dict[str, Any], parent_data: dict, batch_size: int, user_guidelines: str = "", feedback: str = "") -> List[dict]:
    feedback_str = f"\nUser feedback modification: {feedback}" if feedback else ""
    guidelines_str = f"\nCustom instructions: {user_guidelines}" if user_guidelines else ""
    
    prompt = f"Generate JSON data for {table_rule['table_name']}.\nSchema:\n{json.dumps(table_rule['columns'])}{guidelines_str}{feedback_str}\nParent context keys available: {json.dumps(parent_data)}\nReturn a raw array of objects matching columns directly."
    
    # Trace Generation execution via Langfuse
    
    response = client.models.generate_content(
        model='gemini-3.5-flash',
        contents=prompt,
        config=types.GenerateContentConfig(temperature=0.7, response_mime_type="application/json")
    )
    
    clean_text = response.text.strip().lstrip("```json").rstrip("```").strip()
    return json.loads(clean_text)

def push_dataframe_to_postgres(table_name: str, df: Any):
    """Pushes generated data directly to PostgreSQL tables to satisfy system persistence rules."""
    df.to_sql(table_name.lower(), con=db_engine, if_exists='replace', index=False)
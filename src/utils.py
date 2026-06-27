import pandas as pd
import db_connections as dbc
from concurrent.futures import ThreadPoolExecutor
import re
import unicodedata


# ---- Function to execute one SQL query ----

def run_query(q, dbl):

    # Run de query against an Oracle DB
    df = dbc.oracle2pd('ecap', q)

    # Return a dataframe when data else None
    if df is not None and not df.empty:
        df['dbl'] = dbl
        return df

    return None


# ---- Function to execute many SQL queries in parallel ----

def extract_data_parallel(q_ini, dblinks, user_chunks=None, max_workers=6):

    # Will store all queries to execute
    tasks = []

    # Build all SQL queries first
    for dbl in dblinks:

        if user_chunks is None:

            q = q_ini.format(dbl=dbl)       # replace dbl for dbl in SQL
            tasks.append((q, dbl))          # add query

        else:

            for ids_sql in user_chunks:     # user_chunks are the CTE generated for chunks of patients

                q = q_ini.format(
                    dbl=dbl,                # replace dbl for dbl in SQL
                    usuaris=ids_sql         # replace usuaris for the chunk of patients
                )

                tasks.append((q, dbl))  # add query

    dfs = []

    # Run queries in parallel
    with ThreadPoolExecutor(max_workers=max_workers) as executor:       # max_workers = queries to run at the same time
                                                                        # with: start the thread pool and clean it up when done                
        for df in executor.map(lambda x: run_query(*x), tasks):         # x = task = (q, dbl) * unpaks the tuple
                                                                        # executor: multiple tasks executed at the same time
            # Collect results when data is obtained                     # each df is a result of run_query
            if df is not None:
                dfs.append(df)

    # If list of dfs is not empty merge all dataframes in a single final dataframe
    if dfs:
        df = pd.concat(dfs, ignore_index=True)
        df.columns = df.columns.str.lower()
        return df

    return pd.DataFrame()


# ---- Function to convert a Dataframe into a temporary SQL table ----

def build_patient_window_sql(df, cip_ini):

    rows = []

    # Loop through each row of the dataframe
    for _, r in df.iterrows():

        # Extract interesting values
        cip = r[cip_ini]
        start = r["start_window"].strftime("%Y-%m-%d")
        end = r["end_window"].strftime("%Y-%m-%d")

        # Creat SQL snippet and save it
        rows.append(
            f"""
            SELECT '{cip}' AS cip,
                   DATE '{start}' AS start_window,
                   DATE '{end}' AS end_window
            FROM dual
            """
        )

    # Combine all snippets from all rows - Temporary table in SQL
    return "\nUNION ALL\n".join(rows)


# ---- Function to create chunks of patients and convert them to temporary SQL tables ----

def create_chunks(df, chunk_size, cip_type):

    # Split a dataframe into chunks (list of snippets of the original dataframe)
    patient_chunks = [
        df.iloc[i:i+chunk_size]
        for i in range(0, len(df), chunk_size)
    ]

    # Convert each chunk into a temporary SQL table (list of SQL blocks)
    user_chunks = [build_patient_window_sql(c, cip_type) for c in patient_chunks]

    return user_chunks


# ---- Function to clean column names

def clean_columns(df):
    df = df.copy()
    cleaned = []

    for col in df.columns:
        # Remove accents
        col = unicodedata.normalize('NFKD', str(col)).encode('ascii', 'ignore').decode('utf-8')

        # Replace special chars/spaces with _
        col = re.sub(r'[^A-Za-z0-9_]+', '_', col)

        # Remove leading/trailing _
        col = col.strip('_')

        cleaned.append(col)

    df.columns = cleaned

    return df

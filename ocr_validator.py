# Save this as main_integrated.py
import streamlit as st
import pandas as pd
import io
import time
import concurrent.futures
from datetime import datetime, time as dt_time
from typing import List, Dict, Any

from azure.storage.blob import BlobServiceClient
from pymongo import MongoClient
from dateutil import parser
from dotenv import load_dotenv
import os

# ---------------- Load Secrets from .env ----------------
# Loads variables from a .env file into the environment
load_dotenv()
AZURE_CONN_STR = os.getenv("AZURE_CONN_STR")
MONGO_CONN_STR = os.getenv("MONGO_CONN_STR")
MONGO_DB = os.getenv("MONGO_DB_NAME")
ALLOWED_COLLECTIONS = ["processed_tenders", "E_TENDERS_L1", "E_TENDERS_L2",
                       "E_TENDERS_L3", "E_PROCURE_L1", "E_PROCURE_L2",
                       "E_PROCURE_L3", "IREPS_L3"]

# ---------------- Page Config ----------------
st.set_page_config(
    page_title="OCR Validator",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------- Custom CSS ----------------
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .stats-card {
        background-color: #ffffff;
        border: 1px solid #dee2e6;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# ---------------- Session State Initialization ----------------
# Connection Status
if 'azure_connected' not in st.session_state:
    st.session_state.azure_connected = False
if 'mongo_connected' not in st.session_state:
    st.session_state.mongo_connected = False
if 'connections_initialized' not in st.session_state:
    st.session_state.connections_initialized = False

# Results State
if 'single_validation_results' not in st.session_state:
    st.session_state.single_validation_results = []
if 'batch_validation_results' not in st.session_state:
    st.session_state.batch_validation_results = []
if 'csv_validation_results' not in st.session_state:
    st.session_state.csv_validation_results = []
if 'mongo_validation_results' not in st.session_state:
    st.session_state.mongo_validation_results = []
if 'mongo_tender_ids' not in st.session_state:
    st.session_state.mongo_tender_ids = []


# ---------------- Connection Logic ----------------
def initialize_connections():
    """
    Attempts to connect to Azure and MongoDB using credentials from .env file.
    This function should only run once per session unless a retry is triggered.
    """
    # Reset connection errors at the start of an attempt
    st.session_state.pop("azure_connection_error", None)
    st.session_state.pop("mongo_connection_error", None)

    if not AZURE_CONN_STR:
        st.session_state.azure_connection_error = "AZURE_CONN_STR not found in environment variables."
        st.session_state.azure_connected = False
    else:
        try:
            blob_service_client = BlobServiceClient.from_connection_string(AZURE_CONN_STR)
            next(blob_service_client.list_containers(), None)
            st.session_state.azure_connected = True
        except Exception as e:
            st.session_state.azure_connection_error = f"Azure connection failed: {e}"
            st.session_state.azure_connected = False

    if not MONGO_CONN_STR:
        st.session_state.mongo_connection_error = "MONGO_CONN_STR not found in environment variables."
        st.session_state.mongo_connected = False
    else:
        try:
            client = MongoClient(MONGO_CONN_STR, serverSelectionTimeoutMS=5000)
            client.server_info()
            st.session_state.mongo_connected = True
        except Exception as e:
            st.session_state.mongo_connection_error = f"Mongo connection failed: {e}"
            st.session_state.mongo_connected = False

    st.session_state.connections_initialized = True


def display_connection_status():
    """
    Displays the connection status in the sidebar and provides a retry button on failure.
    """
    st.sidebar.header("🔗 Connection Status")

    # --- Azure Status ---
    st.sidebar.subheader("Azure Storage")
    if st.session_state.azure_connected:
        st.sidebar.success("✅ Connected to Azure")
    else:
        error_msg = st.session_state.get("azure_connection_error", "Connection not established.")
        st.sidebar.error(f"❌ {error_msg}")

    # --- Mongo Status ---
    st.sidebar.subheader("MongoDB")
    if st.session_state.mongo_connected:
        st.sidebar.success("✅ Connected to MongoDB")
    else:
        error_msg = st.session_state.get("mongo_connection_error", "Connection not established.")
        st.sidebar.error(f"❌ {error_msg}")

    # --- Retry Button ---
    if not st.session_state.azure_connected or not st.session_state.mongo_connected:
        if st.sidebar.button("🔄 Retry Connections"):
            with st.spinner("Re-attempting connections..."):
                initialize_connections()
            st.rerun()


# ---------------- Azure & Validation Logic ----------------
def list_blobs_with_ext(container_client, prefix: str, extension: str):
    """Lists blobs with a specific extension in a given prefix."""
    extension = extension.lower()
    return [b for b in container_client.list_blobs(name_starts_with=prefix) if b.name.lower().endswith(extension)]


def validate_tender_blob(container_client, tender_id: str):
    """
    Validates the OCR status of a single tender ID in Azure Blob Storage.
    Checks for the existence of 'Original' and 'ocr' folders and their contents.
    """
    result = {
        "Tender ID": tender_id,
        "Status": "",
        "OCR folder Exists": "No",
        "Count of files in 'Original'": 0,
        "Count of files in 'OCR'": 0,
        "Notes": ""
    }

    tender_prefix = f"{tender_id}/"
    tender_blobs = list(container_client.list_blobs(name_starts_with=tender_prefix))

    if not tender_blobs:
        result["Status"] = "Not Found"
        result["Notes"] = f"Tender ID '{tender_id}' not found in Azure"
        return result

    original_prefix = f"{tender_id}/Original/"
    ocr_prefix = f"{tender_id}/ocr/"

    pdf_blobs = list_blobs_with_ext(container_client, original_prefix, ".pdf")
    txt_blobs = list_blobs_with_ext(container_client, ocr_prefix, ".txt")

    result["Count of files in 'Original'"] = len(pdf_blobs)
    result["Count of files in 'OCR'"] = len(txt_blobs)

    ocr_folder_exists = any(b.name.startswith(ocr_prefix) for b in tender_blobs)
    original_folder_exists = any(b.name.startswith(original_prefix) for b in tender_blobs)

    if txt_blobs:
        result["Status"] = "Correct"
        result["OCR folder Exists"] = "Yes"
        result["Notes"] = "OCR Data Present"
    else:
        if not ocr_folder_exists and original_folder_exists:
            result["Status"] = "No OCR folder"
            result["Notes"] = "OCR folder does not exist"
        elif ocr_folder_exists and not txt_blobs:
            result["Status"] = "Empty OCR folder"
            result["Notes"] = "OCR folder contains no .txt files"
        elif not ocr_folder_exists and not original_folder_exists:
            result["Status"] = "FAIL"
            result["Notes"] = "Neither OCR nor Original folder exists"
        else:
            result["Status"] = "Not Found"
            result["Notes"] = "Unknown state"

    return result


@st.cache_data(ttl=600)
def validate_tender_blob_cached(tender_id: str):
    """
    A cached wrapper for the validation function to speed up repeated checks.
    """
    try:
        blob_service_client = BlobServiceClient.from_connection_string(AZURE_CONN_STR)
        container_client = blob_service_client.get_container_client("tender")
        return validate_tender_blob(container_client, tender_id)
    except Exception as e:
        return {"Tender ID": tender_id, "Status": "Error", "Notes": f"Failed during cached validation: {str(e)}"}


# ---------------- Mongo Helpers ----------------
def parse_bid_date(s, tender_id=None):
    """
    Parses a date string with a primary format and a fallback parser.
    """
    s = str(s).strip()
    if not s:
        return pd.NaT

    try:
        return datetime.strptime(s, "%d-%b-%Y %I:%M %p")
    except ValueError:
        try:
            return parser.parse(s)
        except Exception:
            if tender_id:
                print(f"❗ Unparseable date for tender '{tender_id}': '{s}'")
            return pd.NaT


@st.cache_data(ttl=600)
def fetch_tenders_from_mongo(collection_name: str, cutoff_date: datetime) -> List[str]:
    """
    Fetches tenders from MongoDB, filtering by date.
    """
    try:
        client = MongoClient(MONGO_CONN_STR)
        db = client[MONGO_DB]
        collection = db[collection_name]

        query = {
            "$or": [
                {"bid_submission_end_date": {"$exists": True, "$ne": "", "$ne": None}},
                {"closing_date": {"$exists": True, "$ne": "", "$ne": None}}
            ]
        }
        projection = {"tender_id": 1, "bid_submission_end_date": 1, "closing_date": 1}
        cursor = collection.find(query, projection)

        valid_tender_ids = []
        for doc in cursor:
            tender_id = doc.get("tender_id")
            if not tender_id:
                continue

            date_str = doc.get("bid_submission_end_date") or doc.get("closing_date")
            if date_str:
                parsed_date = parse_bid_date(date_str, tender_id)
                if pd.notna(parsed_date):
                    if parsed_date.tzinfo is not None:
                        parsed_date = parsed_date.replace(tzinfo=None)
                    if parsed_date >= cutoff_date:
                        valid_tender_ids.append(str(tender_id))

        return sorted(list(set(valid_tender_ids)))
    except Exception as e:
        st.error(f"Failed to fetch from MongoDB: {e}")
        return []


# ---------------- Concurrency & Results Display ----------------
def process_tenders_concurrently(tender_ids: List[str]) -> List[Dict[str, Any]]:
    """
    Processes a list of tender IDs concurrently using a thread pool.
    """
    results = []
    total_count = len(tender_ids)
    if total_count == 0:
        st.warning("No tender IDs to process.")
        return []

    status_text = st.empty()
    progress_bar = st.progress(0)
    start_time = time.time()

    with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
        future_to_tender = {executor.submit(validate_tender_blob_cached, tid): tid for tid in tender_ids}
        for i, future in enumerate(concurrent.futures.as_completed(future_to_tender)):
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                tender_id = future_to_tender[future]
                results.append({"Tender ID": tender_id, "Status": "Error", "Notes": str(e)})
            progress_bar.progress((i + 1) / total_count)
            status_text.text(f"Validating... ({i + 1}/{total_count})")

    elapsed_time = time.time() - start_time
    status_text.text(f"Validation completed in {elapsed_time:.2f} seconds!")
    st.success(f"Validated {total_count} tender IDs in {elapsed_time:.2f} seconds!")
    return sorted(results, key=lambda x: tender_ids.index(x['Tender ID']))


def display_results(results, title="Validation Results"):
    """
    Displays validation results with interactive filters and download options.
    """
    if not results:
        st.info("No results to display.")
        return

    st.subheader(title)
    df = pd.DataFrame(results)

    # --- Interactive Filtering ---
    status_options = sorted(df['Status'].unique())
    selected_statuses = st.multiselect("Filter by Status:", options=status_options)

    if selected_statuses:
        filtered_df = df[df['Status'].isin(selected_statuses)]
    else:
        filtered_df = df

    stats = {
        "Total": len(df),
        "Correct": len(df[df['Status'] == 'Correct']),
        "No OCR Folder": len(df[df['Status'] == 'No OCR folder']),
        "Empty OCR": len(df[df['Status'] == 'Empty OCR folder']),
        "FAIL": len(df[df['Status'] == 'FAIL']),
        "NOT FOUND": len(df[df['Status'] == 'Not Found']),
    }
    cols = st.columns(len(stats))
    for col, (key, value) in zip(cols, stats.items()):
        col.metric(key, value)

    def color_status(val):
        color_map = {
            "Correct": "background-color: #d4edda",
            "Not Found": "background-color: #fff3cd",
            "Error": "background-color: #f5c6cb",
            "No OCR folder": "background-color: #f8d7da",
            "Empty OCR folder": "background-color: #f8d7da",
            "FAIL": "background-color: #dc3545; color: white;"
        }
        return color_map.get(val, "")

    st.dataframe(filtered_df.style.map(color_status, subset=['Status']), use_container_width=True)

    # ---- Export Options ----
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    csv = filtered_df.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Download as CSV", data=csv, file_name=f"validation_results_{timestamp}.csv", mime="text/csv")

    excel_buffer = io.BytesIO()
    with pd.ExcelWriter(excel_buffer, engine="openpyxl") as writer:
        filtered_df.to_excel(writer, index=False, sheet_name="Results")
    st.download_button("📊 Download as Excel", data=excel_buffer.getvalue(),
                       file_name=f"validation_results_{timestamp}.xlsx",
                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

    json_data = filtered_df.to_json(orient="records", indent=2).encode('utf-8')
    st.download_button("🗄️ Download as JSON", data=json_data, file_name=f"validation_results_{timestamp}.json",
                       mime="application/json")


# ---------------- Tabs UI ----------------
@st.cache_data(ttl=60)
def get_mongo_collection_names() -> List[str]:
    """Connects to Mongo and returns a list of all collection names in the DB."""
    try:
        client = MongoClient(MONGO_CONN_STR, serverSelectionTimeoutMS=3000)
        db = client[MONGO_DB]
        names = db.list_collection_names()
        client.close()
        return names
    except Exception as e:
        st.error(f"Could not fetch collection list from MongoDB: {e}")
        return []


def mongo_auto_validation():
    """UI and logic for the Mongo Auto-Validation tab."""
    st.header("🗄️ Mongo Auto-Validation")
    if not st.session_state.azure_connected or not st.session_state.mongo_connected:
        st.warning("Please ensure both Azure and MongoDB are connected successfully to use this feature.")
        return

    use_manual_entry = st.toggle("Enter collection name manually")
    if use_manual_entry:
        collection_name = st.text_input("Enter MongoDB Collection Name", placeholder="e.g., E_TENDERS_L3")
    else:
        collection_name = st.selectbox("Select Collection", ALLOWED_COLLECTIONS)

    c1, c2 = st.columns(2)
    selected_date = c1.date_input("Select Cutoff Date", datetime.today())
    time_str = c2.text_input("Enter Time (HH:MM, 24h format)", "09:00")
    try:
        selected_time = datetime.strptime(time_str, "%H:%M").time()
    except ValueError:
        st.error("Please enter time in HH:MM format (24-hour).")
        selected_time = dt_time(9, 0)
    cutoff = datetime.combine(selected_date, selected_time)

    if st.button("Fetch Tender IDs from Mongo", type="primary"):
        st.session_state.mongo_tender_ids = []
        st.session_state.mongo_validation_results = []
        if not collection_name:
            st.warning("Please select or enter a collection name.")
            return

        with st.spinner("Verifying collection..."):
            available_collections = get_mongo_collection_names()
            if not available_collections:
                st.error("Could not verify collection. Please check Mongo connection.")
                return
            if collection_name not in available_collections:
                st.error(f"Error: Collection '{collection_name}' does not exist in the database '{MONGO_DB}'.")
                st.info(f"Available collections are: `{'`, `'.join(available_collections)}`")
                return

        with st.spinner(f"Fetching tenders from '{collection_name}'..."):
            tender_ids = fetch_tenders_from_mongo(collection_name, cutoff)
            st.session_state.mongo_tender_ids = tender_ids

        if tender_ids:
            st.success(f"Found {len(tender_ids)} tenders from collection '{collection_name}'.")
            st.dataframe(pd.DataFrame({"Tender IDs Preview": tender_ids[:100]}), use_container_width=True)
        else:
            st.warning(f"No tenders found in collection '{collection_name}' for the selected criteria.")

    if st.session_state.mongo_tender_ids:
        if st.button("Run Azure Validation on Fetched IDs", type="primary"):
            results = process_tenders_concurrently(st.session_state.mongo_tender_ids)
            st.session_state.mongo_validation_results = results

    if st.session_state.mongo_validation_results:
        display_results(st.session_state.mongo_validation_results, "Mongo Validation Results")


def single_tender_validation():
    """UI and logic for the Single Tender Validation tab."""
    st.header("🔍 Single Tender Validation")
    if not st.session_state.azure_connected:
        st.warning("Please ensure Azure is connected successfully.")
        return
    tender_id = st.text_input("Enter Tender ID", placeholder="e.g., 2023_ABC_12345_1")
    if st.button("Validate Single Tender", type="primary") and tender_id:
        with st.spinner("Validating..."):
            result = validate_tender_blob_cached(tender_id.strip())
            st.session_state.single_validation_results = [result]
    if st.session_state.single_validation_results:
        display_results(st.session_state.single_validation_results, "Single Validation Result")


def batch_validation():
    """UI and logic for the Batch Validation tab."""
    st.header("📋 Batch Validation")
    if not st.session_state.azure_connected:
        st.warning("Please ensure Azure is connected successfully.")
        return
    tender_input = st.text_area("Enter tender IDs (one per line or comma-separated)", height=150)
    if st.button("Validate Batch", type="primary", disabled=not tender_input):
        lines = tender_input.strip().split('\n')
        tender_ids = []
        for line in lines:
            tender_ids.extend([tid.strip() for tid in line.split(',') if tid.strip()])
        unique_ids = sorted(list(set(tender_ids)))
        if unique_ids:
            results = process_tenders_concurrently(unique_ids)
            st.session_state.batch_validation_results = results
        else:
            st.warning("Please enter valid tender IDs.")
    if st.session_state.batch_validation_results:
        display_results(st.session_state.batch_validation_results, "Batch Validation Results")


def csv_upload_validation():
    """UI and logic for the CSV Upload Validation tab."""
    st.header("📁 CSV Upload Validation")
    if not st.session_state.azure_connected:
        st.warning("Please ensure Azure is connected successfully.")
        return
    uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'])
    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
            st.dataframe(df.head())

            # --- Flexible Column Selection ---
            col_to_use = st.selectbox("Select the column containing Tender IDs:", df.columns)

            if col_to_use:
                tender_ids = sorted(list(set(df[col_to_use].dropna().astype(str).tolist())))
                st.success(f"Found {len(tender_ids)} unique tender IDs in column '{col_to_use}'.")

                if st.button("Validate from CSV", type="primary"):
                    results = process_tenders_concurrently(tender_ids)
                    st.session_state.csv_validation_results = results
        except Exception as e:
            st.error(f"Error processing CSV file: {e}")

    if st.session_state.csv_validation_results:
        display_results(st.session_state.csv_validation_results, "CSV Validation Results")


# ---------------- Main App ----------------
def main():
    """Main function to run the Streamlit app."""
    st.markdown('<h1 class="main-header">📋 OCR Validator</h1>', unsafe_allow_html=True)

    if not st.session_state.connections_initialized:
        initialize_connections()

    display_connection_status()

    if not st.session_state.azure_connected and not st.session_state.mongo_connected:
        st.error(
            "Connections to Azure and MongoDB failed. Please check your .env file and network, then use the 'Retry Connections' button in the sidebar.")
    elif not st.session_state.azure_connected:
        st.error(
            "Azure connection failed. Please check your `AZURE_CONN_STR` and use the retry button. Validation features are disabled.")
    else:
        tab1, tab2, tab3, tab4 = st.tabs(
            ["🗄️ Mongo Auto-Validation", "🔍 Single Tender", "📋 Batch Validation", "📁 CSV Upload"])
        with tab1:
            mongo_auto_validation()
        with tab2:
            single_tender_validation()
        with tab3:
            batch_validation()
        with tab4:
            csv_upload_validation()

    st.markdown("""
        <style>.signature {
            position: fixed; bottom: 10px; right: 20px; text-align: right;
            font-size: 0.85rem; opacity: 0.5; z-index: 9999;
        }
        [data-theme="dark"] .signature { color: #cccccc; }
        [data-theme="light"] .signature { color: #666666; }
        </style>
        <div class="signature">- <b>Shreyansh Srivastava 🚀</b></div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()

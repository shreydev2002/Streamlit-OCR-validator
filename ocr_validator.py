import streamlit as st
import pandas as pd
import io
import time
import json
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any

from azure.storage.blob import BlobServiceClient
import base64

# Page configuration
st.set_page_config(
    page_title="Tender OCR Validator",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .status-success {
        color: #28a745;
        font-weight: bold;
    }
    .status-error {
        color: #dc3545;
        font-weight: bold;
    }
    .status-warning {
        color: #ffc107;
        font-weight: bold;
    }
    .progress-container {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
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

# Initialize session state
if 'azure_connected' not in st.session_state:
    st.session_state.azure_connected = False
if 'connection_string' not in st.session_state:
    st.session_state.connection_string = ""
if 'single_validation_results' not in st.session_state:
    st.session_state.single_validation_results = []
if 'batch_validation_results' not in st.session_state:
    st.session_state.batch_validation_results = []
if 'csv_validation_results' not in st.session_state:
    st.session_state.csv_validation_results = []
if 'processing_stats' not in st.session_state:
    st.session_state.processing_stats = {}
if 'current_tab' not in st.session_state:
    st.session_state.current_tab = "Single Tender"


# Azure Blob Storage Helper Functions
def get_container_client():
    """Get Azure container client"""
    if not st.session_state.azure_connected:
        raise RuntimeError("Azure connection not established. Please set connection string first.")

    try:
        blob_service_client = BlobServiceClient.from_connection_string(st.session_state.connection_string)
        return blob_service_client.get_container_client("tender")
    except Exception as e:
        st.error(f"Failed to connect to Azure: {str(e)}")
        return None


def list_blobs_with_ext(container_client, prefix: str, extension: str):
    """List blobs with specific extension"""
    extension = extension.lower()
    return [b for b in container_client.list_blobs(name_starts_with=prefix) if b.name.lower().endswith(extension)]


def get_latest_modified_timestamp(blobs):
    """Get latest modified timestamp from blobs"""
    if not blobs:
        return ""
    latest_blob = max(blobs, key=lambda b: b.last_modified)
    return latest_blob.last_modified.strftime("%Y-%m-%d %H:%M")


def validate_tender_blob(container_client, tender_id: str):
    """Core validation logic for a single tender ID"""
    result = {
        "Tender ID": tender_id,
        "Status": "",
        "OCR folder Exists": "No",
        "Count of files in 'Original'": 0,
        "Count of files in 'OCR'": 0,
        "Mismatch Analysis": "",
        "Missing Files": "",
        "Missing File Timestamps": "",
        "Original Folder Last Modified": "",
        "Notes": ""
    }

    # Check if tender_id folder exists
    tender_prefix = f"{tender_id}/"
    tender_blobs = list(container_client.list_blobs(name_starts_with=tender_prefix))

    if not tender_blobs:
        result["Status"] = "Not Found"
        result["Notes"] = f"Tender ID '{tender_id}' not found in Azure"
        result["Mismatch Analysis"] = "Tender ID does not exist"
        return result

    original_prefix = f"{tender_id}/Original/"
    ocr_prefix = f"{tender_id}/ocr/"

    pdf_blobs = list_blobs_with_ext(container_client, original_prefix, ".pdf")
    txt_blobs = list_blobs_with_ext(container_client, ocr_prefix, ".txt")
    pdf_stems = {Path(blob.name).stem: blob for blob in pdf_blobs}
    txt_stems = {Path(blob.name).stem for blob in txt_blobs}

    result["Count of files in 'Original'"] = len(pdf_blobs)
    result["Count of files in 'OCR'"] = len(txt_blobs)
    result["OCR folder Exists"] = "Yes" if txt_blobs else "No"
    result["Original Folder Last Modified"] = get_latest_modified_timestamp(pdf_blobs)

    missing_in_ocr = sorted(set(pdf_stems.keys()) - txt_stems)
    missing_in_original = sorted(set(txt_stems) - pdf_stems.keys())

    # Validation logic
    if not pdf_blobs:
        result["Notes"] = "No PDF files found"
        result["Status"] = "Mismatch"
        result["Mismatch Analysis"] = "Original folder empty"
        return result

    if not txt_blobs:
        result["Notes"] = "OCR folder missing or no .txt files"
        result["Status"] = "Mismatch"
        result["Mismatch Analysis"] = "OCR folder not present or empty"
        return result

    if not missing_in_ocr and not missing_in_original:
        result["Status"] = "Correct"
        result["Mismatch Analysis"] = "All matched"
    else:
        result["Status"] = "Mismatch"
        parts = []
        if missing_in_ocr:
            parts.append(f"{len(missing_in_ocr)} missing in OCR")
        if missing_in_original:
            parts.append(f"{len(missing_in_original)} missing in Original")
        result["Mismatch Analysis"] = ", ".join(parts)

        details = []
        if missing_in_ocr:
            details.append("OCR:\n" + "\n".join([f"{m}.pdf" for m in missing_in_ocr]))
        if missing_in_original:
            details.append("Original:\n" + "\n".join([f"{m}.txt" for m in missing_in_original]))
        result["Missing Files"] = "\n\n".join(details)

        timestamps = [f"{m}.pdf: {pdf_stems[m].last_modified.strftime('%Y-%m-%d %H:%M')}" for m in missing_in_ocr if
                      m in pdf_stems]
        result["Missing File Timestamps"] = " | ".join(timestamps)

        result["Notes"] = "Mismatch detected"

    return result


# Azure Connection Management
def setup_azure_connection():
    """Setup Azure connection in sidebar"""
    st.sidebar.header("🔗 Azure Connection")

    if not st.session_state.azure_connected:
        conn_str = st.sidebar.text_input(
            "Azure Connection String",
            type="password",
            help="Enter your Azure Storage connection string"
        )

        if st.sidebar.button("Connect to Azure", type="primary"):
            if conn_str:
                try:
                    # Test connection
                    blob_service_client = BlobServiceClient.from_connection_string(conn_str)
                    _ = list(blob_service_client.list_containers())[:1]

                    st.session_state.connection_string = conn_str
                    st.session_state.azure_connected = True
                    st.sidebar.success("✅ Connected to Azure successfully!")
                    st.rerun()
                except Exception as e:
                    st.sidebar.error(f"❌ Connection failed: {str(e)}")
            else:
                st.sidebar.warning("Please enter a connection string")
    else:
        st.sidebar.success("✅ Connected to Azure")
        if st.sidebar.button("Disconnect"):
            st.session_state.azure_connected = False
            st.session_state.connection_string = ""
            st.session_state.single_validation_results = []
            st.session_state.batch_validation_results = []
            st.session_state.csv_validation_results = []
            st.rerun()


# Progress Tracking
def update_progress_stats(results):
    """Update processing statistics"""
    if not results:
        return {}

    total = len(results)
    processed = total
    found = len([r for r in results if r["Status"] != "Not Found"])
    not_found = len([r for r in results if r["Status"] == "Not Found"])
    correct = len([r for r in results if r["Status"] == "Correct"])
    mismatch = len([r for r in results if r["Status"] == "Mismatch"])
    errors = len([r for r in results if r["Status"] == "Error"])

    return {
        "total": total,
        "processed": processed,
        "found": found,
        "not_found": not_found,
        "correct": correct,
        "mismatch": mismatch,
        "errors": errors
    }


# Results Display
def display_results(results, title="Validation Results"):
    """Display validation results with filtering and download options"""
    if not results:
        st.info("No results to display")
        return

    st.subheader(title)

    # Convert to DataFrame
    df = pd.DataFrame(results)

    # Compact validation status summary in corner
    stats = update_progress_stats(results)
    with st.container():
        col1, col2, col3, col4, col5 = st.columns([1, 1, 1, 1, 2])
        with col1:
            st.metric("Total", stats["total"], help="Total records processed")
        with col2:
            st.metric("Found", stats["found"], help="Records found in Azure")
        with col3:
            st.metric("Correct", stats["correct"], help="Correct matches")
        with col4:
            st.metric("Mismatch", stats["mismatch"], help="Mismatched records")

    # JSON view toggle above table
    show_json = st.checkbox("📄 Show JSON", key=f"json_toggle_{title.replace(' ', '_').lower()}")

    if show_json:
        st.json(df.to_dict(orient='records'))

    # Display results
    st.write(f"Showing {len(df)} results")

    # Color coding for status
    def color_status(val):
        if val == "Correct":
            return "background-color: #d4edda"
        elif val == "Mismatch":
            return "background-color: #f8d7da"
        elif val == "Not Found":
            return "background-color: #fff3cd"
        elif val == "Error":
            return "background-color: #f5c6cb"
        return ""

    styled_df = df.style.map(color_status, subset=['Status'])
    st.dataframe(styled_df, use_container_width=True)

    # Download options
    st.write("### Download Results")
    col1, col2, col3 = st.columns(3)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    with col1:
        st.download_button(
            label="📥 Download CSV",
            data=df.to_csv(index=False),
            file_name=f"validation_results_{timestamp}.csv",
            mime="text/csv",
            key=f"csv_download_{title.replace(' ', '_').lower()}"
        )

    with col2:
        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
            df.to_excel(writer, index=False, sheet_name='Results')
        buffer.seek(0)
        st.download_button(
            label="📥 Download Excel",
            data=buffer.getvalue(),
            file_name=f"validation_results_{timestamp}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            key=f"excel_download_{title.replace(' ', '_').lower()}"
        )

    with col3:
        st.download_button(
            label="📥 Download JSON",
            data=df.to_json(orient='records', indent=2),
            file_name=f"validation_results_{timestamp}.json",
            mime="application/json",
            key=f"json_download_{title.replace(' ', '_').lower()}"
        )


# Single Tender Validation
def single_tender_validation():
    """Single tender validation tab"""
    st.header("🔍 Single Tender Validation")

    if not st.session_state.azure_connected:
        st.warning("Please connect to Azure first using the sidebar.")
        return

    # Use form to handle Enter key
    with st.form("single_tender_form"):
        tender_id = st.text_input("Enter Tender ID", placeholder="e.g., 12345")
        submitted = st.form_submit_button("Validate", type="primary")

        if submitted and tender_id:
            with st.spinner("Validating tender ID..."):
                try:
                    container_client = get_container_client()
                    if container_client:
                        result = validate_tender_blob(container_client, tender_id.strip())
                        st.session_state.single_validation_results = [result]
                        st.success("Validation completed!")
                except Exception as e:
                    st.error(f"Validation failed: {str(e)}")

    # Display results - simplified for single tender
    if st.session_state.single_validation_results:
        result = st.session_state.single_validation_results[0]
        df = pd.DataFrame([result])

        # JSON view toggle above table
        show_json = st.checkbox("📄 Show JSON", key="json_toggle_single")

        if show_json:
            st.json(df.to_dict(orient='records'))

        # Color coding for status
        def color_status(val):
            if val == "Correct":
                return "background-color: #d4edda"
            elif val == "Mismatch":
                return "background-color: #f8d7da"
            elif val == "Not Found":
                return "background-color: #fff3cd"
            elif val == "Error":
                return "background-color: #f5c6cb"
            return ""

        styled_df = df.style.map(color_status, subset=['Status'])
        st.dataframe(styled_df, use_container_width=True, height=400)


# Batch Validation
def batch_validation():
    """Batch validation tab"""
    st.header("📋 Batch Validation")

    if not st.session_state.azure_connected:
        st.warning("Please connect to Azure first using the sidebar.")
        return

    # Input tender IDs
    st.write("### Enter Tender IDs")
    tender_input = st.text_area(
        "Enter tender IDs (one per line or comma-separated)",
        height=150,
        placeholder="12345\n67890\n11111\nor\n12345, 67890, 11111"
    )

    col1, col2 = st.columns([1, 1])

    with col1:
        if st.button("Validate Batch", type="primary", disabled=not tender_input):
            if tender_input:
                # Parse tender IDs
                lines = tender_input.strip().split('\n')
                tender_ids = []
                for line in lines:
                    if ',' in line:
                        tender_ids.extend([tid.strip() for tid in line.split(',') if tid.strip()])
                    else:
                        if line.strip():
                            tender_ids.append(line.strip())

                if tender_ids:
                    st.session_state.batch_validation_results = []

                    # Progress bar
                    progress_bar = st.progress(0)
                    status_text = st.empty()

                    results = []
                    for i, tender_id in enumerate(tender_ids):
                        status_text.text(f"Validating {tender_id}... ({i + 1}/{len(tender_ids)})")

                        try:
                            container_client = get_container_client()
                            if container_client:
                                result = validate_tender_blob(container_client, tender_id)
                                results.append(result)
                        except Exception as e:
                            results.append({
                                "Tender ID": tender_id,
                                "Status": "Error",
                                "Notes": str(e)
                            })

                        progress_bar.progress((i + 1) / len(tender_ids))

                    st.session_state.batch_validation_results = results
                    status_text.text("Validation completed!")
                    st.success(f"Validated {len(tender_ids)} tender IDs!")
                else:
                    st.warning("Please enter valid tender IDs")

    with col2:
        if st.button("Clear Results"):
            st.session_state.batch_validation_results = []
            st.rerun()

    # Display results
    if st.session_state.batch_validation_results:
        display_results(st.session_state.batch_validation_results, "Batch Validation Results")


# CSV Upload Validation
def csv_upload_validation():
    """CSV upload validation tab"""
    st.header("📁 CSV Upload Validation")

    if not st.session_state.azure_connected:
        st.warning("Please connect to Azure first using the sidebar.")
        return

    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type=['csv'],
        help="Upload a CSV file with tender IDs"
    )

    if uploaded_file is not None:
        # Parse CSV
        try:
            df = pd.read_csv(uploaded_file)

            # Detect tender ID column
            col = None
            for possible in ["tender_id", "Tender ID", "tenderId", "id"]:
                if possible in df.columns:
                    col = possible
                    break
            if col is None:
                col = df.columns[0]

            # Extract tender IDs
            tender_ids = df[col].dropna().astype(str).tolist()
            st.write(f"**Found {len(tender_ids)} tender IDs**")

            if st.button("Validate", type="primary"):
                if tender_ids:
                    st.session_state.csv_validation_results = []

                    # Progress tracking
                    progress_bar = st.progress(0)
                    status_text = st.empty()

                    results = []
                    start_time = time.time()

                    for i, tender_id in enumerate(tender_ids):
                        status_text.text(f"Validating {tender_id}... ({i + 1}/{len(tender_ids)})")

                        try:
                            container_client = get_container_client()
                            if container_client:
                                result = validate_tender_blob(container_client, tender_id)
                                results.append(result)
                        except Exception as e:
                            results.append({
                                "Tender ID": tender_id,
                                "Status": "Error",
                                "Notes": str(e)
                            })

                        # Update progress
                        progress = (i + 1) / len(tender_ids)
                        progress_bar.progress(progress)

                    elapsed_time = time.time() - start_time
                    status_text.text(f"Validation completed in {elapsed_time:.2f} seconds!")
                    st.success(f"Validated {len(tender_ids)} tender IDs in {elapsed_time:.2f} seconds!")

                    # Store results
                    st.session_state.csv_validation_results = results

                else:
                    st.warning("No valid tender IDs found in the CSV file")

        except Exception as e:
            st.error(f"Error reading CSV file: {str(e)}")

    # Display results
    if st.session_state.csv_validation_results:
        display_results(st.session_state.csv_validation_results, "CSV Validation Results")


# Main Application
def main():
    """Main application function"""
    # Header
    st.markdown('<h1 class="main-header">📋 Tender OCR Validator</h1>', unsafe_allow_html=True)

    # Setup Azure connection
    setup_azure_connection()

    # Main content
    if st.session_state.azure_connected:
        # Tab selection
        tabs = st.tabs(["🔍 Single Tender", "📋 Batch Validation", "📁 CSV Upload"])

        with tabs[0]:
            single_tender_validation()

        with tabs[1]:
            batch_validation()

        with tabs[2]:
            csv_upload_validation()
    else:
        st.info("👈 Please connect to Azure using the sidebar to start validating tender IDs.")


if __name__ == "__main__":
    main()


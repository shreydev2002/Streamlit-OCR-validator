# ocr_validator.py
import streamlit as st
from azure.storage.blob import BlobServiceClient
from pathlib import Path
from typing import List, Dict, Any
import pandas as pd
import io
import datetime

st.set_page_config(page_title="Tender OCR Validator", layout="wide")

# ---------------------------
# Session state initialization
# ---------------------------
if "AZURE_CONN_STR" not in st.session_state:
    st.session_state["AZURE_CONN_STR"] = ""
if "connected" not in st.session_state:
    st.session_state["connected"] = False
if "single_result" not in st.session_state:
    st.session_state["single_result"] = None
if "multi_results" not in st.session_state:
    st.session_state["multi_results"] = []
if "csv_results" not in st.session_state:
    st.session_state["csv_results"] = []

# ---------------------------
# Helper: Azure container client
# ---------------------------
def get_container_client():
    conn_str = st.session_state.get("AZURE_CONN_STR")
    if not conn_str:
        raise RuntimeError("Azure connection string not set. Enter it in the sidebar and press 'Set Connection'.")
    blob_service_client = BlobServiceClient.from_connection_string(conn_str)
    # container name assumed "tender" from your backend code
    return blob_service_client.get_container_client("tender")


# ---------------------------
# Blob utilities (same logic as your backend)
# ---------------------------
def list_blobs_with_ext(container_client, prefix: str, extension: str):
    extension = extension.lower()
    try:
        blobs = [
            b for b in container_client.list_blobs(name_starts_with=prefix)
            if b.name.lower().endswith(extension)
        ]
        return blobs
    except Exception as e:
        raise


def get_latest_modified_timestamp(blobs):
    if not blobs:
        return ""
    latest_blob = max(blobs, key=lambda b: b.last_modified)
    # convert to readable local string
    try:
        return latest_blob.last_modified.strftime("%Y-%m-%d %H:%M")
    except Exception:
        return str(latest_blob.last_modified)


def validate_tender_blob(container_client, tender_id: str) -> Dict[str, Any]:
    """
    Core logic kept identical to your FastAPI validate_tender_blob function,
    returns a dict with fields for the result row.
    """
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

    original_prefix = f"{tender_id}/Original/"
    ocr_prefix = f"{tender_id}/ocr/"

    # list blobs
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

    # Cases
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

        timestamps = [
            f"{m}.pdf: {pdf_stems[m].last_modified.strftime('%Y-%m-%d %H:%M')}"
            for m in missing_in_ocr if m in pdf_stems
        ]
        result["Missing File Timestamps"] = " | ".join(timestamps)
        result["Notes"] = "Mismatch detected"

    return result


def validate_batch(tender_ids: List[str]) -> List[Dict[str, Any]]:
    container_client = get_container_client()
    results = []
    for tid in tender_ids:
        try:
            results.append(validate_tender_blob(container_client, tid))
        except Exception as e:
            results.append({"Tender ID": tid, "Status": "Error", "Notes": str(e)})
    return results


# ---------------------------
# Styling helpers
# ---------------------------
def style_status_column(df: pd.DataFrame) -> str:
    """
    Return HTML string for a styled dataframe where the Status column
    is color coded. We'll produce df.to_html() and inject tiny CSS.
    """
    def color_for_status(val):
        if pd.isna(val):
            return ""
        v = str(val).lower()
        if "correct" in v:
            return "background-color:#d7f3d7;color:#0b6b0b"  # soft green
        if "mismatch" in v:
            return "background-color:#ffdede;color:#8b0000"  # soft red
        if "error" in v or "err" in v:
            return "background-color:#fff4d6;color:#7a5500"  # soft orange/yellow
        return ""

    # create style map for each cell in Status column
    styled = df.copy()
    # convert lists/long fields to strings to keep table compact
    for c in styled.columns:
        styled[c] = styled[c].apply(lambda x: x if (isinstance(x, (int, float)) or pd.isna(x)) else str(x))
    # Build HTML with inline styles for Status column
    html = styled.to_html(index=False, escape=False)
    # We'll inject per-row inline styling by re-creating the table rows.
    # Simpler approach: use pandas Styler if available
    try:
        styler = styled.style.applymap(lambda v: color_for_status(v) if styled.columns.tolist().index("Status") >= 0 and isinstance(v, str) else "", subset=["Status"])
        # Return full HTML
        return styler.to_html()
    except Exception:
        # Fallback: manual replacement for Status cell values with span-wrapping
        # This is a conservative fallback in case Styler fails for any env.
        html_out = "<style>table{border-collapse:collapse;} table th, table td{padding:6px; border:1px solid #ddd;}</style>\n"
        html_out += "<table>\n<thead>\n<tr>"
        for col in styled.columns:
            html_out += f"<th>{col}</th>"
        html_out += "</tr>\n</thead>\n<tbody>\n"
        for _, row in styled.iterrows():
            html_out += "<tr>"
            for col in styled.columns:
                val = row[col] if not pd.isna(row[col]) else ""
                if col == "Status":
                    style = color_for_status(val)
                    html_out += f"<td style='{style}'>{val}</td>"
                else:
                    html_out += f"<td>{val}</td>"
            html_out += "</tr>\n"
        html_out += "</tbody>\n</table>"
        return html_out


# ---------------------------
# Layout
# ---------------------------

# Sidebar: only connection
with st.sidebar:
    st.header("Azure Connection")
    conn_input = st.text_input("Azure connection string (input-based)", type="password", value=st.session_state["AZURE_CONN_STR"])
    if st.button("Set Connection"):
        if not conn_input:
            st.error("Enter a connection string.")
        else:
            try:
                # quick validate
                BlobServiceClient.from_connection_string(conn_input)
                st.session_state["AZURE_CONN_STR"] = conn_input
                st.session_state["connected"] = True
                st.success("✅ Connection validated and set.")
            except Exception as e:
                st.session_state["connected"] = False
                st.error(f"Invalid connection: {e}")
    st.markdown("---")
    if st.session_state.get("connected"):
        st.info("Connected to Azure storage (container: `tender`).")
    else:
        st.warning("Not connected. Set connection to use validators.")


st.title("📄 Azure OCR Validator")
st.caption(" - Shreyansh Srivastava")

# --- Single Tender Section ---
st.subheader("1) Single Tender")
col1, col2, col3 = st.columns([3, 1, 1])
with col1:
    single_id = st.text_input("Enter tender ID (single)", key="single_id")
with col2:
    if st.button("Validate Single"):
        if not single_id:
            st.warning("Enter a tender ID first.")
        else:
            try:
                res = validate_batch([single_id])[0]
                st.session_state["single_result"] = res
            except Exception as e:
                st.session_state["single_result"] = {"Tender ID": single_id, "Status": "Error", "Notes": str(e)}
with col3:
    # placeholder for spacing or future controls
    pass

# display single result area (default table)
if st.session_state.get("single_result") is not None:
    st.markdown("**Result (Single Tender)**")
    single_view_json = st.checkbox("Show JSON", key="single_json_toggle")
    single_result = st.session_state["single_result"]
    df_single = pd.DataFrame([single_result])
    if not single_view_json:
        try:
            # Try Styler path
            styled_html = style_status_column(df_single)
            st.write(styled_html, unsafe_allow_html=True)
        except Exception:
            st.dataframe(df_single)
    else:
        st.json(single_result)
    # download
    csv_bytes = df_single.to_csv(index=False).encode("utf-8")
    st.download_button("Download single result CSV", data=csv_bytes, file_name=f"result_{single_result.get('Tender ID','single')}.csv", mime="text/csv")

st.markdown("---")

# --- Multiple Tenders Section ---
st.subheader("2) Multiple Tenders")
multi_col1, multi_col2 = st.columns([4,1])
with multi_col1:
    multi_input = st.text_area("Enter multiple tender IDs (comma or newline separated)", key="multi_input", height=120, placeholder="e.g. 2025_ABC_1234_1, 2025_DEF_9876_1")
with multi_col2:
    if st.button("Validate Multiple"):
        ids_raw = multi_input or ""
        ids = [tid.strip() for tid in ids_raw.replace(",", "\n").split("\n") if tid.strip()]
        if not ids:
            st.warning("Enter at least one tender ID for batch validation.")
        else:
            try:
                results = validate_batch(ids)
                st.session_state["multi_results"] = results
            except Exception as e:
                st.session_state["multi_results"] = [{"Tender ID": "N/A", "Status": "Error", "Notes": str(e)}]

if st.session_state.get("multi_results"):
    st.markdown("**Results (Multiple Tenders)**")
    multi_view_json = st.checkbox("Show JSON", key="multi_json_toggle")
    multi_results = st.session_state["multi_results"]
    df_multi = pd.DataFrame(multi_results)
    if not multi_view_json:
        try:
            styled_html = style_status_column(df_multi)
            st.write(styled_html, unsafe_allow_html=True)
        except Exception:
            st.dataframe(df_multi)
    else:
        st.json(multi_results)
    csv_bytes = df_multi.to_csv(index=False).encode("utf-8")
    st.download_button("Download multiple results CSV", data=csv_bytes, file_name="batch_results.csv", mime="text/csv")

st.markdown("---")

# --- CSV Upload Section ---
st.subheader("3) CSV Upload")
uploaded_file = st.file_uploader("Upload CSV with tender IDs (first matching column used)", type=["csv"], key="csv_uploader")
if uploaded_file is not None:
    try:
        df_in = pd.read_csv(uploaded_file)
        # find column name to use
        col = None
        for possible in ["tender_id", "Tender ID", "tenderId", "id"]:
            if possible in df_in.columns:
                col = possible
                break
        if col is None:
            col = df_in.columns[0]
        tender_ids = df_in[col].dropna().astype(str).tolist()
        if st.button("Validate CSV"):
            if not tender_ids:
                st.warning("CSV didn't contain any tender ids.")
            else:
                try:
                    csv_results = validate_batch(tender_ids)
                    st.session_state["csv_results"] = csv_results
                except Exception as e:
                    st.session_state["csv_results"] = [{"Tender ID": "N/A", "Status": "Error", "Notes": str(e)}]
    except Exception as e:
        st.error(f"Failed to parse CSV: {e}")

if st.session_state.get("csv_results"):
    st.markdown("**Results (CSV Upload)**")
    csv_view_json = st.checkbox("Show JSON", key="csv_json_toggle")
    csv_results = st.session_state["csv_results"]
    df_csv = pd.DataFrame(csv_results)
    if not csv_view_json:
        try:
            styled_html = style_status_column(df_csv)
            st.write(styled_html, unsafe_allow_html=True)
        except Exception:
            st.dataframe(df_csv)
    else:
        st.json(csv_results)
    csv_bytes = df_csv.to_csv(index=False).encode("utf-8")
    st.download_button("Download CSV results", data=csv_bytes, file_name="csv_results.csv", mime="text/csv")

st.markdown("---")
st.caption(f"App last loaded: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


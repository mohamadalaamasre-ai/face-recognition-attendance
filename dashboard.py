"""
Attendance dashboard (Streamlit). It only READS files and never touches the camera:
  status.json            live state written by attendance_camera.py (about once per second)
  attendance.csv         current session: who is present and when they arrived
  attendance_archive/    past sessions (one CSV per reset)
  gallery/               registered people
 
Run from the face_recognition/ folder, in a SECOND terminal while the camera runs:
  streamlit run dashboard.py
"""
import json
import time
from datetime import datetime
from pathlib import Path
 
import pandas as pd
import streamlit as st
 
import config as C
from gallery import Gallery
 
STATUS_FILE = Path("status.json")
OFFLINE_AFTER_S = 5        # no status update for this long -> camera shown as stopped
REFRESH_S = 2              # live section refresh period (seconds)
EMPTY = pd.DataFrame(columns=["name", "date", "time"])
 
st.set_page_config(page_title="Attendance Dashboard", page_icon="🎥", layout="wide")
 
 
def read_status():
    """Latest live state from the camera, or None if the file is missing or being replaced."""
    try:
        return json.loads(STATUS_FILE.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return None
 
 
def read_attendance(path):
    """One attendance CSV as a table (empty table if the file is missing or has only the header)."""
    try:
        return pd.read_csv(path, encoding="utf-8-sig")
    except (FileNotFoundError, pd.errors.EmptyDataError):
        return EMPTY.copy()
 
 
def session_label(path):
    """attendance_20261122_143512.csv -> '2026-11-22 14:35'."""
    try:
        stamp = datetime.strptime(Path(path).stem.replace("attendance_", ""), "%Y%m%d_%H%M%S")
        return stamp.strftime("%Y-%m-%d %H:%M")
    except ValueError:
        return Path(path).stem
 
 
def card(column, label, value):
    with column.container(border=True):
        st.metric(label, value)
 
 
@st.fragment(run_every=REFRESH_S)
def live_section():
    """Re-runs by itself every REFRESH_S seconds; the rest of the page stays as it is."""
    status = read_status()
    online = bool(status and status.get("running") and time.time() - status.get("updated", 0) < OFFLINE_AFTER_S)
    present = read_attendance(C.ATTENDANCE_CSV)
 
    st.subheader("🟢 Camera running" if online else "🔴 Camera stopped")
    c1, c2, c3, c4 = st.columns(4)
    card(c1, "In front of the camera now", status["in_view"] if online else 0)
    card(c2, "Present this session", len(present))
    card(c3, "Unknown people seen", status["unknown_seen"] if status else 0)
    card(c4, "FPS", status["fps"] if online else "-")
    if online and status.get("labels"):
        st.caption("Now in view: " + ", ".join(status["labels"]))
 
    left, right = st.columns([2, 1])
    with left:
        st.markdown("#### Present now")
        if present.empty:
            st.info("Nobody has been logged in this session yet.")
        else:
            st.dataframe(present.sort_values(["date", "time"]), hide_index=True)
    with right:
        st.markdown("#### Latest arrivals")
        if present.empty:
            st.write("—")
        for _, row in present.sort_values(["date", "time"], ascending=False).head(6).iterrows():
            st.write(f"✅ **{row['name']}** at {row['time']}")
 
 
def history_section():
    files = sorted(Path(C.ATTENDANCE_ARCHIVE_DIR).glob("attendance_*.csv"), reverse=True)
    if not files:
        st.info("No past sessions yet. A session is archived each time attendance is reset (r, r).")
        return
    counts = pd.DataFrame({
        "session": [session_label(f) for f in files],
        "people present": [len(read_attendance(f)) for f in files],
    }).iloc[::-1]                                          # oldest on the left
    st.markdown("#### People present per session")
    st.bar_chart(counts.set_index("session"))
 
    choice = st.selectbox("Open a session", files, format_func=session_label)
    st.dataframe(read_attendance(choice), hide_index=True)
    st.download_button("Download this session (CSV)", choice.read_bytes(), file_name=choice.name, mime="text/csv")
 
 
def gallery_section():
    g = Gallery()
    if not len(g):
        st.info("Nobody is registered yet. Press e in the camera window to enroll someone.")
        return
    st.dataframe(pd.DataFrame({"name": g.names, "face samples": g.counts}), hide_index=True)
 
 
st.title("🎥 Attendance Dashboard")
st.caption(f"Face recognition + person segmentation • live part refreshes every {REFRESH_S} s")
live_section()
st.divider()
tab_history, tab_people = st.tabs(["📁 Past sessions", "👤 Registered people"])
with tab_history:
    history_section()
with tab_people:
    gallery_section()
 
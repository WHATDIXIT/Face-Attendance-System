import streamlit as st
import firebase_admin
from firebase_admin import credentials, db
import os
import json
from datetime import datetime

# Safe imports for packages that may not work on cloud
try:
    import cv2
    import pickle
    import face_recognition
    import numpy as np
    import cvzone
    CV_AVAILABLE = True
except ImportError:
    CV_AVAILABLE = False

# ── 1. Page config (must be first Streamlit command) ─────────────────────────
st.set_page_config(
    page_title="Central Attendance ERP",
    page_icon="🛡️",
    layout="wide"
)

# ── 2. Firebase initialization (Cloud + Local hybrid) ────────────────────────
if not firebase_admin._apps:
    try:
        if "firebase_key" in st.secrets:
            # Cloud mode: load from Streamlit Secrets
            secret = st.secrets["firebase_key"]
            key_dict = json.loads(secret) if isinstance(secret, str) else dict(secret)
            cred = credentials.Certificate(key_dict)
        else:
            # Local mode: load from file
            cred = credentials.Certificate("serviceAccountKey.json")

        firebase_admin.initialize_app(cred, {
    "databaseURL": "https://face-based-attendance-38ee8-default-rtdb.asia-southeast1.firebasedatabase.app"
})

    except Exception as e:
        st.error(f"❌ Firebase Configuration Error: {e}")
        st.stop()

# ── 3. Session state defaults ─────────────────────────────────────────────────
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

# ── 4. Login gate (if/else — avoids render freeze) ───────────────────────────
if not st.session_state.logged_in:

    st.markdown(
        "<h2 style='text-align:center; margin-top:60px;'>🔒 Admin Authentication Gate</h2>",
        unsafe_allow_html=True
    )

    _, col_center, _ = st.columns([1, 2, 1])
    with col_center:
        with st.form("login_gate", clear_on_submit=False):
            admin_name     = st.text_input("Admin Username")
            admin_password = st.text_input("Security Access Password", type="password")
            login_btn      = st.form_submit_button("🔑 Authenticate Access")

            if login_btn:
                if admin_name == "Utkarsh Dixit" and admin_password == "DeepLearning":
                    st.session_state.logged_in = True
                    st.toast("✅ Access Granted! Welcome back, Admin.")
                    st.rerun()
                else:
                    st.error("❌ Invalid credentials. Access denied.")

else:
    # ══════════════════════════════════════════════════════════════════════════
    #  MAIN DASHBOARD — only renders when logged_in = True
    # ══════════════════════════════════════════════════════════════════════════

    # ── Sidebar ───────────────────────────────────────────────────────────────
    st.sidebar.title("🛡️ Admin Command Panel")
    st.sidebar.write("Logged in as: **Utkarsh Dixit**")
    st.sidebar.markdown("---")

    page = st.sidebar.radio(
        "Navigate Workspace",
        ["📊 Attendance Analytics", "📸 Live Attendance Camera", "🎓 Enroll New Student"]
    )

    st.sidebar.markdown("---")
    if st.sidebar.button("🚪 Secure Logout", use_container_width=True):
        st.session_state.logged_in = False
        st.rerun()

    # ── PAGE 1: Attendance Analytics ──────────────────────────────────────────
    if page == "📊 Attendance Analytics":
        st.title("📊 Attendance Monitoring Metrics")
        st.caption("Real-time data pulled from Firebase Realtime Database.")

        try:
            students_ref = db.reference("Students").get()

            if students_ref:
                total_students       = len(students_ref)
                total_attendance_logs = sum(
                    data.get("attendance", 0)
                    for data in students_ref.values()
                    if isinstance(data, dict)
                )

                col1, col2 = st.columns(2)
                col1.metric("Total Enrolled Students",   total_students,        "Active Directory")
                col2.metric("Cumulative Attendance Hits", total_attendance_logs, "Live Syncing")

                st.markdown("---")
                st.subheader("📋 Student Roster")

                table_data = []
                for s_id, data in students_ref.items():
                    if data and isinstance(data, dict):
                        table_data.append({
                            "Student ID":             s_id,
                            "Full Name":              data.get("name",            "N/A"),
                            "Branch":                 data.get("branch",          "N/A"),
                            "Section":                data.get("section",         "N/A"),
                            "Year":                   data.get("year",            "N/A"),
                            "Total Attendance":       data.get("attendance",       0),
                            "Last Check-In":          data.get("last_attendance", "N/A"),
                        })

                st.dataframe(table_data, use_container_width=True)

            else:
                st.info("📂 Database is empty. Register students in the Enroll tab.")

        except Exception as e:
            st.error(f"❌ Failed to fetch database: {e}")

    # ── PAGE 2: Live Attendance Camera ────────────────────────────────────────
    elif page == "📸 Live Attendance Camera":
        st.title("📸 Biometric Face-Scanning Terminal")

        st.info("💡 **Cloud Restriction:** Live webcam scanning requires local deployment.")
        st.warning(
            "⚠️ Cloud servers have no physical camera hardware. "
            "To run the live facial recognition scanner, use your local machine:"
        )
        st.code("streamlit run app.py", language="bash")

        st.markdown("---")
        st.markdown("### 📋 Local Setup Reminder")
        st.markdown("""
        1. Connect your webcam or Iriun mobile camera
        2. Make sure `encodings.p` and `images/` folder are present locally
        3. Run the command above in your terminal
        4. The scanner window will open automatically
        """)

    # ── PAGE 3: Enroll New Student ────────────────────────────────────────────
    elif page == "🎓 Enroll New Student":
        st.title("🎓 Student Registration Portal")
        st.caption("Register a new student and upload their face reference image.")

        with st.form("registration_form", clear_on_submit=True):
            st.subheader("Personal & Academic Details")

            student_id = st.text_input("Student ID (Unique Roll Number)", placeholder="e.g., 452331")
            name       = st.text_input("Full Name",                        placeholder="e.g., Rohan Sharma")
            branch     = st.text_input("Branch / Department",              placeholder="e.g., CSE")
            section    = st.text_input("Section",                          placeholder="e.g., A")

            col1, col2 = st.columns(2)
            with col1:
                year         = st.number_input("Current Year",    min_value=1,    max_value=4,    value=1,    step=1)
            with col2:
                joining_year = st.number_input("Year of Joining", min_value=2020, max_value=2030, value=2026, step=1)

            st.markdown("---")
            st.subheader("📸 Biometric Image Upload")
            uploaded_file = st.file_uploader(
                "Upload Student Face Photo (JPG / PNG)",
                type=["jpg", "jpeg", "png"]
            )

            submit = st.form_submit_button("✅ Register Student")

        if submit:
            if not student_id or not name or not branch or not section or not uploaded_file:
                st.error("❌ Please fill in all fields and upload a face image.")
            else:
                try:
                    # Save image locally
                    images_dir = os.path.join(os.getcwd(), "images")
                    os.makedirs(images_dir, exist_ok=True)

                    ext           = os.path.splitext(uploaded_file.name)[1]
                    save_path     = os.path.join(images_dir, f"{student_id}{ext}")
                    with open(save_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())

                    # Push to Firebase
                    student_data = {
                        "name":            name,
                        "branch":          branch,
                        "section":         section,
                        "year":            int(year),
                        "yearOfJoining":   int(joining_year),
                        "attendance":      0,
                        "last_attendance": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    }
                    db.reference(f"Students/{student_id}").set(student_data)

                    st.success(f"🎉 {name} ({student_id}) registered successfully!")
                    st.balloons()

                except Exception as e:
                    st.error(f"❌ Registration Error: {e}")

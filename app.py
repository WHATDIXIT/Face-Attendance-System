import streamlit as st
import firebase_admin
from firebase_admin import credentials, db
import os
import json
from datetime import datetime

# ── Safe imports (only available locally, not on cloud) ───────────────────────
try:
    import cv2
    import pickle
    import face_recognition
    import numpy as np
    CV_AVAILABLE = True
except ImportError:
    CV_AVAILABLE = False

# ══════════════════════════════════════════════════════════════════════════════
#  1. Page Config  (must be FIRST Streamlit command)
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Central Attendance ERP",
    page_icon="🛡️",
    layout="wide"
)

# ══════════════════════════════════════════════════════════════════════════════
#  2. Firebase Initialization  (Cloud + Local hybrid)
# ══════════════════════════════════════════════════════════════════════════════
if not firebase_admin._apps:
    try:
        if "firebase_key" in st.secrets:
            # ── Cloud mode: read from Streamlit Secrets ──────────────────────
            secret = st.secrets["firebase_key"]
            key_dict = json.loads(secret) if isinstance(secret, str) else dict(secret)
            cred = credentials.Certificate(key_dict)
        else:
            # ── Local mode: read from file ────────────────────────────────────
            cred = credentials.Certificate("serviceAccountKey.json")

        firebase_admin.initialize_app(cred, {
            "databaseURL": "https://face-based-attendance-38ee8-default-rtdb.asia-southeast1.firebasedatabase.app"
        })

    except Exception as e:
        st.error(f"❌ Firebase Configuration Error: {e}")
        st.stop()

# ══════════════════════════════════════════════════════════════════════════════
#  3. Session State
# ══════════════════════════════════════════════════════════════════════════════
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if "marked_today" not in st.session_state:
    st.session_state.marked_today = set()

# ══════════════════════════════════════════════════════════════════════════════
#  4. Login Gate  (if/else — no st.stop() freeze)
# ══════════════════════════════════════════════════════════════════════════════
if not st.session_state.logged_in:

    st.markdown(
        "<h2 style='text-align:center; margin-top:80px;'>🔒 Admin Authentication Gate</h2>",
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

# ══════════════════════════════════════════════════════════════════════════════
#  5. Main Dashboard  (only when logged in)
# ══════════════════════════════════════════════════════════════════════════════
else:

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
        st.session_state.marked_today = set()
        st.rerun()

    # ══════════════════════════════════════════════════════════════════════════
    #  PAGE 1 — Attendance Analytics
    # ══════════════════════════════════════════════════════════════════════════
    if page == "📊 Attendance Analytics":
        st.title("📊 Attendance Monitoring Metrics")
        st.caption("Real-time data pulled from Firebase Realtime Database.")

        try:
            students_ref = db.reference("Students").get()

            if students_ref:
                total_students        = len(students_ref)
                total_attendance_logs = sum(
                    data.get("attendance", 0)
                    for data in students_ref.values()
                    if isinstance(data, dict)
                )

                col1, col2 = st.columns(2)
                col1.metric("Total Enrolled Students",    total_students,        "Active Directory")
                col2.metric("Cumulative Attendance Hits", total_attendance_logs, "Live Syncing")

                st.markdown("---")
                st.subheader("📋 Student Roster")

                table_data = []
                for s_id, data in students_ref.items():
                    if data and isinstance(data, dict):
                        table_data.append({
                            "Student ID":       s_id,
                            "Full Name":        data.get("name",            "N/A"),
                            "Branch":           data.get("branch",          "N/A"),
                            "Section":          data.get("section",         "N/A"),
                            "Year":             data.get("year",            "N/A"),
                            "Total Attendance": data.get("attendance",       0),
                            "Last Check-In":    data.get("last_attendance", "N/A"),
                        })

                st.dataframe(table_data, use_container_width=True)

            else:
                st.info("📂 Database is empty. Register students in the Enroll tab first.")

        except Exception as e:
            st.error(f"❌ Failed to fetch database: {e}")

    # ══════════════════════════════════════════════════════════════════════════
    #  PAGE 2 — Live Attendance Camera
    # ══════════════════════════════════════════════════════════════════════════
    elif page == "📸 Live Attendance Camera":
        st.title("📸 Biometric Face-Scanning Terminal")

        tab1, tab2 = st.tabs(["📷 Browser Camera", "🖥️ Local Webcam / Iriun Guide"])

        # ── Tab 1: Browser Camera (works on Cloud + Local) ────────────────────
        with tab1:
            st.info("💡 Uses your device camera through the browser. Works on cloud and locally.")

            img_file = st.camera_input("📸 Click to take a photo for attendance")

            if img_file is not None:

                col_img, col_result = st.columns(2)

                with col_img:
                    st.image(img_file, caption="Captured Photo", use_container_width=True)

                with col_result:
                    if CV_AVAILABLE:
                        # ── Load encodings ────────────────────────────────────
                        if not os.path.exists("encodings.p"):
                            st.warning("⚠️ `encodings.p` not found. Generate it locally first.")
                        else:
                            with st.spinner("🔍 Scanning face..."):
                                try:
                                    with open("encodings.p", "rb") as f:
                                        encode_list_known, student_ids = pickle.load(f)

                                    # Decode image bytes
                                    bytes_data = img_file.getvalue()
                                    nparr      = np.frombuffer(bytes_data, np.uint8)
                                    img_cv     = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                                    img_rgb    = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
                                    img_small  = cv2.resize(img_rgb, (0, 0), fx=0.5, fy=0.5)

                                    # Detect faces
                                    face_locs = face_recognition.face_locations(img_small)
                                    face_encs = face_recognition.face_encodings(img_small, face_locs)

                                    if not face_encs:
                                        st.warning("⚠️ No face detected. Try again with better lighting.")

                                    else:
                                        for face_enc in face_encs:
                                            matches    = face_recognition.compare_faces(
                                                encode_list_known, face_enc, tolerance=0.6
                                            )
                                            face_dists = face_recognition.face_distance(
                                                encode_list_known, face_enc
                                            )
                                            best_match = int(np.argmin(face_dists))

                                            if matches[best_match]:
                                                student_id = student_ids[best_match]
                                                confidence = round((1 - face_dists[best_match]) * 100, 1)

                                                # Fetch student from Firebase
                                                student_data = db.reference(
                                                    f"Students/{student_id}"
                                                ).get()

                                                if student_data:
                                                    name = student_data.get("name", student_id)

                                                    # Mark attendance (once per session)
                                                    if student_id not in st.session_state.marked_today:
                                                        current_att = student_data.get("attendance", 0)
                                                        db.reference(f"Students/{student_id}").update({
                                                            "attendance":      current_att + 1,
                                                            "last_attendance": datetime.now().strftime(
                                                                "%Y-%m-%d %H:%M:%S"
                                                            )
                                                        })
                                                        st.session_state.marked_today.add(student_id)
                                                        st.success(
                                                            f"✅ Attendance marked!\n\n"
                                                            f"**{name}** (ID: {student_id})\n\n"
                                                            f"Confidence: {confidence}%"
                                                        )
                                                        st.balloons()
                                                    else:
                                                        st.info(
                                                            f"ℹ️ Already marked today\n\n"
                                                            f"**{name}** (ID: {student_id})"
                                                        )
                                                else:
                                                    st.error("❌ Student not found in database.")
                                            else:
                                                st.error("❌ Face not recognised. Not registered.")

                                except Exception as e:
                                    st.error(f"Recognition error: {e}")

                    else:
                        # Cloud — face_recognition not installed
                        st.warning(
                            "⚠️ Face recognition is only available when running locally.\n\n"
                            "The photo was captured successfully. "
                            "Run `streamlit run app.py` on your laptop for full recognition."
                        )

        # ── Tab 2: Local Webcam / Iriun Guide ─────────────────────────────────
        with tab2:
            st.subheader("🖥️ Local Webcam or Iriun Camera")

            st.markdown("### 📷 How Camera Auto-Detection Works")
            st.markdown("""
            When you run `python local_camera.py` locally, the system automatically
            tries cameras in this order:

            | Priority | Index | Camera Type |
            |----------|-------|-------------|
            | 1st | 0 | Built-in laptop webcam |
            | 2nd | 1 | Iriun / External USB cam |
            | 3rd | 2 | Second external camera |

            The first working camera is used automatically — no manual selection needed.
            """)

            st.markdown("---")
            st.markdown("### 📱 Iriun Setup Steps")
            st.markdown("""
            1. Install **Iriun Webcam** app on your phone (Play Store / App Store)
            2. Install **Iriun PC client** on your laptop from [iriun.com](https://iriun.com)
            3. Connect phone and laptop to the **same WiFi network**
            4. Open Iriun on phone first → then open on laptop
            5. Run the local scanner — it detects Iriun automatically
            """)

            st.markdown("---")
            st.markdown("### ▶️ Run Local Scanner")
            st.code("python local_camera.py", language="bash")

            st.markdown("---")
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**✅ Available on Cloud**")
                st.markdown("- Browser camera snapshot\n- Database sync\n- Student enrollment\n- Analytics dashboard")
            with col2:
                st.markdown("**🖥️ Available Locally Only**")
                st.markdown("- Live video face recognition\n- Iriun mobile camera\n- Real-time bounding boxes\n- Continuous scanning loop")

    # ══════════════════════════════════════════════════════════════════════════
    #  PAGE 3 — Enroll New Student
    # ══════════════════════════════════════════════════════════════════════════
    elif page == "🎓 Enroll New Student":
        st.title("🎓 Student Registration Portal")
        st.caption("Register a new student and upload their face reference image.")

        with st.form("registration_form", clear_on_submit=True):
            st.subheader("📋 Personal & Academic Details")

            student_id = st.text_input("Student ID (Unique Roll Number)", placeholder="e.g., 452331")
            name       = st.text_input("Full Name",                        placeholder="e.g., Rohan Sharma")
            branch     = st.text_input("Branch / Department",              placeholder="e.g., CSE")
            section    = st.text_input("Section",                          placeholder="e.g., A")

            col1, col2 = st.columns(2)
            with col1:
                year         = st.number_input("Current Academic Year", min_value=1,    max_value=4,    value=1,    step=1)
            with col2:
                joining_year = st.number_input("Year of Joining",       min_value=2020, max_value=2030, value=2026, step=1)

            st.markdown("---")
            st.subheader("📸 Biometric Image Upload")
            uploaded_file = st.file_uploader(
                "Upload Student Face Photo (JPG / PNG)",
                type=["jpg", "jpeg", "png"]
            )

            submit = st.form_submit_button("✅ Register Student")

        if submit:
            if not student_id or not name or not branch or not section or not uploaded_file:
                st.error("❌ Please fill in all fields and upload a face photo.")
            else:
                try:
                    # Save image to local images/ folder
                    images_dir = os.path.join(os.getcwd(), "images")
                    os.makedirs(images_dir, exist_ok=True)

                    ext       = os.path.splitext(uploaded_file.name)[1]
                    save_path = os.path.join(images_dir, f"{student_id}{ext}")
                    with open(save_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())

                    # Push student record to Firebase
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

                    st.success(f"🎉 **{name}** (ID: {student_id}) registered successfully!")
                    st.balloons()

                except Exception as e:
                    st.error(f"❌ Registration Error: {e}")

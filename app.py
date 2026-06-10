import streamlit as st
import firebase_admin
from firebase_admin import credentials, db
import os
import json
import pickle
from datetime import datetime

# ── Biometric and Audio-Visual Processing Packages ────────────────────────────
import cv2
import numpy as np
import face_recognition
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, WebRtcMode

# ══════════════════════════════════════════════════════════════════════════════
#  1. Page Config  (must be FIRST Streamlit command)
# ═══════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Central Attendance ERP",
    page_icon="🛡️",
    layout="wide"
)

# ══════════════════════════════════════════════════════════════════════════════
#  2. Firebase Initialization  (Cloud + Local Hybrid Core Router)
# ══════════════════════════════════════════════════════════════════════════════
if not firebase_admin._apps:
    try:
        if "firebase_key" in st.secrets:
            # Cloud mode: read from Streamlit Secrets encrypted registry
            secret = st.secrets["firebase_key"]
            key_dict = json.loads(secret) if isinstance(secret, str) else dict(secret)
            cred = credentials.Certificate(key_dict)
        else:
            # Local mode: read from file fallback system
            cred = credentials.Certificate("serviceAccountKey.json")

        # FIX: Pointing cleanly to your active US-Central Database Node Pathway
        firebase_admin.initialize_app(cred, {
            "databaseURL": "https://face-based-attendance-38ee8-default-rtdb.firebaseio.com"
        })

    except Exception as e:
        st.error(f"❌ Firebase Configuration Error: {e}")
        st.stop()

# ══════════════════════════════════════════════════════════════════════════════
#  3. Session State Registry
# ══════════════════════════════════════════════════════════════════════════════
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if "marked_today" not in st.session_state:
    st.session_state.marked_today = set()

# ══════════════════════════════════════════════════════════════════════════════
#  4. Cloud-Safe WebRTC Frame Processor Class Module
# ══════════════════════════════════════════════════════════════════════════════
class BiometricRecognitionTransformer(VideoTransformerBase):
    def __init__(self):
        self.encode_list_known = []
        self.student_ids = []
        
        # Pre-load vectorized embedding models into instance memory threads
        if os.path.exists("encodings.p"):
            with open("encodings.p", "rb") as f:
                self.encode_list_known, self.student_ids = pickle.load(f)

    def transform(self, frame):
        # Convert raw video socket packet into an active array matrix layout
        img = frame.to_ndarray(format="bgr24")
        
        if len(self.encode_list_known) == 0:
            # Return raw video frame clean if matching signatures don't exist yet
            return img

        # Compress canvas framework scale to optimize network compute throughput
        img_small = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)
        img_rgb = cv2.cvtColor(img_small, cv2.COLOR_BGR2RGB)

        # Execute 128-Dimension facial coordinate tracking passes
        face_locs = face_recognition.face_locations(img_rgb)
        face_encs = face_recognition.face_encodings(img_rgb, face_locs)

        for face_loc, face_enc in zip(face_locs, face_encs):
            matches = face_recognition.compare_faces(self.encode_list_known, face_enc, tolerance=0.55)
            face_dists = face_recognition.face_distance(self.encode_list_known, face_enc)
            
            if len(face_dists) > 0:
                best_match = int(np.argmin(face_dists))

                if matches[best_match]:
                    student_id = self.student_ids[best_match]
                    
                    # Compute spatial bounding markers back to raw display size canvas scales
                    top, right, bottom, left = [v * 2 for v in face_loc]
                    
                    # Draw visual interface overlay borders tracking the matching user profile
                    cv2.rectangle(img, (left, top), (right, bottom), (0, 255, 0), 2)
                    cv2.putText(img, f"ID: {student_id}", (left, top - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                    try:
                        # Synchronize database check-in loops inside backend threads safely
                        student_ref = db.reference(f"Students/{student_id}")
                        student_data = student_ref.get()
                        
                        if student_data and isinstance(student_data, dict):
                            current_attendance = student_data.get("attendance", 0)
                            last_attendance_str = student_data.get("last_attendance", "")

                            # Enforce a secure cooling time constraint window threshold limit 
                            should_update = True
                            if last_attendance_str:
                                last_time = datetime.strptime(last_attendance_str, "%Y-%m-%d %H:%M:%S")
                                if (datetime.now() - last_time).total_seconds() < 30:
                                    should_update = False

                            if should_update:
                                student_ref.update({
                                    "attendance": current_attendance + 1,
                                    "last_attendance": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                                })
                    except Exception:
                        pass
                else:
                    # Render red alert bounding fields to signal unknown tracking vector metrics
                    top, right, bottom, left = [v * 2 for v in face_loc]
                    cv2.rectangle(img, (left, top), (right, bottom), (0, 0, 255), 2)
                    cv2.putText(img, "UNKNOWN ENTRY", (left, top - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        return img

# ══════════════════════════════════════════════════════════════════════════════
#  5. Login Gate Framework (Seamless Switch Layer Layout)
# ══════════════════════════════════════════════════════════════════════════════
if not st.session_state.logged_in:
    st.markdown(
        "<h2 style='text-align:center; margin-top:80px;'>🔒 Admin Authentication Gate</h2>",
        unsafe_allow_html=True
    )

    _, col_center, _ = st.columns([1, 2, 1])
    with col_center:
        with st.form("login_gate", clear_on_submit=False):
            admin_name = st.text_input("Admin Username")
            admin_password = st.text_input("Security Access Password", type="password")
            login_btn = st.form_submit_button("🔑 Authenticate Access")

        if login_btn:
            if admin_name == "Utkarsh Dixit" and admin_password == "DeepLearning":
                st.session_state.logged_in = True
                st.toast("✅ Access Granted! Welcome back, Admin.")
                st.rerun()
            else:
                st.error("❌ Invalid credentials. Access denied.")

# ══════════════════════════════════════════════════════════════════════════════
#  6. Main Dashboard Controls (Authenticated Access State Route Mapping)
# ══════════════════════════════════════════════════════════════════════════════
else:
    # Sidebar Operational Elements
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

    # ── PAGE 1 — Attendance Analytics ─────────────────────────────────────────
    if page == "📊 Attendance Analytics":
        st.title("📊 Attendance Monitoring Metrics")
        st.caption("Real-time data pulled from Firebase Realtime Database.")

        try:
            students_ref = db.reference("Students").get()

            if students_ref:
                total_students = len(students_ref)
                total_attendance_logs = sum(
                    data.get("attendance", 0)
                    for data in students_ref.values()
                    if isinstance(data, dict)
                )

                col1, col2 = st.columns(2)
                col1.metric("Total Enrolled Students", total_students, "Active Directory")
                col2.metric("Cumulative Attendance Hits", total_attendance_logs, "Live Syncing")

                st.markdown("---")
                st.subheader("📋 Student Roster")

                table_data = []
                for s_id, data in students_ref.items():
                    if data and isinstance(data, dict):
                        table_data.append({
                            "Student ID": s_id,
                            "Full Name": data.get("name", "N/A"),
                            "Branch": data.get("branch", "N/A"),
                            "Section": data.get("section", "N/A"),
                            "Year": data.get("year", "N/A"),
                            "Total Attendance": data.get("attendance", 0),
                            "Last Check-In": data.get("last_attendance", "N/A"),
                        })

                st.dataframe(table_data, use_container_width=True)
            else:
                st.info("📂 Database is empty. Register students in the Enroll tab first.")

        except Exception as e:
            st.error(f"❌ Failed to fetch database: {e}")

    # ── PAGE 2 — Live Attendance Camera (UPGRADED CLOUD WEB RTC ENGINE) ────────
    elif page == "📸 Live Attendance Camera":
        st.title("📸 Biometric Face-Scanning Terminal")
        st.write("---")

        col_feed, col_info = st.columns([2, 1])

        with col_feed:
            st.subheader("📹 Cloud Portal Live Scanning Console")
            
            if not os.path.exists("encodings.p"):
                st.error("⚠️ System Offline: `encodings.p` vector matrix signature payload missing from tracking directory. System cannot verify tracking records.")
            else:
                # Upgraded operational loop layer: opens active stream routes natively on the web dashboard page directly
                ctx = webrtc_streamer(
                    key="biometric-attendance-stream",
                    mode=WebRtcMode.SENDRECV,
                    video_transformer_factory=BiometricRecognitionTransformer,
                    async_transform=True,
                    rtc_configuration={
                        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
                    }
                )

        with col_info:
            st.subheader("⚡ Core Link Stream Status")
            st.info("💡 Real-time WebRTC architecture active. Click 'Start' inside the workspace display module frame panel to activate biometric hardware device streaming authorizations directly via your browser terminal link.")
            
            st.markdown("#### Database Gateway Link")
            st.success("🛰️ Connected Target Node Hub: Firebase Live Sync Active")

    # ── PAGE 3 — Enroll New Student ───────────────────────────────────────────
    elif page == "🎓 Enroll New Student":
        st.title("🎓 Student Registration Portal")
        st.caption("Register a new student and upload their face reference image.")

        with st.form("registration_form", clear_on_submit=True):
            st.subheader("📋 Personal & Academic Details")

            student_id = st.text_input("Student ID (Unique Roll Number)", placeholder="e.g., 452331")
            name = st.text_input("Full Name", placeholder="e.g., Rohan Sharma")
            branch = st.text_input("Branch / Department", placeholder="e.g., CSE")
            section = st.text_input("Section", placeholder="e.g., A")

            col1, col2 = st.columns(2)
            with col1:
                year = st.number_input("Current Academic Year", min_value=1, max_value=4, value=1, step=1)
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
                st.error("❌ Please fill in all fields and upload a face photo.")
            else:
                try:
                    images_dir = os.path.join(os.getcwd(), "images")
                    os.makedirs(images_dir, exist_ok=True)

                    ext = os.path.splitext(uploaded_file.name)[1]
                    save_path = os.path.join(images_dir, f"{student_id}{ext}")
                    with open(save_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())

                    student_data = {
                        "name": name,
                        "branch": branch,
                        "section": section,
                        "year": int(year),
                        "yearOfJoining": int(joining_year),
                        "attendance": 0,
                        "last_attendance": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    }
                    db.reference(f"Students/{student_id}").set(student_data)

                    st.success(f"🎉 **{name}** (ID: {student_id}) registered successfully!")
                    st.balloons()

                except Exception as e:
                    st.error(f"❌ Registration Error: {e}")

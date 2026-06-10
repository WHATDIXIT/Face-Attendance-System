import streamlit as st
import firebase_admin
from firebase_admin import credentials, db
import os
import json
import base64
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
#  2. Firebase Initialization
# ══════════════════════════════════════════════════════════════════════════════
if not firebase_admin._apps:
    try:
        if "firebase_key" in st.secrets:
            secret   = st.secrets["firebase_key"]
            key_dict = json.loads(secret) if isinstance(secret, str) else dict(secret)
            cred     = credentials.Certificate(key_dict)
        else:
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
for key, default in [("logged_in", False), ("marked_today", set()), ("camera_frame", None)]:
    if key not in st.session_state:
        st.session_state[key] = default

# ══════════════════════════════════════════════════════════════════════════════
#  4. Helper: process a raw image (bytes/base64) and mark attendance
# ══════════════════════════════════════════════════════════════════════════════
def process_face_image(image_bytes: bytes):
    """
    Run face recognition on image_bytes.
    Returns (student_id, name, confidence) on success, or raises RuntimeError.
    """
    if not CV_AVAILABLE:
        raise RuntimeError("face_recognition library is not installed in this environment.")

    if not os.path.exists("encodings.p"):
        raise RuntimeError("`encodings.p` not found. Generate it locally first using `python encode_faces.py`.")

    with open("encodings.p", "rb") as f:
        encode_list_known, student_ids = pickle.load(f)

    nparr     = np.frombuffer(image_bytes, np.uint8)
    img_cv    = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    img_rgb   = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
    img_small = cv2.resize(img_rgb, (0, 0), fx=0.5, fy=0.5)

    face_locs = face_recognition.face_locations(img_small)
    face_encs = face_recognition.face_encodings(img_small, face_locs)

    if not face_encs:
        raise RuntimeError("No face detected. Try again with better lighting or move closer.")

    face_enc   = face_encs[0]
    matches    = face_recognition.compare_faces(encode_list_known, face_enc, tolerance=0.6)
    face_dists = face_recognition.face_distance(encode_list_known, face_enc)
    best_match = int(np.argmin(face_dists))

    if not matches[best_match]:
        raise RuntimeError("Face not recognised. This person is not registered.")

    student_id = student_ids[best_match]
    confidence = round((1 - face_dists[best_match]) * 100, 1)
    return student_id, confidence


def mark_attendance(student_id: str, confidence: float):
    """
    Fetch student from Firebase and update attendance counter.
    Returns (name, already_marked).
    """
    student_data = db.reference(f"Students/{student_id}").get()
    if not student_data:
        raise RuntimeError(f"Student ID {student_id} not found in database.")

    name = student_data.get("name", student_id)

    if student_id in st.session_state.marked_today:
        return name, True

    current_att = student_data.get("attendance", 0)
    db.reference(f"Students/{student_id}").update({
        "attendance":      current_att + 1,
        "last_attendance": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    })
    st.session_state.marked_today.add(student_id)
    return name, False


# ══════════════════════════════════════════════════════════════════════════════
#  5. Camera HTML Component (works on cloud + local, asks for permission)
# ══════════════════════════════════════════════════════════════════════════════
CAMERA_HTML = """
<style>
  #camera-container {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 12px;
    font-family: sans-serif;
  }
  #video {
    width: 100%;
    max-width: 540px;
    border-radius: 12px;
    border: 2px solid #444;
    background: #111;
  }
  #canvas { display: none; }
  .cam-btn {
    padding: 10px 28px;
    font-size: 15px;
    font-weight: 600;
    border: none;
    border-radius: 8px;
    cursor: pointer;
    transition: opacity .2s;
  }
  .cam-btn:hover { opacity: 0.85; }
  #startBtn  { background: #2ecc71; color: #fff; }
  #captureBtn{ background: #3498db; color: #fff; display: none; }
  #status    { font-size: 13px; color: #aaa; min-height: 20px; }
  #preview   { max-width: 540px; border-radius: 8px; display: none; margin-top: 6px; }
</style>

<div id="camera-container">
  <video id="video" autoplay playsinline muted></video>
  <canvas id="canvas"></canvas>
  <div style="display:flex; gap:10px;">
    <button class="cam-btn" id="startBtn"   onclick="startCamera()">📷 Enable Camera</button>
    <button class="cam-btn" id="captureBtn" onclick="capturePhoto()">🔍 Capture & Scan</button>
  </div>
  <div id="status">Click "Enable Camera" to request camera access.</div>
  <img id="preview" alt="Captured frame"/>
</div>

<script>
  let stream = null;

  async function startCamera() {
    const status = document.getElementById('status');
    status.textContent = '⏳ Requesting camera permission…';
    try {
      stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: 'user', width: { ideal: 1280 }, height: { ideal: 720 } },
        audio: false
      });
      document.getElementById('video').srcObject = stream;
      document.getElementById('startBtn').style.display   = 'none';
      document.getElementById('captureBtn').style.display = 'inline-block';
      status.textContent = '✅ Camera active. Position face in frame and click Capture.';
    } catch (err) {
      status.textContent = '❌ Camera access denied: ' + err.message;
    }
  }

  function capturePhoto() {
    const video  = document.getElementById('video');
    const canvas = document.getElementById('canvas');
    canvas.width  = video.videoWidth  || 640;
    canvas.height = video.videoHeight || 480;
    canvas.getContext('2d').drawImage(video, 0, 0);

    // Show preview
    const dataURL = canvas.toDataURL('image/jpeg', 0.92);
    const preview = document.getElementById('preview');
    preview.src     = dataURL;
    preview.style.display = 'block';

    document.getElementById('status').textContent = '📡 Sending image for recognition…';

    // Send base64 data to Streamlit via postMessage
    const base64 = dataURL.split(',')[1];
    window.parent.postMessage({ type: 'streamlit:setComponentValue', value: base64 }, '*');
  }
</script>
"""


# ══════════════════════════════════════════════════════════════════════════════
#  6. Login Gate
# ══════════════════════════════════════════════════════════════════════════════
if not st.session_state.logged_in:
    st.markdown(
        "<h2 style='text-align:center; margin-top:80px;'>🔒 Admin Authentication Gate</h2>",
        unsafe_allow_html=True,
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
#  7. Main Dashboard
# ══════════════════════════════════════════════════════════════════════════════
else:
    # ── Sidebar ───────────────────────────────────────────────────────────────
    st.sidebar.title("🛡️ Admin Command Panel")
    st.sidebar.write("Logged in as: **Utkarsh Dixit**")
    st.sidebar.markdown("---")

    page = st.sidebar.radio(
        "Navigate Workspace",
        ["📊 Attendance Analytics", "📸 Live Attendance Camera", "🎓 Enroll New Student"],
    )

    st.sidebar.markdown("---")
    if st.sidebar.button("🚪 Secure Logout", use_container_width=True):
        st.session_state.logged_in      = False
        st.session_state.marked_today   = set()
        st.session_state.camera_frame   = None
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

        tab1, tab2, tab3 = st.tabs([
            "🌐 Browser Camera (Cloud + Local)",
            "📷 Quick Snapshot",
            "🖥️ Local Webcam Guide",
        ])

        # ── Tab 1: Full Browser Camera with JS permission request ─────────────
        with tab1:
            st.info(
                "💡 This camera runs entirely in your browser — works on Streamlit Cloud "
                "and locally. Your browser will ask for camera permission on first use."
            )

            st.markdown("#### Live Camera Feed")

            # Render the JS camera widget
            st.components.v1.html(CAMERA_HTML, height=520, scrolling=False)

            st.markdown("---")
            st.markdown("#### 📤 Paste Captured Image for Recognition")
            st.caption(
                "After clicking **Capture & Scan** above, copy the Base64 string from the "
                "browser console (or use the Upload method below if your deployment blocks postMessage)."
            )

            # ── Method A: Upload file (most reliable cross-origin fallback) ───
            uploaded_file = st.file_uploader(
                "Upload a captured photo (JPG/PNG) — use this if the live scan doesn't auto-submit",
                type=["jpg", "jpeg", "png"],
                key="live_upload",
            )

            if uploaded_file:
                col_img, col_result = st.columns(2)
                with col_img:
                    st.image(uploaded_file, caption="Uploaded Frame", use_container_width=True)

                with col_result:
                    if not CV_AVAILABLE:
                        st.warning(
                            "⚠️ Face recognition libraries are not installed in this environment.\n\n"
                            "Run `streamlit run app.py` locally with `face_recognition`, "
                            "`opencv-python`, and `numpy` installed for full recognition."
                        )
                    else:
                        with st.spinner("🔍 Running face recognition…"):
                            try:
                                student_id, confidence = process_face_image(uploaded_file.getvalue())
                                name, already_marked   = mark_attendance(student_id, confidence)

                                if already_marked:
                                    st.info(
                                        f"ℹ️ **Already marked today**\n\n"
                                        f"**{name}** (ID: `{student_id}`)\n\n"
                                        f"Confidence: {confidence}%"
                                    )
                                else:
                                    st.success(
                                        f"✅ **Attendance Marked!**\n\n"
                                        f"**{name}** (ID: `{student_id}`)\n\n"
                                        f"Confidence: {confidence}%"
                                    )
                                    st.balloons()

                            except RuntimeError as e:
                                st.error(f"❌ {e}")
                            except Exception as e:
                                st.error(f"❌ Recognition error: {e}")

        # ── Tab 2: Quick Snapshot (st.camera_input) ───────────────────────────
        with tab2:
            st.info("💡 Uses the browser's built-in camera snapshot widget.")

            img_file = st.camera_input("📸 Click to take a snapshot")

            if img_file is not None:
                col_img, col_result = st.columns(2)
                with col_img:
                    st.image(img_file, caption="Captured Photo", use_container_width=True)

                with col_result:
                    if not CV_AVAILABLE:
                        st.warning(
                            "⚠️ Face recognition is only available when running locally.\n\n"
                            "The photo was captured successfully. Run `streamlit run app.py` "
                            "on your laptop for full recognition."
                        )
                    else:
                        with st.spinner("🔍 Scanning face…"):
                            try:
                                student_id, confidence = process_face_image(img_file.getvalue())
                                name, already_marked   = mark_attendance(student_id, confidence)

                                if already_marked:
                                    st.info(
                                        f"ℹ️ **Already marked today**\n\n"
                                        f"**{name}** (ID: `{student_id}`)\n\n"
                                        f"Confidence: {confidence}%"
                                    )
                                else:
                                    st.success(
                                        f"✅ **Attendance Marked!**\n\n"
                                        f"**{name}** (ID: `{student_id}`)\n\n"
                                        f"Confidence: {confidence}%"
                                    )
                                    st.balloons()

                            except RuntimeError as e:
                                st.error(f"❌ {e}")
                            except Exception as e:
                                st.error(f"❌ Recognition error: {e}")

        # ── Tab 3: Local Webcam / Iriun Guide ─────────────────────────────────
        with tab3:
            st.subheader("🖥️ Continuous Local Webcam Scanner")
            st.markdown("""
            When you run `python local_camera.py` on your machine, the system
            tries cameras in this order:

            | Priority | Index | Camera Type |
            |----------|-------|-------------|
            | 1st | 0 | Built-in laptop webcam |
            | 2nd | 1 | Iriun / External USB cam |
            | 3rd | 2 | Second external camera |

            The first working camera is used automatically.
            """)

            st.markdown("---")
            st.markdown("### 📱 Iriun Mobile Camera Setup")
            st.markdown("""
            1. Install **Iriun Webcam** on your phone (Play Store / App Store)
            2. Install **Iriun PC client** from [iriun.com](https://iriun.com)
            3. Connect phone and laptop to the **same Wi-Fi network**
            4. Open Iriun on phone first → then on laptop
            5. Run the local scanner — Iriun is detected automatically
            """)

            st.markdown("---")
            st.markdown("### ▶️ Run Local Scanner")
            st.code("python local_camera.py", language="bash")

            st.markdown("---")
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**✅ Available on Cloud**")
                st.markdown(
                    "- Browser camera with permission request\n"
                    "- Snapshot upload + recognition\n"
                    "- Database sync\n"
                    "- Student enrollment\n"
                    "- Analytics dashboard"
                )
            with col2:
                st.markdown("**🖥️ Available Locally Only**")
                st.markdown(
                    "- Continuous live video loop\n"
                    "- Real-time bounding boxes\n"
                    "- Iriun mobile camera\n"
                    "- No-click automatic detection"
                )

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
                type=["jpg", "jpeg", "png"],
            )

            submit = st.form_submit_button("✅ Register Student")

        if submit:
            if not student_id or not name or not branch or not section or not uploaded_file:
                st.error("❌ Please fill in all fields and upload a face photo.")
            else:
                try:
                    images_dir = os.path.join(os.getcwd(), "images")
                    os.makedirs(images_dir, exist_ok=True)

                    ext       = os.path.splitext(uploaded_file.name)[1]
                    save_path = os.path.join(images_dir, f"{student_id}{ext}")
                    with open(save_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())

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

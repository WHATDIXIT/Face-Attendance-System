import streamlit as st
import firebase_admin
from firebase_admin import credentials, db
import cv2
import pickle
import face_recognition
import numpy as np
import cvzone
import os
from datetime import datetime
import json

# Page Layout Setup
st.set_page_config(page_title="Central Attendance ERP", page_icon="🛡️", layout="wide")

# Initialize Firebase safely (Cloud & Local Hybrid Mode)
if not firebase_admin._apps:
    try:
        # FIX: Directly read from st.secrets without the broken sub-nested key wrapper
        if "firebase_key" in st.secrets:
            secret_json = st.secrets["firebase_key"]
            # Check if secrets manager passed it as a dictionary already or raw text string
            if isinstance(secret_json, str):
                key_dict = json.loads(secret_json)
            else:
                key_dict = dict(secret_json)
            cred = credentials.Certificate(key_dict)
        else:
            # Local fallback for running on your personal laptop station terminal
            cred = credentials.Certificate("YourserviceAccountKey.json")
            
        firebase_admin.initialize_app(cred, {
            "databaseURL": "https://face-based-attendance-38ee8-default-rtdb.asia-southeast1.firebasedatabase.app/"
        })
    except Exception as e:
        st.error(f"❌ Firebase Configuration Error: {e}")
        st.stop()

# --- SESSION STATE LOCK FOR LOGIN SYSTEM ---
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

# --- COOKIE-STYLE LOGIN INTERFACE ---
if not st.session_state.logged_in:
    st.markdown("<h2 style='text-align: center;'>🔒 Admin Authentication Gate</h2>", unsafe_allow_html=True)
    
    with st.form("login_gate", clear_on_submit=False):
        admin_name = st.text_input("Admin Username")
        admin_password = st.text_input("Security Access Password", type="password")
        login_btn = st.form_submit_button("Authenticate Access Link")
        
        if login_btn:
            if admin_name == "Utkarsh Dixit" and admin_password == "DeepLearning":
                st.session_state.logged_in = True
                st.toast("Access Granted! Welcome back Admin.", icon="✅")
                st.rerun()
            else:
                st.error("❌ Invalid Admin Username or Password. Security access denied.")
    st.stop() 

# --- MAIN DASHBOARD CONTROL PORTAL (LOGGED IN) ---
st.sidebar.title("🛡️ Admin Command Panel")
st.sidebar.write(f"Logged in as: **Utkarsh Dixit**")

# Sidebar Application Navigation Links
page = st.sidebar.radio("Navigate Workspace", ["📊 Attendance Analytics", "📸 Live Attendance Camera", "🎓 Enroll New Student"])

# Logout Utility
if st.sidebar.button("Secure Logout"):
    st.session_state.logged_in = False
    st.rerun()

# ==============================================================================
# PAGE 1: ATTENDANCE ANALYTICS
# ==============================================================================
if page == "📊 Attendance Analytics":
    st.title("📊 Attendance Monitoring Metrics")
    st.write("Real-time telemetry metrics pulled straight from the Firebase Realtime Database cloud nodes.")
    
    students_ref = db.reference('Students').get()
    
    if students_ref:
        total_students = len(students_ref)
        total_attendance_logs = sum([data.get('attendance', 0) for data in students_ref.values() if isinstance(data, dict)])
        
        col1, col2 = st.columns(2)
        col1.metric(label="Total Enrolled Students", value=total_students, delta="Active Directory")
        col2.metric(label="Cumulative Attendance Hits", value=total_attendance_logs, delta="Live Syncing", delta_color="inverse")
        
        st.markdown("---")
        st.subheader("📋 Core Roster Logs Directory")
        
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
                    "Last Check-In Timestamp": data.get("last_attendance", "N/A")
                })
        st.dataframe(table_data, use_container_width=True)
    else:
        st.info("📂 Database directory tree is currently empty. Go to the Enrollment tab to register new students.")

# ==============================================================================
# PAGE 2: LIVE ATTENDANCE CAMERA
# ==============================================================================
elif page == "📸 Live Attendance Camera":
    st.title("📸 Biometric Face-Scanning Core Terminal")
    st.write("Click 'Start Camera' below to capture video input frames from your tethered Iriun Mobile camera feed.")
    
    # SAFETY: Explicit cloud hardware warning notification block
    st.warning("⚠️ Cloud Deployment Warning: The live camera loop can only track frame input streams when executed locally via your laptop station console (`localhost:8501`). Running it directly inside the web browser framework will trigger a hardware connection error.")
    
    run_cam = st.checkbox("🟢 Toggle Live Tracking System Node Active")
    FRAME_WINDOW = st.image([]) 
    
    if run_cam:
        # Safe model loading sequence wrapper
        if not os.path.exists('encodings.p'):
            st.error("❌ System Error: 'encodings.p' model vector map missing from the application root folder. Please run your local generator script first.")
            st.stop()
            
        try:
            with open('encodings.p', 'rb') as file:
                encodeListKnown, studentIds = pickle.load(file)
            
            capture = cv2.VideoCapture(0, cv2.CAP_DSHOW)
            capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            
            st.toast("AI Encodings linked successfully! Camera is warming up.", icon="🚀")
            counter = 0
            id = -1
            
            while run_cam:
                success, img = capture.read()
                if not success:
                    st.error("Failed to read image array from camera connection. Ensure your webcam link is not occupied by another background app.")
                    break
                
                imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
                imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)
                
                faceCurFrame = face_recognition.face_locations(imgS)
                encodeCurFrame = face_recognition.face_encodings(imgS, faceCurFrame)
                
                if faceCurFrame:
                    for encodeFace, faceLoc in zip(encodeCurFrame, faceCurFrame):
                        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
                        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
                        matchIndex = np.argmin(faceDis)
                        
                        if matches[matchIndex]:
                            y1, x2, y2, x1 = faceLoc
                            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
                            cvzone.cornerRect(img, (x1, y1, x2 - x1, y2 - y1), rt=1, colorC=(0, 255, 0))
                            
                            id = studentIds[matchIndex]
                            if counter == 0:
                                counter = 1
                    
                    if counter != 0:
                        if counter == 1:
                            StudentInfo = db.reference(f'Students/{id}').get()
                            if StudentInfo:
                                datetimeObj = datetime.strptime(StudentInfo.get('last_attendance', datetime.now().strftime("%Y-%m-%d %H:%M:%S")), "%Y-%m-%d %H:%M:%S")
                                secondElapsed = (datetime.now() - datetimeObj).total_seconds()
                                
                                if secondElapsed > 30:
                                    ref = db.reference(f'Students/{id}')
                                    current_attendance = StudentInfo.get('attendance', 0) + 1
                                    ref.child('attendance').set(current_attendance)
                                    ref.child('last_attendance').set(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
                                    st.toast(f"✅ Attendance Logged: {StudentInfo.get('name', 'Student')}", icon="🎉")
                            counter = 2
                        
                        if 'StudentInfo' in locals() and StudentInfo:
                            cv2.putText(img, f"Name: {StudentInfo.get('name', 'N/A')}", (20, 40), cv2.FONT_HERSHEY_COMPLEX, 0.6, (255, 255, 255), 1)
                            cv2.putText(img, f"Attendance: {StudentInfo.get('attendance', 0)}", (20, 70), cv2.FONT_HERSHEY_COMPLEX, 0.6, (0, 255, 0), 1)
                else:
                    counter = 0
                
                FRAME_WINDOW.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            
            capture.release()
        except Exception as e:
            st.error(f"Hardware initialization failed: {e}. Note: Live local camera access requires hosting the script on your local machine.")

# ==============================================================================
# PAGE 3: ENROLL NEW STUDENT
# ==============================================================================
elif page == "🎓 Enroll New Student":
    st.title("🎓 Student Registration Portal")
    st.write("Fill out the credentials below to register a new student profile into the cloud directory.")
    
    with st.form("registration_form", clear_on_submit=True):
        st.subheader("Personal & Academic Details")
        student_id = st.text_input("Student ID (Unique Roll Number/Code)", placeholder="e.g., 452331")
        name = st.text_input("Full Name", placeholder="e.g., Rohan Sharma")
        branch = st.text_input("Branch / Department", placeholder="e.g., CSE")
        section = st.text_input("Section", placeholder="e.g., A")
        
        col1, col2 = st.columns(2)
        with col1:
            year = st.number_input("Current Academic Year", min_value=1, max_value=4, value=1, step=1)
        with col2:
            joining_year = st.number_input("Year of Joining", min_value=2020, max_value=2030, value=2026, step=1)
            
        st.markdown("---")
        st.subheader("Biometric Image Upload")
        uploaded_file = st.file_uploader("Upload Student Face Reference Image (JPG/PNG)", type=["jpg", "jpeg", "png"])
        
        submit_button = st.form_submit_button("Securely Register Student")
    
    if submit_button:
        if not student_id or not name or not branch or not section or not uploaded_file:
            st.error("❌ Registration Failed: Please fill in all fields and upload a face reference image.")
        else:
            try:
                images_dir = os.path.join(os.getcwd(), "images")
                os.makedirs(images_dir, exist_ok=True)
                
                file_extension = os.path.splitext(uploaded_file.name)[1]
                target_filename = f"{student_id}{file_extension}"
                full_save_path = os.path.join(images_dir, target_filename)
                
                with open(full_save_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                    
                student_data = {
                    "name": name,
                    "branch": branch,
                    "section": section,
                    "year": int(year),
                    "yearOfJoining": int(joining_year),
                    "attendance": 0,
                    "last_attendance": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
                
                ref = db.reference(f"Students/{student_id}")
                ref.set(student_data)
                
                st.success(f"🎉 Success! {name} ({student_id}) has been registered into the cloud database system.")
            except Exception as e:
                st.error(f"System Error: {e}")

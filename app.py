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

# 1. Page Layout Setup (Must be the very first Streamlit command)
st.set_page_config(page_title="Central Attendance ERP", page_icon="🛡️", layout="wide")

# 2. Initialize Firebase safely (Cloud & Local Hybrid Mode)
if not firebase_admin._apps:
    try:
        if "firebase_key" in st.secrets:
            secret_json = st.secrets["firebase_key"]
            if isinstance(secret_json, str):
                key_dict = json.loads(secret_json)
            else:
                key_dict = dict(secret_json)
            cred = credentials.Certificate(key_dict)
        else:
            cred = credentials.Certificate("YourserviceAccountKey.json")
            
        # FIX: Removed the trailing slash from the database URL string
        firebase_admin.initialize_app(cred, {
            "databaseURL": "https://face-based-attendance-38ee8-default-rtdb.asia-southeast1.firebasedatabase.app"
        })
    except Exception as e:
        st.error(f"❌ Firebase Configuration Error: {e}")
        st.stop()

# 3. Session State Lock for Login System
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

# 4. Security Gateway Login Panel
if not st.session_state.logged_in:
    st.markdown("<h2 style='text-align: center;'>🔒 Admin Authentication Gate</h2>", unsafe_allow_html=True)
    
    _, col_center, _ = st.columns([1, 2, 1])
    with col_center:
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
    
    # FIX: Explicitly stop the execution flow here so Streamlit forces the login page HTML to draw
    st.stop() 

# ==============================================================================
# MAIN DASHBOARD CONTROL PORTAL (Runs only when logged_in is True)
# ==============================================================================
st.sidebar.title("🛡️ Admin Command Panel")
st.sidebar.write(f"Logged in as: **Utkarsh Dixit**")

page = st.sidebar.radio("Navigate Workspace", ["📊 Attendance Analytics", "📸 Live Attendance Camera", "🎓 Enroll New Student"])

if st.sidebar.button("Secure Logout"):
    st.session_state.logged_in = False
    st.rerun()

# --- PAGE 1: ATTENDANCE ANALYTICS ---
if page == "📊 Attendance Analytics":
    st.title("📊 Attendance Monitoring Metrics")
    st.write("Real-time telemetry metrics pulled straight from the Firebase Realtime Database cloud nodes.")
    
    try:
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
    except Exception as e:
        st.error(f"Failed to pull metrics from database node: {e}")

# --- PAGE 2: LIVE ATTENDANCE CAMERA ---
elif page == "📸 Live Attendance Camera":
    st.title("📸 Biometric Face-Scanning Core Terminal")
    st.info("💡 Local Terminal Deployment Active")
    st.warning("⚠️ Cloud Hardware Restriction: To run the live webcam scanning loop with your tethered Iriun Mobile camera feed, run this script locally on your laptop station console using: `streamlit run app.py`")
    st.write("### System Ready for Local Connection")
    st.code("Command: streamlit run app.py", language="bash")

# --- PAGE 3: ENROLL NEW STUDENT ---
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

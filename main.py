import os
import cv2
import pickle
import face_recognition
import numpy as np
import cvzone
import firebase_admin
from firebase_admin import credentials
from firebase_admin import db
from datetime import datetime

# 1. Initialize Firebase with your Service Key and Realtime Database
cred = credentials.Certificate("YourserviceAccountKey.json")
firebase_admin.initialize_app(cred, {
    "databaseURL": "https://face-based-attendance-38ee8-default-rtdb.asia-southeast1.firebasedatabase.app/"
})

# 2. Initialize your Iriun Mobile Camera via DirectShow (Index 0)
capture = cv2.VideoCapture(0, cv2.CAP_DSHOW)
capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# 3. Load your pre-compiled face encodings
file = open('encodings.p', 'rb')
encodeListWithIds = pickle.load(file)
file.close()
encodeListKnown, studentIds = encodeListWithIds
print("Encoding loaded successfully. Attendance System is Live!")

id = -1
counter = 0

while True:
    success, img = capture.read()
    if not success:
        print("Failed to grab frame from your phone camera.")
        break

    # Scale down the image frame for faster real-time processing performance
    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    faceCurFrame = face_recognition.face_locations(imgS)
    encodeCurFrame = face_recognition.face_encodings(imgS, faceCurFrame)

    if faceCurFrame:
        for encodeFace, faceLoc in zip(encodeCurFrame, faceCurFrame):
            matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
            faceDis = face_distance = face_recognition.face_distance(encodeListKnown, encodeFace)
            matchIndex = np.argmin(faceDis)

            if matches[matchIndex]:
                # Scale the face bounding box coordinates back up to match original size
                y1, x2, y2, x1 = faceLoc
                y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
                
                # Draw a clean tracking corner box around your detected face
                cvzone.cornerRect(img, (x1, y1, x2 - x1, y2 - y1), rt=1, colorC=(0, 255, 0))
                
                id = studentIds[matchIndex]
                if counter == 0:
                    counter = 1

        if counter != 0:
            if counter == 1:
                # Fetch student data records directly from the database
                StudentInfo = db.reference(f'Students/{id}').get()
                
                # Double-check the 30-second security cooldown window
                datetimeObj = datetime.strptime(StudentInfo['last_attendance'], "%Y-%m-%d %H:%M:%S")
                secondElapsed = (datetime.now() - datetimeObj).total_seconds()

                if secondElapsed > 30:
                    ref = db.reference(f'Students/{id}')
                    StudentInfo['attendance'] += 1
                    ref.child('attendance').set(StudentInfo['attendance'])
                    ref.child('last_attendance').set(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
                    print(f"✅ Attendance updated for {StudentInfo['name']}. Total: {StudentInfo['attendance']}")
                else:
                    print(f"⏱️ Cooldown active for {StudentInfo['name']}. Please wait.")
                
                counter = 2

            # Overlay student info variables clearly onto your live camera window feed
            cv2.putText(img, f"Name: {StudentInfo['name']}", (20, 40), cv2.FONT_HERSHEY_COMPLEX, 0.6, (255, 255, 255), 1)
            cv2.putText(img, f"ID: {id}", (20, 70), cv2.FONT_HERSHEY_COMPLEX, 0.6, (255, 255, 255), 1)
            cv2.putText(img, f"Attendance: {StudentInfo['attendance']}", (20, 100), cv2.FONT_HERSHEY_COMPLEX, 0.6, (0, 255, 0), 1)
    else:
        counter = 0

    # Display the direct standalone OpenCV feed window frame
    cv2.imshow('Face Attendance System (No-GUI Mode)', img)

    # Press the 'q' key on your keyboard to instantly close the camera loop window
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

capture.release()
cv2.destroyAllWindows()
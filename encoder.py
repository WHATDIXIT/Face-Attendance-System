import cv2
import face_recognition 
import pickle
import os
import firebase_admin
from firebase_admin import credentials
from firebase_admin import db

# Initialize Firebase with your actual Singapore Realtime Database link
cred = credentials.Certificate("YourserviceAccountKey.json")
firebase_admin.initialize_app(cred, {
    "databaseURL": "https://face-based-attendance-38ee8-default-rtdb.asia-southeast1.firebasedatabase.app/"
})

# Absolute path safeguard pointing directly to your local folder
imgpath = r'C:\Users\HP\Desktop\Face Based Attedance\face-recognition-attendance-main\images'
pathlist = os.listdir(imgpath)
imglist = []
studentIds = []

for i in pathlist:
    imglist.append(cv2.imread(os.path.join(imgpath, i)))
    studentIds.append(os.path.splitext(i)[0])

print(f"Total student images found locally: {len(imglist)}")

def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encodes = face_recognition.face_encodings(img)
        if len(encodes) > 0:
            encodeList.append(encodes[0])
        else:
            print("No face found in one image, skipping.")
    return encodeList

print("Encoding started...")
encodeListknown = findEncodings(imglist)
print("Encoding complete")

encodeListWithIds = [encodeListknown, studentIds]
file = open('encodings.p', 'wb')
pickle.dump(encodeListWithIds, file)
file.close()

print("All encodings saved successfully to encodings.p")
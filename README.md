\# Smart Face-Recognition Attendance System 🚀



A real-time, lightweight computer vision application that detects faces via a connected camera feed, matches them against pre-compiled facial vector encodings, and logs attendance instantly to a cloud-based Firebase Realtime Database with an automated security cooldown mechanism.



\---



\## ✨ Features



Wired/Wireless Camera Tethering: Fully optimized to utilize mobile device lenses over USB via Iriun Webcam, ensuring high FPS tracking on laptops with missing or low-resolution webcams.

Rapid Matrix Scaling: Downsamples live image vectors to $0.25\\times$ scale during vector scanning for optimal real-time matching, scaling coordinate blocks back up smoothly to render bounding markers.

Anti-Spam Cooldown Loop: Engages a strict 30-second security window between check-ins by parsing delta timestamps mathematically:

&#x20;   $$\\Delta t = t\_{\\text{current}} - t\_{\\text{last\\\_attendance}} > 30\\text{ seconds}$$

Standalone GUI-less Overlay: Overlays tracking metrics (`Name`, `ID`, `Attendance Count`) as a live head-up text array directly over the video matrix frame.



\---



\## 📋 Directory Structure



```text

face-recognition-attendance-main/

├── images/                      # Storage folder for raw student reference images

│   └── 452331.png               # Reference photo labeled with Student ID

├── AddDatatoDatabase.py         # Administrative script to push/edit profile nodes

├── encoder.py                   # Facial landmark vector compiler script

├── main.py                      # Core live tracking and attendance application

├── encodings.p                  # Compiled binary matrix of processed facial vectors

└── YourserviceAccountKey.json   # Private Firebase SDK security credentials key



```



\---



\## 🚀 Installation \& Setup



\### 1. Environment Configuration



Ensure you have Python 3.11+ installed. Run the following deployment script to configure the exact version-harmonized package ecosystem:



```cmd

pip install opencv-python==4.8.1.78 opencv-contrib-python==4.8.1.78 face-recognition cvzone firebase-admin pygrabber "numpy<2"



```



\### 2. Hardware Video Stream Tethering (Mobile Device)



1\. Install Iriun Webcam on both your PC and your mobile device.

2\. Enable USB Debugging inside your phone's Developer Options settings panel.

3\. Plug your phone into your PC via a USB cable, disable your mobile hotspot, and open Iriun on both devices to bind the hardware stream.



\### 3. Initialize the Student Records



1\. Place a square, clear photo of the student's face inside the `images/` directory. Name the file exactly after their identification number (e.g., `452331.png`).

2\. Open `AddDatatoDatabase.py` and populate the student profile parameters dictionary mapping to your active database instance. Run the initialization script:

```cmd

python AddDatatoDatabase.py



```







\### 4. Compile Vector Encodings



Run the feature extraction script to parse the images directory, isolate spatial face vectors, and compile the matching model bin:



```cmd

python encoder.py



```



\---



\## 🖥️ Execution



To launch the real-time tracking engine and start taking attendance live:



```cmd

python main.py



```



To Exit: Click onto the active tracking window and tap the `q` key on your keyboard to release camera locks gracefully.



\---



\## 🔒 Security Note



Do not commit or upload `YourserviceAccountKey.json` to public repositories. Keep private cloud infrastructure certificates secure and added to your `.gitignore` configuration.



\---



\## 👤 Author



Utkarsh - Initial Work \& Core Developer - \[WHATDIXIT](https://github.com/WHATDIXIT)



```


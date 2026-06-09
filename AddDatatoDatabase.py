import firebase_admin
from firebase_admin import credentials
from firebase_admin import db

# 1. Initialize Firebase connection safely within your project environment
cred = credentials.Certificate("YourserviceAccountKey.json")
firebase_admin.initialize_app(cred, {
    "databaseURL": "https://face-based-attendance-38ee8-default-rtdb.asia-southeast1.firebasedatabase.app/"
})

# 2. Reference the exact student ID slot in your database tree
# (If your photo file name is different from 452331, change the number below!)
ref = db.reference('Students/452331')

# 3. Update the fields on the cloud server
ref.child('name').set('Utkarsh Dixit')

print("🎉 Database successfully updated! Name changed to Utkarsh Dixit.")
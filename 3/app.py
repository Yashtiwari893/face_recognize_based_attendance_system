import cv2
import os
from flask import Flask, request, render_template, session, redirect, url_for, Response
from datetime import date, datetime
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import joblib
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from werkzeug.security import generate_password_hash, check_password_hash
from functools import wraps
import time
import threading

# --- App & DB Initialization ---
app = Flask(__name__)
app.secret_key = 'f8e1a6c4b7d9a8c3e2f1b0a9d8c7e6f5'

# --- Google Sheet Connection ---
try:
    scope = ["https://spreadsheets.google.com/feeds", 'https://www.googleapis.com/auth/spreadsheets',
             "https://www.googleapis.com/auth/drive.file", "https://www.googleapis.com/auth/drive"]
    creds = ServiceAccountCredentials.from_json_keyfile_name("credentials.json", scope)
    client = gspread.authorize(creds)
    user_sheet = client.open("AttendanceAppUsers").sheet1
    attendance_sheet = client.open("AttendanceAppUsers").worksheet("Attendance")
    print("Successfully connected to Google Sheets.")
except Exception as e:
    print(f"Error connecting to Google Sheets: {e}")
    user_sheet = None
    attendance_sheet = None

# --- Basic Setup ---
nimgs = 10
face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

if not os.path.isdir('static'):
    os.makedirs('static')
if not os.path.isdir('static/faces'):
    os.makedirs('static/faces')

# --- Helper Functions, Login & Role Management ---
def totalreg():
    if not os.path.isdir('static/faces'):
        return 0
    return len(os.listdir('static/faces'))

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'username' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

def admin_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if session.get('role') != 'admin':
            return "Access Denied: You are not authorized to view this page.", 403
        return f(*args, **kwargs)
    return decorated_function

def find_user_by_username(username):
    if not user_sheet:
        return None
    try:
        users = user_sheet.get_all_records()
        for user in users:
            if user['username'] == username:
                return user
    except gspread.exceptions.APIError as e:
        print(f"Google Sheets API Error: {e}")
        return None
    return None

def update_attendance_in_sheet(username, userid):
    current_time_str = datetime.now().strftime("%H:%M:%S")
    today_date_str = date.today().strftime("%Y-%m-%d")

    if not attendance_sheet:
        print("Attendance sheet not available.")
        return None

    try:
        all_records = attendance_sheet.get_all_records()
        user_row_index = -1
        punch_type = ''

        for i, record in enumerate(all_records):
            if record.get('Date') == today_date_str and record.get('Username') == username:
                user_row_index = i
                break
        
        if user_row_index == -1:
            new_row = [today_date_str, username, userid, current_time_str, '']
            attendance_sheet.append_row(new_row)
            punch_type = 'IN'
            print(f"IN-PUNCH: User {username} marked IN at {current_time_str}")
        else:
            if all_records[user_row_index].get('Out_Time') and all_records[user_row_index]['Out_Time'] != '':
                print(f"User {username} has already marked OUT today.")
                return 'ALREADY_MARKED'
            
            row_to_update = user_row_index + 2 
            attendance_sheet.update_cell(row_to_update, 5, current_time_str)
            punch_type = 'OUT'
            print(f"OUT-PUNCH: User {username} marked OUT at {current_time_str}")

        return punch_type
    except gspread.exceptions.APIError as e:
        print(f"Google Sheets API Error while updating attendance: {e}")
        return None


# --- Routes ---
@app.route('/')
@login_required
def home():
    if session.get('role') == 'admin':
        return redirect(url_for('admin_dashboard'))
    else:
        return redirect(url_for('user_dashboard'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        user_data = find_user_by_username(username)
        if user_data and check_password_hash(user_data['password'], password):
            session['username'] = user_data['username']
            session['role'] = user_data['role']
            session['userid'] = user_data.get('userid', '')
            return redirect(url_for('home'))
        else:
            return render_template('login.html', error="Invalid username or password")
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

@app.route('/admin/dashboard')
@login_required
@admin_required
def admin_dashboard():
    return render_template('admin_dashboard.html', username=session.get('username'), totalreg=totalreg())

@app.route('/user/dashboard')
@login_required
def user_dashboard():
    message = request.args.get('message')
    return render_template('user_dashboard.html', username=session.get('username'), message=message)
    
@app.route('/admin/add_user', methods=['GET', 'POST'])
@login_required
@admin_required
def add_user():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        role = request.form['role']
        userid = request.form['userid']
        if find_user_by_username(username):
            return render_template('add_user.html', error="User already exists!")
        hashed_password = generate_password_hash(password)
        new_user_row = [username, hashed_password, role, userid]
        if user_sheet:
            user_sheet.append_row(new_user_row)
        userimagefolder = f'static/faces/{username}_{userid}'
        if not os.path.isdir(userimagefolder):
            os.makedirs(userimagefolder)
        i, j = 0, 0
        cap = cv2.VideoCapture(0)
        while i < nimgs:
            _, frame = cap.read()
            faces = face_detector.detectMultiScale(frame, 1.3, 5)
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 20), 2)
                cv2.putText(frame, f'Images Captured: {i}/{nimgs}', (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 20), 2, cv2.LINE_AA)
                if j % 10 == 0:
                    name = f'{username}_{i}.jpg'
                    cv2.imwrite(f'{userimagefolder}/{name}', frame[y:y+h, x:x+w])
                    i += 1
                j += 1
            cv2.imshow('Adding new User', frame)
            if cv2.waitKey(1) == 27:
                break
        cap.release()
        cv2.destroyAllWindows()
        print("Training model...")
        train_model()
        return redirect(url_for('admin_dashboard'))
    return render_template('add_user.html')

@app.route('/attendance/view')
@login_required
def view_attendance():
    if not attendance_sheet:
        return "Could not connect to Attendance Sheet", 500
    today_date_str = date.today().strftime("%Y-%m-%d")
    all_records = attendance_sheet.get_all_records()
    todays_attendance = [rec for rec in all_records if rec.get('Date') == today_date_str]
    return render_template('view_attendance.html', attendance_records=todays_attendance, total_records=len(todays_attendance), datetoday=today_date_str)

# --- Video Streaming and Face Recognition ---

# --- >>> THIS IS THE UPDATED FUNCTION <<< ---
def generate_frames(username, userid):
    """Video streaming generator with corrected logic for success and mismatch."""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Cannot open camera")
        return

    try:
        model = joblib.load('static/face_recognition_model.pkl')
    except FileNotFoundError:
        print("Error: Model file not found!")
        return

    start_time = time.time()
    
    # We will try for 20 seconds
    while time.time() - start_time < 20:
        success, frame = cap.read()
        if not success:
            continue
        
        faces = face_detector.detectMultiScale(frame, 1.3, 5)
        
        # If no face is detected, keep sending the frame
        if len(faces) == 0:
            cv2.putText(frame, "Show your face to camera", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            ret, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            continue # Go to the next frame

        # If a face is detected, check it
        for (x, y, w, h) in faces:
            face_img = cv2.resize(frame[y:y+h, x:x+w], (50, 50))
            identified_person = model.predict(face_img.reshape(1, -1))[0]
            identified_username = identified_person.split('_')[0]

            if identified_username == username:
                # --- SUCCESS LOGIC ---
                # 1. Mark attendance in the background
                thread = threading.Thread(target=update_attendance_in_sheet, args=(username, userid))
                thread.start()

                # 2. Draw Success message on the frame
                cv2.putText(frame, "Success! Attendance Marked.", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

                # 3. Send the success frame to the browser
                ret, buffer = cv2.imencode('.jpg', frame)
                frame_bytes = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                
                # 4. Wait 2 seconds so the user can read the message
                time.sleep(2)

                # 5. Release the camera and stop the function
                print("Releasing camera and stopping stream...")
                cap.release()
                return # Returning here is essential

            else:
                # --- MISMATCH LOGIC ---
                cv2.putText(frame, "Mismatch! Access Denied.", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
        
        # Send the mismatch frame to the browser
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    # If face is not detected in 20 seconds, also release the camera
    print("Timeout! Releasing camera...")
    cap.release()


@app.route('/attendance/take')
@login_required
def take_attendance():
    if 'face_recognition_model.pkl' not in os.listdir('static'):
        return "Model has not been trained yet. Please ask an admin to add a user first to train the model.", 404
    return render_template('video_feed.html')

@app.route('/video_feed')
@login_required
def video_feed():
    username = session.get('username')
    userid = session.get('userid')
    if not username or not userid:
        return "User information not found in session", 400
    return Response(generate_frames(username, userid),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

# --- Model Training Function ---
def train_model():
    faces, labels = [], []
    if not os.path.isdir('static/faces'):
        return
    userlist = os.listdir('static/faces')
    for user in userlist:
        user_dir = f'static/faces/{user}'
        for imgname in os.listdir(user_dir):
            img_path = f'{user_dir}/{imgname}'
            img = cv2.imread(img_path)
            if img is None:
                continue
            resized_face = cv2.resize(img, (50, 50))
            faces.append(resized_face.ravel())
            labels.append(user)
    if not faces:
        return
    faces = np.array(faces)
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(faces, labels)
    joblib.dump(knn, 'static/face_recognition_model.pkl')
    print("Model training complete.")


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
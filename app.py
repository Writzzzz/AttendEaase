import cv2
import os
from flask import Flask, request, render_template
from datetime import date, datetime
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import joblib

#### Initializing Flask App
app = Flask(__name__)

#### Save Date Today in Different Formats
datetoday = date.today().strftime("%m_%d_%y")
datetoday2 = date.today().strftime("%d-%B-%Y")

#### Initialize Face Detector & Camera
face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

#### Create Required Directories if Not Present
os.makedirs('Attendance', exist_ok=True)
os.makedirs('static/faces', exist_ok=True)
os.makedirs('static', exist_ok=True)

# Initialize Attendance File if Not Exists
attendance_file = f'Attendance/Attendance-{datetoday}.csv'
if not os.path.exists(attendance_file):
    with open(attendance_file, 'w') as f:
        f.write('Name,Roll,Time\n')

#### Get Total Registered Users
def totalreg():
    return len(os.listdir('static/faces'))

#### Extract Faces from an Image
def extract_faces(img):
    if img is None or img.size == 0:  # Check for empty images
        return np.array([])

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_points = face_detector.detectMultiScale(gray, 1.3, 5)
    return face_points if len(face_points) > 0 else np.array([])

#### Identify Face using Trained ML Model
def identify_face(facearray):
    model_path = 'face_recognition_model.pkl'
    if not os.path.exists(model_path):
        return None  # Return None if model doesn't exist

    model = joblib.load(model_path)
    return model.predict(facearray)

#### Train ML Model with Collected Faces
def train_model():
    faces, labels = [], []
    
    userlist = os.listdir('static/faces')
    for user in userlist:
        for imgname in os.listdir(f'static/faces/{user}'):
            img = cv2.imread(f'static/faces/{user}/{imgname}')
            if img is None:
                continue  # Skip unreadable images

            resized_face = cv2.resize(img, (50, 50)).ravel()
            faces.append(resized_face)
            labels.append(user)

    if faces:  # Only train if faces exist
        knn = KNeighborsClassifier(n_neighbors=5)
        knn.fit(np.array(faces), labels)
        joblib.dump(knn, 'static/face_recognition_model.pkl')

#### Extract Attendance Data from CSV
def extract_attendance():
    df = pd.read_csv(attendance_file)
    return df['Name'], df['Roll'], df['Time'], len(df)

#### Add Attendance for a Recognized User
def add_attendance(name):
    username, userid = name.split('_')
    current_time = datetime.now().strftime("%H:%M:%S")

    df = pd.read_csv(attendance_file)
    
    # Avoid duplicate attendance for the same user
    if int(userid) not in df['Roll'].astype(int).tolist():
        with open(attendance_file, 'a') as f:
            f.write(f'{username},{userid},{current_time}\n')

################## ROUTING FUNCTIONS #########################

#### Home Page
@app.route('/')
def home():
    names, rolls, times, l = extract_attendance()
    return render_template('home.html', names=names, rolls=rolls, times=times, l=l, totalreg=totalreg(), datetoday2=datetoday2)

#### Start Face Recognition Attendance
@app.route('/start', methods=['GET'])
def start():
    if not os.path.exists('static/face_recognition_model.pkl'):
        return render_template('home.html', totalreg=totalreg(), datetoday2=datetoday2, mess='No trained model found. Please add a new face.')

    cap = cv2.VideoCapture(0)
    ret = True

    while ret:
        ret, frame = cap.read()
        if not ret or frame is None:  # Handle camera failure
            continue

        faces = extract_faces(frame)

        if faces is not None and faces.any():
            (x, y, w, h) = faces[0]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 20), 2)
            face = cv2.resize(frame[y:y + h, x:x + w], (50, 50)).reshape(1, -1)

            identified_person = identify_face(face)
            if identified_person is not None:
                add_attendance(identified_person[0])
                cv2.putText(frame, f'{identified_person[0]}', (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 20), 2, cv2.LINE_AA)

        cv2.imshow('Attendance', frame)
        if cv2.waitKey(1) == 27:  # Press 'Esc' to exit
            break

    cap.release()
    cv2.destroyAllWindows()

    names, rolls, times, l = extract_attendance()
    return render_template('home.html', names=names, rolls=rolls, times=times, l=l, totalreg=totalreg(), datetoday2=datetoday2)

#### Add a New User (Fixed Camera Closing Issue)
@app.route('/add', methods=['POST'])
def add():
    newusername = request.form['newusername']
    newuserid = request.form['newuserid']
    userimagefolder = f'static/faces/{newusername}_{newuserid}'

    os.makedirs(userimagefolder, exist_ok=True)

    cap = cv2.VideoCapture(0)
    i, j = 0, 0

    while i < 50:  # Ensure exactly 50 images are captured
        _, frame = cap.read()
        faces = extract_faces(frame)

        if faces is not None and faces.any():
            (x, y, w, h) = faces[0]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 20), 2)
            cv2.putText(frame, f'Images Captured: {i}/50', (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 20), 2, cv2.LINE_AA)

            if j % 10 == 0:
                name = f'{newusername}_{i}.jpg'
                cv2.imwrite(f'{userimagefolder}/{name}', frame[y:y + h, x:x + w])
                i += 1

            j += 1

        cv2.imshow('Adding New User', frame)
        if cv2.waitKey(1) == 27:  # Press 'Esc' to exit manually
            break

   
    cap.release()
    cv2.destroyAllWindows()

    print('Training Model...')
    train_model()

    names, rolls, times, l = extract_attendance()
    return render_template('home.html', names=names, rolls=rolls, times=times, l=l, totalreg=totalreg(), datetoday2=datetoday2)

#### Run Flask App
if __name__ == '__main__':
    app.run(debug=True)

import cv2
import mediapipe as mp
import pyautogui
import time

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1)
mp_draw = mp.solutions.drawing_utils


a = cv2.VideoCapture(0)
b = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

tolerance=50

start=0
take_photo=False

while True:
    check, frame = a.read()
    clean_frame = frame.copy()
    h, w, _ = frame.shape
    center_x = w // 2
    center_y = h // 2
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = b.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=8, minSize=(30, 30))
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)


    for (x, y, w_face, h_face) in faces:


        face_center_x = x + (w_face // 2)
        face_center_y = y + (h_face // 2)
        box_color=(0, 0, 255)
        if abs(face_center_x - center_x) < tolerance and abs(face_center_y - center_y) < tolerance:
             box_color = (0, 255, 0)
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # mp_draw.draw_landmarks(frame, face_landmarks, mp_face_mesh.FACEMESH_CONTOURS)
                lip_left = face_landmarks.landmark[61]
                lip_left_x = int(lip_left.x * w)
                lip_left_y = int(lip_left.y * h)
                lip_right = face_landmarks.landmark[291]
                lip_right_x = int(lip_right.x * w)
                lip_right_y = int(lip_right.y * h)
                lip_dist = lip_right_x - lip_left_x 
                if lip_dist > (w_face * 0.33):
                    radius = w_face // 2
                    cv2.circle(frame, (face_center_x, face_center_y), radius, box_color, 2)
                    cv2.putText(frame, "Smile!", (face_center_x - 40, face_center_y - radius - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

                else:
                    cv2.rectangle(frame, (x, y), (x + w_face, y + h_face), box_color, 2)
            if face_center_x > center_x + tolerance:
                direction_text = "Go Left"
            elif face_center_x < center_x - tolerance:
                direction_text = "Go Right"
            else:
                direction_text = "Centered"
            cv2.putText(frame, direction_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, box_color, 2)

            is_centered = abs(face_center_x - center_x) < tolerance and abs(face_center_y - center_y) < tolerance
            is_smiling = lip_dist > (w_face * 0.33)
            if is_centered and is_smiling:
                if start ==0 and not take_photo:
                    start=time.time()
            else:
                start=0
                take_photo=False
            if start!=0:
                c = time.time() - start
                remaining = 3 - int(c)
                if remaining > 0:
                    cv2.putText(frame, str(remaining), (center_x - 30, center_y), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 255), 5)
                else:
                    cv2.imwrite("selfie.jpg", clean_frame)
                    cv2.putText(frame, "Captured!", (center_x - 120, center_y), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 4)
                    cv2.imshow("Capturing", clean_frame)
                    cv2.waitKey(1000)
                    take_photo=True
                    break

    if take_photo:
        break

        

    cv2.imshow("Capturing", frame)
    key = cv2.waitKey(1)
    if(key == ord('q')):
        break


a.release()
cv2.destroyAllWindows()

img = cv2.imread("selfie.jpg")
if img is not None:
    cv2.imshow("Captured Selfie", img)
    cv2.waitKey(0) 
    cv2.destroyAllWindows()

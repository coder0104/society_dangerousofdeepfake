import cv2
import dlib
import numpy as np

whythisdontworkmotherfucker = "shape_predictor_68_face_landmarks.dat"

detector = dlib.get_frontal_face_detector()
shape_predictor = dlib.shape_predictor(whythisdontworkmotherfucker)

fuck = cv2.imread("noeun.jpg") #여기에다가 넣고싶은 사람 **동의 받아야함 사진 넣기

replacement_faces = detector(fuck)
if len(replacement_faces) == 0:
    raise Exception("No face detected in the replacement image.")

replacement_face = replacement_faces[0]

(x, y, w, h) = (replacement_face.left(), replacement_face.top(), replacement_face.width(), replacement_face.height())
replacement_face_region = fuck[y:y+h, x:x+w]

video_capture = cv2.VideoCapture(0)

previous_points = None

while True:
    ret, frame = video_capture.read()
    if not ret:
        break
    
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    faces = detector(rgb_frame)
    
    for face in faces:
        try:
            shape = shape_predictor(rgb_frame, face)
            points = np.array([[shape.part(n).x, shape.part(n).y] for n in range(68)])

            if previous_points is not None:
                diff = np.linalg.norm(points - previous_points, axis=1).mean()
                if diff > 20: 
                    points = previous_points

            previous_points = points 
            
            (x, y, w, h) = (face.left(), face.top(), face.width(), face.height())
            resized_replacement = cv2.resize(replacement_face_region, (w, h))

            mask = np.zeros_like(frame)
            mask[y:y+h, x:x+w] = resized_replacement

            blended_face = cv2.addWeighted(frame[y:y+h, x:x+w], 0.5, resized_replacement, 0.5, 0)

            frame[y:y+h, x:x+w] = blended_face

            frame = cv2.GaussianBlur(frame, (5, 5), 10)
        
        except Exception as e:
            print(f"Error occurred: {e}")
            continue  

    cv2.imshow('Face Replacement', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()

import time
import tensorflow as tf
import cv2
import face_recognition
import matplotlib.pyplot as plt
import numpy as np

model = tf.keras.models.load_model("./Models/FMClassifier_B")

cap = cv2.VideoCapture(0)

new_frame_time = 0
prev_frame_time = 0

while cap.isOpened():
    _, frame = cap.read()
    rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    #frame = face_recognition.load_image_file('photo.jpg')
    detected_faces = []

    face_locations = face_recognition.face_locations(rgb_image, model='cnn')

    if(len(face_locations) != 0):
        # Loop through each face in this frame of video
        for(top, right, bottom, left) in face_locations:
            #get the face from the image
            face = rgb_image[top:bottom, left:right]
            #resize the face
            face = cv2.resize(face, (100, 100))
            detected_faces.append(face)

        y_pred = model.predict(np.array(detected_faces))
        print(y_pred)

        for(top, right, bottom, left),pred in zip(face_locations, y_pred):
            if(pred[0] > 0.5):
                cv2.rectangle(frame, (left - 35, top - 35), (right + 35, bottom + 35), (0, 255, 0), 2)
                cv2.rectangle(frame, (left - 35 , bottom+35), (right + 35, bottom+70), (0, 255, 0), cv2.FILLED)
                font = cv2.FONT_HERSHEY_DUPLEX
                cv2.putText(frame, 'Mask', (left + 6, bottom + 60), font, 1.0, (255, 255, 255), 2)
            elif(pred[1] > 0.5):
                cv2.rectangle(frame, (left - 35, top - 35), (right + 35, bottom + 35), (0, 0, 255), 2)
                cv2.rectangle(frame, (left - 35 , bottom+35), (right + 35, bottom+70), (0, 0, 255), cv2.FILLED)
                font = cv2.FONT_HERSHEY_DUPLEX
                cv2.putText(frame, 'No Mask', (left + 6, bottom + 60), font, 1.0, (255, 255, 255), 2)
            else:
                cv2.rectangle(frame, (left - 35, top - 35), (right + 35, bottom + 35), (255, 0, 0), 2)
                cv2.rectangle(frame, (left - 35 , bottom+35), (right + 35, bottom+70), (255, 0, 0), cv2.FILLED)
                font = cv2.FONT_HERSHEY_DUPLEX
                cv2.putText(frame, 'None', (left + 6, bottom + 60), font, 1.0, (255, 255, 255), 2)

    font = cv2.FONT_HERSHEY_SIMPLEX
    # time when we finish processing for this frame
    new_frame_time = time.time()
 
    # Calculating the fps
 
    # fps will be number of frame processed in given time frame
    # since their will be most of time error of 0.001 second
    # we will be subtracting it to get more accurate result
    fps = 1/(new_frame_time-prev_frame_time)
    prev_frame_time = new_frame_time
 
    # converting the fps into integer
    fps = int(fps)
 
    # converting the fps to string so that we can display it on frame
    # by using putText function
    fps = str(fps)
 
    # putting the FPS count on the frame
    cv2.putText(frame, fps, (7, 70), font, 3, (100, 255, 0), 3, cv2.LINE_AA)





    cv2.imshow('Face Mask Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

# Display the resulting image
#plt.imshow(frame)
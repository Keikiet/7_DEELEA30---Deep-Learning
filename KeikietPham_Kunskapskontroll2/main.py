from keras.models import load_model
from keras.preprocessing.image import img_to_array
import cv2
import numpy as np

# Load classifiers
face_classifier = cv2.CascadeClassifier(r'C:\Users\Min dator\Desktop\Kunskapskontroll2_DeepLearning\Emotion_Detection_CNN-main\haarcascade_frontalface_default.xml')
emotion_classifier = load_model(r'C:\Users\Min dator\Desktop\Kunskapskontroll2_DeepLearning\Emotion_Detection_CNN-main\my_model_woa_7550.keras')
gender_classifier = load_model(r'C:\Users\Min dator\Desktop\Kunskapskontroll2_DeepLearning\Emotion_Detection_CNN-main\my_model_pt_8284.keras')

# Define labels
emotion_labels = {0: "Angry", 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Neutral', 5: 'Sad', 6: 'Surprise'}
gender_labels = {0: "female", 1: "male"}

# Start video capture
cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    _, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
        
        # Process ROI for emotion detection
        roi_gray = gray[y:y + h, x:x + w]
        roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)
        
        if np.sum([roi_gray]) != 0:
            #roi_gray = roi_gray.astype('float') / 255.0  # Normalize to [0, 1]
            roi = img_to_array(roi_gray.astype('float'))
            roi_gray = img_to_array(roi_gray)
            roi_gray = np.expand_dims(roi_gray, axis=0)

            # Process ROI for gender detection
            roi_color = frame[y:y + h, x:x + w]
            roi_color = cv2.resize(roi_color, (150, 150), interpolation=cv2.INTER_AREA)
            #roi_color = roi_color.astype('float') / 255.0  # Normalize to [0, 1]
            roi_color = img_to_array(roi_color)
            roi_color = np.expand_dims(roi_color, axis=0)

            # Predict emotion and gender
            prediction_emotion = emotion_classifier.predict(roi_gray)
            prediction_gender = gender_classifier.predict(roi_color)

            # Get labels
            emotion_label = emotion_labels[np.argmax(prediction_emotion)]
            gender_label = gender_labels[np.argmax(prediction_gender)]

            # Set positions for labels
            emotion_label_position = (x, y - 10)
            gender_label_position = (x, y + h + 20)

            # Display labels
            cv2.putText(frame, emotion_label, emotion_label_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, gender_label, gender_label_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            cv2.putText(frame, 'No Faces', (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow('Emotion and Gender Detector', frame)
    
    # Break loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close windows
cap.release()
cv2.destroyAllWindows()

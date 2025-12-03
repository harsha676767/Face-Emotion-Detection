import cv2
import torch
import torch.nn.functional as F
from torchvision import transforms
from deep_emotion import Deep_Emotion
from collections import deque

# Load model
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = Deep_Emotion()
model.load_state_dict(torch.load('deep_emotion-100-128-0.005.pt', map_location=device))  # <-- your model file
model.eval()
model.to(device)

# Load Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Define transformation
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

classes = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# Queue for smoothing predictions
predictions_queue = deque(maxlen=5)  # Last 5 predictions

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_gray = cv2.resize(roi_gray, (48, 48))
        face = transform(roi_gray).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(face)
            prediction = torch.argmax(F.softmax(output, dim=1), dim=1)
            predictions_queue.append(prediction.item())

        # Get the most common prediction
        if len(predictions_queue) > 0:
            emotion_counts = [predictions_queue.count(i) for i in range(len(classes))]
            emotion_idx = emotion_counts.index(max(emotion_counts))
            emotion = classes[emotion_idx]
        else:
            emotion = "..."

        # Draw rectangle around face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        # Put emotion label
        cv2.putText(frame, f'{emotion}', (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow('Real-Time Emotion Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

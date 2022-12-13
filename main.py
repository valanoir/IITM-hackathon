import tensorflow as tf
import tensorflow_hub as hub
import cv2
from matplotlib import pyplot as plt
import numpy as np

# Optional if you are using a GPU
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

model = hub.load('https://tfhub.dev/google/movenet/multipose/lightning/1')
movenet = model.signatures['serving_default']
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')

def draw_keypoints(frame, keypoints, confidence_threshold):
    y, x, c = frame.shape
    shaped = np.squeeze(np.multiply(keypoints, [y, x, 1]))

    for kp in shaped:
        ky, kx, kp_conf = kp
        if kp_conf > confidence_threshold:
            cv2.circle(frame, (int(kx), int(ky)), 6, (0, 255, 0), -1)

def draw_connections(frame, keypoints, edges, confidence_threshold):
    y, x, c = frame.shape
    shaped = np.squeeze(np.multiply(keypoints, [y, x, 1]))

    for edge, color in edges.items():
        p1, p2 = edge
        y1, x1, c1 = shaped[p1]
        y2, x2, c2 = shaped[p2]

        if (c1 > confidence_threshold) & (c2 > confidence_threshold):
            cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 4)

EDGES = {
    (0, 1): 'm',
    (0, 2): 'c',
    (1, 3): 'm',
    (2, 4): 'c',
    (0, 5): 'm',
    (0, 6): 'c',
    (5, 7): 'm',
    (7, 9): 'm',
    (6, 8): 'c',
    (8, 10): 'c',
    (5, 6): 'y',
    (5, 11): 'm',
    (6, 12): 'c',
    (11, 12): 'y',
    (11, 13): 'm',
    (13, 15): 'm',
    (12, 14): 'c',
    (14, 16): 'c'
}

def loop_through_people(frame, keypoints_with_scores, edges, confidence_threshold):
    for person in keypoints_with_scores:
        draw_connections(frame, person, edges, confidence_threshold)
        draw_keypoints(frame, person, confidence_threshold)



cap = cv2.VideoCapture('sample.mp4')
fourcc = cv2.VideoWriter_fourcc(*'xvid')
out = cv2.VideoWriter('output.mp4',fourcc,20.0,(640,480))
while cap.isOpened():
    ret, frame = cap.read()
    if(ret==True):
        out.write(frame)






    # Resize image
    img = frame.copy()
    img = tf.image.resize_with_pad(tf.expand_dims(img, axis=0), 384, 640)
    input_img = tf.cast(img, dtype=tf.int32)

    # Detection section
    results = movenet(input_img)
    keypoints_with_scores = results['output_0'].numpy()[:, :, :51].reshape((6, 17, 3))

    # Render keypoints
    loop_through_people(frame, keypoints_with_scores, EDGES, 0.1)

     # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame using a Haar cascade classifier
    faces = face_cascade.detectMultiScale(gray)

    # Loop over the faces
    for (x, y, w, h) in faces:
        # Blur the face
        face_blur = cv2.GaussianBlur(frame[y:y+h, x:x+w], (55, 55), 30)

        # Replace the original face with the blurred face
        frame[y:y+h, x:x+w] = face_blur

    frame = cv2.resize(frame,(1000,1000))
    cv2.imshow('Movenet Multipose', frame)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()


import cv2
import numpy as np

# Load the video
video = cv2.VideoCapture('sample.mp4')
face_cascade = cv2.CascadeClassifier('face_cascade.xml')

# Loop over each frame in the video
while video.isOpened():
    # Read the current frame
    ret, frame = video.read()

    # Check if we have reached the end of the video
    if not ret:
        break

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame using a Haar cascade classifier
    faces = face_cascade.detectMultiScale(gray)

    # Loop over the faces
    for (x, y, w, h) in faces:
        # Blur the face
        face_blur = cv2.GaussianBlur(frame[y:y+h, x:x+w], (55, 55), 30)

        # Replace the original face with the blurred face
        frame[y:y+h, x:x+w] = face_blur

    # Show the frame
    frame = cv2.resize(frame, (640, 480))
    cv2.imshow("Frame", frame)

    # Check if the user pressed the "q" key
    if cv2.waitKey(1) == ord("q"):
        break

# Release the video capture object
video.release()

# Close all windows
cv2.destroyAllWindows()

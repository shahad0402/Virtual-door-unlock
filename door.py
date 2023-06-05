import cv2
import os

# Load the pre-trained face detection model
face_cascade = cv2.CascadeClassifier('/home/varnana/Visual_studio/data/haarcascade_frontalface_default.xml')

# Load the pre-trained face recognition model
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('/home/varnana/Visual_studio/virtual_door_unlock/my_images/my_image.xml')

# Initialize the camera
cam = cv2.VideoCapture(0)

# Load welcome image
welcome_image = cv2.imread('/home/varnana/Visual_studio/virtual_door_unlock/images.jpeg')


# Get the directory path of the door frame images
door_frames_dir = '/home/varnana/Visual_studio/virtual_door_unlock/door_frames'

# Load the door opening frames
door_opening_frames = []
for filename in os.listdir(door_frames_dir):
    if filename.endswith('.jpg'):
        frame_path = os.path.join(door_frames_dir, filename)
        frame = cv2.imread(frame_path)
        if frame is not None:
            door_opening_frames.append(frame)

while True:
    # Read a frame from the camera
    ret, frame = cam.read()

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Perform face detection
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Process each detected face
    for (x, y, w, h) in faces:
        # Extract the face region from the frame
        face = gray[y:y + h, x:x + w]

        # Perform face recognition
        label, confidence = recognizer.predict(face)

        # Check if the face is recognized
        if confidence < 100:
            # Draw a green rectangle around the face
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Simulate door opening
            for door_frame in door_opening_frames:
                cv2.imshow('Virtual Door', door_frame)
                cv2.waitKey(5000)  # Delay between frames
                break
                

            print("Door Unlocked!")


            # Display the "Welcome" image
            cv2.imshow('Virtual Door', welcome_image)
            cv2.waitKey(3000)  # Display the image for 3 seconds
            cv2.destroyAllWindows()
            break
        
        else:
            # Draw a red rectangle around the face
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

            # Display "Unauthorized Access" message
            unauthorized_message = cv2.putText(frame, 'Unauthorized Access', (x, y - 10),
                                               cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
            cv2.imshow('Virtual Door', unauthorized_message)
            cv2.waitKey(10)  # Display the message for .1 seconds

    # Display the frame
    cv2.imshow('Virtual Door', frame)

    # Break the loop if 'esc' key is pressed
    if cv2.waitKey(10) & 0xFF == 27:
        break

# Release the camera and close all windows
cam.release()
cv2.destroyAllWindows()

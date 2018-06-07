####################################################
# Modified by Nazmi Asri                           #
# Original code: http://thecodacus.com/            #
# All right reserved to the respective owner       #
####################################################

# Import OpenCV2 for image processing
import cv2


#https://github.com/nazmiasri95/Face-Recognition

# Start capturing video 
vid_cam = cv2.VideoCapture(0)

# Detect object in video stream using Haarcascade Frontal Face
#face_detector = cv2.CascadeClassifier('haarcascade_licence_plate_rus_16stages.xml')
#face_detector = cv2.CascadeClassifier('haarcascade_upperbody.xml')
#face_detector = cv2.CascadeClassifier('cas1.xml')
#face_detector = cv2.CascadeClassifier('haarcascade_frontalcatface_extended')
eye_detector = cv2.CascadeClassifier('haarcascade_eye.xml')
face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# For each person, one face idq
face_id = 1

# Initialize sample face image
count = 0

dividingFactor = 4

# Start loopingq
while (True):

    # Capture video frame
    _, image_frame = vid_cam.read()

    height, width, layers = image_frame.shape

    # comment this line if you want the fullsize window
    image_frame = cv2.resize(image_frame, (int(width / dividingFactor), int(height / dividingFactor)))

    # Convert frame to grayscale
    gray = cv2.cvtColor(image_frame, cv2.COLOR_BGR2GRAY)

    # Detect frames of different sizes, list of faces rectangles
    faces = face_detector.detectMultiScale(gray, 1.3, 5,10)

    # Detect frames of different sizes, list of eyes rectangles
    eyes = eye_detector.detectMultiScale(gray, 1.3, 5)

    # # Loops for each faces
    # for (x, y, w, h) in eyes:
    #     # Crop the image frame into rectangle
    #     cv2.rectangle(image_frame, (x, y), (x + w, y + h), (0, 0, 255), 3)

        # Loops for each faces
    for (x, y, w, h) in faces:
        # Crop the image frame into rectangle
        cv2.rectangle(image_frame, (x, y), (x + w, y + h), (0, 255, 0), 3)


        # Increment sample face image
        #   count += 1

        # Save the captured image into the datasets folder
        # cv2.imwrite("dataset/User." + str(face_id) + '.' + str(count) + ".jpg", gray[y:y + h, x:x + w])

    # Display the video frame, with bounded rectangle on the person's face
    cv2.imshow('frame', image_frame)

    # To stop taking video, press 'q' for at least 100ms
    if cv2.waitKey(100) & 0xFF == ord('q'):
        break

        # If image taken reach 100, stop taking video
        #  elif count > 100:
        #      break  # Stop video
vid_cam.release()

# Close all started windows
cv2.destroyAllWindows()

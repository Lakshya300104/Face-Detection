import cv2

# Initialize face cascade classifier
face_cap = cv2.CascadeClassifier("C:/Users/Lenovo/AppData/Local/Programs/Python/Python312/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml")
# Initialize video capture
video_cap = cv2.VideoCapture(0)

# Check if the camera opened successfully
if not video_cap.isOpened():
    print("Error: Could not open camera.")
else:
    while True:
        ret, video_data = video_cap.read()

        # Ensure that the frame is successfully captured before processing
        if not ret:
            print("Error: Failed to capture video frame.")
            break

        # Convert the frame to grayscale
        col = cv2.cvtColor(video_data, cv2.COLOR_BGR2GRAY) # convert to grayscale

        # Detect faces in the grayscale image
        faces = face_cap.detectMultiScale(
            col,
            scaleFactor=1.1, 
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        # Draw a rectangle around the detected faces
        for (x, y, w, h) in faces:
            cv2.rectangle(video_data, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Show the video frame with rectangles
        cv2.imshow("video_live", video_data)

        # Break loop if 'a' is pressed
        if cv2.waitKey(10) == ord("a"):
            break

    # Release the video capture object
    video_cap.release()
    cv2.destroyAllWindows()

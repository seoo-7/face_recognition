import cv2 as cv  # type: ignore

capture = cv.VideoCapture(r'D:\opencv\vedios\dog.mp4')

# Check if the video opened successfully
if not capture.isOpened():
    print("Error: Cannot open video file.")
    exit()

while True:
    isTrue, frame = capture.read()
    if not isTrue:
        print("Error: Cannot read frame or end of video reached.")
        break

    cv.imshow('Video', frame)

    # Exit the video window by pressing 'd'
    if cv.waitKey(20) & 0xFF == ord('d'):
        break

capture.release()
cv.destroyAllWindows()



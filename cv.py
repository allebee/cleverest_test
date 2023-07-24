import cv2


def detect_drones(video_path):
    """
    Detects and highlights drones from a video

    Parameters:
        video_path (str): Path to a video file.

    Returns:
        None
    """

    # Open video
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        return

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Define the codec and create VideoWriter object to save the video
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter("/Users/alibiserikbay/Developer/mechta_test/TZ_processed.mp4",
                          fourcc, fps, (frame_width, frame_height))

    """
    Initialize background subtractor for foreground detection.
    It is used to identify the regions of the video that have changed over time.
    """
    bg_subtractor = cv2.createBackgroundSubtractorMOG2()

    while True:
        # Read frame from the video
        ret, frame = cap.read()
        if not ret:
            break

        """
        Resize the frame to a smaller size.
        It reduces computational load.
        """
        small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)

        """
        Apply background subtraction to get the foreground mask.
        This operation computes the difference between the current frame and the learned background model.
        """
        fg_mask = bg_subtractor.apply(small_frame)

        """
        Threshold the foreground mask to filter out shadows. Converts into a binary.
        Sets all pixels with intensity greater than 128 to 255 (white) and the rest to 0 (black)
        """
        _, fg_mask = cv2.threshold(fg_mask, 128, 255, cv2.THRESH_BINARY)

        """
        Apply morphological operations to remove noise and merge nearby regions
        getStructuringElement - creates small matrix that defines the neighborhood for each pixel in the binary image.
        morphologyEx - emoves noise in the binary mask, remaining the overall shape and size of the drone.
        dilate 2 - expands the white regions in the binary mask, joining close areas that likely belong to the same object.
        """
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
        fg_mask = cv2.dilate(fg_mask, kernel, iterations=2)

        """
        Find contours in the foreground mask.
        Detecting the boundaries of connected regions in the binary image.
        """
        contours, _ = cv2.findContours(
            fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            # Ignore small contours
            if cv2.contourArea(contour) < 135:
                continue

            # Finds coordinates of the bounding rectange for the drone
            x, y, w, h = cv2.boundingRect(contour)

            # Filter out large regions (clouds)
            if w * h > 2500:
                continue

            cv2.rectangle(frame, (x * 2, y * 2),
                          ((x + w) * 2, (y + h) * 2), (0, 255, 0), 2)
        out.write(frame)
        cv2.imshow("Drones Detection", frame)

        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    video_path = "/Users/alibiserikbay/Developer/mechta_test/TZ.mp4"
    detect_drones(video_path)

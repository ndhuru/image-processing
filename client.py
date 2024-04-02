import tkinter as tk
import numpy as np
import requests
import threading
from userlog import UserLog
import cv2
from PIL import Image, ImageTk
from lane_detection import pipeline

# read the username from the temporary file
with open("temp_username.txt", "r") as temp_file:
    username = temp_file.read().strip()


class RobotControlApp:
    def __init__(self, root):
        self.root = root
        self.root.geometry("600x600")
        self.root.title("Robot Control App")

        # create four quadrants
        self.quadrant1 = tk.Frame(root, bg="gray", width=300, height=300)
        self.quadrant1.grid(row=0, column=0, rowspan=2, columnspan=2)

        self.quadrant2 = tk.Frame(root, bg="lightgray", width=300, height=300)
        self.quadrant2.grid(row=0, column=2, rowspan=2, columnspan=2)

        self.quadrant3 = tk.Frame(root, bg="lightgray", width=300, height=300)
        self.quadrant3.grid(row=2, column=0, rowspan=2, columnspan=2)

        self.quadrant4 = tk.Frame(root, bg="gray", width=300, height=300)
        self.quadrant4.grid(row=2, column=2, rowspan=2, columnspan=2)

        # create directional buttons, stop, and play buttons in quadrant 2 (top right)
        button_size = ("Helvetica", 12)

        self.forward_button = tk.Button(self.quadrant2, text="↑", command=lambda: self.send_command("forward"),
                                        font=button_size)
        self.forward_button.grid(row=0, column=1)

        self.left_button = tk.Button(self.quadrant2, text="←", command=lambda: self.send_command("left"),
                                     font=button_size)
        self.left_button.grid(row=1, column=0)

        self.right_button = tk.Button(self.quadrant2, text="→", command=lambda: self.send_command("right"),
                                      font=button_size)
        self.right_button.grid(row=1, column=2)

        self.backward_button = tk.Button(self.quadrant2, text="↓", command=lambda: self.send_command("backward"),
                                         font=button_size)
        self.backward_button.grid(row=2, column=1)

        self.stop_button = tk.Button(self.quadrant2, text="⛔", command=lambda: self.send_command("stop"),
                                     font=button_size, foreground='red')
        self.stop_button.grid(row=3, column=0, pady=10)

        self.play_button = tk.Button(self.quadrant2, text="▶", command=lambda: self.send_command("play"),
                                     font=button_size, foreground='green')
        self.play_button.grid(row=3, column=2, pady=10)

        # bind WASD keys to directional commands
        root.bind("<w>", lambda event: self.send_command("forward"))
        root.bind("<a>", lambda event: self.send_command("left"))
        root.bind("<s>", lambda event: self.send_command("backward"))
        root.bind("<d>", lambda event: self.send_command("right"))
        root.bind("<q>", lambda event: self.send_command("stop"))

        # obtained username, so now we need to incorporate it
        self.user_log = UserLog(root, username=username)

        # videostream label(raw)
        self.video_canvas = tk.Canvas(root, width=300, height=300, bg="gray")
        self.video_canvas.grid(row=0, column=0, rowspan=2, columnspan=2)

        # tried threading in order to speed the response time from the api
        # quick fyi, but I took a backend course when I was 13 and learned a little bit about the threading library
        self.video_stream_thread = threading.Thread(target=self.start_video_stream)
        self.video_stream_thread.start()

        # create a label for displaying the video stream with overlay(not raw)
        self.overlay_canvas = tk.Canvas(root, width=300, height=300, bg="gray")
        self.overlay_canvas.grid(row=2, column=0, rowspan=2, columnspan=2)

        # start the video stream thread with the overlay in place
        self.video_overlay_thread = threading.Thread(target=self.start_video_stream_overlay)
        self.video_overlay_thread.start()

    # this was an algorithm that I learned from the following website
    # https://opencv24-python-tutorials.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_houghlines/py_houghlines.html
    # very informative
    # this was the main part of the project that actually really scared me, but as you said I just needed to sit back and read the article
    # and as a result I was able to get this done really quickly

    def apply_line_detection(self, frame):
        # first we convert the frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # then apply GaussianBlur to reduce noise and improve line detection overall
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # then we can use Canny edge detector to find edges in the frame
        edges = cv2.Canny(blurred, 50, 150)

        # lastly we use HoughLinesP to detect lines in the frame
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=50, minLineLength=100, maxLineGap=50)

        # Draw only a limited amount of detected lines
        line_frame = frame.copy()
        for i, line in enumerate(lines):
            if i < 3:  # Draw only the first 100ish lines
                x1, y1, x2, y2 = line[0]
                cv2.line(line_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # merge the original frame with the line-drawn frame
        overlay_frame = cv2.addWeighted(frame, 0.8, line_frame, 1, 0)

        # lastly, just return the overlay of everything that we got on line 110
        return overlay_frame

    # okay, this next function is required for basically anything to work
    # this one is responsible for actually creating the overlay (AKA the meat of the project we were assigned)
    # note: different from the algorithm, this is actually creating it to put on the client
    def start_video_stream_overlay(self):
        # open a video capture object
        cap = cv2.VideoCapture("Car drive.mp4")  # Replace with the URL to the lane detection video

        while True:
            # read a frame from the video capture object
            ret, frame = cap.read()
            if ret:
                # Apply lane detection pipeline
                try:
                    lane_detected_frame = pipeline(frame)
                except:
                    lane_detected_frame = frame

                # Convert the frame from BGR to RGB
                rgb_frame = cv2.cvtColor(lane_detected_frame, cv2.COLOR_BGR2RGB)

                # Resize the frame to fit the canvas
                rgb_frame = cv2.resize(rgb_frame, (300, 300))

                # Convert the frame to a PhotoImage format
                image = Image.fromarray(rgb_frame)
                photo = ImageTk.PhotoImage(image=image)

                # Clear previous overlay if exists
                self.overlay_canvas.delete("all")

                # Update the overlay canvas with the new frame
                self.overlay_canvas.create_image(0, 0, anchor=tk.NW, image=photo)

            # Sleep for a short duration to control the frame rate
            self.root.update()
            self.root.after(10)

        # Release the video capture object when the window is closed
        cap.release()
        # open a video capture object
        cap = cv2.VideoCapture("Car drive.mp4")  # Replace with the URL to the lane detection video

        while True:
            # read a frame from the video capture object
            ret, frame = cap.read()
            if ret:
                # Apply lane detection pipeline
                lane_detected_frame = pipeline(frame)

                # Convert the frame from BGR to RGB
                rgb_frame = cv2.cvtColor(lane_detected_frame, cv2.COLOR_BGR2RGB)

                # Resize the frame to fit the canvas
                rgb_frame = cv2.resize(rgb_frame, (300, 300))

                # Convert the frame to a PhotoImage format
                image = Image.fromarray(rgb_frame)
                photo = ImageTk.PhotoImage(image=image)

                # Clear previous overlay if exists
                self.overlay_canvas.delete("all")

                # Update the overlay canvas with the new frame
                self.overlay_canvas.create_image(0, 0, anchor=tk.NW, image=photo)

            # Sleep for a short duration to control the frame rate
            self.root.update()
            self.root.after(10)

        # Release the video capture object when the window is closed
        cap.release()

    # now for the raw videostream
    # not the overlay
    def start_video_stream(self):
        # open a video capture object (use 0 for the default camera)
        cap = cv2.VideoCapture("Car drive.mp4")  # REPLACE WITH URL TO CAMERA

        while True:
            # read a frame from the video capture object
            ret, frame = cap.read()
            if ret:
                # convert the frame from BGR to RGB
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # calculate scaling factors to fit the video within the canvas
                canvas_width = 300
                canvas_height = 300
                video_width, video_height, _ = rgb_frame.shape
                scale_factor = min(canvas_width / video_width, canvas_height / video_height)

                # resize the frame to fit the canvas while maintaining aspect ratio
                resized_frame = cv2.resize(rgb_frame, None, fx=scale_factor, fy=scale_factor)

                # create a blank canvas to paste the resized frame
                full_frame = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)
                start_x = (canvas_width - resized_frame.shape[1]) // 2
                start_y = (canvas_height - resized_frame.shape[0]) // 2
                full_frame[start_y:start_y + resized_frame.shape[0],
                start_x:start_x + resized_frame.shape[1]] = resized_frame

                # convert the frame to a PhotoImage format
                image = Image.fromarray(full_frame)
                photo = ImageTk.PhotoImage(image=image)

                # update the video canvas with the new frame
                self.video_canvas.create_image(0, 0, anchor=tk.NW, image=photo)

            # sleep for a short duration to control the frame rate
            self.root.update()
            self.root.after(10)

        # release the video capture object when the window is closed
        cap.release()

    def send_command(self, command):
        # log user action before sending the command
        self.user_log.log_action(command)

        # send the command to the robot
        threading.Thread(target=self.send_request, args=(command,)).start()

    # and as always, actually sending the command to the server
    def send_request(self, command):
        api_url = "http://localhost:4444/control"
        payload = {"command": command}

        try:
            response = requests.post(api_url, json=payload)
            if response.status_code == 200:
                print(f"Command '{command}' sent successfully.")
            else:
                print(f"Failed to send command '{command}'.")
        except requests.RequestException as e:
            print(f"Error sending command: {e}")


if __name__ == '__main__':
    root = tk.Tk()
    app = RobotControlApp(root)
    root.mainloop()

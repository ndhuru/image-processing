import tkinter as tk
import numpy as np
import requests
import threading
from userlog import UserLog
import cv2
from PIL import Image, ImageTk
from lane_detection import pipeline  # Importing the pipeline function

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

    def apply_line_detection(self, frame):
        # Apply lane detection pipeline from lane_detection.py
        overlay_frame = pipeline(frame)
        return overlay_frame

    def start_video_stream_overlay(self):
        cap = cv2.VideoCapture("http://127.0.0.1:2345/video_feed")  # REPLACE WITH URL TO CAMERA
        while True:
            ret, frame = cap.read()
            if ret:
                overlay_frame = self.apply_line_detection(frame)
                rgb_frame = cv2.cvtColor(overlay_frame, cv2.COLOR_BGR2RGB)
                rgb_frame = cv2.resize(rgb_frame, (300, 300))
                image = Image.fromarray(rgb_frame)
                photo = ImageTk.PhotoImage(image=image)
                self.overlay_canvas.create_image(0, 0, anchor=tk.NW, image=photo)
            self.root.update()
            self.root.after(10)
        cap.release()

    def start_video_stream(self):
        cap = cv2.VideoCapture("http://127.0.0.1:2345/video_feed")  # REPLACE WITH URL TO CAMERA
        while True:
            ret, frame = cap.read()
            if ret:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                rgb_frame = cv2.resize(rgb_frame, (600, 400))
                image = Image.fromarray(rgb_frame)
                photo = ImageTk.PhotoImage(image=image)
                self.video_canvas.create_image(0, 0, anchor=tk.NW, image=photo)
            self.root.update()
            self.root.after(10)
        cap.release()

    def send_command(self, command):
        self.user_log.log_action(command)
        threading.Thread(target=self.send_request, args=(command,)).start()

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

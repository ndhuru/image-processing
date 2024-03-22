import tkinter as tk
import datetime

class UserLog:
    def __init__(self, root, log_filename="user_log.txt", username=""):
        self.root = root
        self.log_filename = log_filename
        self.log_entries = []
        self.username = username  # Add username as an instance variable

        # Create lower right quadrant window
        self.log_quadrant = tk.Frame(root, bg="lightgray", width=300, height=300)
        self.log_quadrant.grid(row=2, column=2, rowspan=2, columnspan=2)

        # Create a label to display log entries
        self.log_label = tk.Label(self.log_quadrant, text="User Log", font=("Helvetica", 12), justify=tk.LEFT)
        self.log_label.pack(fill=tk.BOTH, expand=True)

    def log_action(self, action):
        timestamp = datetime.datetime.now().strftime("%I:%M %p")
        log_entry = f"{self.username} used {action} at {timestamp}"
        self.log_entries.append(log_entry)

        # Keep only the last 10 log entries
        self.log_entries = self.log_entries[-10:]

        # Update log label with the latest entries
        self.log_label.config(text="\n".join(self.log_entries))

        # Store log entry in the text file
        self.store_log_entry(log_entry)

    def store_log_entry(self, log_entry):
        with open(self.log_filename, "a") as log_file:
            log_file.write(log_entry + "\n")

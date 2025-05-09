"""
Rock Paper Scissors Game with Enhanced GUI, Speech Recognition, and Screenshot Saving
Author: [MDL Rangashan - 22877] (Team Lead - Developer 2)
Date: May 08, 2025
Description: This script integrates gesture detection, image processing, game logic,
speech recognition, and screenshot saving into a user-friendly Rock Paper Scissors game.
It features a beautiful GUI. Speech recognition allows voice commands like "start,"
"proceed," and "reset." Screenshots are saved at game end.
"""

import cv2  # OpenCV for webcam capture and image processing
import mediapipe as mp  # MediaPipe for hand gesture detection
import numpy as np  # NumPy for numerical operations
import tkinter as tk  # Tkinter for GUI creation
from PIL import Image, ImageTk  # Pillow for image handling in Tkinter
import random  # Random for AI choice generation
import time  # Time for timestamp generation
from tkinter import ttk  # Tkinter themed widgets for table display
import speech_recognition as sr  # Speech recognition library
import threading  # Threading for non-blocking speech recognition
import os  # OS for file operations

# Initialize MediaPipe Hands for gesture detection
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.8, min_tracking_confidence=0.8)

# Initialize Speech Recognizer
recognizer = sr.Recognizer()

class GameLogic:
    def __init__(self):
        """Initialize the GameLogic with choices for RPS and RPSLS modes."""
        self.choices = {
            "RPS": ["Rock", "Paper", "Scissors"],
            "RPSLS": ["Rock", "Paper", "Scissors", "Lizard", "Spock"]
        }

    def determine_winner(self, player_choice, ai_choice, game_mode="RPS"):
        """
        Determine the winner of a game round.

        Args:
            player_choice (str): Player's gesture.
            ai_choice (str): AI's gesture.
            game_mode (str): Game mode (default: "RPS").

        Returns:
            str: Result ("Win", "Lose", "Tie", or "Invalid gesture").
        """
        if player_choice not in self.choices[game_mode] or ai_choice not in self.choices[game_mode]:
            return "Invalid gesture"
        if player_choice == ai_choice:
            return "Tie"
        victories = {
            "Rock": ["Scissors", "Lizard"],
            "Paper": ["Rock", "Spock"],
            "Scissors": ["Paper", "Lizard"],
            "Lizard": ["Spock", "Paper"],
            "Spock": ["Scissors", "Rock"]
        } if game_mode == "RPSLS" else {
            "Rock": ["Scissors"],
            "Paper": ["Rock"],
            "Scissors": ["Paper"]
        }
        return "Win" if ai_choice in victories[player_choice] else "Lose"

def detect_gesture(landmarks, game_mode="RPS"):
    """
    Detect hand gestures based on finger positions.

    Args:
        landmarks (list): List of hand landmarks from MediaPipe.
        game_mode (str): Game mode (default: "RPS").

    Returns:
        str: Detected gesture or None.
    """
    thumb_tip = landmarks[4]
    thumb_mcp = landmarks[2]
    index_tip = landmarks[8]
    index_mcp = landmarks[5]
    middle_tip = landmarks[12]
    middle_mcp = landmarks[9]
    ring_tip = landmarks[16]
    ring_mcp = landmarks[13]
    pinky_tip = landmarks[20]
    pinky_mcp = landmarks[17]

    def is_finger_extended(tip, mcp, threshold=0.05):
        return tip.y < mcp.y - threshold

    thumb_extended = is_finger_extended(thumb_tip, thumb_mcp)
    index_extended = is_finger_extended(index_tip, index_mcp)
    middle_extended = is_finger_extended(middle_tip, middle_mcp)
    ring_extended = is_finger_extended(ring_tip, ring_mcp)
    pinky_extended = is_finger_extended(pinky_tip, pinky_mcp)

    if game_mode == "RPS":
        if not index_extended and not middle_extended and not ring_extended and not pinky_extended:
            return "Rock"
        elif index_extended and middle_extended and ring_extended and pinky_extended:
            return "Paper"
        elif index_extended and middle_extended and not ring_extended and not pinky_extended:
            return "Scissors"
    elif game_mode == "RPSLS":
        if not index_extended and not middle_extended and ring_extended and pinky_extended:
            return "Lizard"
        elif index_extended and middle_extended and not ring_extended and pinky_extended:
            return "Spock"
        elif not index_extended and not middle_extended and not ring_extended and not pinky_extended:
            return "Rock"
        elif index_extended and middle_extended and ring_extended and pinky_extended:
            return "Paper"
        elif index_extended and middle_extended and not ring_extended and not pinky_extended:
            return "Scissors"
    return None

def visualize_landmarks(frame, landmarks):
    """
    Visualize hand landmarks on the frame.

    Args:
        frame (numpy.ndarray): Input frame.
        landmarks: Hand landmarks from MediaPipe.

    Returns:
        numpy.ndarray: Frame with landmarks drawn.
    """
    if landmarks:
        contour_img = frame.copy()
        mp_drawing.draw_landmarks(
            contour_img, landmarks, mp_hands.HAND_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2),
            mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2)
        )
        return contour_img
    return frame

def process_image(frame):
    """
    Process frame to isolate hand using greyscale, thresholding, and contours.

    Args:
        frame (numpy.ndarray): Input frame.

    Returns:
        tuple: (grey, thresh, bg_removed).
    """
    grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(grey, 100, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros(grey.shape, dtype=np.uint8)
    for contour in contours:
        if cv2.contourArea(contour) > 2000:
            cv2.drawContours(mask, [contour], -1, 255, -1)
    bg_removed = cv2.bitwise_and(frame, frame, mask=mask)
    return grey, thresh, bg_removed

class GUIDemo:
    def __init__(self, root):
        """
        Initialize the enhanced GUI for the Rock Paper Scissors game.

        Args:
            root (tk.Tk): The root Tkinter window.
        """
        self.root = root
        self.root.title("Rock Paper Scissors Game")
        self.root.geometry("1400x800")
        self.root.configure(bg="#2C3E50")  # Dark blue background

        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("Error: Could not open webcam.")
            return

        self.game_logic = GameLogic()
        self.game_mode = "RPS"
        self.results = []  # Store game results temporarily for display
        self.player_score = 0
        self.ai_score = 0
        self.max_score = 5  # Game ends at 5 points
        self.game_active = False
        self.round_number = 0
        self.showing_preview = False  # Track preview state
        self.preview_start_time = None  # Track when preview started
        self.preview_after_id = None  # Track the after ID for canceling
        self.listening = True  # Control speech recognition thread
        self.screenshot_count = 0  # Track screenshot count
        self.save_path = "screenshots"  # Directory for screenshots
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

        # Title
        title_label = tk.Label(
            self.root, text="Rock Paper Scissors", font=("Helvetica", 24, "bold"),
            fg="#ECF0F1", bg="#2C3E50"
        )
        title_label.pack(pady=10)

        # Instruction for Voice Commands
        instruction_label = tk.Label(
            self.root, text="Voice Commands: Say 'start' to begin, 'proceed' to play a round, 'reset' to restart",
            font=("Helvetica", 12), fg="#BDC3C7", bg="#2C3E50"
        )
        instruction_label.pack(pady=5)

        # Frame for image panels
        panel_frame = tk.Frame(self.root, bg="#2C3E50")
        panel_frame.pack(pady=10)

        # Webcam Feed
        webcam_frame = tk.Frame(panel_frame, bg="#2C3E50")
        webcam_frame.pack(side=tk.LEFT, padx=10)
        tk.Label(webcam_frame, text="Webcam Feed", font=("Helvetica", 12, "bold"),
                 fg="#ECF0F1", bg="#2C3E50").pack()
        self.webcam_label = tk.Label(webcam_frame, bg="#34495E", borderwidth=2, relief="solid")
        self.webcam_label.pack()
        tk.Label(webcam_frame, text="Webcam Feed", font=("Helvetica", 10),
                 fg="#BDC3C7", bg="#2C3E50").pack()

        # Greyscale
        grey_frame = tk.Frame(panel_frame, bg="#2C3E50")
        grey_frame.pack(side=tk.LEFT, padx=10)
        tk.Label(grey_frame, text="Greyscale", font=("Helvetica", 12, "bold"),
                 fg="#ECF0F1", bg="#2C3E50").pack()
        self.grey_label = tk.Label(grey_frame, bg="#34495E", borderwidth=2, relief="solid")
        self.grey_label.pack()
        tk.Label(grey_frame, text="Greyscale", font=("Helvetica", 10),
                 fg="#BDC3C7", bg="#2C3E50").pack()

        # Thresholded
        thresh_frame = tk.Frame(panel_frame, bg="#2C3E50")
        thresh_frame.pack(side=tk.LEFT, padx=10)
        tk.Label(thresh_frame, text="Thresholded", font=("Helvetica", 12, "bold"),
                 fg="#ECF0F1", bg="#2C3E50").pack()
        self.thresh_label = tk.Label(thresh_frame, bg="#34495E", borderwidth=2, relief="solid")
        self.thresh_label.pack()
        tk.Label(thresh_frame, text="Thresholded", font=("Helvetica", 10),
                 fg="#BDC3C7", bg="#2C3E50").pack()

        # Background Removed
        bg_frame = tk.Frame(panel_frame, bg="#2C3E50")
        bg_frame.pack(side=tk.LEFT, padx=10)
        tk.Label(bg_frame, text="Background Removed", font=("Helvetica", 12, "bold"),
                 fg="#ECF0F1", bg="#2C3E50").pack()
        self.bg_label = tk.Label(bg_frame, bg="#34495E", borderwidth=2, relief="solid")
        self.bg_label.pack()
        tk.Label(bg_frame, text="Background Removed", font=("Helvetica", 10),
                 fg="#BDC3C7", bg="#2C3E50").pack()

        # Score and Result Frame
        score_frame = tk.Frame(self.root, bg="#2C3E50")
        score_frame.pack(pady=10)

        # Player Score
        self.player_score_label = tk.Label(
            score_frame, text="Player Score: 0", font=("Helvetica", 14),
            fg="#2ECC71", bg="#2C3E50"
        )
        self.player_score_label.pack(side=tk.LEFT, padx=20)

        # AI Score
        self.ai_score_label = tk.Label(
            score_frame, text="AI Score: 0", font=("Helvetica", 14),
            fg="#E74C3C", bg="#2C3E50"
        )
        self.ai_score_label.pack(side=tk.LEFT, padx=20)

        # Result Display
        self.result_label = tk.Label(
            self.root, text="Result: Click 'Start Game' or say 'start' to begin...", font=("Helvetica", 16),
            fg="#F1C40F", bg="#2C3E50"
        )
        self.result_label.pack(pady=10)

        # Results Table
        table_frame = tk.Frame(self.root, bg="#2C3E50")
        table_frame.pack(pady=10, fill=tk.BOTH, expand=True)
        self.tree = ttk.Treeview(
            table_frame, columns=("Round", "Player", "AI", "Result"), show="headings"
        )
        self.tree.heading("Round", text="Round")
        self.tree.heading("Player", text="Player Choice")
        self.tree.heading("AI", text="AI Choice")
        self.tree.heading("Result", text="Result")
        self.tree.column("Round", width=100, anchor="center")
        self.tree.column("Player", width=150, anchor="center")
        self.tree.column("AI", width=150, anchor="center")
        self.tree.column("Result", width=150, anchor="center")
        style = ttk.Style()
        style.configure("Treeview", background="#34495E", foreground="#ECF0F1", fieldbackground="#34495E")
        style.configure("Treeview.Heading", font=("Helvetica", 12, "bold"))
        self.tree.pack(fill=tk.BOTH, expand=True)

        # Start Game Button
        self.start_button = tk.Button(
            self.root, text="Start Game", font=("Helvetica", 14, "bold"),
            bg="#3498DB", fg="#ECF0F1", activebackground="#2980B9",
            command=self.start_game
        )
        self.start_button.pack(pady=10)

        # Proceed to Round Button (initially hidden)
        self.proceed_button = tk.Button(
            self.root, text="Proceed to Round 1", font=("Helvetica", 14, "bold"),
            bg="#F1C40F", fg="#2C3E50", activebackground="#D4AC0D",
            command=self.proceed_round
        )
        self.proceed_button.pack(pady=5)
        self.proceed_button.pack_forget()  # Hide initially

        # Reset Button
        self.reset_button = tk.Button(
            self.root, text="Reset Game", font=("Helvetica", 14, "bold"),
            bg="#E74C3C", fg="#ECF0F1", activebackground="#C0392B",
            command=self.reset_game
        )
        self.reset_button.pack(pady=5)

        # Start the speech recognition thread
        self.speech_thread = threading.Thread(target=self.listen_for_commands, daemon=True)
        self.speech_thread.start()

    def capture_screenshot(self, frame):
        """
        Capture and save a screenshot from the webcam frame.

        Args:
            frame (numpy.ndarray): The frame to save.
        """
        try:
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            filename = f"game_end_{self.screenshot_count}_{timestamp}.png"
            filepath = os.path.join(self.save_path, filename)
            cv2.imwrite(filepath, frame)
            print(f"Screenshot saved: {filepath}")
            self.screenshot_count += 1
            self.root.after(0, lambda: self.result_label.config(
                text=f"Game Over! {self.result_label.cget('text').split('! ')[1]} Screenshot saved: {filepath}"
            ))
        except Exception as e:
            print(f"Error saving screenshot: {e}")
            self.root.after(0, lambda: self.result_label.config(
                text=f"Game Over! {self.result_label.cget('text').split('! ')[1]} Error saving screenshot."
            ))

    def listen_for_commands(self):
        """Listen for voice commands in a separate thread."""
        with sr.Microphone() as source:
            # Adjust for ambient noise
            recognizer.adjust_for_ambient_noise(source)
            print("Speech recognition started. Say 'start', 'proceed', or 'reset'.")
            while self.listening:
                try:
                    audio = recognizer.listen(source, timeout=1, phrase_time_limit=3)
                    command = recognizer.recognize_google(audio).lower()
                    print(f"Recognized command: {command}")
                    # Map commands to actions, executed on the main thread
                    if "start" in command:
                        self.root.after(0, self.start_game)
                    elif "proceed" in command:
                        self.root.after(0, self.proceed_round)
                    elif "reset" in command:
                        self.root.after(0, self.reset_game)
                    else:
                        self.root.after(0, lambda: self.result_label.config(text="Command not recognized. Try 'start', 'proceed', or 'reset'."))
                except sr.WaitTimeoutError:
                    # Timeout waiting for speech; continue listening
                    continue
                except sr.UnknownValueError:
                    self.root.after(0, lambda: self.result_label.config(text="Could not understand audio. Try again."))
                except sr.RequestError as e:
                    self.root.after(0, lambda: self.result_label.config(text=f"Speech recognition error: {e}"))
                except Exception as e:
                    print(f"Speech recognition error: {e}")
                    self.root.after(0, lambda: self.result_label.config(text="Error in speech recognition."))

    def start_game(self):
        """Start the game by initiating panel previews."""
        if self.game_active:
            return
        self.game_active = True
        self.start_button.pack_forget()  # Hide Start Game button
        self.proceed_button.pack()  # Show Proceed button
        self.result_label.config(text="Panels starting previews...")
        self.preview_start_time = time.time()
        self.show_preview()

    def reset_game(self):
        """Reset the game to initial state."""
        self.game_active = False
        self.player_score = 0
        self.ai_score = 0
        self.round_number = 0
        self.results = []
        self.player_score_label.config(text="Player Score: 0")
        self.ai_score_label.config(text="AI Score: 0")
        self.result_label.config(text="Result: Click 'Start Game' or say 'start' to begin...")
        self.start_button.pack()
        self.proceed_button.pack_forget()
        for item in self.tree.get_children():
            self.tree.delete(item)
        self.showing_preview = False
        self.preview_start_time = None
        # Cancel any pending preview updates
        if self.preview_after_id is not None:
            self.root.after_cancel(self.preview_after_id)
            self.preview_after_id = None

    def update_image(self, label, img):
        """
        Update a Tkinter label with a processed image.

        Args:
            label (tk.Label): The label to update.
            img (numpy.ndarray): The image to display.
        """
        try:
            img = cv2.resize(img, (200, 150))
            if len(img.shape) == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(img)
            photo = ImageTk.PhotoImage(img_pil)
            label.config(image=photo)
            label.image = photo
        except Exception as e:
            print(f"Error updating image: {e}")

    def show_preview(self):
        """Display live preview on panels until Proceed button is clicked."""
        if not self.game_active:
            return
        ret, frame = self.cap.read()
        if not ret:
            print("Error: Could not read frame.")
            self.result_label.config(text="Error: Could not read frame. Restart the game.")
            return
        frame = cv2.flip(frame, 1)
        grey, thresh, bg_removed = process_image(frame)
        self.update_image(self.webcam_label, frame)
        self.update_image(self.grey_label, grey)
        self.update_image(self.thresh_label, thresh)
        self.update_image(self.bg_label, bg_removed)
        if not self.showing_preview and time.time() - self.preview_start_time > 1.0:  # Wait 1 second for stability
            self.result_label.config(text=f"Panels active. Say 'proceed' or click 'Proceed to Round {self.round_number + 1}'...")
            self.showing_preview = True
        # Schedule the next preview update with a longer interval
        self.preview_after_id = self.root.after(50, self.show_preview)

    def proceed_round(self):
        """Proceed to the current round after button click or voice command."""
        if not self.game_active:
            return
        # Cancel any pending preview updates
        if self.preview_after_id is not None:
            self.root.after_cancel(self.preview_after_id)
            self.preview_after_id = None
        self.play_round()

    def play_round(self):
        """Process a single round after Proceed button is clicked or voice command."""
        print("Starting round processing...")
        ret, frame = self.cap.read()
        if not ret:
            print("Error: Failed to read frame from webcam.")
            self.result_label.config(text="Error: Webcam frame not available. Restart the game.")
            return
        frame = cv2.flip(frame, 1)
        grey, thresh, bg_removed = process_image(frame)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)
        if results.multi_hand_landmarks:
            print("Hand landmarks detected.")
            for hand_landmarks in results.multi_hand_landmarks:
                frame = visualize_landmarks(frame, hand_landmarks)
                gesture = detect_gesture(hand_landmarks.landmark, self.game_mode)
                if gesture:
                    print(f"Detected gesture: {gesture}")
                    ai_choice = random.choice(self.game_logic.choices[self.game_mode])
                    result = self.game_logic.determine_winner(gesture, ai_choice, self.game_mode)
                    self.round_number += 1
                    # Update scores
                    if result == "Win":
                        self.player_score += 1
                    elif result == "Lose":
                        self.ai_score += 1
                    # Update labels
                    self.player_score_label.config(text=f"Player Score: {self.player_score}")
                    self.ai_score_label.config(text=f"AI Score: {self.ai_score}")
                    self.result_label.config(text=f"Result: Player: {gesture}, AI: {ai_choice}, {result}")
                    # Add to results and table
                    self.results.append({
                        "round": self.round_number,
                        "player": gesture,
                        "ai": ai_choice,
                        "result": result
                    })
                    self.tree.insert("", tk.END, values=(self.round_number, gesture, ai_choice, result))
                    # Force GUI update
                    self.root.update()
                    # Check for game end
                    if self.player_score >= self.max_score or self.ai_score >= self.max_score:
                        winner = "Player" if self.player_score >= self.max_score else "AI"
                        self.result_label.config(text=f"Game Over! {winner} Wins! Say 'reset' to restart.")
                        self.game_active = False
                        # Capture screenshot
                        self.capture_screenshot(frame)
                        return
                    else:
                        self.result_label.config(text=f"Round {self.round_number} complete! Panels restarting...")
                        self.showing_preview = False
                        self.preview_start_time = time.time()
                        self.proceed_button.config(text=f"Proceed to Round {self.round_number + 1}")
                        # Add a small delay to allow GUI to refresh
                        self.root.after(100, self.show_preview)
                        return
                else:
                    print("No valid gesture detected.")
                    self.result_label.config(text="No valid gesture detected. Try again.")
                    self.root.update()
        else:
            print("No hand landmarks detected.")
            self.result_label.config(text="No hand detected. Try again.")
            self.root.update()
        print("Round processing completed.")
        # Restart preview after a delay
        self.root.after(100, self.show_preview)

    def update_frame(self):
        """Update the GUI with the latest webcam frame (initially empty, handled by show_preview)."""
        if not self.game_active:
            self.root.after(50, self.update_frame)
            return
        self.root.after(50, self.update_frame)

    def cleanup(self):
        """Clean up resources when closing the application."""
        self.listening = False  # Stop the speech recognition thread
        if self.preview_after_id is not None:
            self.root.after_cancel(self.preview_after_id)
        self.cap.release()

# Main execution
if __name__ == "__main__":
    root = tk.Tk()
    app = GUIDemo(root)
    root.protocol("WM_DELETE_WINDOW", lambda: [app.cleanup(), root.destroy()])
    root.mainloop()
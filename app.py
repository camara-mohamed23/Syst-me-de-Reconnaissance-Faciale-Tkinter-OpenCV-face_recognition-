import tkinter as tk
from tkinter import messagebox
import face_recognition
import cv2
import os
import numpy as np
from PIL import Image, ImageTk

# === Initialisation des visages connus ===
known_face_encodings = []
known_face_names = []

# Charger les visages depuis le dossier "known_faces"
def load_known_faces():
    for filename in os.listdir("known_faces"):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image_path = os.path.join("known_faces", filename)
            image = face_recognition.load_image_file(image_path)
            encoding = face_recognition.face_encodings(image)
            if encoding:
                known_face_encodings.append(encoding[0])
                known_face_names.append(os.path.splitext(filename)[0])
            else:
                print(f"[!] Aucun visage détecté dans {filename}")

# === Interface Tkinter ===
class FaceRecognitionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("🧠 Système de Reconnaissance Faciale")
        self.root.geometry("800x600")

        self.label = tk.Label(root, text="Cliquez sur 'Lancer' pour démarrer la caméra", font=("Arial", 16))
        self.label.pack(pady=20)

        self.video_label = tk.Label(root)
        self.video_label.pack()

        self.button_start = tk.Button(root, text="Lancer la reconnaissance", command=self.start_camera, bg="green", fg="white", font=("Arial", 14))
        self.button_start.pack(pady=10)

        self.button_capture = tk.Button(root, text="Capturer visage", command=self.capture_face, bg="blue", fg="white", font=("Arial", 14))
        self.button_capture.pack(pady=10)
        self.button_capture.config(state="disabled")

        self.cap = None
        self.running = False
        self.current_frame = None
        self.current_face_locations = []

    def start_camera(self):
        self.running = True
        self.cap = cv2.VideoCapture(0)
        self.button_capture.config(state="normal")
        self.recognize_faces()

    def recognize_faces(self):
        if not self.running:
            return

        ret, frame = self.cap.read()
        if not ret:
            return

        self.current_frame = frame.copy()  # garder la frame pour la capture
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_frame = small_frame[:, :, ::-1]

        face_locations = face_recognition.face_locations(rgb_frame)
        self.current_face_locations = [(top*4, right*4, bottom*4, left*4) for (top, right, bottom, left) in face_locations]

        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        for (top, right, bottom, left), face_encoding in zip(self.current_face_locations, face_encodings):
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Inconnu"

            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            if len(face_distances) > 0:
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = known_face_names[best_match_index]

            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
            cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 0, 0), 1)

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        imgtk = ImageTk.PhotoImage(image=image)
        self.video_label.imgtk = imgtk
        self.video_label.configure(image=imgtk)
        self.root.after(10, self.recognize_faces)

    def capture_face(self):
        if self.current_frame is None or not self.current_face_locations:
            messagebox.showwarning("Attention", "Aucun visage détecté à capturer.")
            return

        os.makedirs("known_faces", exist_ok=True)
        count = len(os.listdir("known_faces")) + 1

        for (top, right, bottom, left) in self.current_face_locations:
            face_image = self.current_frame[top:bottom, left:right]
            filename = os.path.join("known_faces", f"face_{count}.jpg")
            cv2.imwrite(filename, face_image)
            count += 1

        messagebox.showinfo("Succès", f"{len(self.current_face_locations)} visage(s) capturé(s) et sauvegardé(s).")

# === Lancement de l'application ===
if __name__ == "__main__":
    # S'assurer que le dossier known_faces existe
    os.makedirs("known_faces", exist_ok=True)
    load_known_faces()
    root = tk.Tk()
    app = FaceRecognitionApp(root)
    root.mainloop()

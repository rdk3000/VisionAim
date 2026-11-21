import os
os.environ.setdefault("YOLO_VERBOSE", "False")

import threading
import time
import tkinter as tk
from tkinter import ttk, messagebox

import cv2
import dxcam
import numpy as np
from PIL import Image, ImageTk
from ultralytics import YOLO

APP_TITLE = "VisionAim"
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "yolo11n-pose.pt")
CONFIDENCE = 0.35
IMAGE_SIZE = 640

class VisionAimApp:
    def __init__(self, root):
        self.root = root
        self.root.title(APP_TITLE)
        self.root.geometry("1120x720")
        self.root.minsize(850, 560)
        self.root.configure(bg="#10141c")

        self.running = False
        self.worker = None
        self.camera = None
        self.model = None
        self.photo = None
        self.latest_frame = None
        self.frame_lock = threading.Lock()
        self.status = tk.StringVar(value="Pronto para iniciar")
        self.fps_text = tk.StringVar(value="FPS: --")
        self.people_text = tk.StringVar(value="Pessoas: --")

        self._build_ui()
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.root.after(30, self._refresh_preview)

    def _build_ui(self):
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TButton", font=("Segoe UI", 10), padding=9)
        style.configure("TLabel", background="#10141c", foreground="#e8edf5",
                        font=("Segoe UI", 10))
        style.configure("Title.TLabel", font=("Segoe UI", 22, "bold"),
                        foreground="#ffffff")
        style.configure("Sub.TLabel", foreground="#98a5b8", font=("Segoe UI", 9))

        header = tk.Frame(self.root, bg="#10141c")
        header.pack(fill="x", padx=22, pady=(18, 12))
        ttk.Label(header, text="VISIONAIM", style="Title.TLabel").pack(side="left")
        ttk.Label(header, text="AI  /  VISUAL DIAGNOSTICS", style="Sub.TLabel").pack(
            side="left", padx=14, pady=(9, 0))

        body = tk.Frame(self.root, bg="#10141c")
        body.pack(fill="both", expand=True, padx=22, pady=(0, 18))

        preview_panel = tk.Frame(body, bg="#080b10", highlightbackground="#283243",
                                 highlightthickness=1)
        preview_panel.pack(side="left", fill="both", expand=True)
        self.preview = tk.Label(preview_panel, text="VISIONAIM\n\nPressione INICIAR para começar",
                                bg="#080b10", fg="#7f8da3",
                                font=("Segoe UI", 15), justify="center")
        self.preview.pack(fill="both", expand=True, padx=8, pady=8)

        side = tk.Frame(body, bg="#171e29", width=245)
        side.pack(side="right", fill="y", padx=(14, 0))
        side.pack_propagate(False)

        tk.Label(side, text="CONTROLES", bg="#171e29", fg="#ffffff",
                 font=("Segoe UI", 11, "bold")).pack(anchor="w", padx=17, pady=(20, 14))

        self.start_button = tk.Button(side, text="INICIAR", command=self.start,
            bg="#39d98a", fg="#07130d", activebackground="#2bc47a",
            relief="flat", font=("Segoe UI", 10, "bold"), cursor="hand2")
        self.start_button.pack(fill="x", padx=16, pady=5, ipady=8)

        self.stop_button = tk.Button(side, text="PARAR", command=self.stop,
            bg="#303b4c", fg="#e8edf5", activebackground="#3b4a60",
            relief="flat", font=("Segoe UI", 10, "bold"), cursor="hand2",
            state="disabled")
        self.stop_button.pack(fill="x", padx=16, pady=5, ipady=8)

        tk.Frame(side, bg="#303b4c", height=1).pack(fill="x", padx=16, pady=18)
        tk.Label(side, text="DESEMPENHO", bg="#171e29", fg="#98a5b8",
                 font=("Segoe UI", 9, "bold")).pack(anchor="w", padx=17)
        ttk.Label(side, textvariable=self.fps_text).pack(anchor="w", padx=17, pady=(12, 3))
        ttk.Label(side, textvariable=self.people_text).pack(anchor="w", padx=17, pady=3)

        tk.Frame(side, bg="#303b4c", height=1).pack(fill="x", padx=16, pady=18)
        tk.Label(side, text="STATUS", bg="#171e29", fg="#98a5b8",
                 font=("Segoe UI", 9, "bold")).pack(anchor="w", padx=17)
        tk.Label(side, textvariable=self.status, bg="#171e29", fg="#39d98a",
                 font=("Segoe UI", 9), wraplength=205, justify="left").pack(
                     anchor="w", padx=17, pady=(10, 0))

        tk.Label(side, text="YOLO11 POSE  •  CPU MODE\nSomente análise visual; não controla o mouse.",
                 bg="#171e29", fg="#718096", font=("Segoe UI", 8),
                 justify="left", wraplength=210).pack(side="bottom", anchor="w",
                                                       padx=17, pady=18)

    def start(self):
        if self.running:
            return
        self.running = True
        self.start_button.configure(state="disabled")
        self.stop_button.configure(state="normal")
        self.status.set("Carregando modelo...")
        self.worker = threading.Thread(target=self._worker, daemon=True)
        self.worker.start()

    def stop(self):
        self.running = False
        self.status.set("Parando...")
        self.start_button.configure(state="normal")
        self.stop_button.configure(state="disabled")

    def _worker(self):
        try:
            self.model = YOLO(MODEL_PATH)
            self.camera = dxcam.create(output_color="BGR")
            self.camera.start(target_fps=30, video_mode=True)
            self.root.after(0, lambda: self.status.set("Detectando"))
            last = time.perf_counter()
            smooth_fps = 0.0

            while self.running:
                frame = self.camera.get_latest_frame()
                if frame is None:
                    time.sleep(0.005)
                    continue

                result = self.model.predict(frame, conf=CONFIDENCE,
                    imgsz=IMAGE_SIZE, verbose=False, device="cpu")[0]
                view = frame.copy()
                people = 0

                if result.boxes is not None:
                    boxes = result.boxes.xyxy.cpu().numpy()
                    points = result.keypoints.xy.cpu().numpy() if result.keypoints is not None else []
                    for i, box in enumerate(boxes):
                        people += 1
                        x1, y1, x2, y2 = map(int, box[:4])
                        cv2.rectangle(view, (x1, y1), (x2, y2), (80, 255, 145), 2)
                        if len(points) > i and len(points[i]) > 0:
                            nose = points[i][0]
                            if np.isfinite(nose).all() and (nose > 0).all():
                                cv2.circle(view, tuple(map(int, nose)), 6, (0, 220, 255), -1)
                        cv2.putText(view, "PERSON", (x1, max(22, y1 - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (80, 255, 145), 2)

                now = time.perf_counter()
                current = 1.0 / max(now - last, 1e-6)
                smooth_fps = current if smooth_fps == 0 else smooth_fps * .9 + current * .1
                last = now
                cv2.putText(view, f"VISIONAIM  |  {smooth_fps:.1f} FPS  |  {people} PEOPLE",
                    (18, 32), cv2.FONT_HERSHEY_SIMPLEX, .7, (80, 255, 145), 2)

                with self.frame_lock:
                    self.latest_frame = view
                self.root.after(0, lambda f=smooth_fps, p=people:
                    (self.fps_text.set(f"FPS: {f:.1f}"),
                     self.people_text.set(f"Pessoas: {p}")))

        except Exception as exc:
            self.root.after(0, lambda e=str(exc): self._show_error(e))
        finally:
            try:
                if self.camera is not None:
                    self.camera.stop()
            except Exception:
                pass
            self.camera = None
            self.running = False
            self.root.after(0, self._set_stopped)

    def _show_error(self, error):
        self.status.set("Erro ao iniciar")
        messagebox.showerror("VisionAim", error)

    def _set_stopped(self):
        self.start_button.configure(state="normal")
        self.stop_button.configure(state="disabled")
        if self.status.get() == "Parando...":
            self.status.set("Parado")

    def _refresh_preview(self):
        with self.frame_lock:
            frame = None if self.latest_frame is None else self.latest_frame.copy()
        if frame is not None:
            h, w = frame.shape[:2]
            max_w = max(300, self.preview.winfo_width() - 16)
            max_h = max(220, self.preview.winfo_height() - 16)
            scale = min(max_w / w, max_h / h)
            resized = cv2.resize(frame, (int(w * scale), int(h * scale)))
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            self.photo = ImageTk.PhotoImage(Image.fromarray(rgb))
            self.preview.configure(image=self.photo, text="")
        self.root.after(30, self._refresh_preview)

    def close(self):
        self.running = False
        self.root.after(150, self.root.destroy)

if __name__ == "__main__":
    root = tk.Tk()
    app = VisionAimApp(root)
    root.mainloop()

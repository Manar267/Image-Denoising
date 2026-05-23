print("START")
import tkinter as tk
from tkinter import filedialog, messagebox
import cv2
import cv2 as cv
import numpy as np
import pywt
import matplotlib.pyplot as plt
from PIL import Image, ImageTk
from skimage import restoration
from skimage.util import random_noise
from skimage.restoration import denoise_tv_chambolle
from scipy.signal import convolve2d
import torch
import torch.nn as nn


# =========================
# GUI Application Class
# =========================
class ModernApp:
    def build_cnn(self):
        class SimpleCNN(nn.Module):
            def __init__(self):
                super(SimpleCNN, self).__init__()
                self.seq = nn.Sequential(
                    nn.Conv2d(3, 32, kernel_size=3, padding=1),
                    nn.ReLU(),
                    nn.Conv2d(32, 32, kernel_size=3, padding=1),
                    nn.ReLU(),
                    nn.Conv2d(32, 3, kernel_size=3, padding=1),
                    nn.Sigmoid()
                )
            def forward(self, x):
                return self.seq(x)
        
        model = SimpleCNN()
        model.eval()
        return model

    def __init__(self, root):
        self.root = root
        self.root.title("Image Processing Studio Pro v1.0")
        self.root.geometry("1350x950")
        self.root.configure(bg="#121212")

        self.img = None
        self.output = None
        self.active_slider = None

        # --- Sidebar ---
        self.sidebar = tk.Frame(root, bg="#1e1e1e", width=250)
        self.sidebar.pack(side="left", fill="y")
        self.sidebar.pack_propagate(False)
        self.cnn_model = self.build_cnn()

        tk.Label(
            self.sidebar,
            text="IMAGE STUDIO",
            fg="#00d1b2",
            bg="#1e1e1e",
            font=("Segoe UI", 18, "bold"),
        ).pack(pady=20, padx=20)

        self.add_menu_button("Load Image", self.load_image, "#2ecc71")
        self.add_menu_button("Save Result", self.save_image, "#e67e22")

        tk.Label(
            self.sidebar,
            text="FILTERS",
            fg="gray",
            bg="#1e1e1e",
            font=("Segoe UI", 10, "bold"),
        ).pack(pady=10)

        # =========================
        # قائمة الفلاتر
        # s_range = (min, max, default)
        # =========================
        filters = [
            ("NLM Color", self.apply_nlm_color, "#3498db", (1, 50, 10)),
            ("NLM Grayscale", self.apply_Grayscale_nlm, "#16a085", (1, 31, 5)),
            ("TV Denoising", self.apply_tv, "#4b7bec", (1, 50, 10)),
            ("Mean Filter", self.apply_mean, "#7f8c8d", (1, 15, 5)),
            ("Median Filter", self.apply_median, "#f1c40f", (1, 15, 3)),
            ("Bilateral Filter", self.apply_bilateral, "#1abc9c", (1, 150, 75)),
            ("Anisotropic", self.apply_anisotropic, "#e91e63", (1, 50, 30)),
            ("Diffusion", self.apply_diffusion, "#ff6b35", (1, 50, 20)),  # ← جديد
            ("Guided Filter", self.apply_guided, "#e74c3c", (1, 40, 11)),
            ("FFT", self.apply_fft, "#9b59b6", (5, 100, 40)),
            ("Wavelet Filter", self.apply_wavelet, "#a55eea", (0, 100, 40)),
            ("Wiener Filter", self.apply_wiener_filter, "#34495e", (1, 15, 5)),
            ("Morphology Demo", self.apply_morphology_demo, "#2c3e50", (3, 15, 7)),
            ("Add Noise", self.apply_noise, "#e3ee14", (1, 50, 25)),
            ("Noise + Gaussian", self.apply_noise_gaussian, "#81c11a", (1, 50, 25)),
            ("Gaussian Blur", self.apply_gaussian, "#e70008", (1, 25, 5)),
            ("CNN (Untrained)", self.apply_cnn, "#00bcd4", (0, 0, 0)),
        ]

        for text, cmd, color, s_range in filters:
            self.add_filter_button(text, cmd, color, s_range)

        # --- Main View ---
        self.main_area = tk.Frame(root, bg="#121212")
        self.main_area.pack(side="right", expand=True, fill="both", padx=20, pady=20)
        self.main_area.columnconfigure(0, weight=1)
        self.main_area.columnconfigure(1, weight=1)

        self.panel1 = tk.Label(
            self.main_area, text="Original Image Preview", bg="#1e1e1e", fg="#555"
        )
        self.panel1.grid(row=0, column=0, padx=15, pady=15, sticky="nsew")

        self.panel2 = tk.Label(
            self.main_area, text="Filtered Result Preview", bg="#1e1e1e", fg="#555"
        )
        self.panel2.grid(row=0, column=1, padx=15, pady=15, sticky="nsew")

        self.slider_frame = tk.Frame(self.main_area, bg="#121212")
        self.slider_frame.grid(row=1, column=0, columnspan=2, pady=10)

        self.slider_label = tk.Label(
            self.slider_frame, text="", fg="#00d1b2", bg="#121212"
        )
        self.slider_label.pack()

    # ------------------------------------------------------------------
    def add_menu_button(self, text, command, color):
        btn = tk.Button(
            self.sidebar,
            text=text,
            command=command,
            bg=color,
            fg="white",
            font=("Segoe UI", 10, "bold"),
            width=22,
        )
        btn.pack(pady=4, padx=20)

    def add_filter_button(self, text, command, color, s_range):
        btn = tk.Button(
            self.sidebar,
            text=text,
            command=lambda c=command, s=s_range, n=text: self.setup_slider(n, c, s),
            bg=color,
            fg="white",
            font=("Segoe UI", 10, "bold"),
            width=22,
        )
        btn.pack(pady=4, padx=20)

    def setup_slider(self, name, func, s_range):
        if self.img is None:
            messagebox.showwarning("No Image", "Please load an image first.")
            return
        if self.active_slider:
            self.active_slider.destroy()
        self.slider_label.config(text=f"Intensity: {name}")
        self.active_slider = tk.Scale(
            self.slider_frame,
            from_=s_range[0],
            to=s_range[1],
            orient="horizontal",
            length=400,
            bg="#121212",
            fg="white",
            command=lambda v: func(v),
        )
        self.active_slider.set(s_range[2])
        self.active_slider.pack()
        func(s_range[2])

    # ------------------------------------------------------------------
    def load_image(self):
        path = filedialog.askopenfilename(
            filetypes=[
                ("Image Files", "*.png *.jpg *.jpeg *.bmp *.tiff *.webp"),
                ("All Files", "*.*"),
            ]
        )
        if path:
            bgr = cv2.imread(path)
            if bgr is None:
                messagebox.showerror("Error", "Could not read image file.")
                return
            self.img = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            self.output = None
            self.show_image(self.img, self.panel1)

    def save_image(self):
        if self.output is None:
            messagebox.showwarning("No Output", "Apply a filter first.")
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg"), ("All Files", "*.*")],
        )
        if path:
            cv2.imwrite(path, cv2.cvtColor(self.output, cv2.COLOR_RGB2BGR))

    def show_image(self, image, panel):
        img_res = cv2.resize(np.clip(image, 0, 255).astype(np.uint8), (450, 450))
        image_tk = ImageTk.PhotoImage(Image.fromarray(img_res))
        panel.config(image=image_tk, text="")
        panel.image = image_tk

    # ==================================================================
    # ==================   FILTER IMPLEMENTATIONS   ====================
    # ==================================================================

    # --- Wavelet Filter ---
    def apply_wavelet(self, val):
        if self.img is None:
            return
        threshold_val = float(val)
        chs = cv2.split(self.img.astype(np.float32))
        denoised_chs = []
        for ch in chs:
            coeffs = pywt.wavedec2(ch, "haar", level=3)
            cA = coeffs[0]
            details = coeffs[1:]
            new_details = []
            for cH, cV, cD in details:
                cH = pywt.threshold(cH, threshold_val, mode="soft")
                cV = pywt.threshold(cV, threshold_val, mode="soft")
                cD = pywt.threshold(cD, threshold_val, mode="soft")
                new_details.append((cH, cV, cD))
            denoised_ch = pywt.waverec2([cA] + new_details, "haar")
            denoised_chs.append(denoised_ch)
        self.output = np.clip(cv2.merge(denoised_chs), 0, 255).astype(np.uint8)
        self.show_image(self.output, self.panel2)

    # --- NLM Color ---
    def apply_nlm_color(self, val):
        if self.img is None:
            return
        img_bgr = cv2.cvtColor(self.img, cv2.COLOR_RGB2BGR)
        denoised = cv2.fastNlMeansDenoisingColored(
            img_bgr,
            None,
            h=float(val),
            hColor=float(val),
            templateWindowSize=7,
            searchWindowSize=21,
        )
        self.output = cv2.cvtColor(denoised, cv2.COLOR_BGR2RGB)
        self.show_image(self.output, self.panel2)

    # --- NLM Grayscale ---
    def apply_Grayscale_nlm(self, val):
        if self.img is None:
            return
        gray = cv2.cvtColor(self.img, cv2.COLOR_RGB2GRAY)
        denoised = cv2.fastNlMeansDenoising(
            gray, None, h=float(val), templateWindowSize=7, searchWindowSize=21
        )
        self.output = cv2.cvtColor(denoised, cv2.COLOR_GRAY2RGB)
        self.show_image(self.output, self.panel2)

    # --- TV Denoising ---
    def apply_tv(self, val):
        if self.img is None:
            return
        tv = denoise_tv_chambolle(
            self.img / 255.0, weight=float(val) / 100.0, channel_axis=-1
        )
        self.output = (tv * 255).astype(np.uint8)
        self.show_image(self.output, self.panel2)

    # --- Mean Filter ---
    def apply_mean(self, val):
        if self.img is None:
            return
        v = int(val) | 1
        self.output = cv2.blur(self.img, (v, v))
        self.show_image(self.output, self.panel2)

    # --- Median Filter ---
    def apply_median(self, val):
        if self.img is None:
            return
        v = int(val) | 1
        self.output = cv2.medianBlur(self.img, v)
        self.show_image(self.output, self.panel2)

        # --- Add Noise ---

    def apply_noise(self, val):
        if self.img is None:
            return

        sigma = float(val)
        noise = np.random.normal(0, sigma, self.img.shape)
        noisy_image = self.img + noise

        self.output = np.clip(noisy_image, 0, 255).astype(np.uint8)
        self.show_image(self.output, self.panel2)

    # ________Original | Noisy | Gaussian__________
    def apply_noise_gaussian(self, val):
        if self.img is None:
            return

        sigma = float(val)

        # 1. Add Noise
        noise = np.random.normal(0, sigma, self.img.shape)
        noisy = self.img + noise
        noisy = np.clip(noisy, 0, 255).astype(np.uint8)

        # 2. Gaussian Filter
        k = 5
        gaussian = cv2.GaussianBlur(noisy, (k, k), 0)

        # Output
        self.output = gaussian
        self.show_image(self.output, self.panel2)

    def apply_gaussian(self, val):
        if self.img is None:
            return

        k = int(val) | 1  # لازم فردي
        self.output = cv2.GaussianBlur(self.img, (k, k), 0)
        self.show_image(self.output, self.panel2)

    def apply_wiener_filter(self, val):
        # 1. Prepare Kernel (PSF)
    # The 'val' from the slider now acts as the estimated blur radius.
        k = int(val) | 1
        psf = np.ones((k, k), dtype=np.float32) / (k * k)

        # 2. Process Channels
        # Convert to float32 [0, 1] as required by skimage.restoration
        img_float = self.img.astype(np.float32) / 255.0
        channels = cv2.split(img_float)
        restored_channels = []

        for ch in channels:
            try:
                
                restored, _ = restoration.unsupervised_wiener(ch, psf)
                restored_channels.append(restored)
            except Exception:
                
                restored_channels.append(ch)

        restored_img = cv2.merge(restored_channels)
        self.output = np.clip(restored_img * 255, 0, 255).astype(np.uint8)
        
        
        self.show_image(self.output, self.panel2)


    def apply_morphology_demo(self, val):
        if self.img is None:
            return

        gray = cv2.cvtColor(self.img, cv2.COLOR_RGB2GRAY)
        k = int(val) | 1
        kernel = np.ones((k, k), np.uint8)

        img1 = cv2.GaussianBlur(gray, (k, k), 0)
        erosion = cv2.erode(img1, kernel, iterations=1)
        dilation = cv2.dilate(img1, kernel, iterations=1)

        morph_types = [
            cv.MORPH_OPEN,
            cv.MORPH_CLOSE,
            cv.MORPH_GRADIENT,
            cv.MORPH_BLACKHAT,
            cv.MORPH_TOPHAT,
        ]
        morph_names = [
            "Open",
            "Close",
            "Gradient",
            "BlackHat",
            "TopHat",
        ]

        plt.figure(figsize=(12, 6))
        plt.subplot(241)
        plt.imshow(img1, cmap="gray")
        plt.title("original")
        plt.axis("off")

        plt.subplot(242)
        plt.imshow(erosion, cmap="gray")
        plt.title("Erosion")
        plt.axis("off")

        plt.subplot(243)
        plt.imshow(dilation, cmap="gray")
        plt.title("dilation")
        plt.axis("off")

        for i in range(len(morph_types)):
            plt.subplot(2, 4, i + 4)
            plt.imshow(cv2.morphologyEx(gray, morph_types[i], kernel), cmap="gray")
            plt.title(morph_names[i])
            plt.axis("off")

        plt.tight_layout()
        plt.show()

        self.output = cv2.cvtColor(img1, cv2.COLOR_GRAY2RGB)
        self.show_image(self.output, self.panel2)

    def apply_cnn(self, val):
        if self.img is None:
            return

        # resize
        img = cv2.resize(self.img, (256, 256))
        img = img.astype(np.float32) / 255.0

        # PyTorch expects (N, C, H, W)
        img_input = np.transpose(img, (2, 0, 1))
        img_input = np.expand_dims(img_input, axis=0)
        
        tensor_input = torch.tensor(img_input)

        # predict (UNTRAINED → random behavior)
        with torch.no_grad():
            output = self.cnn_model(tensor_input)

        output_np = output.numpy()[0]
        # back to (H, W, C)
        output_np = np.transpose(output_np, (1, 2, 0))

        output_np = np.clip(output_np, 0, 1)

        self.output = (output_np * 255).astype(np.uint8)
        self.show_image(self.output, self.panel2)

    # --- Bilateral Filter ---
    def apply_bilateral(self, val):
        if self.img is None:
            return
        img_bgr = cv2.cvtColor(self.img, cv2.COLOR_RGB2BGR)
        filtered = cv2.bilateralFilter(
            img_bgr, d=9, sigmaColor=float(val), sigmaSpace=float(val)
        )
        self.output = cv2.cvtColor(filtered, cv2.COLOR_BGR2RGB)
        self.show_image(self.output, self.panel2)

    # --- Anisotropic Diffusion (kappa-controlled, 15 iter) ---
    def apply_anisotropic(self, val):
        if self.img is None:
            return
        kappa = float(val)
        gamma = 0.2
        num_iter = 15

        def ani_logic(img):
            img = img.astype(np.float32)
            for _ in range(num_iter):
                n = np.roll(img, -1, axis=0) - img
                s = np.roll(img, 1, axis=0) - img
                e = np.roll(img, -1, axis=1) - img
                w = np.roll(img, 1, axis=1) - img
                cN = np.exp(-((n / kappa) ** 2))
                cS = np.exp(-((s / kappa) ** 2))
                cE = np.exp(-((e / kappa) ** 2))
                cW = np.exp(-((w / kappa) ** 2))
                img += gamma * (cN * n + cS * s + cE * e + cW * w)
            return img

        channels = cv2.split(self.img)
        filtered = [ani_logic(c) for c in channels]
        self.output = np.clip(cv2.merge(filtered), 0, 255).astype(np.uint8)
        self.show_image(self.output, self.panel2)

    # ==========================================================
    # --- Perona-Malik Diffusion (iteration-controlled) ← جديد ---
    # ==========================================================
    # الـ slider بيتحكم في عدد الـ iterations (1-50)
    # kappa ثابت عند 30 وgamma=0.25 لنتيجة أوضح
    # بيستخدم Quadratic conduction coefficient بدل Exponential
    # عشان يحافظ على الـ edges بشكل أفضل مع iterations كتير
    # ==========================================================
    def apply_diffusion(self, val):
        if self.img is None:
            return

        num_iter = int(val)  # الـ slider بيتحكم في عدد الـ iterations
        kappa = 30.0  # حساسية الـ edges (ثابتة)
        gamma = 0.25  # معدل الـ diffusion في كل step

        def perona_malik(channel):
            img = channel.astype(np.float64)

            for _ in range(num_iter):
                # حساب الفروق في الاتجاهات الأربعة
                delta_n = np.roll(img, -1, axis=0) - img
                delta_s = np.roll(img, 1, axis=0) - img
                delta_e = np.roll(img, -1, axis=1) - img
                delta_w = np.roll(img, 1, axis=1) - img

                # Quadratic conduction coefficient (Perona-Malik Function 2)
                # c(x) = 1 / (1 + (|∇I| / kappa)²)
                # أفضل في الحفاظ على الـ edges من الـ exponential
                cN = 1.0 / (1.0 + (delta_n / kappa) ** 2)
                cS = 1.0 / (1.0 + (delta_s / kappa) ** 2)
                cE = 1.0 / (1.0 + (delta_e / kappa) ** 2)
                cW = 1.0 / (1.0 + (delta_w / kappa) ** 2)

                # تحديث الصورة
                img += gamma * (
                    cN * delta_n + cS * delta_s + cE * delta_e + cW * delta_w
                )

            return img

        # تطبيق على كل channel في الصورة الملونة
        channels = cv2.split(self.img)
        diffused = [perona_malik(c) for c in channels]
        self.output = np.clip(cv2.merge(diffused), 0, 255).astype(np.uint8)
        self.show_image(self.output, self.panel2)

    # --- Guided Filter ---
    def apply_guided(self, val):
        if self.img is None:
            return
        radius = int(val)
        eps = 0.01

        def g_logic(guide, inp_img):
            guide = guide.astype(np.float64) / 255.0
            inp_img = inp_img.astype(np.float64) / 255.0
            k = 2 * radius + 1
            m_g = cv2.boxFilter(guide, -1, (k, k))
            m_i = cv2.boxFilter(inp_img, -1, (k, k))
            cov = cv2.boxFilter(guide * inp_img, -1, (k, k)) - m_g * m_i
            var = cv2.boxFilter(guide * guide, -1, (k, k)) - m_g * m_g
            a = cov / (var + eps)
            b = m_i - a * m_g
            out = cv2.boxFilter(a, -1, (k, k)) * guide + cv2.boxFilter(b, -1, (k, k))
            return np.clip(out * 255, 0, 255).astype(np.uint8)

        chs = cv2.split(self.img)
        self.output = cv2.merge([g_logic(c, c) for c in chs])
        self.show_image(self.output, self.panel2)

    # --- FFT Filter ---
    def apply_fft(self, val):
        if self.img is None:
            return
        sigma = float(val)
        chs = cv2.split(self.img.astype(np.float32))
        outs = []
        for ch in chs:
            rows, cols = ch.shape
            crow, ccol = rows // 2, cols // 2
            f = np.fft.fft2(ch)
            fshift = np.fft.fftshift(f)
            x = np.linspace(-ccol, ccol, cols)
            y = np.linspace(-crow, crow, rows)
            X, Y = np.meshgrid(x, y)
            mask = np.exp(-(X**2 + Y**2) / (2 * sigma**2))
            filtered = fshift * mask
            img_back = np.abs(np.fft.ifft2(np.fft.ifftshift(filtered)))
            img_back = cv2.normalize(img_back, None, 0, 255, cv2.NORM_MINMAX)
            outs.append(img_back)
        self.output = cv2.merge(outs).astype(np.uint8)
        self.show_image(self.output, self.panel2)
    


# =========================
# Run App
# =========================
if __name__ == "__main__":
    root = tk.Tk()
    app = ModernApp(root)
root.mainloop()

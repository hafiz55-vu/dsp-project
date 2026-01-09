# =====================================================================
# FIR Filter Design and Visualization with Audio Playback (Popup Version)
# =====================================================================

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import firwin, lfilter
import librosa
import sounddevice as sd
from tkinter import Tk, ttk, filedialog, StringVar

# -----------------------------------------------------------
# Step 1: Filter setup
# -----------------------------------------------------------
fs = 8000   # Sampling frequency
N = 512     # Filter order
fc_low, fc_band, fc_high = 1000, [1000, 2000], 2000

Wn_low = fc_low / (fs / 2)
Wn_band = [fc_band[0] / (fs / 2), fc_band[1] / (fs / 2)]
Wn_high = fc_high / (fs / 2)

def design_filters():
    """Design FIR filters using Hamming and Blackman windows."""
    return {
        'Low_Hamming': firwin(N + 1, Wn_low, window='hamming', pass_zero='lowpass'),
        'Band_Hamming': firwin(N + 1, Wn_band, window='hamming', pass_zero='bandpass'),
        'High_Hamming': firwin(N + 1, Wn_high, window='hamming', pass_zero='highpass'),
        'Low_Blackman': firwin(N + 1, Wn_low, window='blackman', pass_zero='lowpass'),
        'Band_Blackman': firwin(N + 1, Wn_band, window='blackman', pass_zero='bandpass'),
        'High_Blackman': firwin(N + 1, Wn_high, window='blackman', pass_zero='highpass'),
    }

# -----------------------------------------------------------
# Step 2: Audio loading and filtering
# -----------------------------------------------------------
def load_audio(filepath):
    """Load MP3/WAV/FLAC audio file."""
    x, sr = librosa.load(filepath, sr=fs, mono=True)
    t = np.arange(len(x)) / fs
    return x, t

def apply_filters(x):
    """Apply all designed FIR filters to the signal."""
    filters = design_filters()
    return {name: lfilter(coeffs, 1, x) for name, coeffs in filters.items()}

# -----------------------------------------------------------
# Step 3: Plot visualization (Popup separated)
# -----------------------------------------------------------
def plot_original_signal(x, t):
    """Plot the original signal in a separate popup."""
    plt.style.use('dark_background')
    plt.figure(figsize=(10, 4))
    plt.plot(t, x, color='white')
    plt.title("Original Signal", color='cyan')
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.tight_layout()
    plt.show(block=True)  # Wait for this window to close

def plot_filtered_signals(y_dict, t):
    """Plot all filtered signals in another popup."""
    plt.style.use('dark_background')
    plt.figure(figsize=(12, 10))

    titles = [
        ("Low-pass (Hamming)", y_dict['Low_Hamming'], "cyan"),
        ("Low-pass (Blackman)", y_dict['Low_Blackman'], "orange"),
        ("Band-pass (Hamming)", y_dict['Band_Hamming'], "cyan"),
        ("Band-pass (Blackman)", y_dict['Band_Blackman'], "orange"),
        ("High-pass (Hamming)", y_dict['High_Hamming'], "cyan"),
        ("High-pass (Blackman)", y_dict['High_Blackman'], "orange"),
    ]

    for i, (title, signal, color) in enumerate(titles, start=1):
        plt.subplot(3, 2, i)
        plt.plot(t, signal, color=color)
        plt.title(title, color=color)
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude")

    plt.suptitle("Filtered Outputs — Hamming vs. Blackman", fontsize=14, color='lightgreen')
    plt.tight_layout()
    plt.show(block=False)  # Allow GUI to continue

# -----------------------------------------------------------
# Step 4: Audio Playback
# -----------------------------------------------------------
def play_audio(audio_data, label=""):
    """Play selected audio through sounddevice."""
    sd.stop()
    sd.play(audio_data, fs)
    print(f"Playing: {label}")

def apply_and_plot(filepath, status_var):
    status_var.set("Processing audio... please wait.")
    root.update_idletasks()

    global x, y, t
    x, t = load_audio(filepath)
    y = apply_filters(x)

    # --- Show plots in two separate popups ---
    plot_original_signal(x, t)
    plot_filtered_signals(y, t)

    # Update dropdown with new filter list
    output_choices.set("Original Signal")
    combo["values"] = ["Original Signal"] + list(y.keys())

    status_var.set("Filtering complete! You can now play the signals.")

# -----------------------------------------------------------
# Step 5: GUI setup
# -----------------------------------------------------------
root = Tk()
root.title("FIR Filter Design and Audio Playback")
root.geometry("550x420")
root.resizable(False, False)
root.configure(bg="#1e1e1e")

style = ttk.Style()
style.theme_use('clam')
style.configure("TButton", font=("Segoe UI", 11, "bold"), background="#4fa3ff", foreground="black", padding=8)
style.map("TButton", background=[("active", "#5eff74")])
style.configure("TLabel", background="#1e1e1e", foreground="white", font=("Segoe UI", 10))

# Title
ttk.Label(root, text="FIR Audio Filter",
          font=("Segoe UI", 16, "bold"), foreground="#4fa3ff", background="#1e1e1e").pack(pady=(20, 5))

ttk.Label(root, text="Hamming & Blackman Windows | Low, Band, High-pass",
          font=("Segoe UI", 10), foreground="lightgray", background="#1e1e1e").pack()

ttk.Separator(root, orient='horizontal').pack(fill='x', padx=40, pady=15)

status_var = StringVar(value="Ready to process an audio file.")
ttk.Label(root, textvariable=status_var, foreground="lightgreen").pack(pady=5)

def open_file():
    filepath = filedialog.askopenfilename(
        title="Select an audio file",
        filetypes=[("Audio Files", "*.mp3 *.wav *.flac")]
    )
    if filepath:
        apply_and_plot(filepath, status_var)
    else:
        status_var.set("No file selected bro.")

# Audio control section
ttk.Button(root, text="Choose Audio File bro", command=open_file).pack(pady=15)

output_choices = StringVar(value="Original Signal")
combo = ttk.Combobox(root, textvariable=output_choices, state="readonly", font=("Segoe UI", 10))
combo["values"] = ["Original Signal"]
combo.pack(pady=5) 

def play_selected():
    """Play the currently selected signal."""
    if 'x' not in globals():
        status_var.set("Please load an audio file first bro.")
        return

    selected = output_choices.get()
    if selected == "Original Signal":
        play_audio(x, "Original Signal")
    elif selected in y:
        play_audio(y[selected], selected)
    else:
        status_var.set("Invalid selection.")

ttk.Button(root, text="Play Selected Audio", command=play_selected).pack(pady=10)

ttk.Label(root, text="Select & play: Original • Low-pass • Band-pass • High-pass (Hamming/Blackman)",
          foreground="gray", font=("Segoe UI", 9)).pack(pady=10)

# Footer
ttk.Label(root, text="Developed by ROMIM - ATIK - HAFIZ | DSP Project",
          foreground="#888", font=("Segoe UI", 8)).pack(side="bottom", pady=10)

root.mainloop()

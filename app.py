import os
import time
import numpy as np
import scipy.signal
import torch
import matplotlib.pyplot as plt
import streamlit as st

st.set_page_config(page_title="SWIFT-Mamba Denoising", layout="wide")
st.title("🚁 SWIFT-Mamba: Non-contact Wavefront Sensing Denoising")
st.markdown("Welcome to the interactive demonstration for the SWIFT-Mamba algorithm. Please upload your noisy data to see the denoising results.")
st.divider()

from swift_mamba_model import SWIFTMambaNet

device = torch.device("cpu") 
MODEL_PATH = "best_model_weights.pth"

class Config:
    def __init__(self):
        self.fs = 20000
        self.nperseg = 64
        self.nfft = 128
        self.nt = 1024

config = Config()

@st.cache_resource
def load_model():
    model = SWIFTMambaNet(
        ablation_mode="dual_softclip", 
        mask_bound=10.0
    ).to(device)
    
    if os.path.exists(MODEL_PATH):
        try:
            checkpoint = torch.load(MODEL_PATH, map_location=device)
            if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
                state_dict = checkpoint["state_dict"]
            elif isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                state_dict = checkpoint["model_state_dict"]
            else:
                state_dict = checkpoint
            
            cleaned_state_dict = {}
            for key, value in state_dict.items():
                new_key = key.replace("module.", "").replace("mamba1.", "block1.").replace("mamba2.", "block2.")
                cleaned_state_dict[new_key] = value

            model.load_state_dict(cleaned_state_dict, strict=False)
            model.eval()
            return model, "Model weights loaded successfully!"
        except Exception as e:
            return None, f"Failed to load weights: {e}"
    else:
        model.eval()
        return model, f"Warning: {MODEL_PATH} not found. Running with randomly initialized weights for demonstration purposes only."

def process_signal(model, noisy_raw):
    if len(noisy_raw) > config.nt:
        noisy_raw = noisy_raw[:config.nt]
    elif len(noisy_raw) < config.nt:
        noisy_raw = np.pad(noisy_raw, (0, config.nt - len(noisy_raw)))
    
    noisy_raw = noisy_raw - np.mean(noisy_raw)

    _, _, noisy_stft = scipy.signal.stft(
        noisy_raw, fs=config.fs, nperseg=config.nperseg, nfft=config.nfft, boundary="zeros"
    )

    freq_bins = config.nfft // 2
    noisy_stft_trunc = noisy_stft[:freq_bins, :32]
    
    std_val = np.std(noisy_stft_trunc) + 1e-8
    normalized = noisy_stft_trunc / std_val

    x_input = np.stack([np.abs(normalized), normalized.real, normalized.imag], axis=-1)
    x_tensor = torch.tensor(x_input, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(device)

    start_time = time.perf_counter()
    with torch.no_grad():
        pred_mask = model(x_tensor)[0].cpu().numpy()
    latency = (time.perf_counter() - start_time) * 1000

    mask_complex = pred_mask[0] + 1j * pred_mask[1]
    denoised_stft_trunc = noisy_stft_trunc * mask_complex

    denoised_stft = np.pad(denoised_stft_trunc, ((0, 1), (0, 0)), mode="constant")
    _, denoised_wave = scipy.signal.istft(
        denoised_stft,
        fs=config.fs,
        nperseg=config.nperseg,
        nfft=config.nfft,
        boundary=True 
    )
    
    if len(denoised_wave) > config.nt:
        denoised_wave = denoised_wave[:config.nt]
    elif len(denoised_wave) < config.nt:
        denoised_wave = np.pad(denoised_wave, (0, config.nt - len(denoised_wave)))
    
    return noisy_raw, denoised_wave[:len(noisy_raw)], latency

model, model_status = load_model()

if "Warning" in model_status:
    st.warning(model_status)
elif "Failed" in model_status:
    st.error(model_status)
    st.stop()
elif "successfully" in model_status:
    st.success(model_status)

col1, col_gap, col2 = st.columns([1, 0.2, 2.5])

with col1:
    st.header("1. Upload Data")
    uploaded_file = st.file_uploader("Upload Noisy Signal (.npz or .csv)", type=['npz', 'csv'])
    st.info("Instructions: Upload a 1D array of approximately 1024 points.")

with col2:
    st.header("2. Results")
    
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith('.npz'):
                data = np.load(uploaded_file)
                array_key = 'data' if 'data' in data else data.files[0]
                noisy_raw = np.squeeze(data[array_key])
            else:
                noisy_raw = np.loadtxt(uploaded_file, delimiter=',')
                noisy_raw = np.squeeze(noisy_raw)

            if noisy_raw.ndim > 1:
                noisy_raw = noisy_raw.flatten()

            with st.spinner('Processing signal...'):
                noisy_processed, denoised_wave, latency = process_signal(model, noisy_raw)
            
            fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
            time_axis = np.arange(len(noisy_processed)) / config.fs * 1000
            
            axes[0].plot(time_axis, noisy_processed, color='#ef4444', linewidth=1, label='Noisy Input')
            axes[0].set_title('Uploaded Noisy Signal', fontsize=12, fontweight='bold')
            axes[0].legend(loc='upper right')
            axes[0].grid(True, linestyle='--', alpha=0.5)

            axes[1].plot(time_axis, denoised_wave, color='#3b82f6', linewidth=1, label='SWIFT-Mamba Denoised')
            axes[1].set_title('Denoised Output Reconstruction', fontsize=12, fontweight='bold')
            axes[1].set_xlabel('Time (ms)', fontsize=10)
            axes[1].legend(loc='upper right')
            axes[1].grid(True, linestyle='--', alpha=0.5)
            plt.tight_layout()

            st.pyplot(fig)

        except Exception as e:
            st.error(f"An error occurred while processing the file: {e}")
    else:
        st.write("Awaiting file upload...")

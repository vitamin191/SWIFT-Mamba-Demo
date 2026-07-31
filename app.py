import os
import time
import numpy as np
import scipy.signal
import torch
import matplotlib.pyplot as plt
import streamlit as st

# 导入您的模型
from swift_mamba_model import SWIFTMambaNet

# =========================================================================
# 1. 全局配置
# =========================================================================
device = torch.device("cpu") # 强制使用 CPU 进行云端推理
MODEL_PATH = "best_model_weights.pth"

class Config:
    def __init__(self):
        self.fs = 20000
        self.nperseg = 64
        self.nfft = 128
        self.nt = 1024

config = Config()

# =========================================================================
# 2. 缓存模型加载 (避免每次点击都重新加载权重)
# =========================================================================
@st.cache_resource
def load_model():
    model = SWIFTMambaNet(
        ablation_mode="dual_softclip", 
        mask_bound=10.0
    ).to(device)
    
    if os.path.exists(MODEL_PATH):
        try:
            checkpoint = torch.load(MODEL_PATH, map_location=device)
            # 兼容不同的保存格式
            if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
                state_dict = checkpoint["state_dict"]
            elif isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                state_dict = checkpoint["model_state_dict"]
            else:
                state_dict = checkpoint
            
            # 清理权重字典键名 (处理 DDP module. 等前缀)
            cleaned_state_dict = {}
            for key, value in state_dict.items():
                new_key = key.replace("module.", "").replace("mamba1.", "block1.").replace("mamba2.", "block2.")
                cleaned_state_dict[new_key] = value

            # 【关键修改】：将 strict=True 改为 strict=False，允许忽略不匹配的 bias 权重
            model.load_state_dict(cleaned_state_dict, strict=False)
            model.eval()
            return model, "模型权重加载成功！"
        except Exception as e:
            return None, f"权重加载失败: {e}"
    else:
        model.eval()
        return model, f"警告: 未找到 {MODEL_PATH}，当前使用随机初始化权重仅作界面演示。"

# =========================================================================
# 3. 信号处理核心逻辑
# =========================================================================
def process_signal(model, noisy_raw):
    # 截断或填充到 1024
    if len(noisy_raw) > config.nt:
        noisy_raw = noisy_raw[:config.nt]
    elif len(noisy_raw) < config.nt:
        noisy_raw = np.pad(noisy_raw, (0, config.nt - len(noisy_raw)))
    
    # 去均值
    noisy_raw = noisy_raw - np.mean(noisy_raw)

    # STFT 变换
    _, _, noisy_stft = scipy.signal.stft(
        noisy_raw, fs=config.fs, nperseg=config.nperseg, nfft=config.nfft, boundary="zeros"
    )

    freq_bins = config.nfft // 2
    noisy_stft_trunc = noisy_stft[:freq_bins, :32]
    
    # 标准化
    std_val = np.std(noisy_stft_trunc) + 1e-8
    normalized = noisy_stft_trunc / std_val

    # 堆叠输入张量 [1, 3, H, W]
    x_input = np.stack([np.abs(normalized), normalized.real, normalized.imag], axis=-1)
    x_tensor = torch.tensor(x_input, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(device)

    # 模型推理计算掩码
    start_time = time.perf_counter()
    with torch.no_grad():
        pred_mask = model(x_tensor)[0].cpu().numpy()
    latency = (time.perf_counter() - start_time) * 1000

    # 应用掩码与 ISTFT 逆变换
    mask_complex = pred_mask[0] + 1j * pred_mask[1]
    denoised_stft_trunc = noisy_stft_trunc * mask_complex

    denoised_stft = np.pad(denoised_stft_trunc, ((0, 1), (0, 0)), mode="constant")
    _, denoised_wave = scipy.signal.istft(
        denoised_stft,
        fs=config.fs,
        nperseg=config.nperseg,
        nfft=config.nfft,
        boundary=True # 保持与默认行为一致，但关键在下面强制修复长度
    )
    
    # 【关键修改】：强制调整输出波形的长度到 1024，解决绘图尺寸不匹配的问题
    if len(denoised_wave) > config.nt:
        denoised_wave = denoised_wave[:config.nt]
    elif len(denoised_wave) < config.nt:
        denoised_wave = np.pad(denoised_wave, (0, config.nt - len(denoised_wave)))
    
    return noisy_raw, denoised_wave[:len(noisy_raw)], latency

# --- 修复后的逻辑块开始 ---
model, model_status = load_model()

if "警告" in model_status:
    st.warning(model_status)
elif "失败" in model_status:
    st.error(model_status)
    st.stop()
elif "成功" in model_status:
    st.success(model_status)
# --- 修复后的逻辑块结束 ---

col1, col2 = st.columns([1, 2])

with col1:
    st.header("1. Upload Data")
    uploaded_file = st.file_uploader("Upload Noisy Signal (.npz or .csv)", type=['npz', 'csv'])
    st.info("Instructions: Upload a 1D array of approximately 1024 points.")

with col2:
    st.header("2. Results")
    
    if uploaded_file is not None:
        try:
            # 读取数据
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

            st.success(f"Denoising complete! Inference latency: {latency:.2f} ms")

            # 绘制对比图
            fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
            time_axis = np.arange(len(noisy_processed)) / config.fs * 1000 # 转为毫秒
            
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

            # 提供 CSV 下载按钮
            csv_data = "\n".join([str(val) for val in denoised_wave])
            st.download_button(
                label="Download Denoised Data (.csv)",
                data=csv_data,
                file_name="denoised_output.csv",
                mime="text/csv"
            )

        except Exception as e:
            st.error(f"An error occurred while processing the file: {e}")
    else:
        st.write("Awaiting file upload...")

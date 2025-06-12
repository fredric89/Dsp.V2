import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import tempfile
import os
from scipy.signal import butter, lfilter
import soundfile as sf
from scipy.interpolate import interp1d

# Page state control
if "page" not in st.session_state:
    st.session_state.page = "home"

# ---------------- LANDING PAGE ----------------
if st.session_state.page == "home":
    st.set_page_config(page_title="Voice Pitch Detector", layout="centered")
    st.markdown("""
        <style>
        .stApp {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        }
        .nav-container {
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            z-index: 1000;
            background: rgba(255, 255, 255, 0.1);
            backdrop-filter: blur(10px);
            border-bottom: 1px solid rgba(255, 255, 255, 0.2);
            padding: 1rem 2rem;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.5);
        }
        .nav-content { max-width: 1200px; margin: 0 auto; display: flex; justify-content: center; align-items: center; }
        .nav-logo { color: white; font-size: 1.8rem; font-weight: bold; text-shadow: 3px 3px 6px rgba(0,0,0,0.6); }
        .main-content {
            background: rgba(255, 255, 255, 0.15);
            backdrop-filter: blur(10px);
            border-radius: 20px;
            border: 1px solid rgba(255, 255, 255, 0.2);
            padding: 3rem 2rem;
            margin: 8rem auto 2rem;
            max-width: 600px;
            box-shadow: 0 12px 48px rgba(0, 0, 0, 0.4);
            text-align: center;
        }
        .title { color: white; font-size: 3rem; font-weight: bold; margin-bottom: 1rem; text-shadow: 3px 3px 6px rgba(0,0,0,0.6); }
        .subtitle { color: rgba(255, 255, 255, 0.95); font-size: 1.2rem; margin-bottom: 3rem; }
        #MainMenu, footer, header { visibility: hidden; }
        </style>
    """, unsafe_allow_html=True)

    st.markdown("""
        <div class="nav-container"><div class="nav-content"><div class="nav-logo">
        🎵 PitchScope: Your Voice Frequency Visualizer
        </div></div></div>
        <div class="main-content">
            <div class="title">Voice Pitch Detector</div>
            <div class="subtitle">Analyze and visualize your voice pitch with ease.</div>
        </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("Start Analysis", key="start_btn", use_container_width=True):
            st.session_state.page = "app"
            st.rerun()

    st.markdown("""
        <style>
        div.stButton > button:first-child {
            font-size: 1.5rem; padding: 1rem 2rem;
            border-radius: 12px; background-color: #ffffff;
            color: #222222; font-weight: bold; border: none;
        }
        div.stButton > button:hover { background-color: #dddddd; }
        </style>
    """, unsafe_allow_html=True)

    st.stop()

# ---------------- PITCH DETECTION PAGE ----------------
elif st.session_state.page == "app":
    st.set_page_config(page_title="Voice Pitch Detection", layout="wide")

    st.markdown("<h2 style='text-align: center; color: white;'>Upload Audio File</h2>", unsafe_allow_html=True)
    _, col2, _ = st.columns([1, 2, 1])
    with col2:
        audio_file = st.file_uploader("Choose a WAV or MP3 file", type=["wav", "mp3"])

    st.markdown("<h2 style='text-align: center; color: white;'>Bandpass Filter Settings</h2>", unsafe_allow_html=True)
    _, col5, _ = st.columns([1, 2, 1])
    with col5:
        lowcut = st.slider("Lowcut Frequency (Hz)", 20, 500, 50, 10)
        highcut = st.slider("Highcut Frequency (Hz)", 480, 2000, 1000, 10)

    _, col8, _ = st.columns([1, 2, 1])
    with col8:
        if st.button("Back to Home"):
            st.session_state.page = "home"
            st.rerun()

    def butter_bandpass(l, h, fs, order=4):
        nyq = fs / 2
        return butter(order, [l/nyq, h/nyq], btype='band')

    def bandpass_filter(data, l, h, fs, order=4):
        b, a = butter_bandpass(l, h, fs, order)
        return lfilter(b, a, data)

    def autocorrelation_pitch(y, sr, frame_size, hop_size):
        # ... (same as before) ...
        return times, pitches

    if audio_file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as f:
            f.write(audio_file.read())
            tmp = f.name

        try:
            y, sr = librosa.load(tmp, sr=None, mono=True)
            duration = librosa.get_duration(y=y, sr=sr)

            c1, c2 = st.columns(2)
            with c1: st.audio(audio_file)
            with c2:
                st.metric("Duration", f"{duration:.2f} sec")
                st.metric("Sampling Rate", f"{sr} Hz")

            st.subheader("Original Audio Waveform")
            fig_raw, ax_raw = plt.subplots(figsize=(12,3))
            librosa.display.waveshow(y, sr, ax=ax_raw)
            ax_raw.set_title("Original Audio")
            ax_raw.grid(True, alpha=0.3)
            st.pyplot(fig_raw); plt.close(fig_raw)

            yf = bandpass_filter(y, lowcut, highcut, sr)
            yf = np.nan_to_num(yf)
            if np.max(np.abs(yf)) > 1e-5: yf /= np.max(np.abs(yf))
            filtered_path = tmp.replace(".wav", "_f.wav")
            sf.write(filtered_path, yf, sr)

            st.subheader("Filtered Audio")
            st.audio(filtered_path)

            if np.all(np.abs(yf) < 1e-5):
                st.warning("Filtered signal is silent. Try adjusting the filter.")
            else:
                fs = int(sr * 0.03); hop = fs//2
                with st.spinner("Analyzing pitch..."):
                    times, pitches = autocorrelation_pitch(yf, sr, fs, hop)

                st.subheader("Pitch Analysis Results")
                fig, ax = plt.subplots(2, 1, sharex=True, figsize=(12,8))
                for a in ax:
                    a.set_facecolor((0.95, 0.98, 1))  # light, uniform gradient-like color
                    a.grid(color='gray', linestyle='--', alpha=0.3)

                librosa.display.waveshow(yf, sr, ax=ax[0], color='dodgerblue')
                ax[0].set_title("Filtered Audio Waveform")
                ax[1].plot(times, pitches, color='mediumvioletred', linewidth=2, label="Pitch (Hz)")
                ax[1].set_title("Pitch Over Time")
                ax[1].set_xlabel("Time (s)")
                ax[1].set_ylabel("Pitch (Hz)")
                ax[1].legend()
                ax[1].set_ylim(0, max(1000, np.max(pitches)*1.1) if np.any(pitches>0) else 1000)

                fig.patch.set_facecolor((0.95, 0.98, 1))
                st.pyplot(fig); plt.close(fig)

                if np.any(pitches>0):
                    vp = pitches[pitches>0]
                    avg_p = vp.mean(); std_p = vp.std()
                    st.subheader("Pitch Statistics")
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Average Pitch", f"{avg_p:.2f} Hz")
                    c2.metric("Min Pitch", f"{vp.min():.2f} Hz")
                    c3.metric("Max Pitch", f"{vp.max():.2f} Hz")
                    c4.metric("Std Deviation", f"{std_p:.2f} Hz")

                    # Interpretation
                    st.subheader("Pitch Analysis Summary")
                    if avg_p < 150:
                        st.info("Low average pitch — typical of deeper voices.")
                    elif avg_p < 300:
                        st.info("Mid‑range pitch — typical of conversational voices.")
                    else:
                        st.info("High average pitch — common in higher-pitched singing/voices.")
                    if std_p < 20:
                        st.info("Pitch is very stable.")
                    elif std_p < 80:
                        st.info("Moderate pitch variation.")
                    else:
                        st.info("High pitch variability detected.")

                else:
                    st.error("No valid pitch detected. Try clearer audio or filter tweaks.")

            os.unlink(tmp)
            if os.path.exists(filtered_path):
                os.unlink(filtered_path)

        except Exception as e:
            st.error(f"Error processing file: {e}")
            if os.path.exists(tmp):
                os.unlink(tmp)
    else:
        st.info("Upload an audio file to start analysis.")

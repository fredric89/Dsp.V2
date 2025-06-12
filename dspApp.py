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

    # Styling
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
        .nav-content {
            max-width: 1200px;
            margin: 0 auto;
            display: flex;
            justify-content: center;
            align-items: center;
        }
        .nav-logo {
            color: white;
            font-size: 1.8rem;
            font-weight: bold;
            text-shadow: 3px 3px 6px rgba(0,0,0,0.6);
        }
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
        .title {
            color: white;
            font-size: 3rem;
            font-weight: bold;
            margin-bottom: 1rem;
            text-shadow: 3px 3px 6px rgba(0,0,0,0.6);
        }
        .subtitle {
            color: rgba(255, 255, 255, 0.95);
            font-size: 1.2rem;
            margin-bottom: 3rem;
        }
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        </style>
    """, unsafe_allow_html=True)

    # Navbar
    st.markdown("""
        <div class="nav-container">
            <div class="nav-content">
                <div class="nav-logo">🎵 PitchScope: Your Voice Frequency Visualizer</div>
            </div>
        </div>
    """, unsafe_allow_html=True)

    # Main Content
    st.markdown("""
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
            font-size: 1.5rem;
            padding: 1rem 2rem;
            border-radius: 12px;
            background-color: #ffffff;
            color: #222222;
            font-weight: bold;
            border: none;
        }
        div.stButton > button:first-child:hover {
            background-color: #dddddd;
        }
        </style>
    """, unsafe_allow_html=True)

    st.stop()

# ---------------- PITCH DETECTION PAGE ----------------
elif st.session_state.page == "app":
    st.set_page_config(page_title="Voice Pitch Detection", layout="wide")

    st.markdown("<h2 style='text-align: center; color: white;'>Upload Audio File</h2>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        audio_file = st.file_uploader("Choose a WAV or MP3 file", type=["wav", "mp3"])

    st.markdown("<h2 style='text-align: center; color: white;'>Bandpass Filter Settings</h2>", unsafe_allow_html=True)
    col4, col5, col6 = st.columns([1, 2, 1])
    with col5:
        lowcut = st.slider("Lowcut Frequency (Hz)", min_value=20, max_value=500, value=50, step=10)
        highcut = st.slider("Highcut Frequency (Hz)", min_value=480, max_value=2000, value=1000, step=10)

    st.markdown("<br>", unsafe_allow_html=True)
    col7, col8, col9 = st.columns([1, 2, 1])
    with col8:
        if st.button("Back to Home"):
            st.session_state.page = "home"
            st.rerun()

    def butter_bandpass(lowcut, highcut, fs, order=4):
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(order, [low, high], btype='band')
        return b, a

    def bandpass_filter(data, lowcut, highcut, fs, order=4):
        b, a = butter_bandpass(lowcut, highcut, fs, order=order)
        return lfilter(b, a, data)

    def autocorrelation_pitch(y, sr, frame_size, hop_size):
        num_frames = 1 + int((len(y) - frame_size) / hop_size)
        pitches = np.zeros(num_frames)
        times = np.zeros(num_frames)
        for i in range(num_frames):
            start = i * hop_size
            frame = y[start:start+frame_size]
            if np.all(frame == 0):
                pitches[i] = 0
                times[i] = start / sr
                continue
            frame -= np.mean(frame)
            autocorr = np.correlate(frame, frame, mode='full')[frame_size:]
            d = np.diff(autocorr)
            start_peak_candidates = np.where(d > 0)[0]
            if start_peak_candidates.size == 0:
                pitches[i] = 0
                times[i] = start / sr
                continue
            start_peak = start_peak_candidates[0]
            peaks = [j for j in range(start_peak, len(autocorr) - 1)
                     if autocorr[j] > autocorr[j - 1] and autocorr[j] > autocorr[j + 1]]
            if peaks and autocorr[peaks[0]] > 0:
                pitch = sr / peaks[0]
                pitches[i] = pitch if 50 < pitch < 1000 else 0
            else:
                pitches[i] = 0
            times[i] = start / sr
        if np.any(pitches > 0):
            valid = pitches > 0
            if np.sum(valid) > 1:
                interp = interp1d(times[valid], pitches[valid], kind='linear', fill_value='extrapolate')
                pitches = interp(times)
        return times, pitches

    if audio_file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
            tmp_file.write(audio_file.read())
            tmp_path = tmp_file.name

        try:
            y, sr = librosa.load(tmp_path, sr=None, mono=True)
            duration = librosa.get_duration(y=y, sr=sr)

            col1, col2 = st.columns(2)
            with col1:
                st.audio(audio_file, format='audio/wav')
            with col2:
                st.metric("Duration", f"{duration:.2f} seconds")
                st.metric("Sampling Rate", f"{sr} Hz")

            st.subheader("Original Audio Waveform")
            fig_raw, ax_raw = plt.subplots(figsize=(12, 3))
            librosa.display.waveshow(y, sr=sr, ax=ax_raw)
            ax_raw.set_title('Original Audio (Before Filtering)')
            ax_raw.grid(True, alpha=0.3)
            st.pyplot(fig_raw)
            plt.close(fig_raw)

            y_filtered = bandpass_filter(y, lowcut, highcut, sr)
            y_filtered = np.nan_to_num(y_filtered)
            if np.max(np.abs(y_filtered)) > 1e-5:
                y_filtered /= np.max(np.abs(y_filtered))

            filtered_path = tmp_path.replace(".wav", "_filtered.wav")
            sf.write(filtered_path, y_filtered, sr)

            st.subheader("Filtered Audio")
            st.audio(filtered_path, format='audio/wav')

            if np.all(np.abs(y_filtered) < 1e-5):
                st.warning("Filtered signal is too quiet or empty. Try adjusting the bandpass filter range.")
            else:
                frame_size = int(sr * 0.03)
                hop_size = frame_size // 2
                with st.spinner("Analyzing pitch..."):
                    times, pitches = autocorrelation_pitch(y_filtered, sr, frame_size, hop_size)

                st.subheader("Pitch Analysis Results")
                fig, ax = plt.subplots(nrows=2, sharex=True, figsize=(12, 8))
                librosa.display.waveshow(y_filtered, sr=sr, ax=ax[0])
                ax[0].set_title('Filtered Audio Waveform')
                ax[0].grid(True, alpha=0.3)
                ax[1].plot(times, pitches, label='Estimated Pitch (Hz)', color='red', linewidth=2)
                ax[1].set_title('Pitch Over Time')
                ax[1].set_xlabel('Time (s)')
                ax[1].set_ylabel('Pitch (Hz)')
                ax[1].legend()
                ax[1].grid(True, alpha=0.3)
                ax[1].set_ylim(0, max(1000, np.max(pitches) * 1.1) if np.any(pitches > 0) else 1000)
                st.pyplot(fig)
                plt.close(fig)

                if np.any(pitches > 0):
                    valid_pitches = pitches[pitches > 0]
                    st.subheader("Pitch Statistics")
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Average Pitch", f"{np.mean(valid_pitches):.2f} Hz")
                    with col2:
                        st.metric("Min Pitch", f"{np.min(valid_pitches):.2f} Hz")
                    with col3:
                        st.metric("Max Pitch", f"{np.max(valid_pitches):.2f} Hz")
                    with col4:
                        st.metric("Std Deviation", f"{np.std(valid_pitches):.2f} Hz")

                    # Pitch Analysis Summary
                    st.subheader("Pitch Interpretation")
                    analysis = ""
                    avg_pitch = np.mean(valid_pitches)
                    std_pitch = np.std(valid_pitches)

                    if avg_pitch < 160:
                        analysis += "🔹 Your average pitch is relatively **low**, which is typical for male voices or deeper vocal tones.\n\n"
                    elif 160 <= avg_pitch <= 250:
                        analysis += "🔹 Your average pitch falls in the **mid-range**, which is typical for many adult voices (especially female or higher-pitched male voices).\n\n"
                    else:
                        analysis += "🔹 Your average pitch is relatively **high**, which may indicate a higher-pitched voice, such as those in children or soprano-range voices.\n\n"

                    if std_pitch < 20:
                        analysis += "🔸 Your pitch is **very stable**, showing consistent vocal tone.\n\n"
                    elif 20 <= std_pitch < 50:
                        analysis += "🔸 Your pitch shows **moderate variation**, which is common in natural speech and expressive talking.\n\n"
                    else:
                        analysis += "🔸 Your pitch is **highly variable**, which might suggest emotional expression, emphasis, or even background noise affecting detection.\n\n"

                    st.markdown(analysis)

                else:
                    st.error("No valid pitch detected. Try uploading a clearer audio sample or adjusting the filter settings.")

            os.unlink(tmp_path)
            if os.path.exists(filtered_path):
                os.unlink(filtered_path)

        except Exception as e:
            st.error(f"Error processing audio file: {str(e)}")
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
    else:
        st.info("Please upload an audio file to start pitch detection.")

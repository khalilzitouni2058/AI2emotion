"""Streamlit web application for emotion analysis."""

import json
from dataclasses import asdict
from pathlib import Path

import librosa
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from app.services.emotion_service import EmotionService


# ==================== PAGE CONFIG ====================
st.set_page_config(
    page_title="Emotion Analysis",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
        .stTabs [data-baseweb="tab-list"] button {
            font-size: 16px;
            padding: 10px;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# ==================== HELPERS ====================
def result_to_dict(result):
    """Convert dataclass result to a plain dictionary."""
    return asdict(result)


def load_audio_preview(audio_path: str):
    """Load audio for waveform preview."""
    audio_array, sampling_rate = librosa.load(audio_path, sr=None)
    duration = len(audio_array) / sampling_rate
    return audio_array, sampling_rate, duration


def build_segments_table(result_dict: dict):
    """Prepare segment rows for display."""
    rows = []
    for index, segment in enumerate(result_dict.get("emotion_segments", []), 1):
        rows.append(
            {
                "Segment": index,
                "Start (s)": f"{segment['start_time']:.2f}",
                "End (s)": f"{segment['end_time']:.2f}",
                "Duration (s)": f"{segment['duration']:.2f}",
                "Emotion": segment["emotion"],
                "Confidence": f"{segment.get('confidence', 0):.1%}",
            }
        )
    return rows


def build_emotion_duration_distribution(result_dict: dict):
    """Aggregate emotion distribution by segment duration."""
    distribution = {}
    for segment in result_dict.get("emotion_segments", []):
        emotion = segment["emotion"]
        duration = segment["end_time"] - segment["start_time"]
        distribution[emotion] = distribution.get(emotion, 0) + duration
    return distribution


# ==================== SESSION STATE ====================
if "current_tab" not in st.session_state:
    st.session_state.current_tab = 0

if "analysis_result" not in st.session_state:
    st.session_state.analysis_result = None

if "audio_duration" not in st.session_state:
    st.session_state.audio_duration = None


# ==================== SIDEBAR ====================
st.sidebar.title("⚙️ System Info")
st.sidebar.markdown("### Architecture")
st.sidebar.info(
    """
**Analysis Engine:** EmotionService  
**UI Layer:** Streamlit  
**Inference:** Hugging Face audio classification model  
**Processing:** Centralized in service layer
"""
)

st.sidebar.divider()

st.sidebar.markdown("### Notes")
st.sidebar.info(
    """
- Upload WAV files only
- The UI does not perform inference logic directly
- All processing goes through the service layer
"""
)


# ==================== TITLE ====================
st.title("🎵 Emotion Analysis System")
st.markdown("Upload audio files and analyze emotional content.")


# ==================== NAVIGATION ====================
col1, col2, col3 = st.columns(3)

with col1:
    if st.button("📥 Upload & Analyze", use_container_width=True):
        st.session_state.current_tab = 0
        st.rerun()

with col2:
    if st.button("📊 Results", use_container_width=True):
        st.session_state.current_tab = 1
        st.rerun()

with col3:
    if st.button("📚 Documentation", use_container_width=True):
        st.session_state.current_tab = 2
        st.rerun()

st.divider()


# ==================== TAB 1: UPLOAD ====================
if st.session_state.current_tab == 0:
    st.header("Upload & Analyze Audio")

    uploaded_file = st.file_uploader(
        "Upload a WAV audio file",
        type=["wav"],
        help="Only WAV files are supported.",
    )

    if uploaded_file is not None:
        temp_dir = Path("temp_uploads")
        temp_dir.mkdir(exist_ok=True)

        temp_file_path = temp_dir / uploaded_file.name

        with open(temp_file_path, "wb") as file:
            file.write(uploaded_file.getbuffer())

        st.success(f"✅ File uploaded: {uploaded_file.name}")

        try:
            audio_array, sampling_rate, duration = load_audio_preview(str(temp_file_path))

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Duration", f"{duration:.2f}s")
            with col2:
                st.metric("Sample Rate", f"{sampling_rate} Hz")
            with col3:
                st.metric("File Size", f"{uploaded_file.size / 1024:.1f} KB")

            st.subheader("🎵 Waveform Preview")

            fig, ax = plt.subplots(figsize=(12, 4))
            time_axis = np.linspace(0, duration, len(audio_array))
            ax.plot(time_axis, audio_array, linewidth=0.5)
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Amplitude")
            ax.set_title("Audio Waveform")
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)

            if st.button("🚀 Analyze Emotion", type="primary", use_container_width=True):
                try:
                    service = EmotionService()

                    with st.spinner("⏳ Running emotion analysis..."):
                        result = service.analyze(str(temp_file_path))

                    result_dict = result_to_dict(result)

                    result_dict["metadata"]["audio_file"] = uploaded_file.name

                    st.session_state.analysis_result = result_dict
                    st.session_state.audio_duration = duration

                    st.success("✅ Analysis complete!")

                    col_left, col_center, col_right = st.columns([1, 2, 1])
                    with col_center:
                        if st.button("📊 View Results", use_container_width=True):
                            st.session_state.current_tab = 1
                            st.rerun()

                except Exception as exc:
                    st.error(f"❌ Error during analysis: {exc}")

        except Exception as exc:
            st.error(f"❌ Error loading audio file: {exc}")


# ==================== TAB 2: RESULTS ====================
elif st.session_state.current_tab == 1:
    st.header("Analysis Results")

    if st.session_state.analysis_result is None:
        st.info("Upload and analyze an audio file first.")
    else:
        result = st.session_state.analysis_result
        audio_duration = st.session_state.audio_duration or 0.0
        segments = result.get("emotion_segments", [])
        transitions = result.get("emotion_transitions", [])
        summary = result.get("summary", {})

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Total Segments", len(segments))

        with col2:
            if segments:
                dominant_emotion = max(segments, key=lambda item: item["confidence"])["emotion"]
                st.metric("Dominant Emotion", dominant_emotion)
            else:
                st.metric("Dominant Emotion", "N/A")

        with col3:
            if segments:
                avg_confidence = np.mean([segment["confidence"] for segment in segments])
                st.metric("Avg Confidence", f"{avg_confidence:.1%}")
            else:
                st.metric("Avg Confidence", "0.0%")

        with col4:
            st.metric("Emotion Changes", len(transitions))

        if segments:
            st.subheader("📊 Emotion Distribution")

            distribution = build_emotion_duration_distribution(result)

            fig, ax = plt.subplots(figsize=(8, 6))
            ax.pie(
                distribution.values(),
                labels=distribution.keys(),
                autopct="%1.1f%%",
                startangle=90,
            )
            ax.set_title("Emotion Distribution by Duration")
            st.pyplot(fig)

            st.subheader("📈 Emotion Timeline")

            fig, ax = plt.subplots(figsize=(12, 4))

            emotion_colors = {
                "angry": "#FF6B6B",
                "disgust": "#FFD93D",
                "fear": "#6BCB77",
                "happy": "#4D96FF",
                "neutral": "#A0A0A0",
                "sad": "#9B59B6",
                "surprise": "#FF9F43",
            }

            for segment in segments:
                emotion = segment["emotion"]
                start = segment["start_time"]
                end = segment["end_time"]
                confidence = segment.get("confidence", 0.0)

                color = emotion_colors.get(emotion, "#808080")

                ax.barh(
                    0,
                    end - start,
                    left=start,
                    height=0.5,
                    color=color,
                    alpha=0.7,
                    edgecolor="black",
                    linewidth=0.5,
                )

                mid_point = (start + end) / 2
                ax.text(
                    mid_point,
                    0,
                    f"{confidence:.0%}",
                    ha="center",
                    va="center",
                    fontsize=8,
                    fontweight="bold",
                )

            ax.set_xlim(0, audio_duration)
            ax.set_ylim(-0.5, 0.5)
            ax.set_xlabel("Time (s)")
            ax.set_title("Emotion Timeline with Confidence Scores")
            ax.set_yticks([])
            ax.grid(True, alpha=0.3, axis="x")
            st.pyplot(fig)

            st.subheader("📋 Detailed Segments")
            st.dataframe(build_segments_table(result), use_container_width=True, height=400)

            st.subheader("⬇️ Download Results")
            json_str = json.dumps(result, indent=2)
            st.download_button(
                label="📥 Download JSON Results",
                data=json_str,
                file_name="emotion_analysis_results.json",
                mime="application/json",
                use_container_width=True,
            )

        else:
            st.warning("No segments were generated for this analysis.")


# ==================== TAB 3: DOCUMENTATION ====================
elif st.session_state.current_tab == 2:
    st.header("📚 Documentation")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("System Overview")
        st.write(
            """
This system:
- analyzes WAV audio files for emotion recognition
- uses a centralized service layer for processing
- tracks emotion changes over time
- supports smoothing and temporal analysis
"""
        )

        st.subheader("Supported Emotions")
        emotions_info = {
            "😠 Angry": "High-energy negative emotion",
            "🤢 Disgust": "Reaction of offense or revulsion",
            "😨 Fear": "Anxiety and apprehension",
            "😊 Happy": "Joy and positive mood",
            "😐 Neutral": "No strong emotion",
            "😢 Sad": "Low-energy negative emotion",
            "😮 Surprise": "Sudden unexpected emotion",
        }

        for emotion, description in emotions_info.items():
            st.write(f"**{emotion}**: {description}")

    with col2:
        st.subheader("Usage Notes")
        st.write(
            """
- Use clear WAV audio
- Better results come from cleaner recordings
- Long audio is automatically analyzed over time
- JSON output can be reused later in APIs or dashboards
"""
        )

        st.subheader("Architecture Benefit")
        st.write(
            """
The UI is now separated from the analysis logic.
That means:
- easier maintenance
- easier API creation
- easier Unreal integration later
"""
        )

# ==================== FOOTER ====================
st.divider()
st.markdown(
    """
    <div style='text-align: center; color: gray; font-size: 12px;'>
        Emotion Analysis System | Built with Streamlit and a clean service architecture
    </div>
    """,
    unsafe_allow_html=True,
)
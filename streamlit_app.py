"""Streamlit web application for emotion analysis."""

import json
from dataclasses import asdict
from pathlib import Path

import librosa
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from app.services.emotion_service import EmotionService
from dotenv import load_dotenv

load_dotenv()


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


def build_emotion_probability_distribution(result_dict: dict):
    """Return probability distribution for text/single-result mode."""
    segments = result_dict.get("emotion_segments", [])
    if not segments:
        return {}
    return segments[0].get("probabilities", {})


def get_top_segment(result_dict: dict):
    """Return the first or best segment."""
    segments = result_dict.get("emotion_segments", [])
    if not segments:
        return None
    return max(segments, key=lambda item: item.get("confidence", 0.0))


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
**Supported Inputs:** Audio + Text  
**UI Layer:** Streamlit  
**Processing:** Centralized in service layer
"""
)

st.sidebar.divider()

st.sidebar.markdown("### Notes")
st.sidebar.info(
    """
- Audio input supports WAV files
- Text input supports multilingual text
- UI does not perform inference directly
- All processing goes through the backend services
"""
)


# ==================== TITLE ====================
st.title("🎵 Emotion Analysis System")
st.markdown("Analyze emotion from either **audio** or **text**.")


# ==================== NAVIGATION ====================
col1, col2, col3 = st.columns(3)

with col1:
    if st.button("📥 Analyze Input", use_container_width=True):
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


# ==================== TAB 1: INPUT ====================
if st.session_state.current_tab == 0:
    st.header("Analyze Emotion")

    input_mode = st.radio(
        "Choose input type",
        ["Audio", "Text"],
        horizontal=True,
    )

    service = EmotionService()

    if input_mode == "Audio":
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

                if st.button("🚀 Analyze Audio", type="primary", use_container_width=True):
                    try:
                        with st.spinner("⏳ Running audio emotion analysis..."):
                            result = service.analyze(audio_path=str(temp_file_path))

                        result_dict = result_to_dict(result)
                        result_dict["metadata"]["source_name"] = uploaded_file.name

                        st.session_state.analysis_result = result_dict
                        st.session_state.audio_duration = duration

                        st.success("✅ Audio analysis complete!")

                        col_left, col_center, col_right = st.columns([1, 2, 1])
                        with col_center:
                            if st.button("📊 View Results", key="view_audio_results", use_container_width=True):
                                st.session_state.current_tab = 1
                                st.rerun()

                    except Exception as exc:
                        st.error(f"❌ Error during analysis: {exc}")

            except Exception as exc:
                st.error(f"❌ Error loading audio file: {exc}")

    elif input_mode == "Text":
        text_input = st.text_area(
            "Enter text to analyze emotion",
            height=180,
            placeholder="Type text here... مثال: أشعر بالسعادة اليوم / I feel very happy today.",
        )

        if st.button("🚀 Analyze Text", type="primary", use_container_width=True):
            try:
                with st.spinner("⏳ Running text emotion analysis..."):
                    result = service.analyze(text=text_input)

                result_dict = result_to_dict(result)
                result_dict["metadata"]["source_name"] = "Typed text"

                st.session_state.analysis_result = result_dict
                st.session_state.audio_duration = None

                st.success("✅ Text analysis complete!")

                col_left, col_center, col_right = st.columns([1, 2, 1])
                with col_center:
                    if st.button("📊 View Results", key="view_text_results", use_container_width=True):
                        st.session_state.current_tab = 1
                        st.rerun()

            except Exception as exc:
                st.error(f"❌ Error during analysis: {exc}")


# ==================== TAB 2: RESULTS ====================
elif st.session_state.current_tab == 1:
    st.header("Analysis Results")

    if st.session_state.analysis_result is None:
        st.info("Analyze an audio file or text first.")
    else:
        result = st.session_state.analysis_result
        metadata = result.get("metadata", {})
        input_type = metadata.get("input_type", "audio")
        segments = result.get("emotion_segments", [])
        transitions = result.get("emotion_transitions", [])
        source_name = metadata.get("source_name", "N/A")

        st.caption(f"Input type: **{input_type}** | Source: **{source_name}**")

        col1, col2, col3, col4 = st.columns(4)

        top_segment = get_top_segment(result)

        with col1:
            st.metric("Total Segments", len(segments))

        with col2:
            if top_segment:
                st.metric("Dominant Emotion", top_segment["emotion"])
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

        # ==================== AUDIO RESULTS ====================
        if input_type == "audio":
            audio_duration = st.session_state.audio_duration or metadata.get("total_duration", 0.0)

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
                    "Angry": "#FF6B6B",
                    "Fear": "#6BCB77",
                    "Happy": "#4D96FF",
                    "Neutral": "#A0A0A0",
                    "Sad": "#9B59B6",
                    "Surprise": "#FF9F43",
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

            else:
                st.warning("No segments were generated for this audio analysis.")

        # ==================== TEXT RESULTS ====================
        elif input_type == "text":
            if top_segment:
                st.subheader("🧠 Text Emotion Result")

                col_a, col_b = st.columns(2)
                with col_a:
                    st.metric("Predicted Emotion", top_segment["emotion"])
                with col_b:
                    st.metric("Confidence", f"{top_segment['confidence']:.1%}")

                st.subheader("📊 Emotion Probability Distribution")
                probability_distribution = build_emotion_probability_distribution(result)

                fig, ax = plt.subplots(figsize=(10, 5))
                labels = list(probability_distribution.keys())
                values = list(probability_distribution.values())

                ax.bar(labels, values)
                ax.set_ylabel("Probability")
                ax.set_xlabel("Emotion")
                ax.set_title("Mapped Emotion Probabilities")
                ax.set_ylim(0, max(values) * 1.15 if values else 1)
                ax.grid(True, axis="y", alpha=0.3)
                plt.xticks(rotation=20)
                st.pyplot(fig)

                st.subheader("📋 Text Analysis Details")
                text_length = metadata.get("text_length", 0)
                st.write(f"**Text length:** {text_length} characters")
                st.write(f"**Processing mode:** {metadata.get('processing_mode', 'text')}")

            else:
                st.warning("No text prediction was generated.")

        # ==================== DOWNLOAD ====================
        st.subheader("⬇️ Download Results")
        json_str = json.dumps(result, indent=2, ensure_ascii=False)
        st.download_button(
            label="📥 Download JSON Results",
            data=json_str,
            file_name="emotion_analysis_results.json",
            mime="application/json",
            use_container_width=True,
        )


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
- analyzes multilingual text input for emotion recognition
- uses a centralized service layer for processing
- supports consistent emotion categories across models
"""
        )

        st.subheader("Supported Inputs")
        st.write(
            """
- **Audio:** WAV files
- **Text:** multilingual typed text, including Arabic and English
"""
        )

    with col2:
        st.subheader("Architecture Benefit")
        st.write(
            """
The UI is separated from the analysis logic.
That means:
- easier maintenance
- easier API creation
- easier Unreal integration
- consistent outputs across audio and text
"""
        )

        st.subheader("Output Format")
        st.write(
            """
The result structure includes:
- metadata
- emotion segments
- transitions
- summary

Both audio and text use the same top-level contract.
"""
        )

# ==================== FOOTER ====================
st.divider()
st.markdown(
    """
    <div style='text-align: center; color: gray; font-size: 12px;'>
        Emotion Analysis System | Audio + Text | Clean service architecture
    </div>
    """,
    unsafe_allow_html=True,
)
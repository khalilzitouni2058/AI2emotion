"""
Streamlit web application for emotion analysis.
"""

import streamlit as st
import librosa
import numpy as np
import matplotlib.pyplot as plt
import json
import io
from pathlib import Path
from processor import run_analysis
from audio_utils import get_audio_duration, group_emotion_ranges, sliding_window_segmentation
from emotion_model import analyze_emotion_over_time, predict_emotion_full, _average_subwindow_probabilities, id2label, _probs_to_dict
from config import CHUNK_WINDOW_SIZE, CHUNK_HOP_SIZE, AUDIO_DURATION_THRESHOLD

# Configure page
st.set_page_config(
    page_title="Emotion Analysis",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .stTabs [data-baseweb="tab-list"] button {
        font-size: 16px;
        padding: 10px;
    }
    .emotion-badge {
        display: inline-block;
        padding: 5px 10px;
        border-radius: 5px;
        margin: 5px;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# ==================== SIDEBAR: Configuration ====================
st.sidebar.title("⚙️ Configuration")
st.sidebar.markdown("### System Settings")

# Initialize session state for configuration (using defaults from config)
if 'config_window_size' not in st.session_state:
    st.session_state.config_window_size = CHUNK_WINDOW_SIZE
if 'config_hop_size' not in st.session_state:
    st.session_state.config_hop_size = CHUNK_HOP_SIZE
if 'config_duration_threshold' not in st.session_state:
    st.session_state.config_duration_threshold = AUDIO_DURATION_THRESHOLD
if 'config_smoothing_mode' not in st.session_state:
    st.session_state.config_smoothing_mode = "offline"
if 'config_smoothing_method' not in st.session_state:
    st.session_state.config_smoothing_method = "interpolate"

# Display current configuration (read-only)
st.sidebar.markdown("**Analysis Mode**")
st.sidebar.info(f"""
**Duration Threshold:** {st.session_state.config_duration_threshold:.1f}s  
Audio under this uses full prediction, over uses chunking
""")

st.sidebar.divider()

st.sidebar.markdown("**Chunking Parameters**")
st.sidebar.info(f"""
**Window Size:** {st.session_state.config_window_size:.1f}s  
**Hop Size:** {st.session_state.config_hop_size:.1f}s  
Used for audio over the threshold
""")

st.sidebar.divider()

st.sidebar.markdown("**Model Information**")
st.sidebar.info("""
**Model:** Whisper Large v3  
**Sample Rate:** 16 kHz  
**Processing:** GPU/CPU Auto-detect
""")

# App title
st.title("🎵 Emotion Analysis System")
st.markdown("Upload audio files and analyze emotional content in real-time")

# Initialize session state for tab control
if 'current_tab' not in st.session_state:
    st.session_state.current_tab = 0  # 0 = Upload, 1 = Results, 2 = Documentation

# Tab selection buttons
col1, col2, col3 = st.columns(3)
with col1:
    if st.button("📥 Upload & Analyze", width='stretch', key="tab_upload"):
        st.session_state.current_tab = 0
        st.rerun()
with col2:
    if st.button("📊 Results", width='stretch', key="tab_results"):
        st.session_state.current_tab = 1
        st.rerun()
with col3:
    if st.button("📚 Documentation", width='stretch', key="tab_docs"):
        st.session_state.current_tab = 2
        st.rerun()

st.divider()

# ==================== TAB 0: Upload & Analyze ====================
if st.session_state.current_tab == 0:
    st.header("Upload & Analyze Audio")
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Upload a WAV audio file",
        type=["wav"],
        help="Select a WAV format audio file (MP3 not supported)"
    )
    
    if uploaded_file is not None:
        # Save uploaded file temporarily
        temp_dir = Path("temp_uploads")
        temp_dir.mkdir(exist_ok=True)
        temp_file_path = temp_dir / uploaded_file.name
        
        with open(temp_file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        st.success(f"✅ File uploaded: {uploaded_file.name}")
        
        # Display file info
        col1, col2, col3 = st.columns(3)
        
        try:
            audio_duration, audio_array, sampling_rate = get_audio_duration(str(temp_file_path))
            
            with col1:
                st.metric("Duration", f"{audio_duration:.2f}s")
            with col2:
                st.metric("Sample Rate", f"{sampling_rate} Hz")
            with col3:
                st.metric("File Size", f"{uploaded_file.size / 1024:.1f} KB")
            
            # Waveform preview
            st.subheader("🎵 Waveform Preview")
            
            fig, ax = plt.subplots(figsize=(12, 4))
            time_axis = np.linspace(0, audio_duration, len(audio_array))
            ax.plot(time_axis, audio_array, linewidth=0.5, color='#1f77b4')
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Amplitude")
            ax.set_title("Audio Waveform")
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
            
            # Analyze button
            if st.button("🚀 Analyze Emotion", type="primary", width='stretch'):
                import time
                import torch
                from datetime import datetime
                from smoothing import apply_smoothing, hysteresis_filter, interpolate_probabilities
                from config import (
                    SMOOTHING_MODE, OFFLINE_SMOOTHING_METHOD, STREAMING_SMOOTHING_METHOD,
                    STREAMING_USE_HYSTERESIS, SMOOTHING_METHOD, SMOOTHING_WINDOW_SIZE,
                    SMOOTHING_EMA_ALPHA, SMOOTHING_CONFIDENCE_THRESHOLD, SMOOTHING_MIN_FRAMES,
                    SUB_WINDOW_SIZE, SUB_HOP_SIZE
                )
                
                try:
                    # Create containers for progress display
                    progress_container = st.container()
                    status_container = st.container()
                    
                    # Check audio duration to determine loading strategy
                    # Use configured threshold from sidebar
                    duration_threshold = st.session_state.config_duration_threshold
                    if audio_duration < duration_threshold:
                        # Short audio: simple loading spinner
                        with status_container:
                            st.info(f"🎵 Audio is {audio_duration:.2f}s (under {duration_threshold}s). Processing full audio prediction...")
                        
                        with st.spinner("⏳ Analyzing..."):
                            analysis_result = predict_emotion_full(
                                audio_path=str(temp_file_path),
                                audio_duration=audio_duration
                            )
                    else:
                        # Long audio: show real-time chunk processing
                        # Use configured parameters from sidebar
                        window_size = st.session_state.config_window_size
                        hop_size = st.session_state.config_hop_size
                        
                        with status_container:
                            st.info(f"🎵 Audio is {audio_duration:.2f}s (over {duration_threshold}s). Processing with chunked analysis...")
                        
                        with progress_container:
                            # Generate chunks
                            chunks = sliding_window_segmentation(audio_duration, window_size, hop_size)
                            total_chunks = len(chunks)
                            
                            # Progress bar
                            progress_bar = st.progress(0)
                            status_text = st.empty()
                            
                            # Predict emotion for each chunk
                            predictions = []
                            raw_emotions = []
                            confidences = []
                            avg_probs_list = []
                            
                            for i, (start, end) in enumerate(chunks):
                                start_sample = int(start * sampling_rate)
                                end_sample = int(end * sampling_rate)
                                segment_audio = audio_array[start_sample:end_sample]
                                center_time = start + ((end - start) / 2.0)
                                
                                # Update progress
                                progress = (i + 1) / total_chunks
                                progress_bar.progress(progress)
                                status_text.info(f"🔄 Processing chunk {i+1}/{total_chunks} | Time: {start:.2f}s - {end:.2f}s")
                                
                                if len(segment_audio) > 0:
                                    avg_probs = _average_subwindow_probabilities(segment_audio, sampling_rate)
                                    if avg_probs is None:
                                        continue

                                    avg_probs_list.append(avg_probs)
                                    predicted_id = int(torch.argmax(avg_probs).item())
                                    emotion = id2label[predicted_id]
                                    confidence = float(avg_probs[predicted_id].item())

                                    raw_emotions.append(emotion)
                                    confidences.append(confidence)
                                    predictions.append({
                                        "chunk_id": i + 1,
                                        "start_time": round(start, 3),
                                        "end_time": round(end, 3),
                                        "timestamp": round(center_time, 3),
                                        "duration": round(end - start, 3),
                                        "emotion": emotion,
                                        "confidence": round(confidence, 3),
                                        "probabilities": _probs_to_dict(avg_probs),
                                    })
                            
                            # Apply smoothing
                            status_text.info("🔄 Applying smoothing...")
                            smoothing_method_used = None
                            
                            # Use configured smoothing settings from sidebar
                            smoothing_mode = st.session_state.config_smoothing_mode
                            smoothing_method = st.session_state.config_smoothing_method
                            
                            if len(raw_emotions) > 1 and smoothing_mode != 'none':
                                if smoothing_mode == 'offline':
                                    if smoothing_method == 'interpolate':
                                        smoothed_probs = interpolate_probabilities(avg_probs_list)
                                        for i, pred in enumerate(predictions):
                                            pred["emotion_raw"] = pred["emotion"]
                                            pred["confidence_raw"] = pred["confidence"]
                                            pred["probabilities_raw"] = pred["probabilities"]
                                            predicted_id = int(torch.argmax(smoothed_probs[i]).item())
                                            pred["emotion"] = id2label[predicted_id]
                                            pred["confidence"] = round(float(smoothed_probs[i][predicted_id].item()), 3)
                                            pred["probabilities"] = _probs_to_dict(smoothed_probs[i])
                                        smoothing_method_used = 'interpolate'
                                    elif smoothing_method == 'median':
                                        smoothed_emotions = apply_smoothing(
                                            raw_emotions,
                                            method='median',
                                            window_size=SMOOTHING_WINDOW_SIZE,
                                        )
                                        for i, pred in enumerate(predictions):
                                            pred["emotion_raw"] = pred["emotion"]
                                            pred["emotion"] = smoothed_emotions[i]
                                        smoothing_method_used = 'median'
                                elif smoothing_mode == 'streaming':
                                    if smoothing_method == 'ema':
                                        smoothed_emotions = apply_smoothing(
                                            raw_emotions,
                                            method='ema',
                                            alpha=SMOOTHING_EMA_ALPHA,
                                        )
                                        if STREAMING_USE_HYSTERESIS:
                                            smoothed_emotions = hysteresis_filter(
                                                smoothed_emotions,
                                                confidences,
                                                confidence_threshold=SMOOTHING_CONFIDENCE_THRESHOLD,
                                                min_consecutive_frames=SMOOTHING_MIN_FRAMES,
                                            )
                                            smoothing_method_used = 'ema+hysteresis'
                                        else:
                                            smoothing_method_used = 'ema'

                                        for i, pred in enumerate(predictions):
                                            pred["emotion_raw"] = pred["emotion"]
                                            pred["emotion"] = smoothed_emotions[i]
                            
                            # Track emotion transitions
                            transitions = []
                            for i in range(1, len(predictions)):
                                if predictions[i]["emotion"] != predictions[i-1]["emotion"]:
                                    transitions.append({
                                        "transition_id": len(transitions) + 1,
                                        "from_emotion": predictions[i-1]["emotion"],
                                        "to_emotion": predictions[i]["emotion"],
                                        "transition_time": predictions[i].get("timestamp", predictions[i]["start_time"]),
                                        "previous_segment": predictions[i-1]["chunk_id"],
                                        "current_segment": predictions[i]["chunk_id"]
                                    })
                            
                            # Calculate emotion distribution
                            emotion_distribution = {}
                            for pred in predictions:
                                emotion = pred["emotion"]
                                if emotion not in emotion_distribution:
                                    emotion_distribution[emotion] = 0
                                emotion_distribution[emotion] += 1
                            
                            # Create result
                            analysis_result = {
                                "metadata": {
                                    "total_duration": round(audio_duration, 3),
                                    "total_chunks": len(predictions),
                                    "window_size": window_size,
                                    "hop_size": hop_size,
                                    "sampling_rate": sampling_rate,
                                    "smoothing_method": smoothing_method_used or smoothing_method,
                                    "analysis_timestamp": datetime.now().isoformat(),
                                    "sub_window_size": SUB_WINDOW_SIZE,
                                    "sub_hop_size": SUB_HOP_SIZE,
                                },
                                "emotion_segments": predictions,
                                "emotion_transitions": transitions,
                                "summary": {
                                    "total_transitions": len(transitions),
                                    "emotion_distribution": emotion_distribution
                                }
                            }
                    
                    # Add metadata
                    analysis_result['metadata']['audio_file'] = uploaded_file.name
                    
                    # Store in session state
                    st.session_state.analysis_result = analysis_result
                    st.session_state.audio_duration = audio_duration
                    
                    # Clear progress display and show success
                    progress_container.empty()
                    status_container.empty()
                    
                    # Final success message
                    st.success("✅ Analysis Complete!")
                    
                    # Add button to navigate to results
                    st.divider()
                    col1, col2, col3 = st.columns([1, 2, 1])
                    with col2:
                        if st.button("📊 View Results", key="view_results", width='stretch'):
                            st.session_state.current_tab = 1  # Switch to Results tab
                            st.rerun()
                    
                except Exception as e:
                    progress_container.empty()
                    status_container.empty()
                    st.error(f"❌ Error during analysis: {str(e)}")
                    st.info("Make sure the audio file is valid and not too large.")
        
        except Exception as e:
            st.error(f"❌ Error loading audio file: {str(e)}")
            st.info("Please ensure the file is a valid WAV format.")

# ==================== TAB 2: Results ====================
elif st.session_state.current_tab == 1:
    st.header("Analysis Results")
    
    if 'analysis_result' in st.session_state:
        analysis_result = st.session_state.analysis_result
        audio_duration = st.session_state.audio_duration
        
        # Summary statistics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Segments", len(analysis_result.get('emotion_segments', [])))
        
        with col2:
            if analysis_result.get('emotion_segments'):
                dominant_emotion = max(
                    analysis_result['emotion_segments'],
                    key=lambda x: x['confidence']
                )['emotion']
                st.metric("Dominant Emotion", dominant_emotion)
        
        with col3:
            avg_confidence = np.mean([
                s['confidence'] for s in analysis_result.get('emotion_segments', [])
                if 'confidence' in s
            ]) if analysis_result.get('emotion_segments') else 0
            st.metric("Avg Confidence", f"{avg_confidence:.1%}")
        
        with col4:
            transitions = len(analysis_result.get('emotion_transitions', []))
            st.metric("Emotion Changes", transitions)
        
        # Emotion Distribution Pie Chart
        if analysis_result.get('emotion_segments'):
            st.subheader("📊 Emotion Distribution")
            
            emotion_counts = {}
            for segment in analysis_result['emotion_segments']:
                emotion = segment['emotion']
                duration = segment['end_time'] - segment['start_time']
                emotion_counts[emotion] = emotion_counts.get(emotion, 0) + duration
            
            fig, ax = plt.subplots(figsize=(8, 6))
            colors = plt.cm.Set3(np.linspace(0, 1, len(emotion_counts)))
            ax.pie(
                emotion_counts.values(),
                labels=emotion_counts.keys(),
                autopct='%1.1f%%',
                colors=colors,
                startangle=90
            )
            ax.set_title("Emotion Distribution by Duration")
            st.pyplot(fig)
        
        # Emotion Timeline
        st.subheader("📈 Emotion Timeline")
        
        if analysis_result.get('emotion_segments'):
            # Create timeline data
            timeline_data = []
            emotions = []
            times = []
            
            for segment in analysis_result['emotion_segments']:
                emotions.append(segment['emotion'])
                times.append(segment['start_time'])
            
            # Plot timeline
            fig, ax = plt.subplots(figsize=(12, 4))
            
            emotion_colors = {
                'angry': '#FF6B6B',
                'disgust': '#FFD93D',
                'fear': '#6BCB77',
                'happy': '#4D96FF',
                'neutral': '#A0A0A0',
                'sad': '#9B59B6',
                'surprise': '#FF9F43'
            }
            
            for i, segment in enumerate(analysis_result['emotion_segments']):
                emotion = segment['emotion']
                start = segment['start_time']
                end = segment['end_time']
                confidence = segment.get('confidence', 0.5)
                
                color = emotion_colors.get(emotion, '#808080')
                ax.barh(0, end - start, left=start, height=0.5, 
                       color=color, alpha=0.7, edgecolor='black', linewidth=0.5)
                
                # Add confidence label
                mid_point = (start + end) / 2
                ax.text(mid_point, 0, f"{confidence:.0%}", 
                       ha='center', va='center', fontsize=8, fontweight='bold')
            
            ax.set_xlim(0, audio_duration)
            ax.set_ylim(-0.5, 0.5)
            ax.set_xlabel("Time (s)")
            ax.set_title("Emotion Timeline with Confidence Scores")
            ax.set_yticks([])
            ax.grid(True, alpha=0.3, axis='x')
            st.pyplot(fig)
        
        # Detailed Segments Table
        st.subheader("📋 Detailed Segments")
        
        if analysis_result.get('emotion_segments'):
            segments_data = []
            for i, segment in enumerate(analysis_result['emotion_segments'], 1):
                segments_data.append({
                    'Segment': i,
                    'Start (s)': f"{segment['start_time']:.2f}",
                    'End (s)': f"{segment['end_time']:.2f}",
                    'Duration (s)': f"{segment['end_time'] - segment['start_time']:.2f}",
                    'Emotion': segment['emotion'],
                    'Confidence': f"{segment.get('confidence', 0):.1%}",
                })
            
            st.dataframe(segments_data, width='stretch', height=400)
        
        # Download results
        st.subheader("⬇️ Download Results")
        
        json_str = json.dumps(analysis_result, indent=2)
        st.download_button(
            label="📥 Download JSON Results",
            data=json_str,
            file_name="emotion_analysis_results.json",
            mime="application/json",
            width='stretch'
        )
    
    else:
        st.info("👈 Upload and analyze an audio file in the 'Upload & Analyze' tab to see results here.")

# ==================== TAB 3: Documentation ====================
elif st.session_state.current_tab == 2:
    st.header("📚 Documentation")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("System Overview")
        st.write("""
        This emotion analysis system:
        - Uses advanced deep learning model for emotion classification
        - Processes audio using 1.9s sliding windows
        - Applies smoothing for temporal consistency
        - Tracks emotion transitions over time
        - Supports multiple emotions: Angry, Disgust, Fear, Happy, Neutral, Sad, Surprise
        """)
        
        st.subheader("Supported Emotions")
        emotions_info = {
            "😠 Angry": "High energy negative emotion",
            "🤢 Disgust": "Reaction of offense or revulsion",
            "😨 Fear": "Anxiety and apprehension",
            "😊 Happy": "Joy and positive mood",
            "😐 Neutral": "No strong emotion",
            "😢 Sad": "Low energy negative emotion",
            "😮 Surprise": "Sudden unexpected emotion"
        }
        for emotion, desc in emotions_info.items():
            st.write(f"**{emotion}**: {desc}")
    
    with col2:
        st.subheader("Tips for Best Results")
        st.write("""
        - **Audio Quality**: Use clear audio without background noise
        - **File Format**: Only WAV format is supported
        - **Duration**: Optimal audio length is 5-60 seconds
        - **Volume**: Ensure consistent audio volume
        - **Content**: Clear emotional expression works best
        """)
        
        st.subheader("Output Format")
        st.write("""
        The JSON output includes:
        - **Metadata**: File info, analysis timestamp
        - **Segments**: Time ranges with detected emotion
        - **Confidence**: Prediction confidence for each segment
        - **Transitions**: When emotion changes occur
        - **Statistics**: Overall emotion distribution
        """)
    
    st.divider()
    
    st.subheader("Advanced Configuration")
    
    with st.expander("View Configuration Details"):
        config_info = {
            "Chunk Window": f"{st.session_state.config_window_size}s",
            "Chunk Hop": f"{st.session_state.config_hop_size}s",
            "Sample Rate": "16 kHz",
            "Processing": "GPU (if available) / CPU"
        }
        
        for key, value in config_info.items():
            st.write(f"• **{key}**: {value}")

# Footer
st.divider()
st.markdown("""
<div style='text-align: center; color: gray; font-size: 12px;'>
    Emotion Analysis System | Built with Streamlit and PyTorch
</div>
""", unsafe_allow_html=True)

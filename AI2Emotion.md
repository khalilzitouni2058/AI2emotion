# Emotion Analysis Web Application

A real-time speech emotion recognition system with an interactive web interface built with Streamlit. Analyze audio files and visualize emotional content using deep learning.

## 🎯 Features

### Core Capabilities
- **Real-time Emotion Detection** - Analyze WAV audio files for 7 emotions: Angry, Disgust, Fear, Happy, Neutral, Sad, Surprise
- **Smart Processing** - Automatically switches between full audio analysis (<15s) and chunked analysis (≥15s)
- **Live Progress Tracking** - See real-time chunk processing with progress bars
- **Interactive Visualizations** - Emotion distribution charts, timeline graphs, and confidence scores
- **Result Export** - Download detailed analysis as JSON

### User Interface
- **📥 Upload & Analyze** - Drag-and-drop audio upload with waveform preview
- **📊 Results** - Comprehensive visualizations and detailed segment tables
- **📚 Documentation** - Built-in help and system information
- **⚙️ Configuration Sidebar** - View current system settings and parameters

## 🚀 Getting Started

### Prerequisites
- Python 3.8 or higher
- WAV audio files (MP3 not supported)
- 2GB+ RAM recommended

### Installation

1. **Clone or navigate to the project directory**
```bash
cd path/to/ISS
```

2. **Install dependencies**
```bash
pip install -r requirements_streamlit.txt
```

3. **Run the application**
```bash
streamlit run streamlit_app.py
```

The app will automatically open in your default browser at `http://localhost:8501`

## 📖 How It Works

### Analysis Process

1. **Audio Upload** - Upload a WAV file through the web interface
2. **Automatic Mode Selection**:
   - **Short Audio (<15s)**: Full audio prediction for faster results
   - **Long Audio (≥15s)**: Chunked analysis with sliding windows for detailed emotion tracking
3. **Real-time Processing** - Watch progress as each chunk is analyzed
4. **Smoothing Applied** - Offline interpolation smoothing reduces prediction noise
5. **Results Display** - View comprehensive analysis with charts and tables

### Configuration

The sidebar displays current system settings (read-only):

**Analysis Mode**
- Duration Threshold: 15.0 seconds
- Determines when to use full vs chunked processing

**Chunking Parameters**
- Window Size: 1.9 seconds (size of each audio segment)
- Hop Size: 0.5 seconds (overlap between segments)

**Model Information**
- Model: Whisper Large v3 (Speech Emotion Recognition)
- Sample Rate: 16 kHz
- Processing: Automatic GPU/CPU detection

## 💡 Usage Tips

### Best Results
- Use **clear audio** without background noise
- **Optimal duration**: 5-60 seconds
- Ensure **consistent volume** throughout
- **Emotional expression** should be clear and distinct

### File Requirements
- Format: WAV only
- Sample Rate: Any (automatically resampled to 16 kHz)
- Channels: Mono or Stereo (converted to mono)
- File Size: Recommend under 10MB

## 📊 Understanding Results

### Summary Metrics
- **Total Segments**: Number of analyzed chunks
- **Dominant Emotion**: Most confident emotion detected
- **Avg Confidence**: Average prediction confidence (0-100%)
- **Emotion Changes**: Number of emotion transitions

### Emotion Distribution
Pie chart showing the proportion of time spent in each emotion state.

### Emotion Timeline
Visual timeline with:
- Color-coded emotion segments
- Confidence scores for each segment
- Time markers for precise tracking

### Detailed Segments Table
Complete breakdown including:
- Segment number and time range
- Detected emotion and confidence
- Duration of each segment

## 🔧 Advanced Configuration

### Command Line Options

**Custom Port**
```bash
streamlit run streamlit_app.py --server.port 8502
```

**Disable CUDA (Force CPU)**
```bash
# Windows Command Prompt
set CUDA_VISIBLE_DEVICES=
streamlit run streamlit_app.py

# PowerShell
$env:CUDA_VISIBLE_DEVICES = ""
streamlit run streamlit_app.py

# Linux/Mac
CUDA_VISIBLE_DEVICES= streamlit run streamlit_app.py
```

### Modifying Parameters

To change analysis parameters, edit `config.py`:

```python
AUDIO_DURATION_THRESHOLD = 15.0  # Threshold for chunking
CHUNK_WINDOW_SIZE = 1.9          # Window size in seconds
CHUNK_HOP_SIZE = 0.5             # Hop size in seconds
SMOOTHING_MODE = 'offline'        # Smoothing mode
OFFLINE_SMOOTHING_METHOD = 'interpolate'  # Smoothing method
```

## 🐛 Troubleshooting

### Common Issues

**Port Already in Use**
```bash
streamlit run streamlit_app.py --server.port 8502
```

**Memory Issues with Large Files**
- Split audio files over 10 minutes into smaller segments
- Close other applications to free up RAM
- Use shorter window sizes if needed

**Slow Processing**
- First run downloads the model (~3GB) - this is normal
- GPU acceleration significantly improves speed
- Ensure you're not running multiple analyses simultaneously

**Audio File Not Accepted**
- Convert MP3/M4A files to WAV format first
- Use tools like FFmpeg: `ffmpeg -i input.mp3 output.wav`

**Unexpected Emotion Results**
- Ensure audio has clear emotional expression
- Check for background noise or poor audio quality
- Try with different audio samples for comparison

## 🎓 Supported Emotions

| Emotion | Description | Characteristics |
|---------|-------------|-----------------|
| 😠 Angry | High-energy negative | Loud, sharp, aggressive tone |
| 🤢 Disgust | Offense or revulsion | Harsh, rejecting quality |
| 😨 Fear | Anxiety and apprehension | Tense, trembling, high-pitched |
| 😊 Happy | Joy and positive mood | Upbeat, energetic, bright |
| 😐 Neutral | No strong emotion | Calm, steady, matter-of-fact |
| 😢 Sad | Low-energy negative | Slow, quiet, low-pitched |
| 😮 Surprise | Sudden unexpected reaction | Sudden pitch changes, exclamations |

## 📁 Output Format

Results are saved as JSON with the following structure:

```json
{
  "metadata": {
    "audio_file": "example.wav",
    "total_duration": 25.5,
    "total_chunks": 50,
    "smoothing_method": "interpolate"
  },
  "emotion_segments": [
    {
      "chunk_id": 1,
      "start_time": 0.0,
      "end_time": 1.9,
      "emotion": "happy",
      "confidence": 0.87
    }
  ],
  "emotion_transitions": [
    {
      "transition_id": 1,
      "from_emotion": "happy",
      "to_emotion": "neutral",
      "transition_time": 5.2
    }
  ]
}
```

## 🔬 Technical Details

### Model Architecture
- Base: OpenAI Whisper Large v3
- Fine-tuned for Speech Emotion Recognition
- Input: 16 kHz mono audio
- Output: 7-class emotion probabilities

### Processing Pipeline
1. Audio preprocessing (resampling, normalization)
2. Feature extraction (mel-spectrogram)
3. Model inference (transformer encoder)
4. Probability smoothing (interpolation)
5. Emotion label assignment

### Performance
- **Short audio (<15s)**: ~2-5 seconds processing time
- **Long audio (≥15s)**: ~0.1s per chunk on GPU, ~0.5s per chunk on CPU
- **GPU**: Nvidia GPU with CUDA support recommended
- **CPU**: Works but slower, suitable for testing

## 📝 Keyboard Shortcuts

While using the Streamlit app:
- `c` - Clear output and cache
- `r` - Rerun the application
- `p` - Show settings

## 🤝 Support

For issues or questions:
1. Check the Documentation tab in the app
2. Review the troubleshooting section above
3. Verify your audio file meets requirements
4. Check console logs for error details

## 📄 License

This project uses the Whisper Large v3 model from OpenAI. Please refer to the original model license for usage terms.

---

**Built with:** Streamlit • PyTorch • Transformers • Librosa

**Last Updated:** March 2026

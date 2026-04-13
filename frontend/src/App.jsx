import { useEffect, useMemo, useRef, useState } from 'react';

// Animated counter component
function AnimatedCounter({ from = 0, to, duration = 800 }) {
  const [count, setCount] = useState(from);

  useEffect(() => {
    if (typeof to !== 'number') return;

    let startTime = null;
    const animate = (now) => {
      if (!startTime) startTime = now;
      const progress = Math.min((now - startTime) / duration, 1);
      setCount(Math.floor(from + (to - from) * progress));
      if (progress < 1) requestAnimationFrame(animate);
    };
    requestAnimationFrame(animate);
  }, [to, duration, from]);

  return <span>{count}</span>;
}

// Animated stat card component with HUD styling
function AnimatedStatCard({ label, value, delay = 0, isCounter = false }) {
  const numValue = isCounter && typeof value === 'number' ? value : null;

  return (
    <div className="result-card animated" style={{ '--delay': `${delay}ms` }}>
      <span className="label">{label}</span>
      <span className="value hud-glow">
        {isCounter ? <AnimatedCounter to={numValue} /> : value}
      </span>
      <div className="card-scanline" />
      <div className="card-glow-overlay" />
    </div>
  );
}

function App() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [profile, setProfile] = useState('main');
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState('');
  const [dragOver, setDragOver] = useState(false);
  const fileInputRef = useRef(null);
  const resultPanelRef = useRef(null);

  useEffect(() => {
    if (result && resultPanelRef.current) {
      resultPanelRef.current.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }
  }, [result]);

  const normalized = result?.data || {};
  const metadata = normalized.metadata || {};
  const summary = normalized.summary || {};
  const segments = Array.isArray(normalized.emotion_segments) ? normalized.emotion_segments : [];
  const distribution = summary.emotion_distribution || {};
  const distributionEntries = Object.entries(distribution);
  const totalSegments = segments.length;
  const dominantEmotion = useMemo(() => {
    if (!distributionEntries.length) return 'N/A';
    return [...distributionEntries].sort((a, b) => Number(b[1]) - Number(a[1]))[0][0];
  }, [distributionEntries]);

  function handleFile(file) {
    if (!file) return;
    setSelectedFile(file);
    setError('');
    setResult(null);
  }

  async function runAnalysis() {
    if (!selectedFile || loading) return;

    setLoading(true);
    setError('');

    try {
      const formData = new FormData();
      formData.append('file', selectedFile);
      formData.append('profile', profile);
      formData.append('source_name', selectedFile.name);

      const response = await fetch('/api/v1/analyze/audio', {
        method: 'POST',
        body: formData,
      });

      const payload = await response.json();
      if (!response.ok || !payload.success) {
        throw new Error(payload.error || 'Analysis failed');
      }

      setResult(payload);
      setSelectedFile(null);
      if (fileInputRef.current) {
        fileInputRef.current.value = '';
      }
    } catch (err) {
      setError(err.message || 'Unexpected error');
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="app-shell">
      <section className="hero">
        <div className="hero-card">
          <div className="eyebrow">Neon Forge Emotion Lab</div>
          <h1 className="title">Emotion inference with a gaming-grade interface</h1>
          <p className="subtitle">
            Upload audio, run the same analysis pipeline as <strong>main.py</strong>, and view the final emotion result in a polished React dashboard.
          </p>
          <div className="stats-grid">
            <div className="stat-chip">
              <span className="label">Mode</span>
              <span className="value">{profile === 'main' ? 'Main' : 'Fast'}</span>
            </div>
            <div className="stat-chip">
              <span className="label">Result</span>
              <span className="value">{result ? dominantEmotion : 'Waiting'}</span>
            </div>
            <div className="stat-chip">
              <span className="label">Status</span>
              <span className="value">{loading ? 'Loading' : 'Ready'}</span>
            </div>
          </div>
        </div>

      </section>

      <section className="layout">
        <div className="panel">
          <h2>Audio Upload</h2>
          <div
            className={`upload-area ${dragOver ? 'dragover' : ''}`}
            onClick={() => fileInputRef.current?.click()}
            onDragOver={(event) => {
              event.preventDefault();
              setDragOver(true);
            }}
            onDragLeave={() => setDragOver(false)}
            onDrop={(event) => {
              event.preventDefault();
              setDragOver(false);
              handleFile(event.dataTransfer.files?.[0]);
            }}
          >
            <div className="upload-stack">
              <div className="upload-icon">🎮</div>
              <div className="upload-headline">Drop your audio file here</div>
              <div className="upload-copy">WAV input recommended. Click or drag to load a test clip.</div>
              {selectedFile && (
                <div className="selected-file">
                  <span>✓</span>
                  <span>{selectedFile.name}</span>
                </div>
              )}
              <input
                ref={fileInputRef}
                type="file"
                accept="audio/*"
                onChange={(event) => handleFile(event.target.files?.[0])}
              />
            </div>
          </div>

          <select className="profile-select" value={profile} onChange={(event) => setProfile(event.target.value)}>
            <option value="main">Main Workflow</option>
            <option value="fast">Fast Workflow</option>
          </select>

          <button className="upload-btn" onClick={runAnalysis} disabled={!selectedFile || loading}>
            {loading ? 'Analyzing...' : 'Run Analysis'}
          </button>

          <div className={`loading-shell ${loading ? 'visible' : 'hidden'}`} aria-hidden={!loading}>
            <div>
              <div className="spinner" />
              <div className="loading-text">Analyzing audio</div>
            </div>
          </div>

          <div className={`error-banner ${error ? 'visible' : ''}`}>{error}</div>
          <div className="ghost-tip">Loading only. The result panel stays focused on the final output.</div>
        </div>

        <div className="panel" ref={resultPanelRef}>
          <div className="result-title-row">
            <h2>Final Emotion Result</h2>
            <span className={`badge ${result ? 'success' : 'neutral'}`}>{result ? 'Completed' : 'Idle'}</span>
          </div>

          {!result && (
            <div className="empty-state">
              <div className="empty-state-icon">📊</div>
              <div>No inference result yet.</div>
              <div>Upload an audio file to see the emotion output.</div>
            </div>
          )}

          {result && (
            <>
              <div className="result-grid">
                <AnimatedStatCard 
                  label="Dominant Emotion" 
                  value={summary.overall_emotion || dominantEmotion}
                  delay={0}
                />
                <AnimatedStatCard 
                  label="Energy Level" 
                  value={summary.energy_level || 'N/A'}
                  delay={80}
                />
                <AnimatedStatCard 
                  label="Total Duration" 
                  value={`${Number(metadata.total_duration || 0).toFixed(2)}s`}
                  delay={160}
                />
              </div>

              <div className="result-grid">
                <AnimatedStatCard 
                  label="Segments" 
                  value={totalSegments}
                  delay={0}
                  isCounter={true}
                />
                <AnimatedStatCard 
                  label="Transitions" 
                  value={Array.isArray(normalized.emotion_transitions) ? normalized.emotion_transitions.length : 0}
                  delay={80}
                  isCounter={true}
                />
                <AnimatedStatCard 
                  label="Processing Mode" 
                  value={metadata.processing_mode || 'N/A'}
                  delay={160}
                />
              </div>

              <div className="result-section">
                <div className="section-head">
                  <h3>Emotion Distribution</h3>
                  <span className="badge neutral">{distributionEntries.length} labels</span>
                </div>
                <div className="distribution-list">
                  {distributionEntries.length ? distributionEntries.map(([emotion, count]) => {
                    const value = Number(count) || 0;
                    const max = Math.max(...distributionEntries.map(([, c]) => Number(c) || 0), 1);
                    return (
                      <div className="distribution-row" key={emotion}>
                        <div className="name">{emotion}</div>
                        <div className="distribution-bar">
                          <div className="distribution-fill" style={{ width: `${(value / max) * 100}%` }} />
                        </div>
                        <div className="segment-meta">{value}</div>
                      </div>
                    );
                  }) : <div className="empty-state">No distribution data available.</div>}
                </div>
              </div>

              <div className="result-section">
                <div className="section-head">
                  <h3>Emotion Segments</h3>
                  <span className="badge neutral">First {Math.min(10, totalSegments)} segments</span>
                </div>
                <div className="segment-list">
                  {segments.length ? segments.slice(0, 10).map((segment) => (
                    <div className="segment-row" key={segment.chunk_id}>
                      <div className="chunk">Chunk {segment.chunk_id}</div>
                      <div className="segment-meta">
                        {Number(segment.start_time || 0).toFixed(2)}s - {Number(segment.end_time || 0).toFixed(2)}s
                      </div>
                      <div className="segment-meta">
                        {segment.emotion} · {Number(segment.confidence || 0).toFixed(3)}
                      </div>
                    </div>
                  )) : <div className="empty-state">No segments returned.</div>}
                </div>
              </div>
            </>
          )}
        </div>
      </section>
    </div>
  );
}

export default App;

import { useEffect, useMemo, useState } from 'react'

const API_URL = import.meta.env.VITE_API_URL || 'http://127.0.0.1:8000'

const adviceSections = (text) => {
  if (!text) return []
  const lines = text.split('\n').map((l) => l.trim()).filter(Boolean)
  const sections = []
  let current = null

  const pushCurrent = () => {
    if (current && current.body.length) sections.push(current)
  }

  for (const line of lines) {
    const match = line.match(/^(WHAT|WHY|NOW|PREVENT|URGENCY)\s*:?\s*(.*)$/i)
    if (match) {
      pushCurrent()
      current = { title: match[1].toUpperCase(), body: [match[2]].filter(Boolean) }
    } else if (current) {
      current.body.push(line)
    }
  }
  pushCurrent()

  if (!sections.length) return [{ title: 'ADVICE', body: lines }]
  return sections
}

export default function App() {
  const [image, setImage] = useState(null)
  const [crop, setCrop] = useState('Tomato')
  const [district, setDistrict] = useState('Guntur')
  const [stateName, setStateName] = useState('Andhra Pradesh')
  const [season, setSeason] = useState('Kharif')
  const [language, setLanguage] = useState('Telugu')
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState(null)
  const [error, setError] = useState(null)
  const [supportedCrops, setSupportedCrops] = useState([])

  useEffect(() => {
    const fetchSupported = async () => {
      try {
        const res = await fetch(`${API_URL}/supported-crops`)
        const data = await res.json()
        setSupportedCrops(data?.supported_crops || [])
      } catch {
        setSupportedCrops([])
      }
    }
    fetchSupported()
  }, [])

  const imageUrl = useMemo(() => {
    if (!image) return null
    return URL.createObjectURL(image)
  }, [image])

  const handleSubmit = async (e) => {
    e.preventDefault()
    setError(null)
    setResult(null)

    if (!image) {
      setError('Please upload a clear leaf image.')
      return
    }

    const form = new FormData()
    form.append('file', image)
    form.append('crop', crop)
    form.append('district', district)
    form.append('state', stateName)
    form.append('season', season)
    form.append('language', language)

    try {
      setLoading(true)
      const res = await fetch(`${API_URL}/diagnose`, { method: 'POST', body: form })
      const data = await res.json()
      if (!res.ok) throw new Error(data?.error || 'Request failed')
      setResult(data)
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  const sections = adviceSections(result?.advice)
  const confidence = typeof result?.confidence === 'number' ? result.confidence : null
  const status = result?.status || ''
  const topPreds = result?.top_predictions || []

  const shareOnWhatsApp = () => {
    if (!result) return
    const text = [
      'CropGuard AI Diagnosis',
      `Crop: ${crop}`,
      `Location: ${(result.district || district)}, ${(result.state || stateName)}`,
      `Disease: ${result.disease}`,
      `Confidence: ${typeof confidence === 'number' ? confidence.toFixed(1) : result.confidence}%`,
      '',
      result.advice || ''
    ].join('\n')
    window.open(`https://wa.me/?text=${encodeURIComponent(text)}`, '_blank')
  }

  return (
    <div className="page">
      <header className="hero">
        <div className="badge">ProtoThon 2026</div>
        <div className="title">CropGuard AI</div>
        <div className="subtitle">Instant crop disease diagnosis + climate-aware treatment advice</div>
      </header>

      <main className="container">
        <form className="card" onSubmit={handleSubmit}>
          <div className="card-title">Get Diagnosis</div>

          {supportedCrops.length > 0 && (
            <div className="support-note">
              Supported crops in this MVP: <strong>{supportedCrops.join(', ')}</strong>
            </div>
          )}

          <div className="grid">
            <label>
              Crop
              <input value={crop} onChange={(e) => setCrop(e.target.value)} />
            </label>
            <label>
              District
              <input value={district} onChange={(e) => setDistrict(e.target.value)} />
            </label>
            <label>
              State
              <input value={stateName} onChange={(e) => setStateName(e.target.value)} />
            </label>
            <label>
              Season
              <select value={season} onChange={(e) => setSeason(e.target.value)}>
                <option>Kharif</option>
                <option>Rabi</option>
                <option>Zaid</option>
              </select>
            </label>
            <label className="full">
              Language
              <select value={language} onChange={(e) => setLanguage(e.target.value)}>
                <option>Telugu</option>
                <option>Hindi</option>
                <option>English</option>
              </select>
            </label>
          </div>

          <label className="upload">
            <input type="file" accept="image/*" onChange={(e) => setImage(e.target.files?.[0] || null)} />
            <span>{image ? image.name : 'Upload leaf image'}</span>
          </label>

          {imageUrl && <img src={imageUrl} alt="Leaf preview" className="preview-image" />}

          <button className="btn" type="submit" disabled={loading}>
            {loading ? 'Diagnosing...' : 'Diagnose'}
          </button>

          {error && <div className="error">{error}</div>}
        </form>

        <section className="card results">
          <div className="card-title">Result</div>
          {!result && !loading && <div className="muted">No result yet.</div>}
          {loading && <div className="muted">Analyzing image and generating advisory...</div>}

          {result && (
            <div className="result-body">
              <div className="pill-row">
                <div className="pill">
                  <span>Disease</span>
                  <strong>{result.disease}</strong>
                </div>
                <div className="pill">
                  <span>Location</span>
                  <strong>{(result.district || district) + ', ' + (result.state || stateName)}</strong>
                </div>
                <div className="pill">
                  <span>Language</span>
                  <strong>{result.language}</strong>
                </div>
              </div>

              {confidence !== null && (
                <div className="confidence">
                  <div className="confidence-label">
                    Confidence <strong>{confidence.toFixed(1)}%</strong>
                  </div>
                  <div className="bar">
                    <div className="bar-fill" style={{ width: `${Math.min(confidence, 100)}%` }} />
                  </div>
                </div>
              )}

              {status === 'needs_confirmation' && topPreds.length > 1 && (
                <div className="warning">
                  Possible match between <strong>{topPreds[0].disease}</strong> and <strong>{topPreds[1].disease}</strong>. Please verify visually.
                </div>
              )}

              {(status === 'unsupported_crop' || status === 'uncertain' || status === 'possible_non_plant') && (
                <div className="warning">
                  Diagnosis is uncertain for this input. Upload a close-up leaf image and choose a supported crop.
                </div>
              )}

              {result.climate_context && <div className="climate-box">{result.climate_context}</div>}

              <div className="advice-grid">
                {sections.map((sec) => (
                  <div key={sec.title} className="advice-card">
                    <div className="advice-title">{sec.title}</div>
                    <div className="advice-body">
                      {sec.body.map((line, i) => (
                        <div key={i}>{line}</div>
                      ))}
                    </div>
                  </div>
                ))}
              </div>

              <button type="button" className="btn secondary" onClick={shareOnWhatsApp}>
                Share on WhatsApp
              </button>
            </div>
          )}
        </section>
      </main>
    </div>
  )
}

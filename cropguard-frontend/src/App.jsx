import { useEffect, useMemo, useState } from 'react'

const API_URL = import.meta.env.VITE_API_URL || 'http://127.0.0.1:8000'
const TOKEN_KEY = 'cropguard_token'
const MODE_KEY = 'cropguard_mode'
const DETAIL_KEY = 'cropguard_detail'

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
  const [token, setToken] = useState(localStorage.getItem(TOKEN_KEY) || '')
  const [user, setUser] = useState(null)
  const [diagnosisMode, setDiagnosisMode] = useState(localStorage.getItem(MODE_KEY) || 'Balanced')
  const [detailLevel, setDetailLevel] = useState(localStorage.getItem(DETAIL_KEY) || 'Standard')

  const [authMode, setAuthMode] = useState('login')
  const [authName, setAuthName] = useState('')
  const [authPhone, setAuthPhone] = useState('')
  const [authEmail, setAuthEmail] = useState('')
  const [authPassword, setAuthPassword] = useState('')
  const [authOtp, setAuthOtp] = useState('')
  const [authNewPassword, setAuthNewPassword] = useState('')
  const [authError, setAuthError] = useState(null)
  const [authInfo, setAuthInfo] = useState(null)

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

  useEffect(() => {
    const fetchMe = async () => {
      if (!token) return
      try {
        const res = await fetch(`${API_URL}/auth/me`, { headers: { Authorization: `Bearer ${token}` } })
        if (!res.ok) throw new Error('Session expired')
        const data = await res.json()
        setUser(data.user)
      } catch {
        localStorage.removeItem(TOKEN_KEY)
        setToken('')
        setUser(null)
      }
    }
    fetchMe()
  }, [token])

  useEffect(() => {
    localStorage.setItem(MODE_KEY, diagnosisMode)
  }, [diagnosisMode])

  useEffect(() => {
    localStorage.setItem(DETAIL_KEY, detailLevel)
  }, [detailLevel])

  const imageUrl = useMemo(() => {
    if (!image) return null
    return URL.createObjectURL(image)
  }, [image])

  const handleAuth = async (e) => {
    e.preventDefault()
    setAuthError(null)
    setAuthInfo(null)
    try {
      if (authMode === 'register') {
        const res = await fetch(`${API_URL}/auth/register`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ name: authName, phone: authPhone, email: authEmail, password: authPassword })
        })
        const data = await res.json()
        if (!res.ok) throw new Error(data?.error || 'Registration failed')
        localStorage.setItem(TOKEN_KEY, data.token)
        setToken(data.token)
        setUser(data.user)
      } else if (authMode === 'login') {
        const res = await fetch(`${API_URL}/auth/login`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ phone: authPhone, password: authPassword })
        })
        const data = await res.json()
        if (!res.ok) throw new Error(data?.error || 'Login failed')
        localStorage.setItem(TOKEN_KEY, data.token)
        setToken(data.token)
        setUser(data.user)
      } else if (authMode === 'forgot') {
        const res = await fetch(`${API_URL}/auth/forgot-password/request`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ email: authEmail })
        })
        const data = await res.json()
        if (!res.ok) throw new Error(data?.error || 'OTP request failed')
        setAuthInfo(data?.message || 'OTP sent')
        setAuthMode('reset')
      } else if (authMode === 'reset') {
        const res = await fetch(`${API_URL}/auth/forgot-password/verify`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ email: authEmail, otp: authOtp, new_password: authNewPassword })
        })
        const data = await res.json()
        if (!res.ok) throw new Error(data?.error || 'Password reset failed')
        setAuthInfo('Password updated. Please login.')
        setAuthMode('login')
        setAuthOtp('')
        setAuthNewPassword('')
      }
    } catch (err) {
      setAuthError(err.message)
    }
  }

  const logout = async () => {
    try {
      if (token) {
        await fetch(`${API_URL}/auth/logout`, { method: 'POST', headers: { Authorization: `Bearer ${token}` } })
      }
    } finally {
      localStorage.removeItem(TOKEN_KEY)
      setToken('')
      setUser(null)
      setResult(null)
    }
  }

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
    form.append('diagnosis_mode', diagnosisMode)
    form.append('response_detail', detailLevel)

    try {
      setLoading(true)
      const res = await fetch(`${API_URL}/diagnose`, {
        method: 'POST',
        headers: { Authorization: `Bearer ${token}` },
        body: form,
      })
      const data = await res.json()
      if (!res.ok) throw new Error(data?.error || data?.detail || 'Request failed')
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

  if (!token || !user) {
    return (
      <div className="page">
        <header className="hero">
          <div className="badge">Secure Access</div>
          <div className="title">CropGuard AI</div>
          <div className="subtitle">Login to diagnose crop diseases</div>
        </header>

        <main className="container single">
          <form className="card" onSubmit={handleAuth}>
            <div className="auth-tabs">
              <button type="button" className={`tab ${authMode === 'login' ? 'active' : ''}`} onClick={() => setAuthMode('login')}>Login</button>
              <button type="button" className={`tab ${authMode === 'register' ? 'active' : ''}`} onClick={() => setAuthMode('register')}>Register</button>
              <button type="button" className={`tab ${authMode === 'forgot' ? 'active' : ''}`} onClick={() => setAuthMode('forgot')}>Forgot</button>
            </div>

            {authMode === 'register' && (
              <label>
                Name
                <input value={authName} onChange={(e) => setAuthName(e.target.value)} required />
              </label>
            )}

            {(authMode === 'login' || authMode === 'register') && (
              <label>
                Phone Number
                <input value={authPhone} onChange={(e) => setAuthPhone(e.target.value)} placeholder="+91XXXXXXXXXX" required />
              </label>
            )}

            {(authMode === 'register' || authMode === 'forgot' || authMode === 'reset') && (
              <label>
                Email
                <input type="email" value={authEmail} onChange={(e) => setAuthEmail(e.target.value)} placeholder="you@example.com" required />
              </label>
            )}

            {(authMode === 'login' || authMode === 'register') && (
              <label>
                Password
                <input type="password" value={authPassword} onChange={(e) => setAuthPassword(e.target.value)} required />
              </label>
            )}

            {authMode === 'reset' && (
              <>
                <label>
                  OTP
                  <input value={authOtp} onChange={(e) => setAuthOtp(e.target.value)} required />
                </label>
                <label>
                  New Password
                  <input type="password" value={authNewPassword} onChange={(e) => setAuthNewPassword(e.target.value)} required />
                </label>
              </>
            )}

            {authError && <div className="error">{authError}</div>}
            {authInfo && <div className="support-note">{authInfo}</div>}

            <button className="btn" type="submit">
              {authMode === 'register' && 'Create Account'}
              {authMode === 'login' && 'Login'}
              {authMode === 'forgot' && 'Send OTP'}
              {authMode === 'reset' && 'Reset Password'}
            </button>

            {authMode === 'forgot' && (
              <button type="button" className="btn secondary" onClick={() => setAuthMode('reset')}>I have OTP</button>
            )}
          </form>
        </main>
      </div>
    )
  }

  return (
    <div className="page">
      <header className="hero">
        <div className="badge">Authenticated</div>
        <div className="title">CropGuard AI</div>
        <div className="subtitle">Hi {user.name}. You are logged in with {user.phone}</div>
        <button className="btn logout" type="button" onClick={logout}>Logout</button>
      </header>

      <main className="container">
        <form className="card" onSubmit={handleSubmit}>
          <div className="profile-box">
            <div className="profile-title">Profile</div>
            <div><strong>Name:</strong> {user.name || '-'}</div>
            <div><strong>Phone:</strong> {user.phone || '-'}</div>
            <div><strong>Email:</strong> {user.email || '-'}</div>
            <div className="profile-title settings">Settings</div>
            <label>
              Diagnosis Mode
              <select value={diagnosisMode} onChange={(e) => setDiagnosisMode(e.target.value)}>
                <option>Balanced</option>
                <option>Strict</option>
                <option>Fast</option>
              </select>
            </label>
            <label>
              Response Detail
              <select value={detailLevel} onChange={(e) => setDetailLevel(e.target.value)}>
                <option>Standard</option>
                <option>Detailed</option>
                <option>Short</option>
              </select>
            </label>
          </div>

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
                <option>Kannada</option>
                <option>Tamil</option>
                <option>Malayalam</option>
                <option>Marathi</option>
                <option>Gujarati</option>
                <option>Hindi</option>
                <option>English</option>
                <option>Bengali</option>
                <option>Punjabi</option>
                <option>Odia</option>
                <option>Urdu</option>
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

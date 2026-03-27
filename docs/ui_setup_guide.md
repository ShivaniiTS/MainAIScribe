# AI Scribe Enterprise — UI Setup Guide

This document provides step-by-step instructions to start every user-facing application: the admin web UI, the provider-facing web portal, and the mobile app.

---

## 1. Prerequisites

Before starting any UI, ensure the backend API is running:

```bash
cd /home/sanket/code/ai-scribe-enterprise
source .venv/bin/activate

# Two terminals — one for each server role:
AI_SCRIBE_SERVER_ROLE=provider-facing uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
AI_SCRIBE_SERVER_ROLE=processing-pipeline uvicorn api.main:app --reload --host 0.0.0.0 --port 8100
```

Verify the API is running:
```bash
curl http://localhost:8000/health
curl http://localhost:8000/providers   # Should return JSON array
```

---

## 2. Port Map

| Component | Port | URL | Purpose |
|-----------|------|-----|---------|
| Provider-facing API | **8000** | `http://localhost:8000` | Client API — encounters, providers, patients, pipeline proxy |
| Processing-pipeline API | **8100** | `http://localhost:8100` | Pipeline execution, admin CRUD |
| Admin Web UI | **3100** | `http://localhost:3100` | Full admin dashboard (connects to pipeline API) |
| Provider Web Portal | **3000** | `http://localhost:3000` | Read-only provider dashboard (connects to provider-facing API) |
| Expo Dev Server | **8081** | `http://localhost:8081` | Mobile app Metro bundler |
| Ollama | **11434** | `http://localhost:11434` | LLM inference (required for pipeline) |

**Firewall / network access requirements:**
- Port **8000** must be accessible from the mobile app device (phone/tablet)
- Port **8081** must be accessible from the mobile app device (Expo dev server)
- If testing on a physical device over the network, use the machine's LAN IP (e.g., `192.168.1.42`) instead of `localhost`
- For remote mobile testing, use a tunnel (see [Section 6](#6-mobile-app-remote-access-via-tunnel))

---

## 3. Admin Web UI (Pipeline Team)

**Purpose:** Full management dashboard — create/edit providers, templates, specialties, view quality metrics, trigger pipeline re-runs.

**Code location:** `client/web/`

```
client/web/
├── app/                        # Next.js App Router pages
│   ├── layout.tsx              # Root layout with sidebar navigation
│   ├── page.tsx                # Dashboard (KPI tiles, quality trends)
│   ├── globals.css             # Tailwind v4 + design tokens
│   ├── capture/page.tsx        # Audio upload + pipeline trigger
│   ├── providers/
│   │   ├── page.tsx            # Provider list (CRUD)
│   │   ├── new/page.tsx        # Create provider form
│   │   └── [id]/page.tsx       # Provider detail + edit
│   ├── samples/
│   │   ├── page.tsx            # Sample browser with filters
│   │   └── [id]/page.tsx       # 6-tab detail (transcript, note, comparison, gold, quality, versions)
│   ├── specialties/
│   │   ├── page.tsx            # Specialty list (CRUD)
│   │   ├── new/page.tsx        # Create specialty
│   │   └── [id]/page.tsx       # Specialty detail + inline dictionary editor
│   └── templates/
│       ├── page.tsx            # Template list (CRUD)
│       ├── new/page.tsx        # Create template with section builder
│       └── [id]/page.tsx       # Template detail + section editor
├── components/                 # Shared React components
│   ├── Sidebar.tsx             # Left nav (logo, all sections)
│   ├── DashboardCharts.tsx     # KPI cards + recharts
│   ├── SampleDetailTabs.tsx    # 6-tab encounter viewer
│   ├── ProviderEditForm.tsx    # Create/edit provider form
│   ├── RerunButton.tsx         # Pipeline re-run trigger
│   ├── FeatureGate.tsx         # Role-based feature wrapper
│   └── ...
├── lib/
│   ├── api.ts                  # TypeScript API client (all endpoints)
│   └── useFeatures.ts          # Feature flag hook
├── .env.local                  # API URL config
└── package.json
```

### Setup

```bash
cd client/web
npm install

# Set the API URL (pipeline server for admin UI)
echo 'NEXT_PUBLIC_API_URL=http://localhost:8100' > .env.local

# Start the dev server on port 3100
npm run dev -- --port 3100
```

Open **http://localhost:3100** in your browser.

### Pages available

| Page | URL | Features |
|------|-----|----------|
| Dashboard | `/` | KPI cards, quality trend chart, dimension radar, recent encounters |
| Samples | `/samples` | Filterable table (version, mode, score), click → detail |
| Sample Detail | `/samples/[id]` | 6 tabs: Transcript, Clinical Note, Comparison, Gold Standard, Quality, Compare Versions |
| Providers | `/providers` | Provider cards + "New Provider" button |
| Provider Detail | `/providers/[id]` | Edit form: specialty, templates, vocabulary, style directives |
| Specialties | `/specialties` | List + create/edit + inline dictionary editor |
| Templates | `/templates` | List + create/edit + section builder with drag-reorder |
| Capture | `/capture` | Upload audio → trigger pipeline |

---

## 4. Provider Web Portal (Provider Team)

**Purpose:** Read-only dashboard for clinicians — browse encounters, view generated notes and transcripts, capture new audio.

**Code location:** `client/provider/`

```
client/provider/
├── app/
│   ├── layout.tsx              # Root layout (no admin nav items)
│   ├── page.tsx                # Dashboard (read-only)
│   ├── globals.css
│   ├── capture/page.tsx        # Record/upload audio + patient selection
│   ├── providers/
│   │   ├── page.tsx            # Provider list (read-only)
│   │   └── [id]/page.tsx       # Provider detail (read-only)
│   └── samples/
│       ├── page.tsx            # Encounter browser (read-only)
│       └── [id]/page.tsx       # Encounter detail (read-only)
├── components/                 # Subset of web/ components (no admin CRUD)
│   ├── Sidebar.tsx             # Filtered nav (no Specialties, Templates)
│   ├── DashboardCharts.tsx
│   ├── SamplesTable.tsx
│   ├── SampleDetailTabs.tsx
│   ├── MarkdownViewer.tsx
│   └── ...
├── lib/
│   └── api.ts                  # Read-only API client (no create/update mutations)
├── .env.local                  # API URL config
└── package.json
```

### Setup

```bash
cd client/provider
npm install

# Set the API URL (provider-facing server)
echo 'NEXT_PUBLIC_API_URL=http://localhost:8000' > .env.local

# Start the dev server on port 3000
npm run dev -- --port 3000
```

Open **http://localhost:3000** in your browser.

### Key differences from admin UI

| Feature | Admin UI (3100) | Provider Portal (3000) |
|---------|:-:|:-:|
| View encounters/notes/transcripts | Yes | Yes |
| View quality scores | Yes | Yes |
| Capture/upload audio | Yes | Yes |
| Create/edit providers | Yes | No |
| Create/edit templates | Yes | No |
| Create/edit specialties | Yes | No |
| Gold standard comparison tab | Yes | No |
| Re-run pipeline button | Yes | No |

---

## 5. Mobile App (Provider Team)

**Purpose:** Record audio from a phone/tablet, view encounters and clinical notes, trigger pipeline.

**Code location:** `client/mobile/`

```
client/mobile/
├── App.tsx                     # Root: tab navigator (Record, Encounters, Providers, Settings)
├── index.ts                    # Expo entry point
├── app.json                    # Expo config (app name, permissions, icons)
├── src/
│   ├── screens/
│   │   ├── RecordScreen.tsx    # Provider + patient select → record/upload audio
│   │   ├── EncountersScreen.tsx # Recent encounter list
│   │   ├── EncounterDetailScreen.tsx # Transcript + note viewer (tabs)
│   │   ├── ProvidersScreen.tsx # Provider list with quality scores
│   │   └── SettingsScreen.tsx  # API URL config + Test Connection button
│   ├── components/
│   │   ├── Badge.tsx           # Quality score badge
│   │   ├── Card.tsx            # Card container
│   │   └── ProgressBar.tsx     # Pipeline progress indicator
│   ├── lib/
│   │   ├── api.ts              # API client (dynamic URL from settings store)
│   │   └── theme.ts            # Design tokens (colors, spacing, fonts)
│   └── store/
│       ├── settings.ts         # Zustand: apiUrl + AsyncStorage persistence
│       └── offline.ts          # Offline audio upload queue
└── package.json
```

### Setup

```bash
cd client/mobile
npm install
```

### Option A: Same machine (emulator or USB-connected device)

```bash
npx expo start
```

This starts the Metro bundler on port **8081**. Scan the QR code with:
- **iOS:** Camera app → opens Expo Go
- **Android:** Expo Go app → scan QR

The mobile app auto-detects the dev machine's LAN IP from the Expo host URI and connects to port **8000** on that IP. For example, if your machine is `192.168.1.42`, the app will call `http://192.168.1.42:8000`.

**Requirements for this to work:**
- The API server must be running on port **8000** with `--host 0.0.0.0` (not just `localhost`)
- Port **8000** must be reachable from the phone (same WiFi network)
- Port **8081** must be reachable from the phone (Expo dev server)

### Option B: Physical device on the same network

1. Find your machine's LAN IP:
   ```bash
   hostname -I | awk '{print $1}'
   # Example output: 192.168.1.42
   ```

2. Start the API server bound to all interfaces:
   ```bash
   uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
   ```

3. Start Expo:
   ```bash
   npx expo start
   ```

4. Scan QR code with phone. The app should auto-detect the LAN IP.

5. If the app can't reach the API, go to **Settings** tab in the app and manually enter:
   ```
   http://192.168.1.42:8000
   ```
   Then tap **Save** and **Test Connection**.

### Option C: Remote device (different network)

See [Section 6](#6-mobile-app-remote-access-via-tunnel) below.

### Mobile app screens

| Tab | Screen | API Endpoints Used |
|-----|--------|-------------------|
| Record | RecordScreen | `GET /providers`, `GET /patients/search`, `POST /encounters`, `POST /encounters/{id}/upload` |
| Encounters | EncountersScreen | `GET /encounters` |
| Encounters | EncounterDetailScreen | `GET /encounters/{id}`, `GET /encounters/{id}/note`, `GET /encounters/{id}/transcript`, `GET /encounters/{id}/audio` |
| Providers | ProvidersScreen | `GET /providers` |
| Settings | SettingsScreen | Manual URL entry + test via `GET /providers` |

### Settings screen

The Settings tab lets you configure the API server URL at runtime:

- **Provider Server URL:** The URL of the provider-facing API (default: auto-detected LAN IP on port 8000)
- **Test Connection:** Pings `GET /providers` with a 5-second timeout
- **Save:** Persists to AsyncStorage (survives app restarts)
- **Reset to Default:** Reverts to auto-detected LAN IP

---

## 6. Mobile App Remote Access via Tunnel

When the phone is on a different network than the dev machine, use a Cloudflare tunnel:

### Step 1: Install cloudflared

```bash
# Download (if not already installed)
curl -L https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 -o /tmp/cloudflared
chmod +x /tmp/cloudflared
```

### Step 2: Create a tunnel for the API

```bash
/tmp/cloudflared tunnel --url http://localhost:8000
```

This prints a URL like: `https://something-random.trycloudflare.com`

### Step 3: Start Expo with tunnel mode

```bash
cd client/mobile
npx expo start --tunnel
```

This creates a tunnel for the Expo dev server (port 8081) and prints a QR code with the tunnel URL.

### Step 4: Configure the mobile app

1. Scan the Expo tunnel QR code with your phone
2. Once the app loads, go to **Settings** tab
3. Enter the Cloudflare tunnel URL from Step 2 (e.g., `https://something-random.trycloudflare.com`)
4. Tap **Test Connection** — should show "Connected to provider-facing server"
5. Tap **Save**

The app now routes all API calls through the tunnel. The tunnel URLs are ephemeral — they change each time you restart `cloudflared`.

---

## 7. Running Everything Together

### Development mode (single machine, two API instances)

Open 6 terminals:

```bash
# Terminal 1: Provider-facing API
cd /home/sanket/code/ai-scribe-enterprise
source .venv/bin/activate
AI_SCRIBE_SERVER_ROLE=provider-facing uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

# Terminal 2: Processing-pipeline API
cd /home/sanket/code/ai-scribe-enterprise
source .venv/bin/activate
AI_SCRIBE_SERVER_ROLE=processing-pipeline uvicorn api.main:app --reload --host 0.0.0.0 --port 8100

# Terminal 3: Admin Web UI → pipeline API
cd /home/sanket/code/ai-scribe-enterprise/client/web
NEXT_PUBLIC_API_URL=http://localhost:8100 npm run dev -- --port 3100

# Terminal 4: Provider Web Portal → provider-facing API
cd /home/sanket/code/ai-scribe-enterprise/client/provider
NEXT_PUBLIC_API_URL=http://localhost:8000 npm run dev -- --port 3000

# Terminal 5: Mobile App → provider-facing API (auto-detects port 8000)
cd /home/sanket/code/ai-scribe-enterprise/client/mobile
npx expo start

# Terminal 6: Ollama (if not already running as a service)
ollama serve
```

**Result:**
- Admin UI at **http://localhost:3100** (full CRUD, connects to pipeline API)
- Provider Portal at **http://localhost:3000** (read-only, connects to provider-facing API)
- Mobile app via QR code (connects to port 8000 on LAN IP)

---

## 8. Troubleshooting

### Web UI shows "Failed to fetch" or empty dashboard

- Verify the API is running: `curl http://localhost:8000/providers`
- Check `.env.local` has the correct `NEXT_PUBLIC_API_URL`
- Restart the Next.js dev server after changing `.env.local`

### Mobile app: providers/patients list is empty

- Go to Settings tab and verify the API URL
- Tap **Test Connection** — must show green "Connected"
- Ensure port 8000 is accessible from the phone's network
- If URL was recently changed, the app auto-re-fetches; wait 2-3 seconds

### Mobile app: Test Connection succeeds but data doesn't load

- Kill the app completely and reopen (AsyncStorage race condition fix requires app restart after first URL save)
- Clear Metro cache: `npx expo start --clear`

### Mobile app: "Network request failed" on upload

- Audio uploads require the API to accept multipart form data
- Ensure the API is running with `--host 0.0.0.0`, not `--host 127.0.0.1`
- If using a tunnel, ensure it's still running (ephemeral tunnels expire)

### Expo QR code doesn't connect

- Phone and dev machine must be on the same WiFi (for local mode)
- Try `npx expo start --tunnel` for cross-network access
- Install Expo Go from App Store / Play Store on the phone

### Port already in use

```bash
# Find what's using the port
lsof -i :8000
lsof -i :3000
lsof -i :8081

# Kill the process
kill -9 <PID>
```

---

## 9. Environment Variable Reference

| Variable | Where | Default | Purpose |
|----------|-------|---------|---------|
| `NEXT_PUBLIC_API_URL` | `client/web/.env.local` | `http://localhost:8000` | API URL for admin web UI |
| `NEXT_PUBLIC_API_URL` | `client/provider/.env.local` | `http://localhost:8000` | API URL for provider portal |
| `AI_SCRIBE_SERVER_ROLE` | Backend process | `provider-facing` | Server role: `provider-facing` or `processing-pipeline` |
| `AI_SCRIBE_INTER_SERVER_SECRET` | Backend process | (none) | Shared secret for inter-server auth |
| `HF_TOKEN` | Backend process | (required) | HuggingFace token for pyannote diarization |

The mobile app does not use environment variables — the API URL is configured at runtime via the Settings screen and persisted to AsyncStorage.

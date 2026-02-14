# Face Lock: Full Deployment Guide

Complete guide to deploying the Face Lock biometric consistency bot on a Dell OptiPlex (or any Linux box) with Docker.

---

## Architecture Overview

```
Telegram User
    |
    | sends photo
    v
Telegram Bot API
    |
    v
server.js (Telegraf)          <-- Node.js: receives photos, routes to skill
    |
    v
skills/face_lock/index.js    <-- Downloads image, checks premium, shells out
    |
    | python3 run_analysis.py --image <path> --platform <p> --json-output
    v
run_analysis.py               <-- CLI bridge: imports the real Python package
    |
    v
llama_index.packs.face_lock_biometric
    |-- face_lock.py           <-- MediaPipe 468-landmark mesh, 11 measurements
    |-- prompt_generators.py   <-- Platform-specific positive/negative prompts
    |-- bipa_consent.py        <-- BIPA consent records + audit trail
    |-- analog_film.py         <-- Kodachrome/Agfa/Vision3 film emulation
    |-- workflows.py           <-- Synthetic + real-face workflow orchestration
    |
    v
JSON stdout {
    gonial_angle, canthal_tilt, facial_index, nasal_tip_rotation,
    zygomatic_prominence, eye_shape, fitzpatrick_type, lip_fullness_ratio,
    brow_arch_height, face_symmetry_score, philtrum_length,
    positive_prompt, negative_prompt
}
    |
    v
index.js formats Telegram message
    |
    | Free tier:  geometry only, prompt codes locked
    | Premium:    full metrics + positive/negative prompts
    v
Telegram User receives formatted biometric report
```

---

## Prerequisites

| Item | Requirement |
|------|-------------|
| Hardware | Any x86_64 Linux machine (Dell OptiPlex, VPS, etc). 2GB+ RAM, 2+ CPU cores |
| OS | Ubuntu 22.04+ / Debian 12+ / any Docker-capable Linux |
| Docker | Docker Engine 24+ and Docker Compose v2 |
| Telegram | Bot token from @BotFather |
| Network | Outbound HTTPS (Telegram API polling), port 3000 internal only |

---

## Step 1: Get the Telegram Bot Token

1. Open Telegram, search for **@BotFather**
2. Send `/newbot`
3. Choose a display name (e.g., `Face Lock`)
4. Choose a username ending in `bot` (e.g., `FaceLockBiometricBot`)
5. Copy the token: `7123456789:AAH...` — you'll need this in Step 4

Optional commands to send @BotFather:
```
/setdescription — Biometric face analysis for AI character consistency
/setabouttext — Send a face photo. Get prompt codes for Reve/Flux/Midjourney.
/setuserpic — Upload a logo
```

---

## Step 2: Prepare the Host Machine

```bash
# Install Docker if not present
curl -fsSL https://get.docker.com | sh
sudo usermod -aG docker $USER
# Log out and back in, then verify:
docker --version
docker compose version

# Clone the repo (or copy the package directory)
git clone https://github.com/morongosteve/llama_index.git
cd llama_index/llama-index-packs/llama-index-packs-face-lock-biometric
```

---

## Step 3: Stage the Python Package for Docker

The Dockerfile needs the Python package inside its build context. Copy it:

```bash
cd openclaw

# Create a staging directory for the Python package
mkdir -p face_lock_pkg/llama_index/packs/face_lock_biometric

# Copy package files
cp -r ../llama_index/packs/face_lock_biometric/* face_lock_pkg/llama_index/packs/face_lock_biometric/
cp ../pyproject.toml face_lock_pkg/
cp ../README.md face_lock_pkg/

# Ensure Python package structure
touch face_lock_pkg/llama_index/__init__.py
touch face_lock_pkg/llama_index/packs/__init__.py
```

---

## Step 4: Configure Environment

```bash
# Create .env file (never commit this)
cat > .env << 'EOF'
TELEGRAM_BOT_TOKEN=7123456789:AAH_your_actual_token_here
DISCORD_BOT_TOKEN=
EOF

# Verify subscribers.json exists
cat config/subscribers.json
```

To add premium users, edit `config/subscribers.json`:
```json
{
  "premium_users": ["123456789", "987654321"],
  "free_tier_limits": {
    "daily_analyses": 5,
    "platforms": ["generic"]
  },
  "premium_features": {
    "daily_analyses": -1,
    "platforms": ["reve", "flux", "midjourney", "generic"],
    "film_presets": true,
    "drift_detection": true,
    "batch_analysis": true
  }
}
```

Find a user's Telegram ID: have them send `/status` to the bot, or use @userinfobot.

---

## Step 5: Build and Launch

```bash
# From the openclaw/ directory
docker compose --env-file .env up --build -d
```

This will:
1. Pull `node:22-slim` base image
2. Install system libs (OpenCV dependencies, Python3)
3. Install Python packages (mediapipe, opencv-python-headless, numpy, Pillow)
4. Install the face_lock_biometric package from local source
5. Install Node.js dependencies (telegraf)
6. Start the bot in polling mode

Verify it's running:
```bash
# Check container status
docker compose ps

# Watch logs
docker compose logs -f face-lock-bot

# Test health endpoint
curl http://localhost:3000/health
# Expected: {"status":"ok","uptime":12.345}
```

---

## Step 6: Test the Bot

1. Open Telegram, find your bot by its username
2. Send `/start` — should get the welcome message
3. Send a clear frontal face photo
4. Wait for "Scanning..." then the biometric report

**Expected free-tier output:**
```
FACE LOCK: LITE
────────────────
Geometry:
- Gonial Angle: 127.3 deg
- Canthal Tilt: 2.1 deg
- Facial Index: 1.312
- Eye Shape: almond
- Fitzpatrick: Type 3

LOCKED:
- Positive Prompt Codes
- Anti-Drift Negative Prompts
- Zygomatic Prominence
- Lip Fullness Ratio
- Brow Arch Height
- Symmetry Score

Upgrade to unlock full biometrics + prompt codes.
```

**Expected premium output (add your user ID to subscribers.json):**
```
FACE LOCK: PRO
────────────────
Geometry:
- Gonial Angle: 127.3 deg
- Canthal Tilt: 2.1 deg
- Facial Index: 1.312
- Eye Shape: almond
- Fitzpatrick: Type 3

Extended Metrics:
- Nasal Rotation: 96.4 deg
- Zygomatic: 0.167
- Lip Ratio: 0.543
- Brow Arch: 0.089
- Symmetry: 0.934

POSITIVE PROMPT (flux):
photo, oval face shape, moderately defined jawline, moderately defined
cheekbones, almond-shaped eyes, slightly upturned eye corners, straight
nose profile, balanced medium lips, medium skin with warm undertones,
high detail, sharp focus

NEGATIVE PROMPT:
deformed face, asymmetric eyes, blurry, low quality, distorted features,
droopy eyes, downturned eyes, upturned nose, button nose
```

---

## Step 7: Production Hardening

### Restart policy
Already handled — `docker-compose.yml` has `restart: always`.

### Resource limits
Already set — 2GB memory cap, 2 CPU cores max.

### Log rotation
```bash
# Add to docker-compose.yml under the service:
    logging:
      driver: json-file
      options:
        max-size: "10m"
        max-file: "3"
```

### Firewall
Port 3000 is internal only (health check). The bot uses outbound polling, so no inbound ports need to be exposed to the internet.

```bash
# If using ufw:
sudo ufw default deny incoming
sudo ufw default allow outgoing
sudo ufw allow ssh
sudo ufw enable
# Do NOT open port 3000 externally — it's health check only
```

### Auto-start on boot
Docker with `restart: always` handles this. Ensure Docker starts on boot:
```bash
sudo systemctl enable docker
```

---

## File Map

```
llama-index-packs-face-lock-biometric/
|
|-- llama_index/packs/face_lock_biometric/    # Python package (the engine)
|   |-- __init__.py                           # Public API exports
|   |-- face_lock.py                          # FaceAnalyzer, FacialMeasurements, DriftDetector
|   |-- prompt_generators.py                  # PromptGenerator, Platform enum
|   |-- bipa_consent.py                       # BIPAConsent, ConsentStore
|   |-- analog_film.py                        # AnalogFilmProcessor, FilmPreset
|   |-- workflows.py                          # SyntheticWorkflow, RealFaceWorkflow
|   `-- base.py                               # FaceLockBiometricPack (LlamaIndex integration)
|
|-- openclaw/                                 # Docker bot deployment
|   |-- Dockerfile                            # Node 22 + Python + OpenCV + MediaPipe
|   |-- docker-compose.yml                    # Service config, resource limits, healthcheck
|   |-- package.json                          # Node deps (telegraf)
|   |-- server.js                             # Telegram bot entry point
|   |-- config/
|   |   `-- subscribers.json                  # Premium user IDs, tier definitions
|   |-- characters/
|   |   `-- launcher.txt                      # Bot personality / system prompt
|   `-- skills/face_lock/
|       |-- index.js                          # Skill: download image, shell to Python, format output
|       `-- run_analysis.py                   # CLI bridge: imports real package, outputs JSON
|
|-- pyproject.toml                            # Package metadata, dependencies
|-- README.md                                 # Package documentation
`-- DEPLOYMENT.md                             # This file
```

---

## Data Flow: Field-by-Field

The Python engine outputs 11 biometric fields. Here's how each flows through the system:

| Python field name | Computed from (MediaPipe landmarks) | Node.js reads as | Displayed in |
|---|---|---|---|
| `gonial_angle` | Jaw angle: L132-L234-L152 averaged with right | `data.gonial_angle` | Free + Pro |
| `canthal_tilt` | Eye corner slope: L33-L133, L263-L362 averaged | `data.canthal_tilt` | Free + Pro |
| `facial_index` | Face height/width: L10-L152 / L234-L454 | `data.facial_index` | Free + Pro |
| `eye_shape` | Aspect ratio + canthal tilt classification | `data.eye_shape` | Free + Pro |
| `fitzpatrick_type` | ITA angle from LAB cheek samples | `data.fitzpatrick_type` | Free + Pro |
| `nasal_tip_rotation` | Nasion-tip-columella angle: L6-L1-L2 | `data.nasal_tip_rotation` | Pro only |
| `zygomatic_prominence` | Cheekbone projection / IPD: L123-L116 avg | `data.zygomatic_prominence` | Pro only |
| `lip_fullness_ratio` | Upper/lower lip height: L13-L14 / L14-L17 | `data.lip_fullness_ratio` | Pro only |
| `brow_arch_height` | Brow peak offset / IPD: L105 peak vs baseline | `data.brow_arch_height` | Pro only |
| `face_symmetry_score` | Bilateral landmark pair distances vs midline | `data.face_symmetry_score` | Pro only |
| `philtrum_length` | Subnasale to vermilion / IPD: L2-L13 | (not displayed) | Internal |

Plus two generated strings:
| `positive_prompt` | From PromptGenerator based on all measurements | `data.positive_prompt` | Pro only |
| `negative_prompt` | Anti-drift negatives based on measurement ranges | `data.negative_prompt` | Pro only |

---

## Troubleshooting

### "No face detected"
- Image must contain a clearly visible frontal face
- Minimum resolution: ~200x200px face region
- Avoid heavy filters, extreme angles, or occlusion

### "Timeout" (30s exceeded)
- Image is too large. Resize below 2048px on longest edge
- MediaPipe initialization is slow on first run (~5s cold start)

### Container keeps restarting
```bash
docker compose logs face-lock-bot --tail 50
```
Common causes:
- Missing `TELEGRAM_BOT_TOKEN` in .env
- Invalid token (check with @BotFather)
- Python package not installed (check Dockerfile build output)

### "face_lock_biometric package not installed"
The Python package wasn't installed in the container. Verify:
```bash
docker compose exec face-lock-bot python3 -c "from llama_index.packs.face_lock_biometric.face_lock import FaceAnalyzer; print('OK')"
```
If it fails, rebuild: `docker compose up --build --force-recreate -d`

### Memory issues
MediaPipe + OpenCV can spike to ~1.5GB during analysis. If the container is OOM-killed:
- Increase memory limit in docker-compose.yml
- Or process smaller images

---

## Running Without Docker (Development)

```bash
# Terminal 1: Install Python package
cd llama-index-packs/llama-index-packs-face-lock-biometric
pip install -e .

# Verify Python engine
cd openclaw/skills/face_lock
python3 run_analysis.py --mode synthetic --seed 42 --platform flux --json-output

# Terminal 2: Start Node.js bot
cd openclaw
export TELEGRAM_BOT_TOKEN="your-token"
npm install
npm start
```

---

## Running the Streamlit UI (standalone demo)

The Streamlit app at the repo root is a standalone demo with its own simplified analyzer. It is NOT connected to the bot pipeline.

```bash
pip install -r face_lock_requirements.txt
streamlit run face_lock_app.py
# Opens at http://localhost:8501
```

---

## CLI Quick Reference

```bash
# Analyze a real face
python3 run_analysis.py --image photo.jpg --platform reve --json-output

# Generate a synthetic character
python3 run_analysis.py --mode synthetic --seed 7 --platform flux --json-output

# Override specific measurements for synthetic
python3 run_analysis.py --mode synthetic --seed 7 --gonial-angle 122 --canthal-tilt 4.5 --json-output

# Check drift on a generated image
python3 run_analysis.py --drift generated.jpg --target-json locked_profile.json --json-output

# Include raw biometric values in prompts
python3 run_analysis.py --image photo.jpg --platform midjourney --include-technical --json-output
```

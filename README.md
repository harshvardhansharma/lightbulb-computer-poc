# Lightbulb Computer: prototype

A desk/wall prototype of [lightbulbcomputer.com](https://lightbulbcomputer.com/): a **projector + camera + compute**
that turns a table or wall into an interface you use by pointing, pinching and talking.

```
 iPhone (Continuity Camera) ─┐                         ┌─> Projector (2nd display, fullscreen Chrome)
                             ├─> MacBook: Chrome app ──┤
 Mic ────────────────────────┘    │  MediaPipe hands    └─> Speaker (timers, spoken replies)
                                  │  homography cam→projector
                                  └─> server.py ──> Claude Agent SDK ──> local `claude` CLI (your login)
```

## Hardware setup
1. **Projector** → MacBook over HDMI/USB-C. In *System Settings → Displays*, use **Extend** (not Mirror).
   Aim it at a table from above (best) or at a wall.
2. **Phone camera** → MacBook:
   - iPhone: **Continuity Camera** (same Apple ID, Wi-Fi + Bluetooth on). Shows up as “<name>’s iPhone Camera”.
   - Android: DroidCam / Camo / iVCam, which appears as a webcam.
   Mount the phone next to the projector, looking at the **same surface**. It must see the whole projected area.
   A phone clamp or tripod works. Lock it in place, because moving it means recalibrating.
3. Dim the room a bit. Projected light washes out easily.

## Run
Requires [Claude Code](https://claude.com/claude-code) installed and logged in (`claude` works in your terminal).
No API key is used: the server talks to Claude through the Claude Agent SDK, which drives your local `claude` CLI.
```bash
./run.sh               # downloads the hand model on first run, starts http://localhost:8000, opens Safari
```
Optional overrides go in `.env` (see `.env.example`): `LB_MODEL`, `LB_EFFORT`, `LB_CLAUDE_CLI`.
In Chrome, allow **camera + microphone**, then:

| Key | Action |
|---|---|
| **P** | Fullscreen on the projector (Chrome asks for “window management” permission the first time) |
| **C** | Calibrate: projects 9 dots, finds them in the camera, fits a homography. Keep hands out of view. |
| **D** | Debug view: camera feed, hand landmarks, calibrated projection area (green), camera picker |
| **Enter** or **/** | Type a command instead of speaking |
| **W** | Toggle wake word (“Lightbulb, …”) vs. always-listening |
| **M** | Show the mouse cursor (widgets can also be dragged with the mouse) |
| **H** | Help |

## Interactions
- **Point**: a ring is projected onto your fingertip.
- **Dwell**: hold still on a button for ~0.85 s to press it.
- **Pinch** (thumb + index): grab and move a widget; pinch inside a map to pan it.
- **Voice**: “Lightbulb, …”
  - *set a 10 minute timer* / *show the clock* / *clear everything* / *close this*: handled locally, instant
  - *what is this?* (while pointing at something): Claude sees the camera frame with your finger marked
  - *where is Chamonix on this map?* / *which of these is ripe?*: Claude **highlights real objects** (projected rings)
  - *how do I make pancakes?*: step-by-step recipe with per-step timers
  - *show me a map of Kyoto* / *weather in Bengaluru* / *note: buy eggs*

## How it works
- `web/app.js`: camera, calibration, hand tracking (MediaPipe HandLandmarker), gestures, widgets, voice (Web Speech API).
- `web/homography.js`: least-squares DLT homography. Maps camera coordinates to projector pixels for the flat surface.
- `server.py`: one endpoint, `/api/assist`. It runs a Claude Agent SDK `query()` with the request plus a camera JPEG.
  Claude Code's built-in tools are disabled. Claude only gets in-process MCP tools
  (`answer_card`, `highlight`, `timer`, `recipe`, `map`, `weather`, `note`, `clear`, `say`) that record UI actions, and the browser runs them.
  Local settings, CLAUDE.md and hooks are not loaded (`setting_sources=[]`). Each request takes about 5–9 s, including CLI startup.

## Browser: use Safari with an iPhone camera
macOS only gives Continuity Camera video to apps that opt in, and **Chrome doesn't**: it sees the iPhone
microphone but not the camera. **Safari** does, so with an iPhone open the app in Safari
(drag the window onto the projector, then press P to go fullscreen). Chrome works fine with Android/USB
webcams, or with a virtual camera app (e.g. Camo) that re-exposes the iPhone.

## Tips / known limits
- The fingertip hovers slightly above the table, so there's a small parallax offset. Mounting the camera close to the projector lens helps.
- Calibration fails if the camera can't see the dots. Press **D** to check framing, dim the lights, and press **C** again.
- Speech recognition in Chrome uses Google's servers. Use the typed command box if you're offline.
- Maps use OpenStreetMap tiles and Nominatim; weather uses Open-Meteo. None of them need a key.

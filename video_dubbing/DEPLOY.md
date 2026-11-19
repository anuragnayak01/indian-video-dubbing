# Deploy: GitHub → Render (API) + Vercel (UI)

## 1. GitHub
```bash
git init && git add . && git commit -m "Video dubbing recipe"
git branch -M assignment/video-dubbing-anuragnayak01
git remote add origin https://github.com/<you>/<repo>.git && git push -u origin HEAD
```
`.env` is git-ignored; never commit keys. For the Cookbook PR, copy this folder to `examples/video_dubbing/` in your fork.

## 2. Backend on Render
1. Render → New → **Blueprint** → pick the repo (uses `render.yaml`, builds `Dockerfile.web` with ffmpeg).
2. Fill the prompted env vars: the three `BODHAN_*_KEY`, optional `LLM_*`, and `DUB_CORS_ORIGINS` (set after step 3: `https://<your-app>.vercel.app`, no trailing slash).
3. `DUB_ACCESS_TOKEN` is generated for you: copy it from the service's Environment tab.
4. Check `https://<service>.onrender.com/api/health` returns `{"ok":true,"auth":true}`.

Notes: free plan has 512 MB RAM, sleeps when idle (first request ~1 min) and loses files on restart; use Starter + the 5 GB disk for real use. Jobs run one at a time.

## 3. Frontend on Vercel
1. Edit `frontend/config.js`: set `window.DUB_API` to your Render URL; commit and push.
2. Vercel → Add New Project → import the repo → **Root Directory: `frontend`**, Framework: Other, no build command.
3. Add the Vercel URL to `DUB_CORS_ORIGINS` on Render (redeploys automatically).
4. Open the Vercel URL; it asks for the access token once (stored in your browser).

After changing `src/bodhan_dub/web/index.html`, run `make frontend` to refresh the Vercel copy.

## Security
Anyone with the token can spend your Bodhan credit. Keep it private; rotate it in Render if leaked. Media links (`/file/`) are protected only by the unguessable project id.

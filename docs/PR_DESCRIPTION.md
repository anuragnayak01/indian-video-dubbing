Title: `[Agentic Eng Assignment] Video dubbing — <your name>`
Branch: `assignment/video-dubbing-<github-handle>` · Folder: `examples/video_dubbing/`

## Summary
Dubs an English video into any Indic-Speak language and writes an AI-dubbed MP4, SRT, per-sentence JSON, the English transcript and a machine-readable report. All transcription, translation and speech go to Bodhan models; an optional LLM only judges and edits English. A web studio lets a person review and fix any sentence.

## How to run
```bash
cd examples/video_dubbing
make setup && cp .env.example .env        # needs ffmpeg; fill in keys
python -m bodhan_dub --input path/to/video.mp4 --langs hi,ta,bn --out ./out
python -m bodhan_dub.web --port 8000      # optional web studio
```

## Architecture
See the diagram and one line per stage in the [README](../README.md#how-it-works). Components: `segmentation` (VAD, chunks, sentences), `bodhan` (client with retries, cache, cost meter), `llm` (English-only brain with guardrails), `timing` (fit ladder), `qa` (back-transcription checks), `audio` (mix, loudness, mux), `pipeline` (orchestration, per-language isolation), `web` (thin UI).

## Workflow
Flow chart with retries, checks and decision points: [docs/design_notes.md](design_notes.md#flow-chart-with-retries-checks-and-decisions).

## Where the LLM is used, and why
| Decision | Fallback |
|---|---|
| Group fragments into whole sentences | punctuation and pause heuristic |
| Build a glossary of names, acronyms, terms | none; user glossary still applies |
| Shorten English when the dub is too long | cap speed at 1.2x, mark degraded |
| Judge a borderline dub from its back-transcript | trust error-rate metrics |

English only; any output containing Indic script is rejected. Tested with: `<LLM model name>` (fill in after the run).

## Bodhan models used
Indic-Transcribe (hosted: source English and back-transcription), Indic-Translate (hosted: all translation, `codemix` with `--code-mixed`), Indic-Speak (hosted: all speech, the language's own recorded voices, optional `style`), Indic-OCR (hosted, optional `--ocr`: on-screen text in sampled frames, translated and burned in). Chunks are kept to 25 s.

## Results
`<link to demo outputs>` · Quality targets from `report.json`: `<fill in from the real run: sync %, speed range, length diff, LUFS, mean CER, misses and why>`.

## Demo video
`<link, 5 minutes or less>`

## Trade-offs and known limitations
See [README limitations](../README.md#limitations-stated-honestly). In short: speaker split is a pitch heuristic; music under speech is lost; glossary enforcement is partial; QA thresholds need calibration on real output; on-screen text timing follows the frame-sampling interval.

## Cost and runtime
`<fill in from report.json>`. Estimate: about ₹0.6 per video minute per language plus ₹0.10 per video minute once for the source.

## AI assistance
Claude Code (Claude Sonnet) was used for the pipeline, tests, web studio and docs; all code was reviewed and run by the author.

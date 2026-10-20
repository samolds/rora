0. organize this package
1. support detecting other things, not just dogs - humans, pets, objects, etc
2. provide an option to 'invert' the drawing, so that it can be used to draw black lines on a white background. leave the .png background transparent
3. provide a "rainbow" mode, where the lines are drawn to look like a rainbow, not just white or black
4. provide an option to draw the lines in various colors
5. can this project be prepared to be hosted on Google App Engine so there is a simple UI (with rate limiting) that allows people to upload a photo, send it through the pipeline, and return the photo
6. can the whole project be converted to JS, so that it can run on the frontend browser?

---

## Done: two-phase `--refine` + memory fixes

### Why
On this Mac (M2 Pro, 16 GB RAM, ~8 GB to spare next to VS Code, Webex, Code42, Chrome...) the
old single-pass `--refine` loaded FLUX.2 [klein] 4B on top of ~8 PyTorch drawing models and
died; the first two-phase version still peaked at a 17.2 GB footprint and swapped.

### What changed
- Two phases (`lineart.py`): phase 1 draws everything and queues a `RefineJob` per drawing
  (ink + mask bit-packed); then `release_models()` + `free_device_memory()`; phase 2
  (`refine_all`) loads klein once and refines every job.
- `free_device_memory()` after every photo: torch's MPS cache otherwise grows ~1-2 GB per photo.
- `tattoo.release_models()` moves OWLv2 to the CPU: a model left on MPS pins ~4.4 GB of heaps.
- mflux (`refine._load_editor`), which rora calls without the CLI's `MemorySaver`:
  - `mx.set_cache_limit(1 GB)`: the unbounded buffer cache held ~8 GB after loading.
  - `TilingConfig()`: whole-image VAE encode/decode needed +5/+9 GB at ~1 MP. Tiled decode leaves
    the ink unchanged; tiled encode redraws small details (ink IoU ~0.9 vs untiled), accepted.
  - `_free_text_encoder_after_use`: text encoder dropped after encoding each prompt, reloaded
    (~2 s) for the next drawing. Uses mflux 0.23 internals, so `mflux>=0.23,<0.24`.
- `--refine-quantize` default 8 → 4 (8-bit: weights 7.6 GB before the text-encoder change).

### Measured (macOS phys_footprint + MLX counters)
- 3 photos, `--subject all --refine stylize`: peak footprint 17.2 GB (1 photo, before) →
  9.8 GB, now set by phase 1 drawing a two-subject photo; phase 2 MLX peak 6.2 GB with 2.35 GB
  resident (4-bit).
- 8-bit: MLX peak ~7.2 GB. It failed intermittently (2 of 5 runs) with Metal
  `kIOGPUCommandBufferCallbackErrorImpactingInteractivity` (GPU watchdog) under heavy swap; the
  original is kept. 4-bit: no failures.
- A pre-quantized 4-bit copy (`mflux-save --base-model flux2-klein-4b -q 4`, 4.3 GB vs 15 GB
  bf16) gives bit-identical output, ~0.6 GB lower peak; the diffusers backend needs bf16. Decide
  when consolidating with backyard.

### Since then (uncommitted)
- `dog_lineart.py` renamed to `lineart.py` (PNG text key stays `dog_lineart_settings` for old files).
- `--refine-model` (`refine.MODELS`: klein-4b default, klein-9b, kontext-dev) with a memory guard
  (`refine.check_memory`: refuses above RAM - 3 GB unless `--refine-force`, warns above free
  memory). Only klein-4b is downloaded/tested; the others are wired up but untested.
- `rora/seams.py`: after refine, bridges strokes broken across the VAE tile bands and removes
  straight lines along them (mflux only). On 3 real drawings: 1 tiny bridge, 0 lines.

### Next steps
- Phase 1 (drawing) is now the peak (~10-11.5 GB on a two-subject photo at 2500 px).
- Consolidate backyard's image generation (DrawThings) onto mflux + the shared HF cache, then
  remove DrawThings and its model.

### Context for a new session
- venv is `env/` (`env/bin/python`); package code in `rora/`; entry point `lineart.py`.
- Refine model: FLUX.2 [klein] 4B, pinned revision in `rora/refine.py`, downloaded by
  `./setup.sh --refine` into the **shared** HF cache `~/.cache/huggingface/hub` (override with
  `RORA_REFINE_CACHE`); backends mflux (Apple silicon) / diffusers (elsewhere).
- Network: Cisco Secure Access inspects HTTPS here; `rora/env.py` calls
  `truststore.inject_into_ssl()` so Python downloads work.
- Disk is nearly full (~23 GB free, 95% used, with the 15 GB bf16 model in the HF cache).
- Test refine without the model by monkeypatching `refine.check_ready` and `refine._load_editor`.
- `img/stylized/` is untracked output (not in `.gitignore`).

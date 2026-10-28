# rora

Turn photos of dogs, cats and people into print-ready line art: a transparent PNG (and optional
SVG) of just the subjects, drawn in white lines for printing on dark.

The pipeline finds the people, cats and dogs in each photo, cuts them out, and draws them in
one of several styles, including minimalist tattoo-style line art. By default every subject
detected with high confidence is drawn, all together in one image; `--subject auto` draws only
the main one (the biggest, most central), and `--subject dog` (or `cat`, `person`) draws each
one of that kind in its own image. GIFs and videos (MP4, MOV) work too, frame by frame (see
[GIFs and videos](#gifs-and-videos)).

## Sam's Favorite Runs

```
env/bin/python lineart.py img/in -s edges ink noir outline sketch stencil stipple --detail 1 --no-preview --subject dog --refine stylize --sample 1
env/bin/python lineart.py img/peevin -s edges ink noir outline sketch stencil stipple --detail 1 --no-preview --subject all --refine stylize --sample 1
```

```
env/bin/python lineart.py img/in -s edges ink noir outline sketch stencil stipple --detail 1 --no-preview --refine stylize --refine-prompt "alter style to character from epic star wars concept art" --sample 1
```

```
env/bin/python lineart.py img/in -s edges ink noir outline sketch stencil stipple --detail 1 --no-preview --subject dog --invert --color black --sample 1
```

```
for d in 0.1 0.3 0.5 0.8 1; do
  env/bin/python lineart.py img/in -s edges ink noir outline sketch stencil stipple --detail $d --no-preview --sample 1
done
```


## Getting started

Requires Python 3.10+ (tested with 3.12) and about 3 GB of disk for dependencies and models.
Works on macOS (Apple GPU), Linux/Windows (CUDA GPU) or plain CPU.

```sh
./setup.sh                                                        # one time: venv, dependencies, model weights
env/bin/python lineart.py img/in/1.jpg --style tattoo --preview     # try the included demo photo
```

Then add your own photos to `img/in/` and run `env/bin/python lineart.py --preview`.
Results go to `img/out/`, including `_preview.png` (every result on a dark background) and `run.log`.

After `setup.sh` (and `./setup.sh --refine` for `--refine`), runs work fully offline and make no
network calls: once the weights are in `models/`, the Hugging Face libraries and Ultralytics are
switched to offline mode (`HF_HUB_OFFLINE=1`, `YOLO_OFFLINE=1`, set in `rora/env.py`; Ultralytics'
usage analytics are off too), and the refine model loads from its local snapshot. Checked by running
every style, SVG, preview and `--refine` with the network blocked.

Output files are named `<photo>__<style>_d<detail>_w<weight>[_<other settings>].png`, e.g.
`1__edges_d0.4_w1.png`. With `--subject auto` or a kind, the subject is added after the photo
name: `1_dog__edges_d0.4_w1.png`, or `1_dog1__...`, `1_dog2__...` when there are several.
Besides detail and weight, only settings that differ from a normal
run's defaults are added, in short form (e.g. `1__fineline_d0.5_w3_f0.7` for `--set freedom=0.7`).
Re-running with the same settings overwrites the same file; different settings make a new one.
The full settings are also embedded in each PNG (text field `dog_lineart_settings`) and SVG
(`<desc>`), and listed in `run.log`.

<details>
<summary>Short names used in file names</summary>

| Short | Setting | Short | Setting | Short | Setting |
|---|---|---|---|---|---|
| `d` | detail | `fl` | form_lines | `col` | --color |
| `w` | weight | `bs` | max_strokes (body strokes) | `px` | --size |
| `inv` | invert | `hs` | head_strokes | `dpi` | --dpi |
| `th` | threshold | `mv` | min_votes | `pad` | --pad |
| `ct` | contrast | `cv` | coverage | `spk` | --min-speck |
| `eps` | epsilon | `tol` | tolerance | `ol` | --outline |
| `bl` | blur | `ms` | min_score | `aa` | --antialias |
| `ml` | min_length | `jg` | join_gap | `ws` | --work-size |
| `cl` | cell | `sm` | smooth | `mc` | --min-conf |
| `a` | angle | `eb` | edge_band | `det` | --detector |
| `sp` | spacing | `f` | freedom | `seg` | --segmenter |
| `dt` | dot | `ft` | features | `rf` | --refine (refined files only) |
| `sd` | seed | `ae` | almond_eyes | `rsd` | --refine-seed |
| `xh` | crosshatch | `pu` | pupils | `rq` | --refine-quantize |
| `lv` | levels | `ns` | nose_solid | `rp` | --refine-prompt (short hash) |
| `lt` | light | `mb` | max_blobs | `as` | --anim-size |
| `rm` | rim | `dk` | darkness | `fps` | --fps |
| `la` | light_angle | `or` | outline_ratio | `mf` | --max-frames |
| `gr` | grain | `tp` | taper | `sty` | --steady |
| | | `ps` | passes | `hl` | --halo |
| | | `wb` | wobble | `obg` | --opaque-background |
| | | `gp` | gaps | | |
| | | `gs` | gap_size | | |

On/off settings appear as just the short name when on (e.g. `aa`), or with `0` when off.
</details>

Each image is cropped to its drawn lines with an even transparent margin (`--pad`, default
3% of the drawing's size).

Everything (virtual env, model weights, caches) stays inside this folder, except the optional
`--refine` model, which goes in the shared Hugging Face cache (see below).
To uninstall, delete the folder.

## Usage

```sh
source env/bin/activate

# one photo, one style
python lineart.py img/in/1.jpg --style fineline

# several styles, with a contact sheet
python lineart.py img/in/1.jpg -s fineline brush dotwork --preview

# every photo in a folder, every tattoo style, plus SVG
python lineart.py img/in -s tattoo --svg --preview

# 5 photos picked at random from a folder (the seed is logged; --sample-seed repeats a pick)
python lineart.py img/in --sample 5 -s fineline

# fewer / more lines, cleaner / more literal lines
python lineart.py img/in/1.jpg -s fineline --detail 0      # just outline + face
python lineart.py img/in/1.jpg -s fineline --detail 0.6
python lineart.py img/in/1.jpg -s fineline --set freedom=0.9

# black ink for light print
python lineart.py img/in/1.jpg -s bold --color black

# list styles and every tunable setting
python lineart.py --list-styles
```

Useful options: `--detail 0..1` (how much is drawn), `--weight N` (line width),
`--set key=value` (any style setting), `--size` / `--dpi` (print size, default
3600 px at 300 DPI = 12"), `--svg`, `--sticker` (WhatsApp sticker, see
[WhatsApp stickers](#whatsapp-stickers)), `--no-preview`, `--debug` (saves each subject's cut-out),
`--subject all|auto|dog|cat|person` (default `all`: every person, cat and dog detected with at
least `--subject-conf` 0.8 confidence, plus the main one, together in one image; `auto`: only
the main subject; a kind: each one of that kind, in its own image).
Run `python lineart.py --help` for the full list.

### GIFs and videos

GIF, MP4 and MOV inputs (files, or in a folder alongside photos) are drawn frame by frame and
written back in the same format, one per style: `clip.gif` -> `clip__edges_d0.5_w2.gif`.
Videos also get a transparent GIF: `clip.mp4` -> `clip__edges_d0.5_w2.mp4` and `.gif`.
`--sticker` adds an animated WhatsApp sticker, `clip__edges_d0.5_w2.webp` (see
[WhatsApp stickers](#whatsapp-stickers)), and `--mp4` an MP4 for GIF and MOV inputs.
Each frame is drawn on the whole frame (no cropping to the subject), so nothing jumps around.
GIFs are transparent with hard edges; MP4 and MOV have no transparency, so the lines are drawn
on the same dark (or, with `--color black`, light) background as the preview.

`--halo` outlines the lines, a few px wide, in that same background colour, so they show on
white as well as dark wherever transparency turns white, as when a GIF is sent on WhatsApp
(the GIF stays transparent; the name gets `_hl`). `--opaque-background` instead draws the GIF
on that background (soft edges with `--antialias`; the name gets `_obg`).

WhatsApp Web (in the browser) turns uploaded GIF files into a still image; it only animates
GIFs from its own GIF search. To send a clip from there, send the MP4 (`--mp4` for GIF
inputs), or make a `--sticker` and add it with the emoji button -> Stickers -> Create,
picking the `.webp`.

```sh
python lineart.py img/in/clip.mp4 -s ink edges
python lineart.py img/in/clip.gif -s fineline --fps 8 --max-frames 40
```

- `--anim-size` (default 720): long side in px; frames are processed and written at this
  size (`--work-size`, `--size`, `--pad` and `--dpi` are for photos only).
- `--fps` (default 12): at most this many frames per second; the rest are dropped and the
  kept ones shown longer, so the clip keeps its length. `--fps 0` draws every frame.
- `--max-frames N`: only the first N frames (after `--fps`), handy for trying settings.
- `--steady` (default 0.5): every frame is detected on its own, so a subject in an odd pose
  can be missed for a few frames and blink out. Detections are followed from frame to frame,
  and a subject found in more than this fraction of the frames is cut out again in the frames
  where it was missed, from its box in the frames around them (the log lists which frames were
  filled in). A cut-out much bigger or smaller than the subject's nearest detection, or on top
  of another subject, is dropped. `--steady 1` turns this off.
- Each frame is drawn on its own, so lines and outlines change a little from frame to frame.
  Frames with no subject are left empty. With `--subject` other than
  `all`, the matching subjects of every frame are drawn together and the name gets
  `_auto`, `_dog`, etc.
- `--refine`, `--svg` and `--debug` are skipped for them. The preview shows the middle frame.
- Each frame takes a few seconds, so a 10-second clip at 12 fps (120 frames) takes a while.
  The video's sound is not kept.

### WhatsApp stickers

`--sticker` also writes each drawing as a WhatsApp sticker: a 512x512 transparent `.webp`
next to the PNG (animated for GIFs and videos). It's off by default.

- The lines are cropped to where they are drawn (across all frames for a clip, so it doesn't
  jump), fitted inside a 16 px margin, and get a thin halo in the background colour so white
  ink still shows on a light chat background (and black ink on a dark one).
- WhatsApp only takes stickers up to 100 KB (500 KB animated) and 10 s. Longer clips are
  trimmed to 10 s; if a file is too big it's saved lossy, then (for clips) with every other
  frame dropped, playing at the input's frame rate (so the sticker is shorter than the clip).
  The log says when either happened, or if it still doesn't fit.
- To send one, share the `.webp` to WhatsApp (e.g. from Files on a phone), or add it to a
  sticker pack with a sticker-maker app.

With no options, `python lineart.py` runs every photo in `img/in/` with:

```sh
-s edges --detail 0.5 --preview \
  --set max_strokes=50 --set head_strokes=100 --set form_lines=100 \
  --set min_votes=2.5 --set coverage=0.1 --set min_score=0 --set freedom=0.4
```

The `--set` defaults tune the tattoo looks; any `--set` you pass overrides them.
Add `--style-defaults` to drop these command-line defaults and use each style's own
defaults instead (listed below and by `--list-styles`).

### Sweeps

Because each combination of settings gets its own file name, you can run several in a row
into one folder and compare them (`out/` is git-ignored).

Every photo with `edges`, across a grid of detail and line weight (12 combinations):

```sh
for d in 0.2 0.4 0.6 0.8; do
  for w in 1 2 3; do
    env/bin/python lineart.py img/in -o out -s edges --detail $d --weight $w --no-preview
  done
done
```

Every photo in every style, each with that style's own defaults:

```sh
env/bin/python lineart.py img/in -o out -s all --style-defaults
```

Each takes roughly 20–25 minutes for about 50 photos on an Apple M-series Mac.
`--no-preview` keeps each run in the loop from overwriting the same `_preview.png`.

## Tests

High-level tests run the whole pipeline once per style, on the demo photo converted to every
input type (JPEG, PNG, HEIC, WebP, TIFF, and a short GIF/MP4/MOV pan), with different options each time.
They take about 2 minutes, need the models from `setup.sh`, and write only to a temp folder:

```sh
env/bin/python -m unittest -v                                          # all
env/bin/python -m unittest -v tests.test_pipeline.StyleTests.test_edges   # one
RORA_KEEP_TEST_OUTPUT=1 env/bin/python -m unittest -v                   # keep the files to look at
```

## Optional: refine with an image-editing model

`--refine` adds a final pass that runs each finished drawing through an image-editing model
(by default [FLUX.2 [klein] 4B](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B),
Apache-2.0), inside this script, to clean it up or restyle it so it clearly reads as its subject
(dog, cat or person). The original is always written; the refined version is written next to it
with `_rf-<mode>` added to the name (e.g. `1__edges_d0.5_w2_rf-clean.png`).

| Mode | What it does |
|---|---|
| `clean` | light touch-up: removes stray marks, joins broken lines, keeps everything else |
| `stylize` | redraws more clearly in the same style and pose (clear outline, eyes, nose, mouth) |
| `auto` | scores each drawing for "looks like a dog / cat / person" (OWLv2); skips good ones, cleans middling ones, stylizes poor ones |

A refined result is thrown away (and logged) if it looks *less* like its subject than the
original. The scores (0–1) are in `run.log`; tune `auto` with `--refine-threshold` (default 0.5).
The result is clipped to the subject's outline, so the model can't add background.

### Models

Pick one with `--refine-model` (each is a separate download):

| `--refine-model` | Model | Licence | Download | Memory (mflux, 4-bit) |
|---|---|---|---|---|
| `klein-4b` (default) | [FLUX.2 [klein] 4B](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B), Jan 2026, 4 steps | Apache-2.0 | 16 GB | ~8 GB (measured) |
| `klein-9b` | [FLUX.2 [klein] 9B](https://huggingface.co/black-forest-labs/FLUX.2-klein-9B), Jan 2026, 4 steps | non-commercial, gated | 35 GB | ~13 GB (estimated) |
| `kontext-dev` | [FLUX.1 Kontext [dev]](https://huggingface.co/black-forest-labs/FLUX.1-Kontext-dev) 12B, May 2025, 28 steps | non-commercial, gated | 34 GB | ~15 GB (estimated) |

Before anything loads, the run estimates the model's peak memory. If it's more than this
machine's RAM minus ~3 GB for the system, it stops with an explanation (heavy swapping can
freeze or crash the machine); `--refine-force` runs it anyway. If it fits but is more than is
free right now, it logs a warning. Gated models need `env/bin/hf auth login` and accepting the
licence on the model's page. Only `klein-4b` has been run here; the others are wired up the same
way but untested.

### Setup (one time)

The packages come with `requirements.txt`; the model is a separate, optional download
(about 16 GB for the default):

```sh
./setup.sh --refine                 # the default model
./setup.sh --refine=kontext-dev     # another one
```

The model goes in the shared Hugging Face cache, `~/.cache/huggingface/hub` (or `$HF_HUB_CACHE`,
or `$HF_HOME/hub`), where mflux, diffusers and other projects look by default, so it is only
downloaded once per machine and is reused if it's already there. To keep it somewhere else (e.g.
an external drive shared by several projects), set `RORA_REFINE_CACHE` to that folder for both
`setup.sh` and runs. To remove it: `rm -rf ~/.cache/huggingface/hub/models--black-forest-labs--FLUX.2-klein-4B`.

Downloads use the operating system's certificate store, so they also work on networks that
inspect HTTPS (e.g. Cisco Secure Access / Umbrella, Zscaler), where Python's own certificate
list would fail with `CERTIFICATE_VERIFY_FAILED`. To fetch just the model without the rest of
`setup.sh` (same files, same place):

```sh
HF_HUB_OFFLINE=0 env/bin/python -c 'from rora import refine; refine.download("klein-4b")'
```

It runs with one of two backends (`--refine-backend`, default `auto`):

| Backend | Used on | Memory |
|---|---|---|
| [mflux](https://github.com/mflux-community/mflux) (MLX) | Macs with Apple silicon | klein 4B: ~8 GB at the default 4-bit, ~12 GB at 8-bit |
| [diffusers](https://github.com/huggingface/diffusers) (PyTorch) | everything else (CUDA GPU; CPU works but is very slow) | klein 4B: about 20 GB (full precision) |

If the package or the model is missing, the run stops right away with these instructions.

With `--refine` a run works in two phases, so the drawing models and the editing model are never
in memory together: first every photo is drawn and its original written, then the drawing models
are released and every drawing is refined with one load of the editing model. With mflux, the
image is encoded and decoded in tiles and the prompt encoder is only loaded while it's needed.
A last pass repairs what the tiles can leave along their 64 px blend bands: strokes broken
across a band are bridged and straight lines along a band are removed (counts in `run.log`).
On a 16 GB Mac a run peaks around 10 GB (while drawing a photo with several subjects), so it
fits next to an editor and a browser. 8-bit has occasionally failed on a busy 16 GB Mac with a
Metal "Impacting Interactivity" error; the original is kept when that happens.

```sh
env/bin/python lineart.py img/in/1.jpg -s fineline --refine auto
env/bin/python lineart.py img/in/1.jpg -s fineline --refine stylize --refine-prompt "thicker outline"
```

Other options: `--refine-seed`, `--refine-quantize` (mflux: 4 (default), 8 or 0 = full precision).
With `--debug`, the image sent to the model and its raw reply are saved too.

## Styles

**Tattoo line art** (`--style tattoo` for all of them): minimal outlines of the main
features (eyes, nose, mouth, ears, feet) with no shading.

| Style | Look |
|---|---|
| `fineline` | thin, even single-needle line |
| `bold` | thick even line, heavier outline |
| `brush` | tapered strokes, like an ink brush |
| `sketchy` | loose, hand-drawn double strokes |
| `dotwork` | lines made of dots |
| `minimal` | fewest strokes, outline with negative-space gaps |

**Other styles**: `stencil`, `ink`, `edges`, `outline` (line/texture), `halftone`,
`stipple`, `engraving` (photo-real shading), `cartoon`, `comic`, `woodcut`, `noir`, `sketch`.

## How the tattoo styles work

1. **Find the subjects**: YOLO detects people, cats and dogs and SAM 2 cuts each one out.
   The biggest, most central one is the main subject; others need at least `--subject-conf`
   confidence. Each subject is drawn on its own (its own face and outline); with the default
   `--subject all` the drawings are then combined into one image.
   YOLO sometimes labels a curled-up dog as a cat; `--subject dog` avoids that.
2. **Find the face**: OWLv2 finds the head, then the eyes, nose and mouth in a zoomed-in
   crop (with wording for the subject's kind); SAM 2 traces each one's exact shape.
3. **Pass 1, many drawings**: the subject (and, separately, the zoomed-in head) is drawn 7
   different ways (classic edge filters plus pretrained line-art models). They disagree
   about fur but agree about real features.
4. **Pass 2, vote**: only lines most drawings agree on are kept, traced into strokes,
   and ranked.
5. **Pass 3, artistic cleanup**: strokes are smoothed into flowing curves, the outline is
   simplified (fur tufts removed), and clutter is removed. `--set freedom=0..1` controls how
   far it may stylise away from the photo.
6. **Draw**: the eyes (traced outline), nose (for dogs, shape + nostrils; for cats and
   people, a line under the nose) and mouth (lip line) are
   drawn deliberately, then everything is rendered in each line look.
   `--set almond_eyes=1` uses a stylised almond eye shape; `--set pupils=1` adds pupils.

Form lines for volume come from a depth model (Depth Anything V2): where one part of the
subject sits in front of another (a leg over the chest, an ear over the head) an overlap line
is drawn, like an artist would.

The outline and face are always drawn. `--detail` only controls the extra lines:
`--detail 0` is just outline + face; higher values add form lines, then legs, ear edges,
folds, brow and muzzle. `--set form_lines=N` sets the number of form lines directly.
`--set features=0` turns face finding off (lines then come from pass 2 alone).

## Style settings reference

Every setting below can be changed with `--set name=value` (`--detail` and `--weight` also
have their own flags). Defaults shown are each style's own (`--style-defaults`). Sizes in
px are for the 2500 px working image and scale automatically with it.

**All styles**

| Setting | Default | Range | Effect |
|---|---|---|---|
| `detail` | 0.5 | 0 – 1 | how much is drawn: 0 = simplest, 1 = busiest |
| `weight` | varies | 1 – 20 px | line thickness |
| `invert` | off | flag `--invert` | tonal styles only (stencil, halftone, stipple, engraving, cartoon, comic, woodcut, noir): put ink where the subject is dark instead of light |

### Line and texture styles

| Style | Setting | Default | Range | Effect |
|---|---|---|---|---|
| `stencil` | `threshold` | 0.5 | 0 – 1 | lower = more of the subject is inked |
| | `contrast` | 3.0 | 0 – 6 | local contrast boost before thresholding (0 = off) |
| | `weight` | 2 | | not used |
| `ink` | `epsilon` | 0.02 | 0 – 0.1 | lower = more, fainter lines |
| | `weight` | 2 | 1 – 10 | line thickness |
| `edges` | `blur` | 1.0 | 0 – 4 | blur before finding edges; higher = fewer, smoother lines |
| | `weight` | 2 | 1 – 10 | line thickness |
| `outline` | `min_length` | 120 | 20 – 400 px | shortest interior line kept |
| | `weight` | 4 | 1 – 20 | line and silhouette thickness |

### Photo-real tonal styles

| Style | Setting | Default | Range | Effect |
|---|---|---|---|---|
| `halftone` | `cell` | 14 | 4 – 60 px | dot grid size (`detail` also shrinks it) |
| | `angle` | 45 | 0 – 90° | screen angle |
| | `contrast` | 2.0 | 0 – 6 | local contrast boost |
| `stipple` | `spacing` | 8 | 4 – 30 px | distance between dots |
| | `dot` | 5 | 2 – 20 px | dot diameter |
| | `seed` | 0 | any integer | different random dot pattern |
| | `contrast` | 2.0 | 0 – 6 | local contrast boost |
| `engraving` | `spacing` | 10 | 4 – 40 px | distance between engraved lines |
| | `angle` | 30 | -90 – 90° | line direction |
| | `crosshatch` | 0.65 | 0 – 1 | brightness where a second, crossing set of lines starts; lower = more crosshatching |
| | `contrast` | 2.0 | 0 – 6 | local contrast boost |

### Cartoony and dramatic styles

| Style | Setting | Default | Range | Effect |
|---|---|---|---|---|
| `cartoon` | `levels` | 3 | 2 – 6 | number of flat tone bands |
| | `weight` | 5 | 1 – 20 | outline thickness |
| `comic` | `levels` | 3 | 2 – 6 | number of flat tone bands |
| | `cell` | 12 | 4 – 60 px | halftone dot size in mid-tones |
| | `angle` | 45 | 0 – 90° | halftone screen angle |
| | `weight` | 4 | 1 – 20 | outline thickness |
| `woodcut` | `spacing` | 16 | 4 – 60 px | distance between gouge lines |
| | `angle` | -20 | -90 – 90° | gouge direction |
| | `light` | 0.7 | 0 – 1 | brightness above which areas are solid ink; lower = more solid |
| | `contrast` | 3.5 | 0 – 6 | local contrast boost |
| `noir` | `light` | 0.62 | 0 – 1 | highlight threshold; lower = more ink |
| | `rim` | 10 | 0 – 40 px | width of the rim light along the outline |
| | `light_angle` | -45 | -180 – 180° | direction the light comes from |
| | `contrast` | 4.0 | 0 – 6 | local contrast boost |
| `sketch` | `grain` | 0.35 | 0 – 1 | higher = grainier, fewer pencil marks |
| | `weight` | 1 | 1 – 5 | pencil line thickness |

### Tattoo looks (`fineline`, `bold`, `brush`, `sketchy`, `dotwork`, `minimal`)

Shared by all six:

| Group | Setting | Default | Range | Effect |
|---|---|---|---|---|
| How many lines | `detail` | 0.5 (`minimal`: 0.2) | 0 – 1 | scales the three counts below when they are `-1` |
| | `form_lines` | -1 (= 10 × detail) | -1, 0 – 100 | max overlap/volume lines from depth (0 = none) |
| | `max_strokes` | -1 (= 24 × detail) | -1, 0 – 200 | max body lines (0 = none) |
| | `head_strokes` | -1 (= 10 × detail) | -1, 0 – 100 | max extra head lines: brow, muzzle, inner ears (0 = none) |
| Line vote (pass 2) | `min_votes` | 3.5 | 1 – 7 | how many of the 7 drawings must agree on a line; lower = more lines |
| | `coverage` | 0.06 | 0.01 – 0.2 | share of each drawing's strongest lines that get a vote |
| | `tolerance` | 6 | 1 – 20 px | how close lines must be to count as the same line |
| | `min_score` | 0.05 | 0 – 1 | drop lines weaker than this fraction of the best one |
| | `min_length` | 50 | 10 – 300 px | shortest line kept |
| | `join_gap` | 14 | 0 – 50 px | join line ends closer than this into one stroke |
| | `smooth` | 2.5 | 0 – 10 px | smoothing of traced lines |
| | `edge_band` | 12 | 0 – 60 px | ignore interior lines this close to the outline |
| Cleanup (pass 3) | `freedom` | 0.7 | 0 – 1 | 0 = no cleanup (literal); 1 = smoothest, simplest, most stylised |
| Face | `features` | 1 | 0 / 1 | find and draw eyes, nose, mouth deliberately |
| | `almond_eyes` | 0 | 0 / 1 | stylised almond eyes instead of the traced eye shape |
| | `pupils` | 0 | 0 / 1 | add pupils with a highlight |
| | `nose_solid` | 0 (`bold`: 1) | 0 / 1 | solid nose with a highlight instead of outline + nostrils (dogs only) |
| Without face finding | `max_blobs` | 0 | 0 – 5 | fill the darkest round blobs (eyes/nose) found in pass 2; for `features=0` |
| | `darkness` | 0.15 | 0.05 – 0.4 | how much darker than its surroundings a blob must be |

Specific to each look:

| Style | Setting | Default | Range | Effect |
|---|---|---|---|---|
| `fineline` | `weight` | 3 | 1 – 20 | line thickness |
| `bold` | `weight` | 9 | 1 – 30 | line thickness |
| | `outline_ratio` | 1.5 | 1 – 3 | outline thickness relative to interior lines |
| `brush` | `weight` | 10 | 1 – 30 | thickest point of each stroke |
| | `taper` | 0.6 | 0.1 – 2 | higher = longer, thinner stroke ends |
| | `seed` | 1 | any integer | different brush-pressure variation along the outline |
| `sketchy` | `weight` | 3 | 1 – 10 | line thickness |
| | `passes` | 2 | 1 – 5 | pen strokes per line |
| | `wobble` | 4 | 0 – 20 px | how far each pass wanders off the line |
| | `seed` | 2 | any integer | different wobble |
| `dotwork` | `weight` | 6 | 2 – 20 | dot diameter |
| | `spacing` | 2.2 | 1 – 5 | gap between dots, in dot diameters |
| `minimal` | `weight` | 4 | 1 – 20 | line thickness |
| | `gaps` | 6 | 0 – 20 | number of breaks in the outline |
| | `gap_size` | 0.025 | 0 – 0.1 | length of each break, as a share of the outline |
| | `seed` | 3 | any integer | different break positions |

## Project layout

| Path | What |
|---|---|
| `lineart.py` | command-line entry point |
| `rora/` | pipeline package (modules below) |
| `rora/segment.py` | finds the people, cats and dogs and cuts them out |
| `rora/features.py`, `rora/facedraw.py` | find and draw eyes, nose, mouth |
| `rora/depth.py` | depth estimate and form lines (volume) |
| `rora/edgemaps.py`, `rora/strokes.py`, `rora/cleanup.py` | tattoo passes 1, 2 and 3 |
| `rora/tattoo.py`, `rora/styles.py` | line looks and other styles |
| `rora/export.py` | PNG / SVG / preview output |
| `rora/sticker.py` | WhatsApp sticker (`.webp`) output |
| `rora/refine.py` | optional `--refine` pass (FLUX.2 [klein] 4B via mflux or diffusers) |
| `rora/env.py` | keeps model downloads and settings in `./models` |
| `setup.sh` | one-time setup |

Not in git (recreated by `setup.sh` or at run time): `env/`, `models/`, `img/out/`, `out/`, `lab/`.

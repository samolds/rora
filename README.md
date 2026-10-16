# rora

Turn photos of a dog into print-ready line art: a transparent PNG (and optional SVG)
of just the dog, drawn in white lines for printing on dark shirts.

The pipeline finds the dog (ignoring people and background), cuts it out, and draws it
in one of several styles, including minimalist tattoo-style line art.

## Getting started

Requires Python 3.10+ (tested with 3.12) and about 3 GB of disk for dependencies and models.
Works on macOS (Apple GPU), Linux/Windows (CUDA GPU) or plain CPU.

```sh
./setup.sh                                                        # one time: venv, dependencies, model weights
.venv/bin/python dog_lineart.py img/in/1.jpg --style tattoo --preview   # try the included demo photo
```

Then add your own photos to `img/in/` and run `.venv/bin/python dog_lineart.py --preview`.
Results go to `img/out/`, including `_preview.png` (every result on a dark background) and `run.log`.

Everything (virtual env, model weights, caches) stays inside this folder.
To uninstall, delete the folder.

## Usage

```sh
source .venv/bin/activate

# one photo, one style
python dog_lineart.py img/in/1.jpg --style fineline

# several styles, with a contact sheet
python dog_lineart.py img/in/1.jpg -s fineline brush dotwork --preview

# every photo in a folder, every tattoo style, plus SVG
python dog_lineart.py img/in -s tattoo --svg --preview

# fewer / more lines, cleaner / more literal lines
python dog_lineart.py img/in/1.jpg -s fineline --detail 0      # just outline + face
python dog_lineart.py img/in/1.jpg -s fineline --detail 0.6
python dog_lineart.py img/in/1.jpg -s fineline --set freedom=0.9

# black ink for light shirts
python dog_lineart.py img/in/1.jpg -s bold --color black

# list styles and every tunable setting
python dog_lineart.py --list-styles
```

Useful options: `--detail 0..1` (how much is drawn), `--weight N` (line width),
`--set key=value` (any style setting), `--size` / `--dpi` (print size, default
3600 px at 300 DPI = 12"), `--svg`, `--preview`, `--debug` (saves the dog cut-out).
Run `python dog_lineart.py --help` for the full list.

## Styles

**Tattoo line art** (`--style tattoo` for all of them): minimal outlines of the main
features (eyes, nose, mouth, ears, feet) with no shading.

| Style | Look |
|---|---|
| `fineline` | thin, even single-needle line (default) |
| `bold` | thick even line, heavier outline |
| `brush` | tapered strokes, like an ink brush |
| `sketchy` | loose, hand-drawn double strokes |
| `dotwork` | lines made of dots |
| `minimal` | fewest strokes, outline with negative-space gaps |

**Other styles**: `stencil`, `ink`, `edges`, `outline` (line/texture), `halftone`,
`stipple`, `engraving` (photo-real shading), `cartoon`, `comic`, `woodcut`, `noir`, `sketch`.

## How the tattoo styles work

1. **Find the dog**: YOLO detects each dog, SAM 2 cuts it out (one output file per dog).
2. **Find the face**: OWLv2 finds the head, then the eyes, nose and mouth in a zoomed-in
   crop; SAM 2 traces each one's exact shape.
3. **Pass 1, many drawings**: the dog (and, separately, the zoomed-in head) is drawn 7
   different ways (classic edge filters plus pretrained line-art models). They disagree
   about fur but agree about real features.
4. **Pass 2, vote**: only lines most drawings agree on are kept, traced into strokes,
   and ranked.
5. **Pass 3, artistic cleanup**: strokes are smoothed into flowing curves, the outline is
   simplified (fur tufts removed), and clutter is removed. `--set freedom=0..1` controls how
   far it may stylise away from the photo.
6. **Draw**: the eyes (traced shape + pupil + highlight), nose (shape + nostrils) and mouth
   (lip line) are drawn deliberately, then everything is rendered in each line look.
   `--set almond_eyes=1` uses a stylised almond eye shape instead.

The outline and face are always drawn. `--detail` only controls the extra lines:
`--detail 0` is just outline + face, `--detail 1` adds legs, ear edges, folds, brow and muzzle.
`--set features=0` turns face finding off (lines then come from pass 2 alone).

## Project layout

| Path | What |
|---|---|
| `dog_lineart.py` | command-line entry point |
| `segment.py` | dog detection and cut-out |
| `features.py`, `facedraw.py` | find and draw eyes, nose, mouth |
| `edgemaps.py`, `strokes.py`, `cleanup.py` | tattoo passes 1, 2 and 3 |
| `tattoo.py`, `styles.py` | line looks and other styles |
| `export.py` | PNG / SVG / preview output |
| `env.py` | keeps model downloads and settings in `./models` |
| `setup.sh` | one-time setup |

Not in git (recreated by `setup.sh` or at run time): `.venv/`, `models/`, `img/out/`, `lab/`.

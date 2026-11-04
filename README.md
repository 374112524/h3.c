# h3-metal

Native MiniMax-H3 inference for Apple Silicon. The project is being built as a
sequence of working vertical slices: deterministic host/model metadata first,
then portable Metal block parity, prompt encoding, prompt-to-video/audio, and
first/last-frame conditioning and then ordered references.

Prompt-to-video/audio, first/last-frame conditioning, and ordered Ref2VA
image/video/audio references work end to end. The current work is incremental
H3-specific Metal performance and memory optimization on M3 Max and M5 Max.

## Tutorial

### 1. Build and inspect the model

The examples assume that the Hugging Face snapshot is in `./MiniMax-H3` and
that FFmpeg and FFprobe are available on `PATH`.

```sh
make -j8
mkdir -p outputs
./h3 --info -d ./MiniMax-H3
```

`--info` checks the model layout and prints the selected Metal device without
mapping all weights or generating media. Run `./h3 --help` for the complete CLI
reference.

Without `-p`, the same binary starts an Iris-style interactive session:

```sh
./h3 -d ./MiniMax-H3 --width 512 --height 512 --steps 6
```

Type a prompt to generate a numbered video. The session keeps the exact BF16
prompt conditioning, prepared DiT, and video decoder in memory, so repeating a
prompt with another seed avoids loading and encoding them again. Useful commands
are `!status`, `!seed random`, `!seconds 2`, `!show`, `!save output.mp4`, and
`!cache`. Use `!help` for the full, short list.

First/last-frame conditioning is persistent in the session:

```text
h3> !first opening.png
h3> !last ending.png
h3> The camera moves slowly around the subject.
```

Use `!first clear` or `!last clear` to remove an anchor. Generated videos are
written to the session directory printed at startup.

For a general Ref2VA conditioning image, use `!ref-image PATH` instead. Images
are appended in order and exposed to the model as `<Picture 1>`, `<Picture 2>`,
and so on; filenames have no meaning to the model.

```text
h3> !ref-image person.png
h3> Make the person shown in Picture 1 wave to the camera.
```

`!refs` lists the current order, `!ref-remove N` removes one entry, and
`!refs clear` removes them all. Ref2VA references cannot be mixed with
`!first`/`!last` anchors.

### 2. Make a first fast video

Start with the validated balanced preset. It generates 22 frames at 24 fps
(about 0.92 seconds), displays the evolving middle-video frame after every
denoising transition in a supported graphical terminal, and prints phase
timings:

```sh
./h3 --profile \
  -d ./MiniMax-H3 \
  -p "A red fox walks through fresh snow in a pine forest. Medium tracking shot, natural winter light, realistic fur, soft footsteps and wind." \
  --width 512 --height 512 \
  --frames 22 --steps 20 \
  --layers 45 --reuse 2 \
  --show \
  -o outputs/fox-fast.mp4
```

This is deliberately not the most aggressive configuration:

- `--steps 20` performs the default 20 denoising passes.
- `--reuse 2` computes 11 fresh denoiser velocities instead of all 20 and
  extrapolates the skipped transitions.
- `--layers 45` runs 45 of the 50 transformer blocks, reducing both time and
  unified-memory use.
- `--show` is optional. It supports Kitty/Ghostty and
  iTerm2/WezTerm/Konsole graphical protocols. It loads a resident preview VAE,
  displays one representative middle-video frame after every Euler transition,
  and then displays all final frames. Display dimensions default to 2x so the
  image has its intended logical size on macOS Retina screens; use `--zoom 1`
  on a non-HiDPI display. This adds preview decode time and roughly 10 GiB of
  temporary model residency; runs without `--show` are unchanged.
- `--profile` is optional and does not select a different generation path.

The first process invocation also pays model loading and filesystem-cache
costs. Compare performance using repeated runs, and alternate variants when
the machines are warming up because this workload is sensitive to thermal
throttling.

For a very short iteration, request four denoising passes directly:

```sh
./h3 --profile \
  -d ./MiniMax-H3 \
  -p "A red fox walks through fresh snow in a pine forest. Medium tracking shot, natural winter light, realistic fur." \
  --width 512 --height 512 --frames 22 \
  --steps 4 --layers 50 --reuse 1 \
  --show \
  -o outputs/fox-four-step.mp4
```

`--steps N` always means exactly N denoising passes. Four through seven passes
use the same schedule that won the low-budget comparison; increasing from 4
to 7 progressively improves detail and motion. Keep `--reuse 1` at such small
budgets so every requested pass runs the model. `--show` displays one preview
after each pass.

Several tail-heavy schedules were evaluated because most visible cleanup
happens late in a long run. They preserved too few early composition updates
and produced woven texture, weak motion, or clipped colors. The retained mode
uses the released linear base grid with one terminal point. On the 512-square,
22-frame fox test, the selected four-pass result had 0.556 full-video SSIM
against a 29-pass reference; an independent surfer test measured 0.547. The
four-pass denoise took about 3.5 seconds on M5 Max, versus 26.4 seconds for the
reference.

For a low-memory run, add `--ssd-streaming`:

```sh
./h3 --profile \
  -d ./MiniMax-H3 \
  -p "A red fox walks through fresh snow in a pine forest." \
  --width 512 --height 512 --frames 22 --steps 20 \
  --layers 50 --reuse 1 --ssd-streaming \
  -o outputs/fox-ssd.mp4
```

This uses the original BF16 checkpoint without conversion or quantization. It
keeps two DiT blocks in memory and reads the next block from SSD while the GPU
runs the current one. On M5 Max, tracked DiT storage fell from about 36.5 GiB to
2.0 GiB at 512 square and 2.1 GiB at 864x480. A warm 50-block forward measured
1.35 versus 2.49 seconds at 512 square (84% slower), and 2.14 versus 2.68
seconds at 864x480 (26% slower). These are comparisons against the same
full-residency BF16 path, and the results were byte-identical in both checks.

The 2.0--2.1 GiB figure is the DiT's tracked tensor storage, not total system
RAM. Prompt encoding and the two VAEs run in separate phases rather than adding
their full peaks to it; the OS, media buffers, and output resolution still need
headroom. `--show` keeps a preview VAE resident and adds roughly 10 GiB, so omit
it for the lowest-memory run.

SSD streaming is an explicit memory/speed tradeoff and is not the default. It
cannot be combined with `--use-int8-row-fc2`. In an interactive session, use
`!ssd-streaming on`.

### 3. Move toward reference quality

Change one control at a time when evaluating quality. First restore all layers,
then all denoiser evaluations, and finally raise the default 20-pass schedule
to the slower 50-pass reference:

```sh
./h3 --profile \
  -d ./MiniMax-H3 \
  -p "A red fox walks through fresh snow in a pine forest. Medium tracking shot, natural winter light, realistic fur, soft footsteps and wind." \
  --width 512 --height 512 \
  --frames 22 --steps 50 \
  --layers 50 --reuse 1 \
  -o outputs/fox-close.mp4
```

The defaults are `--steps 20 --layers 50 --reuse 1`; keep `--steps 50`
explicit for this close path. It performs 50 complete 50-block denoiser
forwards and is much more expensive than the default, but is the right oracle
when a fast mode changes the subject, anatomy, motion, or composition.
Numerical pixel identity with MLX is not expected because the random-number and
execution engines differ; the depicted content and motion should agree.

### 4. Choose a speed/quality preset

These controls are independent unless noted otherwise:

| Control | Slow reference | Default | Aggressive | Main impact |
|---|---:|---:|---:|---|
| Denoising passes | `--steps 50` | `--steps 20` | `--steps 4..7` | The number always names actual denoising passes. |
| Whole denoiser reuse | `--reuse 1` | `--reuse 2` | `--reuse 3` | At 20 steps: 20, 11, or 8 fresh DiT evaluations. |
| Active DiT blocks | `--layers 50` | `--layers 45` | `--layers 40` | Fewer blocks reduce compute and resident transformer weights. |
| Core residual reuse | `--core-reuse 1` | `--core-reuse 4` | `--core-reuse 6` | Refreshes patch/head work every step but runs the expensive core less often. |
| Token reduction | off | optional | `--token-reduction` | Pairs horizontal video tokens inside middle blocks; faster but may change composition. |
| Internal canvas | output size | `384x384` for 512 square output | `320x320` | Runs DiT/VAE smaller, then upscales with vImage. |

On M5, `--use-int8-row-fc2` uses one activation scale per FC2 row and a single
full-width TensorOps product. It is optional because it is less numerically
conservative than grouped int8. It reduced complete denoiser forwards by about
2.6% in reciprocal tests. Matched four-step fox and surfer videos kept the same
subjects, setting, and motion (full-video SSIM 0.919 and 0.828). In the
interactive session, use `!int8-row-fc2 on`.

`--reuse` and `--core-reuse` are mutually exclusive. Layer thinning can be
combined with either one.

To make the first command faster while keeping its output resolution, add
token reduction:

```sh
./h3 --profile \
  -d ./MiniMax-H3 \
  -p "A surfer riding inside a sharp blue ocean wave, one rider and one white board, realistic spray." \
  --width 512 --height 512 --frames 22 --steps 20 \
  --layers 45 --reuse 2 --token-reduction \
  -o outputs/surfer-fast.mp4
```

At the validated 512 square shape, token reduction cut the `45 layers + reuse
2` denoise profile from 16.69 to 12.60 seconds on the IT M5 Max. Independent
fox and surfer renders stayed coherent, but composition can diverge more from
the close path.

For an aggressive preview, render internally at 320 square and upscale to the
requested 512 square output:

```sh
./h3 --profile \
  -d ./MiniMax-H3 \
  -p "A red fox walking through snow, realistic, tracking shot." \
  --width 512 --height 512 \
  --render-width 320 --render-height 320 \
  --frames 22 --steps 20 --layers 40 --reuse 3 \
  -o outputs/fox-aggressive.mp4
```

This combination produced a clean, recognizable 22-frame fox in validation,
but loses fine detail and can change framing. Do **not** add `--token-reduction`
to both `--layers 40` and `--reuse 3`: that tested combination produced color
ringing, outlines, and ghosted limbs.

As an alternative to whole-velocity reuse, this keeps the timestep-dependent
patch and output heads fresh at every transition:

```sh
./h3 --profile \
  -d ./MiniMax-H3 \
  -p "A surfer riding a blue ocean wave." \
  --width 512 --height 512 --frames 22 --steps 20 \
  --layers 45 --core-reuse 4 \
  -o outputs/surfer-core-reuse.mp4
```

Use `--core-reuse 6` only as an aggressive preview. Values above 6 are not
exposed because validation lost subject fidelity.

### 5. Pick resolution and duration

Width and height must each be multiples of 32, at least 32, and their product
must not exceed `768 * 1344` pixels. Those are mechanical limits, not a promise
that every tiny canvas has good model quality. H3-Base is a 768p model.

| Canvas | Current guidance |
|---|---|
| `512x512` | Safest development size; repeatedly validated with multiple prompts. |
| `768x768` | Validated close-quality square output; substantially more expensive. |
| `1344x768`, `768x1344` | Released 768p-class landscape/portrait limit. |
| `1024x768`, `768x1024` | Valid 4:3 and 3:4 768p-class canvases. |
| `384x384` internal to `512x512` | Validated fast-quality scaling point. |
| `320x320` internal to `512x512` | Validated aggressive scaling point. |
| `256x256` | Native fast-preview canvas with automatic low-resolution RoPE adaptation. |

For a fast native 256-square preview:

```sh
./h3 -d ./MiniMax-H3 \
  -p "A red fox walks through fresh snow in a pine forest." \
  --width 256 --height 256 \
  --frames 22 --steps 20 \
  --layers 50 --reuse 1 \
  -o outputs/fox-256.mp4
```

At 256 square, H3 has only an `8x8` effective spatial-token grid, so it has less
room for fine detail and complex composition. H3 automatically halves spatial
RoPE coordinates at exactly 256 square. This removed repeating lattice
artifacts in long fox renders and stayed coherent on an independent portrait,
without adding tokens or runtime. Use `--use-reference-rope` to restore the
released/MLX coordinates for parity checks. Keep token reduction off at this
size. Native 128 square remains unsupported: its `4x4` token grid did not
recover a recognizable subject even with adjusted RoPE.

`--render-width` and `--render-height` must be set together, must have the same
aspect ratio as the output, and cannot exceed the output dimensions. The model
and VAE use the internal size; terminal frames and the encoded video retain the
requested output size.

H3 emits 24 fps and aligns frame requests upward to `5 + 17*n`:

Use `--seconds N` for a duration-oriented request, or `--frames N` for direct
frame control; the two options are mutually exclusive. Fractional seconds are
accepted. Seconds are converted at 24 fps and then rounded upward to the next
legal H3 temporal shape, so `--seconds 10` produces 243 frames (10.125 seconds).

| Frames | Approximate video duration |
|---:|---:|
| 22 | 0.917 seconds |
| 39 | 1.625 seconds |
| 56 | 2.333 seconds |
| 107 | 4.458 seconds |
| 243 | 10.125 seconds |
| 362 | 15.083 seconds |

Short clips are useful for development. The released workflow is intended for
roughly 4–15 second videos. A request such as `--frames 23` is rounded up to 39
frames rather than producing an arbitrary temporal shape.

### 6. Improve the prompt

A short prompt works, but the released system expects a Context-IR-like
description. State the subject, action, setting, camera, lighting/style, and
desired sound. For example:

```text
Scene: a single red fox in a snow-covered pine forest at dawn.
Action: the fox walks steadily left to right and looks toward the camera once.
Camera: medium-height lateral tracking shot, 50 mm lens, stable framing.
Look: photorealistic fur, cold blue ambient light, warm sunrise rim light.
Audio: soft footsteps in snow, light wind through pine branches, no music.
```

Keep identity and object counts explicit when they matter. `--seed N` controls
the native random stream; the default is 42. Compare options with the same
prompt, seed, resolution, frame count, and step count.

### 7. Preview frames and diagnose performance

- `--show` displays a representative frame after every denoising transition,
  followed by all frames from the completed video. Like Iris, it advertises 2x
  display dimensions by default for Retina terminals; `--zoom N` changes that
  factor without resizing the generated video or the encoded terminal image.
- `--frames-dir DIR` writes final callback frames as PPM files. Intermediate
  `--show` previews are not written there.
- `-o ''` disables MP4 encoding; combine it with `--frames-dir` when FFmpeg is
  unavailable.
- `--profile` reports phase wall time, Metal encoding/wait time, peak live
  tensor storage, cumulative allocation, and dispatch counts.

For example:

```sh
./h3 --profile -d ./MiniMax-H3 -p "A hummingbird hovering over red flowers." \
  --width 512 --height 512 --frames 22 --steps 20 \
  --layers 45 --reuse 2 --frames-dir outputs/hummingbird-frames \
  -o ''
```

### 8. Add image, video, and audio references

First/last-frame anchors select the FL2VA path:

```sh
./h3 -d ./MiniMax-H3 -p "The fox keeps walking through the snow." \
  --width 512 --height 512 --frames 22 --steps 20 \
  --layers 45 --reuse 2 \
  --first-frame fox.png --last-frame fox-later.png \
  -o outputs/fox-anchored.mp4
```

Ordered references select the distinct Ref2VA checkpoint. Use the flag matching
the media semantics:

```sh
# One image reference.
./h3 -d ./MiniMax-H3 -p "Use the animal and setting in the reference." \
  --width 512 --height 512 --frames 22 --steps 20 \
  --ref-image fox.png -o outputs/fox-reference.mp4

# Continue a clip but ignore its soundtrack.
./h3 -d ./MiniMax-H3 -p "Continue the motion in this clip." \
  --width 512 --height 512 --frames 22 --steps 20 \
  --ref-silent-video fox.mp4 -o outputs/fox-video-reference.mp4

# Preserve the clip's embedded audio.
./h3 -d ./MiniMax-H3 -p "Continue this audiovisual scene." \
  --width 512 --height 512 --frames 56 --steps 20 \
  --ref-video fox-with-audio.mp4 -o outputs/fox-video-audio.mp4

# Replace a video's soundtrack explicitly.
./h3 -d ./MiniMax-H3 -p "Continue the scene with the supplied music." \
  --width 512 --height 512 --frames 56 --steps 20 \
  --ref-video-audio silent-fox.mp4 replacement.wav \
  -o outputs/fox-replaced-audio.mp4

# An ordered image plus standalone audio reference.
./h3 -d ./MiniMax-H3 -p "Use the animal and music from the references." \
  --width 512 --height 512 --frames 56 --steps 20 \
  --ref-image fox.png --ref-audio music.wav \
  -o outputs/fox-image-audio.mp4
```

Reference flags may be repeated and their command-line order is preserved.
Standalone audio must accompany an image or video reference. Audio references
must be 2–15 seconds; at most three audio inputs are accepted and their total
decoded duration is capped at 15 seconds.

### 9. Apply a style LoRA

A style LoRA is a safetensors checkpoint with low-rank updates for the
diffusion transformer. The engine merges the updates into the released BF16
DiT weights before the int8 quantization pass, so every generation path
(resident weights, SSD streaming, BF16 fallbacks) uses the adapted model.

```sh
./h3 -d ./MiniMax-H3 -p "A lighthouse in a storm." \
  --lora style/anime-film.safetensors --lora-scale 0.8 \
  --width 512 --height 512 --frames 22 --steps 20 \
  -o outputs/lighthouse-style.mp4
```

`--lora-scale` multiplies the checkpoint alpha and defaults to `1.0`. The
effective strength is `alpha / rank * scale`, where `alpha` comes from the
safetensors metadata (`alpha`, `network_alpha` or `lora_alpha`) and falls back
to the rank when the file carries no metadata. The interactive session offers
the same control with `!lora PATH`, `!lora clear` and `!lora-scale N`; changing
the checkpoint rebuilds the prepared transformer.

Supported adapter names follow the released checkpoint layout:

| Adapter path | Adapted matrix |
| --- | --- |
| `blocks.N.attn.qkv_proj` | fused attention QKV projection |
| `blocks.N.attn.out_proj` | attention output projection |
| `blocks.N.mlp.fc1` | fused MLP gate and up projection |
| `blocks.N.mlp.fc2` | MLP down projection |
| `blocks.N.attn.to_q` / `to_k` / `to_v` | one third of the fused QKV matrix |
| `blocks.N.mlp.gate_proj` / `up_proj` | one half of the fused FC1 matrix |

Common naming conventions are accepted: `lora_A`/`lora_B`, `lora_down`/
`lora_up`, and the `transformer.`, `base_model.model.`, `diffusion_model.` and
`model.` prefixes are stripped. Adapters that do not match the checkpoint
shapes are reported and skipped instead of failing the run.

### 10. Apply a step-acceleration LoRA

Step-acceleration LoRAs (LightX2V, CausVid, Self-Forcing style) are trained
for a small number of denoising steps, so they need the schedule they were
trained with. The engine reads that schedule from the checkpoint metadata and
adopts it automatically:

```sh
./h3 -d ./MiniMax-H3 -p "A lighthouse in a storm." \
  --lora accel/lightx2v-h3.safetensors \
  --width 512 --height 512 --frames 22 -o outputs/lighthouse-fast.mp4
```

The metadata keys `steps` (or `num_steps`, `acceleration_steps`) and `shift`
(or `sigma_shift`, `flow_shift`) carry the recommended values. The engine
prints what it adopted:

```text
h3: LoRA accel/lightx2v-h3.safetensors: 50 adapters, alpha 16, scale 1
h3: LoRA requests 4 denoising steps
h3: LoRA requests a 7 flow shift
```

`--lora-steps N` forces a step count and `--lora-shift S` forces a flow
shift. An explicit `--steps N` keeps the requested step count instead of the
checkpoint recommendation. The interactive session offers the same controls
with `!lora-steps N` and `!lora-shift N`.

The merge itself is a single fused Metal pass that adds the low-rank update to
the BF16 weight in place, so it costs one read and one write of the target
matrix. In `--ssd-streaming` mode the merge repeats with every streamed layer
read, which is the price of keeping only two block slots in memory.

## Tests and runtime requirements

```sh
make test
make parity
```

`make test` runs the deterministic host suite and, when the ignored MLX fixture
is installed under `misc/fixtures/`, compiles the Metal source at runtime and
checks a complete toy H3 block against named MLX outputs. Runtime compilation is
intentional: it follows Iris and does not require Xcode's optional offline Metal
toolchain. The test covers both an F32 diagnosis path and the production BF16
storage path; wide BF16 matrix products and SDPA use cached MPSGraph graphs, with
direct Metal correctness fallbacks. `make parity` runs only those Metal/MLX
checks.

FFmpeg and FFprobe must be available on `PATH` for media inputs and MP4 output
(`H3_FFMPEG` and `H3_FFPROBE` may select explicit executables). Generated RGB24 and
32 kHz stereo F32 PCM are fed through concurrent pipes; no intermediate
uncompressed media file is created.

## Implementation and performance notes

The remainder documents the implementation behind the tutorial presets and the
environment variables retained for exact A/B diagnosis.

### Sampler and DiT controls

The default sampler uses the released shifted video/audio schedule. `--steps`
always names the number of denoising passes, with terminal zero added after the
last pass. Whole-denoiser reuse evaluates the first and last pass plus every
requested interval, then extrapolates skipped video and audio velocities on
their independent schedules. With very small step counts, keep `--reuse 1`.

For the low-budget path, the released linear base grid won against
actual-video-sigma linear spacing,
quadratic and cubic warps, exact 30-point tail subsets, mild power warps,
zero-order held full-grid velocities, linear velocity extrapolation, and RES.
The more tail-heavy candidates often sharpened the subject but damaged motion
or left a repetitive woven background; sparse RES and long extrapolation
intervals failed much more visibly.

Layer thinning ranks the checkpoint's actual AdaLN gates while protecting
structurally important first and final blocks. Unused weights and schedule
tensors are not retained, so `--layers 45` and `--layers 40` reduce both
transformer time and unified-memory use. Core reuse holds the previous full
transformer residual while refreshing the patch projection and timestep-aware
head; it remains mutually exclusive with whole-velocity reuse.

### Exact DiT fusions

Every active DiT block fuses its attention residual gate with the following MLP
AdaLN. The rounded BF16 residual is still written exactly, but the same row is
kept in threadgroup memory for normalization, eliminating one dispatch and one
global reread. Away from token-reduction boundaries, the MLP residual gate also
produces the next block's attention AdaLN and carries that normalized state
across the loop. `H3_DISABLE_FUSED_GATE_ADALN=1` and
`H3_DISABLE_FUSED_CROSS_BLOCK_ADALN=1` restore the two-kernel oracles.
The final audio/video AdaLN kernels bind directly to offsets in the residual
stream, avoiding two slice blits and 18.8 MiB of scratch at 512x512 (29.4 MiB
at the 864-class benchmark shape).
`H3_DISABLE_FUSED_FINAL_SLICE=1` restores the copy-plus-AdaLN oracle at load.
The BF16 final heads then apply AdaLN while loading their 16x16 projection
tiles, preserving the standalone rounding and accumulation order while
removing another equally sized normalized activation. The two optimizations
together save 37.5/58.9 MiB. `H3_DISABLE_FUSED_FINAL_HEAD=1` restores the
offset-AdaLN-plus-linear oracle at load.

### Token-reduction internals

`--token-reduction` is an independent aggressive DiT mode. After block 3 it
pairs adjacent horizontal target-video tokens while leaving text, audio,
conditions, and reference tokens exact. The complete full-resolution state is
kept as a bypass. During the first ten noisy evaluations it restores before
block 40; subsequent detail-forming evaluations restore before block 30. Each
token returns as its original value plus the update learned by its pair, so
within-pair detail is not discarded.
The pooling kernel writes only true-pair baselines into a dense tail of the
already allocated attention scratch buffer; odd-width singleton tokens need no
baseline. The full bypass uses the oversized QKV tail when it fits, with a
guarded dedicated fallback only for reference-heavy layouts. Common text-only
canvases therefore add no activation arena at any token-grid width. Pooling
also snapshots both source tokens while their BF16 values are already in
registers, avoiding a separate full-hidden blit and redundant source read. The
same entry kernel keeps each pooled row in threadgroup memory and emits the
first reduced block's attention AdaLN, eliminating another global residual read.
At the restore boundary, the first full-resolution attention AdaLN is fused
into expansion: a 10.5 KiB threadgroup row avoids a global residual reread while
still writing the exact bypass needed by the following residual branch.
On a thermal-balanced 512x512x22, 19-forward IT M5 Max A/B this reduced denoise
time from 39.13 to 28.06 seconds (28.3%). Final video/audio latent relative L2
was 5.56%/15.14%. First/middle/last fox frames retained one clean muzzle,
coherent legs, and sharp fur; an independent surfer remained consistent with
one rider and board through the wave spray. It changes composition and is
therefore opt-in rather than the close-reference default.
`H3_TOKEN_REDUCTION_BLOCKS` can override the later `4:30` interval;
`H3_TOKEN_REDUCTION_EARLY=STEPS:END` overrides the early schedule and `0`
disables it. `H3_DISABLE_TOKEN_REDUCTION=1` provides an in-context exact oracle.
`H3_DISABLE_FUSED_TOKEN_POOL_ADALN=1` and
`H3_DISABLE_FUSED_TOKEN_ADALN=1` independently restore the two-kernel entry and
exit boundaries for diagnosis.

`H3_TOKEN_REDUCTION_ATTENTION_ONLY=1` replaces the pooled stack with
attention-only merging. The residual stream, attention residual gate, and MLP
stay on the full token grid; each reduced block pools the current residual,
applies attention AdaLN to the pooled row, runs QKV/SDPA/out-projection on the
paired grid, broadcasts that branch to every full-resolution child, and then
gates the full residual and runs the MLP on full tokens. The branch broadcast
uses a dedicated `h3_token_gather_bf16` kernel over the existing parent map, so
text, audio, condition, and reference rows remain exact. Because the full
stream is live, the pool runs at the start of every reduced block and no
full-grid AdaLN may be fused across a reduced-block boundary.
On 864x480x56 turbo8 8@7 the 4:30 stack cut denoise from 204.8 to 133.5 s with
SSIM 0.689 and PSNR 17.78 dB against the untouched run; full token reduction on
the same schedule reached 107.5 s at 0.671 / 16.69 dB. The full-token MLP keeps
noticeably more fur detail, at a smaller speedup. At 12@16 the same stack took
190.4 s against 233.2 s base and 166.7 s for full reduction (SSIM 0.714 versus
0.754), but every `--shift 16` configuration tested is run-to-run
nondeterministic: identical runs land on one of two outputs that differ by
0.75-0.85 SSIM, so those two numbers are single draws. The `--shift 7` 8-step
turbo recipe is bit-identical across repeated runs and carries every other
number in this section. It adds one full-width residual snapshot and one branch
buffer (2 x `sequence * HIDDEN` BF16) and forces the snapshot out of the QKV
tail.

`H3_TOKEN_REDUCTION_ATTENTION_ONLY=kv` is the narrower variant: QKV, the
attention residual, output projection, and the MLP all stay full-resolution and
only the SDPA key/value set is pooled right after RoPE. Queries keep their own
rows, so no branch broadcast is needed; `h3_token_pool_avg_bf16` averages the
paired rows in the projection's layout (head-major or row-major) and
`h3_gpu_sdpa_bf16_split` runs MPSGraph SDPA with unequal query/key lengths.
On 864x480x56 turbo8 8@7 this cut denoise from 204.8 to 161.6 s; on 1280x704x56
it took 486.5 s against 610.0 s untouched and 433.1 s for the broadcast variant,
with SSIM 0.680 versus 0.653. High-frequency energy stays at 0.9-1.1x the
untouched render for both attention-only modes but falls to ~0.7x under full
token reduction, confirming that the full-token MLP is what preserves detail.
Relative SSIM flips between canvases because 8-step trajectories drift; neither
attention-only mode is an exact win.
Token reduction composes cleanly with the validated `--layers 45 --reuse 2`
settings: on the same 512 benchmark it reduced that profile from 16.69 to
12.60 seconds (24.5% marginal), and independent fox and surfer renders stayed
coherent. Do not combine it with both `--layers 40` and `--reuse 3`; that
6.47-second experiment produced chromatic ringing and ghosted limbs despite
acceptable latent norms.

### Internal canvas and video VAE

`--render-width` and `--render-height` run the model and VAE on a lower
same-aspect internal canvas, then high-quality vImage-scale RGB frames to the
requested output size before callbacks, terminal display, and encoding. This is an
explicit quality/speed tradeoff: a measured 384-to-512 prompt render reduced
M5 DiT time by 33% and video-VAE time by 18% while retaining a clean,
recognizable photorealistic result. Both values must be multiples of 32; the
exact output canvas remains the default.
For square 512 output, 384 is the fast-quality point and 320 is the validated
aggressive point. The latter produced a coherent walking fox and repeated at
8.02 seconds of DiT versus about 15.82 seconds natively. Native 256 uses the
same-cost spatial-RoPE adaptation described above; it remains a fast composition
preview rather than a substitute for a 512- or 768-class final render.
The video VAE automatically chooses a 256-320 pixel spatial tile from the
requested canvas geometry, minimizing repeated overlap work while keeping peak
storage bounded. `H3_VAE_TILE_PIXELS=256` restores the original conservative
tile plan for close-reference diagnosis.

### Weight residency and streamed prompt encoding

On M5-class GPUs, persistent transformer weights are mapped directly from their
safetensor shards instead of copied into anonymous shared buffers. This keeps
the 37 GiB model file-backed/reclaimable and slightly improves total transformer
time; M3 uses the faster copied-buffer path. `H3_ZERO_COPY_WEIGHTS=0` disables
the M5 selection for diagnostics.
The streamed Qwen text encoder preallocates a small ring of future layer
buffers and fills them on eight I/O workers while Metal executes the current
layer. The default ring depth is two layers on M3/older hardware and three on
M5, where the target machine has 128 GiB. `H3_QWEN_PREFETCH=0` restores the
single-layer synchronous reference path; values 1-8 select the worker count,
and `H3_QWEN_PREFETCH_DEPTH=1` through `6` overrides the ring depth.

`--ssd-streaming` is a separate, more aggressive residency mode for the DiT.
Only its small per-block normalization weights remain resident. Two complete
BF16 matrix slots alternate while a background reader fills the next slot in
checkpoint-offset order; the current Metal command buffer runs concurrently.
Darwin uncached reads avoid retaining a second copy in the filesystem cache.
The first active block is prefetched again during the final block, so a cached
interactive DiT is ready for its next denoiser evaluation. Measurements reached
about 13--14.6 GiB/s from the internal SSD. `H3_PROFILE=1` reports total bytes,
read throughput, and the part of the read wait that was not hidden by GPU work.

A style LoRA is merged into the same weights, so the merge follows the
residency mode. Resident blocks are merged right after they are read from the
checkpoint and before the int8 quantization pass. Streamed blocks are merged
into their slot as soon as the background reader finishes, inside the command
buffer that runs the next block. The merge evaluates the low-rank update with
a tiled 16x16 product in F32 and rounds the result back to BF16 in the same
pass, so it reads and writes the target matrix exactly once and needs no
large temporary buffers.

### Derived-state cache (opt-in)

`H3_INT8_CACHE=<dir>` persists two derived products of the checkpoint so later
loads skip the work that produced them. Both are written from the exact tensors
the kernels consume, so a hit cannot change a frame.

- `<dir>/<hash>/adaln.bin` holds the materialized AdaLN modulation tables for
  one sigma grid: 50 block tables plus the final table, `time_rows × 96,768`
  BF16 each (65 MB at 4 steps, 215 MB at 20). It removes the ~24 GiB AdaLN
  read and the 50 projections per load. The key covers the step count, both
  flow shifts, the conditioning flags, and the row count.
- `<dir>/<hash>/block-NN.bin` holds each block's post-LoRA int8
  qkv/out/fc1/fc2 matrices and their per-row scales (385 MB per block, 18 GiB
  for all 50). It removes the BF16 reads and the quantization pass. The key
  covers the LoRA file identity and scale; the cache is skipped under
  `--ssd-streaming` and under any `H3_INT8_KEEP_BF16_*` flag.

The hash covers every shard's path, size, and mtime, so a replaced checkpoint
misses. `INT8_CACHE_VERSION` in `h3_dit.c` must be bumped when the quantizer or
the row-scale layout changes; a mismatch is rejected and recomputed rather than
trusted. Enabling the directory only *reads*: a run never writes unless you also
set `H3_INT8_CACHE_WRITE=1`, because one populated variant costs about 18 GiB. On an M5 Pro at
512x512, 22 frames, 4 steps: DiT load 36.6 s without the cache, 24.9 s with the
AdaLN tables, 8.4 s with both; the whole DiT phase drops from 64.4 s to 29.1 s
with byte-identical frames. The first run pays the 18 GiB write, and the cache
directory wants roughly 19 GiB of free space.

`H3_COND_CACHE=<dir>` persists the encoder outputs of a generation - the Qwen
text embedding, the reference-image condition rows from the video VAE encoder,
the DiT modality tags and the reference layout - so a later run with the same
inputs skips the encoders entirely. A hit measured 114 s to 77 s at 864x480 and
22 frames, which is the ~74 s of text, vision and reference encoding a 1344x768
run pays. Entries are small (about 11 MB of embedding plus a few MB of rows) and
`H3_COND_CACHE_WRITE=1` is still required to fill them.

The key covers the prompt, the reference files, the render canvas and
`--reference-image-size`, but **not** the seed - so repeated draws at one canvas
(gacha, same-canvas refinement, chained segments of equal length and size) all
hit. The canvas is in the key because reference images are resampled to the
target area and the text sequence grows with the vision spans.

`H3_COND_CARRY_ACROSS_CANVAS=1` drops the canvas from the key, which is exact
rather than approximate **when the reference images are smaller than the render
area**: `h3_reference_image_canvas` clamps their scale to 1.0, so the encoder
outputs are byte-identical across canvases. Verified with 640x640 refs at
864x480 and 1344x768 - the reference-image conditioning, the audio conditioning,
the modality tags and the reference layout all hashed equal, and only the text
embedding differed (which it does between two identical runs anyway). A
two-pass tail carrying pass 1's conditioning measured 310 s against 342 s for
recomputing it, with the same detail and 0.99 composition correlation. With
1024x1024 refs the scale is below 1.0 at both canvases and the conditioning
genuinely differs, so leave the flag off.

Each write and restore prints a fingerprint of the payload. That is what proves
the round trip faithful, and it also showed that two identical commands encode
*different* conditioning - in F32 as well as BF16 - so the encoders are not
reproducible run to run. A cache hit therefore pins one realization rather than
merely saving time; treat "same seed, same prompt" as insufficient for a quality
A/B at one canvas until that is understood.

### Metal 4 and TensorOps paths

M5 GPUs automatically use native BF16 Metal 4/TensorOps for the DiT QKV and
attention-output projections at sequence lengths up to 2,048. The compact
Morton schedule routes Q/K/V directly into head-major attention inputs, avoids
three MPSGraph input transposes, and is byte-identical to the portable path. It
improves a complete 512x512 50-block forward by about 2% across repeated IT/US
M5 Max runs. For 2,049-3,072 rows, including 864x480, two row-offset Morton
dispatches preserve the efficient tile geometry and improve the complete
forward by about 2% in balanced runs. Still larger sequences stay on MPSGraph.
`H3_NAX=0` disables TensorOps for exact A/B diagnosis. The selection is guarded
at runtime and falls back to the unchanged portable library if compilation is
unavailable.

`H3_NAX=1` forces the broader native BF16 linear path. It passes the complete
50-block MLX fixture, but remains opt-in: exact-shape microbenchmarks favor its
128-row tile while full DiT runs currently favor MPSGraph scheduling. This
keeps a working NAX integration available for later quantized/fused kernels
without making a benchmark regression the default.
`H3_NAX=mlp` selects a more specialized Metal 4 path: paired FC1 gate/up
TensorOps tiles apply SwiGLU in threadgroup memory and write only the
14,336-wide activated intermediate, then FC2 also stays on TensorOps.
`H3_DISABLE_NAX_MLP=1` keeps the MPSGraph MLP in a context created this way for
same-process A/B testing. The path is deliberately opt-in because scheduling
depends on the OS GPU stack: the primary macOS 26.5.2 M5 Max gained 1.3-2.0%
in isolated real-weight MLP runs but lost about 1-3% in a complete 50-block forward,
while an otherwise identical macOS 26.5 M5 Max gained 1.4% in a same-context
forward A/B. The resulting 50-block velocities were close (1.9% video and 2.4%
audio relative L2), but not byte-identical.

### Specialized projection kernels

The narrow DiT audio/video output heads convert their small released F32
weights to BF16 once and use the Iris-derived 16x16 tiled linear directly on
BF16 activations. At the production 320-render geometry, isolated paired-head
measurements are 2.30x faster on M3 Max and 1.83x faster on M5 Max, with
relative L2 `8.64e-4`; the absolute M5 saving is about 0.6 ms per evaluated
step. Full fox and surfer sequences remained clean and measured 29.9/38.4 dB
against the F32-head renders. `H3_DIT_F32_FINAL=1` restores the close-reference
head and its extra activation buffers.
The F32 `96->5376` video and `32->5376` audio patch projections use a dedicated
16x16 cooperative tile, retaining F32 weights, inputs and accumulation while
rounding the tile result directly to BF16.
Paired production-shape measurements are 1.77x faster on M3 and 1.62-1.78x
on M5; the complete generated RGB stream is byte-identical to the scalar path.
Fusing the final cast improves the 2835-row tile itself from 2.499 to 1.734 ms
on M3 and 1.555 to 1.186 ms on M5, and removes 38.27/59.66 MiB of F32 scratch
at 512/864-class geometry. `H3_DISABLE_FUSED_PATCH_CAST=1` restores the tiled
F32 output plus standalone cast; `H3_SCALAR_PATCH=1` selects the scalar
diagnostic path.
The same tile binds its output directly into the packed hidden stream, removing
the BF16 media staging buffers and their blits. This saves another 19.13/29.83
MiB and improves the 2835-row boundary from 1.847 to 1.730 ms on M3 and 1.282
to 1.184 ms on M5. Contiguous T2VA uses byte offsets; FL2VA/Ref2VA use compact
destination-row maps so each modality remains one large dispatch. A complete
six-segment Ref2VA M5 ABBA remained byte-identical and improved 5.067 to 5.033
seconds per measured forward pair. `H3_DISABLE_FUSED_PATCH_PACK=1` restores the
staging buffers and packing blits.

### Scheduling and activation memory

The DiT core is split into ordered Metal command buffers so GPU execution
of one part overlaps CPU encoding of the next. M5 batches eight DiT blocks
per command buffer by default: on an M5 Pro that measured 19% faster at
512-class geometry and 26% faster at 864-class geometry than one buffer per
block, with bit-identical output, because the per-block case spent seconds
per evaluation encoding long sequences. Set
`H3_DIT_CONTINUE_EVERY_BLOCK=1` to restore one buffer per block, or
`H3_DIT_COMMAND_BLOCKS=N` to override the batch size (4-16 measured within a
few percent of each other; `0` emits one buffer per evaluation). Non-M5 GPUs
keep the per-block default and the validated 30/50 batch on request. The
operation order and generated bytes are unchanged by any of these settings.

The serving sampler integrates the flow ODE with a first-order Euler step.
`--sampler ab2` (or `H3_SAMPLER=ab2`) selects a second-order Adams-Bashforth
update that reuses the previous evaluation's velocity at no extra model cost.
On the 512-square, 22-frame fox test, 14-step AB2 ran 30% faster than the
20-step Euler default and matched its trajectory accuracy against a 40-step
Euler reference (18.5 versus 19.1 dB PSNR; plain 14-step Euler fell to
10.7 dB), with frame-to-frame activity closer to the reference (13.5 versus
12.0 mean absolute change). It produces a different sample and is therefore
opt-in; Euler remains the default and both solvers leave the audio stream on
the same schedule. `--shift N` (or the equivalent `--video-shift N`) overrides
the video flow sigma shift (default 12), and `--audio-shift N` sets the audio
stream's shift, previously reachable only through `H3_AUDIO_SHIFT`. Both accept
(0.5, 100) and both values enter the prepared-DiT key, so a cached DiT can
never be reused under a different schedule.

`--knots LIST` replaces the released uniform timestep grid with an explicit
descending list of `steps + 1` timesteps from 1000 down to 0, each mapped
through the video and audio shifts. Adapters that ship their own sampling
schedule need it: Pika H3 Flex records `1000,750,500,250,100,40,0` at video
shift 12 and audio shift 3 and states that generic six-step sampling is not
equivalent — through shift 12 its last two sigmas land on 0.57 and 0.33 while
the uniform grid is still at 0.86 and 0.71. Distills such as the turbo LoRAs
list no knots and assume the uniform grid, so the flag stays unset for them.
Note that this tail-heavy shape is the family that lost the low-budget grid
search above for the *base* model; the question is which adapter is being
sampled, not which grid is generically better. Without knots the builder is
value-for-value the released grid.
Fewer steps prefer a larger shift: on the 512-square fox and surfer tests,
AB2 at 12 steps with `--shift 16` measured 24.6/27.6 dB against a 40-step
reference versus 17.4/18.0 dB at the default shift, i.e. Euler-20 quality
with eight fewer evaluations; ten steps plateaus near 20 dB regardless of
shift. The second-order weight is clamped to +/-1 (`H3_AB2_CLAMP`); removing
the clamp drops the 12-step result by 1.2 dB because adjacent step-size ratios
can reach two orders of magnitude. `H3_SKIP_CACHE=<threshold>` skips a forward when the previous step moved the
video latent by less than the threshold in relative L2 norm (warmup and tail
steps never skip, at most two consecutive skips). At 512-square with AB2 it
stays safe only for one or two late skips (24.9-25.1 dB against a 40-step
reference); six or more skips collapse to about 15 dB because small latent
motion early in the schedule still hides large direction changes. It is off by
default.
`--sampler heun` (predictor-corrector with a trapezoid correction from the
next evaluation) is kept for experiments but loses to the clamped AB2:
12 steps/shift 16 measured 24.6 dB for AB2 versus 24.1 for Heun, and 14/shift
12 measured 24.1 versus 21.5. Below 12 steps the schedule needs more shift and
quality falls quickly: 10 steps/shift 20 measured 22.2 dB and 8 steps at any
shift measured about 20 dB against the 40-step reference, so 12 steps/shift 16
is the practical knee.
DiT activation buffers also follow their actual intra-block lifetimes: the QKV
projection arena is reused first for attention heads and then for the normalized
MLP input, while the current attention-output arena becomes the MLP output after
its branch has been consumed. This removes 61.25 MiB at 512-class geometry and
99.63 MiB at 864-class geometry without changing dispatches or arithmetic.
`H3_DISABLE_DIT_ACTIVATION_ALIAS=1` restores separate diagnostic buffers.
MPSGraph tensor-data wrappers for immutable DiT weights and biases are retained
with their resident buffers. This avoids rebuilding the same binding metadata
for every block and denoiser evaluation without copying tensor storage; measured
ABBA gains were 1.6% on M3 Max and 0.4-1.1% on M5 Max. Activation wrappers stay
transient because retaining them regressed the M5. The outputs remain
byte-identical, and `H3_DISABLE_GRAPH_DATA_CACHE=1` restores transient wrappers
for all tensors.
On M3/older hardware, the four MPSGraph segments in each DiT block also reuse
one `MPSCommandBuffer` wrapper for their shared underlying Metal command buffer.
Repeated thermal-balanced runs measured 1.0-1.6% faster on M3 Max; M5 measured
neutral, so it retains fresh wrappers. `H3_REUSE_MPS_COMMAND=0` or `1` overrides
the automatic selection. Results are byte-identical.
On M5, the serving Euler sampler keeps its patch-packed F32 latents and cached
BF16 velocities in Metal buffers. Each selected denoiser refresh is completed
before the next is encoded, avoiding MPSGraph back-pressure while removing all
intermediate latent/velocity readbacks and repacking. Two warm eight-run A/B
sequences measured small 0.1% and 0.3% gains with byte-identical final latents;
the path also saves roughly 16 bytes of transient host state per video-latent
element (about 136 MB at the 768p shape). M3 and older GPUs retain the CPU
sampler by default. `H3_CPU_SAMPLER=1` restores it on M5;
`H3_GPU_SAMPLER=1` selects the GPU-state path explicitly, and
`H3_GPU_SAMPLER_WINDOW=0` enables the slower unbounded encode-ahead diagnostic.

### Two-pass sampling (opt-in)

`--dump-latent PATH` writes the joint video/audio latent after denoising, and
`--init-latent PATH` starts a run from that dump instead of from pure noise. The
dumped video grid is bilinearly refined to the new canvas and then re-noised to
the schedule's first sigma with the released forward process
`x = (1 - sigma) * x0 + sigma * noise`. Re-noising is the point: resampling the
*noisy* latent would smooth away the high-frequency noise, so the model would be
handed a sample whose noise is not what that sigma means at that resolution.

`--knots` supplies the second pass's sigma ladder. Pika H3 Flex's last two
schedule points are `--knots 100,40,0` (sigma 0.5714, 0.3333, 0 at video shift
12), which reproduces the tail of its own six-step recipe on a larger canvas.
`--audio-frozen` zeroes the audio schedule so every audio Euler delta is zero and
the dialogue recorded by the first pass survives the refinement unchanged; it
requires `--init-latent`. Both passes must keep the same prompt, references and
frame count — the temporal and audio shapes are checked on load, and a knot grid
starting below 1000 is rejected without `--init-latent` because there would be
nothing to re-noise. A missing or truncated dump fails before any weight loads.

`--refine-pixels` changes how the dumped latent reaches the new canvas. Without
it the latent is bilinearly resampled, which measured unusable: at both a
1.5556x and an exact 2x ratio the tail steps rendered the interpolation
structure as dark strokes over fine fur, while the same handoff with no
resampling at all stayed clean — so the problem is leaving the image manifold,
not the ratio. With the flag the pass decodes the dumped latent, resamples the
frames with vImage, and re-encodes them, at the cost of one extra VAE decode and
encode at the target canvas.

Measured on the S17 shot, 480p to 768p, same seed and references. Pass 1 is
Flex six steps unless the row says otherwise; the last three rows stop pass 1 at
sigma 0.923 instead (four steps of `--knots 1000,875,750,625,500`):

| Handoff | Result | Wall (both passes) |
| --- | --- | --- |
| bilinear on the clean latent | dark strokes over fur, colour fringing on the shell | 8.8 min |
| bilinear on a clean latent, exact 2x ratio | worse strokes: the ratio is not the problem | 8.8 min |
| no resampling at all, same handoff sigma | clean: the resampler is the problem | 8.2 min |
| bilinear on a still-noisy sample (continued trajectory) | a full-frame rainbow moire | 11.5 min |
| VAE round-trip (`--refine-pixels`) | clean, but softer than native | 20.4 min |
| sigma 0.923 handoff, **one** tail step | red cast on the shell strap, flatter: not usable | 8.0 min |
| sigma 0.923 handoff, **two** tail steps | **clean and sharp, no ghost** | **9.8 min** |
| sigma 0.923 handoff, three tail steps | clean; indistinguishable from two at 1:1 | 11.2 min |
| native six steps at 768p | the reference | 11.2 min |

Profiled per-phase costs (instrumented runs): pass 1 at 864x480 is 16 s of
reference-image encoding, 15 s of text encoding, 48 s of DiT loading, **25 s per
denoise step**, 31 s of decoding. A 1344x768 tail is 38 s of reference encoding,
34 s of text encoding, 46 s of DiT loading, 80 s of decoding and about **100 s
per step** (the first measured 88 s, the second 114 s, the third 96 s; an earlier
single 7.2 min reading for the three-step arm was an outlier — it profiles at
8.2 min).

**The 768p decode's 80 s is GPU-bound, not command-encoding bound.** Read in the
serial path (`H3_VAE_PIPELINE=0`), where every tile commits through
`h3_gpu_submit` and the wait is accounted honestly, the same decode splits into
8.1 s encoding and **59.9 s waiting on the GPU**. The pipelined path reports that
same 60 s as 62 s of "encode", because `encodeToCommandBuffer` blocks on queue
backpressure while the GPU is behind and the blocked time lands in the encode
counter. The tell was the spread, not the mean: fastest 0.016 ms, slowest 42 ms
against a 9.5 ms average. **So trimming graph-encoding work — pre-transposed
weights, batching tiles — cannot recover more than the 8 s that is genuinely
CPU, and the decode's cost is real compute.**

Those figures are the `--vae-bf16` path, which is what the profiled rows above
use. Benchmarked on the same grid (17x48x84 latent, 56 frames, three chunks,
interleaved twice, machine otherwise idle), the decode is **240 s in F32 and
70 s in BF16 — 3.4x** with 9.55 GiB vs 4.81 GiB Metal live, matching the
44–49 dB near-losslessness measured elsewhere. GPU-bound holds in both
precisions; F32 just multiplies the compute side by ~3.4x. **That precision
switch, not any encode-side restructuring, is the large lever on this stage**,
which is why F32 stays the default (bit-identical output) and speed runs opt in
with `--vae-bf16`.

Two consequences. First, a 768p tail costs about **3.3 minutes of
canvas-independent overhead** plus 1.7 minutes per step, so one step cannot fall
below 5.0 min and two below 6.8 min. Second, and more honestly: **repeating the
same configuration varies by up to 20%** (pass 1 measured 3.0 then 3.6 minutes),
which is larger than the gap to native. Treat two-pass as **cost-neutral against
native** (9.8–10.4 min versus 11.2) and value it for composition iteration, not
for speed.

Each pass is a separate process, so it is a complete generation pipeline with a
preseeded latent — nothing is shared with pass 1 except the dump. Holding that
state in one process instead removes the DiT load and LoRA merge (46 s) and, if
pass 1 does not need to be reviewed as video, its decode (31 s). It does **not**
remove the ~74 s of reference-image, vision and text encoding, because those are
bound to the render canvas, not to the process: reference images are resampled
to the target area and the text sequence grows with the vision spans. Only when
the references are small enough that `h3_reference_image_canvas` clamps their
scale to 1.0 at both canvases is that encoding genuinely canvas-independent —
verified byte-for-byte with 640x640 refs — and in that case a single process
could reuse it too. That reuse needs a layout rebuild that keeps the weight store
resident, since `h3_dit` currently prepares one canvas per instance.

The mechanism, settled by correlating each result's high-pass against the same
base lanczos-upscaled to the target canvas (after an FFT shift search, which
always returned zero offset and 0.95 luminance correlation): **the tail steps
re-invent all texture at every sigma above zero — high-frequency correlation with
the lifted base measured 0.003 to 0.014 — while the composition transfers at
0.95.** A handoff at mid sigma is therefore a superposition of where the lift put
the subject and where the model puts it, which is what the doubled glasses and
doubled heads look like. Lowering the sigma makes it worse, not better.

So two-pass buys **composition control, not detail**. Stop pass 1 high in the
schedule and it inherits composition only, cleanly; stop it low and it tries to
inherit detail and ghosts instead:

```
h3 ... --lora pika_h3_flex_ref2va_comfy_bf16.safetensors --steps 4 --shift 12 \
       --audio-shift 3 --knots 1000,875,750,625,500 --width 864 --height 480 \
       --dump-latent /tmp/seed.latent
h3 ... --lora pika_h3_flex_ref2va_comfy_bf16.safetensors --steps 2 --shift 12 \
       --audio-shift 3 --knots 500,125,0 --init-latent /tmp/seed.latent \
       --audio-frozen --width 1344 --height 768
```

Two tail steps are enough at sigma 0.923 even though one step from sigma 0.333
smears: the model's velocity is well conditioned for a large jump in the
structure regime and badly conditioned for one in the detail regime. The saving
is inside the noise floor of repeated runs, because the 768p fixed cost is paid
either way; the win is that reviewing a composition costs 3.0-3.6 minutes per
attempt instead of 11.2, and only the chosen draft pays the tail.

The SelfLift handoff was tried and removed. It keeps pass 1's sample at its own
sigma instead of projecting it clean, and lets pass 2 consume it verbatim instead
of re-noising, so both passes share one noise realisation. It worked numerically,
but with the bilinear lift it measures worse than re-noising: bilinearly
resampling a noise field correlates it, and the model renders that correlation as
content — a full-frame rainbow moire at 11.5 min. A partial pass therefore always
projects the clean endpoint it predicts at its stop sigma, and every second pass
re-noises. Nothing in the tree carries that opt-out anymore.

What did pay off is the solver: `--sampler renoise` beat Euler on every grid
tested, and the released Flex ladder is
`--knots 1000,750,500,250,100,40,0` (matching the official ComfyUI
`ManualSigmas` to six decimals) — a grid is only valid together with its
solver.

### Canvas beyond 768p (opt-in)

The released package documents 480p/768p as the local output range, and
`H3_MAX_PIXELS` enforces that ceiling. `--oversize` (or `H3_CANVAS_LIMIT=<pixels>`)
raises it to four times the released count. Measured at 1920x1088, 39 frames,
six steps: it runs without hitting the memory wall in 17.2 minutes, but the
extra pixels are not backed by extra detail — texture scale grows with the
canvas instead of resolving more, and the 1:1 crops are softer than a native
768p frame. Treat it as a framing/compositing escape hatch, not a quality path.

### Checkpoint layout and media pipeline

The released checkpoint stores DiT QKV rows interleaved per attention head.
Native Metal consumes that layout directly in the fused QK-normalization/RoPE
kernel, avoiding a checkpoint transpose and extra RAM. The earlier identity
interpretation was the cause of the noisy diagnostic outputs.

The public generation path decodes the joint audio latent with a streamed native
BigVGAN/AudioVAE and writes synchronized H.264 plus 32 kHz stereo AAC. The native
waveform agrees with the corrected MLX oracle to relative L2 `6.94e-5`.
`--first-frame`, `--last-frame`, and their combination use the released visual
VAE encoder, Qwen3-VL vision tower and three-deepstack multimodal presentation,
0.999 condition augmentation, and fixed condition rows in the native DiT. The
first image is stretched to the target canvas; the last image is aspect-cover
scaled and center cropped, matching the reference implementation. `--ref-image`
selects the distinct Ref2VA transformer, preserves ordered `<Picture N>`
presentation, and uses the released down-only aspect-preserving reference canvas.
`--ref-silent-video` additionally performs bounded 24 fps decoding, the visual
VAE's causal `ceil(T/4)` compression, two-frame Qwen sampling, and timestamped
`<Video N>` presentation. `--ref-video` preserves an embedded soundtrack,
`--ref-video-audio VIDEO AUDIO` supplies an explicit replacement, and
`--ref-audio` appends an ordered standalone clip. Reference audio is decoded as
32 kHz stereo F32, encoded by the native AudioVAE posterior-mean path, mixed as
0.999 clean latent plus 0.001 seeded noise, pinned to the audio condition
timestep 1.0, and packed as width-32 rows on the same rotary timeline as visual
references. Audio inputs are 2-15 seconds, at most three are
accepted, their total decoded duration is capped at 15 seconds, and a standalone
audio reference must be combined with an image or video reference.

The native audio encoder matches the corrected MLX oracle at relative L2
`3.59e-6` on a real two-second stereo fixture. The correction is important: the
original MLX reshape interleaved left/right samples, whereas the official
PyTorch/SGLang path folds intact stereo channels into the batch dimension. On
the 128 GB M5 Max, clean end-to-end image+audio and embedded-video+audio renders
completed in 74.58 and 76.99 seconds respectively, each with about a 40.1 GB
peak physical footprint and zero swaps.

### Profiling and diagnostic paths

`--profile` reports each Metal-backed phase separately: wall time, CPU-side
command encoding, complete commit-to-fence wait, root-command GPU timestamps,
peak live tensor storage, cumulative allocation, and dispatch counts. The wait
measurement is the complete command turnaround; the root GPU timestamp alone
can omit child buffers scheduled internally by MPSGraph and is labeled
accordingly.

The DiT fast path evaluates each BF16 `fc1 -> SwiGLU -> fc2` block as one cached
graph, avoiding separate graph boundaries and persistent intermediate tensors.
Set `H3_DISABLE_FUSED_MLP=1` to retain the close-reference operation boundaries
for numerical diagnosis.

On supported M5 Metal 4 TensorOps hardware, the native int8 MLP engine is the
default. It dynamically quantizes activations, uses per-output-channel weight
scales, and gives the sensitive FC2 input one scale per 1,024 channels.
The selected FC2 kernel keeps scaled partial products in private cooperative
fragments instead of repeatedly spilling a 32 KiB threadgroup tile. A fixed
50-layer, 19-transition 512x512 render measured 36.30 seconds with BF16 MPS and
25.80 seconds with int8 on M5 Max. Beginning, middle, and final decoded frames
retained the same subject, composition, and motion; small edge and fur details
can differ. The current diagnostic implementation retains both BF16 and int8
MLP weights only when an A/B diagnostic requests them. Normal int8 loading
releases each block's BF16 FC1/FC2 buffers after their submitted quantization
finishes, reducing measured peak tensor storage to 25.9 GiB from the BF16
path's 36.4 GiB. Runtime weight quantization still adds startup time.

The fastest M5 path also quantizes each DiT QKV projection and writes its
Q/K/V tiles directly in head-major attention layout before the existing Q/K
normalization and RoPE kernel. In a fixed 50-layer, 19-transition 512x512
render this reduced denoising again, from 25.80 to 19.32 seconds. Sampled
beginning, middle, and final frames remained a coherent detailed fox walking
through snow; quantized attention can change framing and fine detail. Use
`--use-slower-bf16-qkv` for the close-reference BF16 projection. Normal int8
loading releases the redundant BF16 QKV weights after quantization.

The following attention-output projection is int8 as well on the default M5
path. Crossed same-model tests improve a complete forward by another 4.5-5.5%
at 512 and 864. A decoded fox render remained clean and closely matched the
int8-QKV-only composition; its thermally hot denoise measured 19.18 seconds.
Use `--use-slower-bf16-attention-output` to retain that projection in BF16.

On that int8 path, SDPA now leaves its result in native
`[head,row,dimension]` order. A specialized 256-thread kernel gathers and
quantizes each H3 row directly into the projection's row-major int8 buffer,
eliminating the intervening full-width BF16 transpose without changing any
output byte. Thermally controlled crossed runs improve complete 512 and 864
forwards by roughly 0.2-1.2%. Use
`--use-slower-row-major-attention-output` to restore the explicit BF16
row-major SDPA output and ordinary quantizer.

The M5 path also folds QKV and MLP activation quantization into the preceding
gated AdaLN kernel. This removes 99 standalone quantizer dispatches per
50-layer forward while preserving the previous output bytes, improving crossed
512/864 measurements by about 0.3-0.6%. Use
`--use-slower-unfused-int8-inputs` to restore the standalone quantizers.

The fused gated-AdaLN path loads its full 5,376-wide H3 rows as BF16x4 vectors
and writes int8x4. It stages the rounded values locally before computing the
original per-thread RMS sequence, so the reduction tree and every output byte
remain unchanged. Crossed measurements save roughly another 0.1-0.5%. The
existing `--use-slower-unfused-int8-inputs` option retains the portable scalar
and standalone-quantizer fallback.

Q/K RMS normalization and RoPE are performed inside the int8 QKV projection
tile as well. The fused epilogue is byte-identical and improves complete
forwards by 2.1-3.2% at 512 and 1.0-1.8% at 864 in crossed M5 measurements.
Use `--use-slower-unfused-qkv-rope` to restore the separate Q/K kernel.

That epilogue processes four adjacent Q/K dimensions per work item with
BF16x4 loads and stores. The per-element arithmetic and BF16 rounding order are
unchanged, while crossed cool-state measurements improve complete forwards by
about 0.4-1.0% at both 512 and 864. The same
`--use-slower-unfused-qkv-rope` option restores the scalar standalone path.

At up to 2,048 rows, the exact RMS loop uses BF16x4 loads followed by four
explicit ordered FMAs. This preserves every output bit and improves 512-class
forwards by another 0.5-0.6%; larger shapes retain scalar loads because the two
forms tie there. Use `--use-slower-scalar-qkv-rms` to force scalar loads.

The int8 attention-output projection caches its 128 row and column scales in
1 KiB of threadgroup memory instead of rereading them for every cooperative
fragment element. Above 2,048 rows the fused QKV kernel uses the same idea and
then recycles that storage for inverse RMS values; smaller QKV shapes retain
direct loads because the two forms tie there. Both are byte-identical and
improve complete forwards by about 0.2-0.7% where selected. Use
`--use-slower-uncached-int8-scales` to restore direct device-scale loads.

For sequences of at most 2,048 rows, the H3 attention-output projection also
compiles its 7,168-by-5,376 shape into the TensorOps kernel. The result remains
byte-identical while saving about 0.2-0.8% in crossed complete 512-forward
measurements. Larger sequences retain the dynamic-shape kernel because the
specialization regresses there. `--use-slower-uncached-int8-scales` restores
the general dynamic, direct-scale-load implementation.

FC1 also uses an H3-specialized, compile-time 5,376-wide TensorOps loop. It is
byte-identical to the generic loop and saves about 0.1-0.4% in crossed complete
forwards. Use `--use-slower-dynamic-fc1-k` to restore the runtime-bound loop.

```sh
./h3 --profile -d ./MiniMax-H3 \
  -p "A red fox walks through fresh snow." \
  --width 512 --height 512 --frames 22 --steps 20 \
  --layers 50 --reuse 1 -o outputs/fox-int8.mp4
```

Use `--use-slower-bf16-mlp` to force the portable close-reference MPS/BF16 MLP
path for numerical comparison. Older Metal hardware selects that path
automatically when the required native TensorOps kernels are unavailable.
For FC2 activation quantization, sequences of at most 2,048 rows use an exact
128-thread reduction. Each thread retains its eight BF16 input values while
computing the group maximum, avoiding a second device-memory read when it emits
the int8 values; crossed M5 measurements improved complete 512 forwards by
about 0.2-0.8% without changing any output byte. Larger sequences retain the
measured 256-thread kernel. `--use-slower-grouped-quantizer` forces the latter
at every size for A/B comparison.

The native baseline targets the original `FL2VA/` and `Ref2VA/` checkpoint
trees. Model phases are loaded and released separately so the 33B transformer,
Qwen encoder, and decoders never have to coexist in unified memory.

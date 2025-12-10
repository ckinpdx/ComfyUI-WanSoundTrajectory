# WanSoundTrajectory

Audio-driven path modulation for WanMove video generation.

## What it does

Takes a path from KJNodes SplineEditor (or any compatible coordinate source) and modulates it based on audio analysis. The result is camera or object movement that reacts to your music - beats push the path, bass makes it wobble, etc.

## Installation

1. Clone or copy this folder to your ComfyUI `custom_nodes` directory:
   ```
   ComfyUI/custom_nodes/ComfyUI-WanSoundTrajectory/
   ```

2. Restart ComfyUI

No additional dependencies beyond numpy (already required by ComfyUI).

## Usage

### Basic workflow:

```
[LoadAudio] ─────────────────────────────┐
                                         ▼
[SplineEditor] ──► coordinates ──► [WanSoundTrajectory] ──► track_coords ──► [WanVideoAddWanMoveTracks]
                                         │
                                    fps, mode,
                                    intensity, etc.
```

### Inputs

| Input | Type | Description |
|-------|------|-------------|
| `audio` | AUDIO | Input audio (waveform + sample_rate) |
| `coordinates` | STRING | Path from SplineEditor - JSON format `[{"x": n, "y": n}, ...]` |
| `fps` | INT | Frames per second - needed to align audio with frames |
| `modulation_mode` | dropdown | Which audio feature to use |
| `modulation_direction` | dropdown | How audio affects the path |
| `intensity` | FLOAT | Strength of modulation in pixels |
| `smoothing` | FLOAT | Temporal smoothing (0-1) |
| `normalize` | BOOL | Normalize audio features to 0-1 range |
| `bidirectional` | BOOL | Allow negative displacement (oscillate around path) |
| `frequency_split_hz` | INT | Cutoff frequency for bass/treble split |

### Modulation Modes

- **envelope** - Overall amplitude/loudness (RMS energy). Reacts to everything - vocals, drums, synths. Good general-purpose option.
- **bass** - Low frequency energy below the `frequency_split_hz` cutoff. Reacts to kicks, bass, low toms. Gives you that punchy, rhythmic movement.
- **treble** - High frequency energy above the cutoff. Reacts to hihats, cymbals, vocal sibilance, synth sparkle. Adds subtle texture and shimmer.
- **onsets** - Transient/attack detection using spectral flux. Fires on any sudden change - drum hits, plucks, note attacks. More percussive than envelope.
- **bass_treble_split** - Analyzes both bands separately (currently uses bass for modulation, treble analysis available for future use).

### Modulation Directions

- **perpendicular** - Pushes points sideways, 90° to the path direction. If your path goes left-to-right, audio pushes it up/down. Creates a wobble/wave effect along the trajectory.
- **radial** - Pushes points toward/away from the path's center point (centroid). Creates a breathing/pulsing effect where the whole path expands and contracts.
- **along_path** - Pushes points in the direction of travel. Effectively speeds up or bunches together the motion on loud parts.

### Key Parameters Explained

#### intensity
How far (in pixels) the path can be displaced at maximum modulation. 
- `50` = subtle movement, good for background elements
- `100-150` = noticeable but controlled
- `200+` = dramatic, may look chaotic

This is your main "how much" knob.

#### smoothing
Temporal smoothing using exponential moving average (0-1).
- `0.0` = no smoothing, instant reaction to audio (can look jittery)
- `0.3` = light smoothing, responsive but less twitchy
- `0.5` = moderate smoothing, flowing movement
- `0.8+` = heavy smoothing, slow/lazy response

Higher values make movement feel more "underwater" or sluggish. Lower values feel more percussive and immediate.

#### bidirectional
Controls how the 0-1 audio values map to displacement:

**Off (default):**
- Audio value 0.0 → no displacement (stays on original path)
- Audio value 1.0 → full displacement in positive direction
- Path only moves "outward" from original, returns to baseline when quiet

**On:**
- Audio values get remapped: `(value * 2) - 1` so 0-1 becomes -1 to +1
- Audio value 0.0 → full negative displacement
- Audio value 0.5 → no displacement (on original path)
- Audio value 1.0 → full positive displacement
- Path oscillates *around* the original trajectory

Use bidirectional when you want symmetrical wobble. Leave it off when you want the path to "punch out" on beats and return to center.

#### normalize
When enabled (default), audio features are scaled so the quietest moment = 0 and loudest = 1. This means:
- Consistent modulation range regardless of source volume
- A quiet song and a loud song produce similar movement intensity

Disable if you want absolute levels to matter (rare).

#### frequency_split_hz
The cutoff frequency (in Hz) that separates "bass" from "treble" modes.
- `200` (default) - standard split, bass gets kick/bass, treble gets everything else
- `100` - only deep sub-bass triggers bass mode
- `300-400` - bass mode includes more low-mid content (toms, low vocals)

Adjust based on your music. Electronic/hip-hop might want 150-200. Rock/acoustic might want 250-300.

#### modulate_static_points
When SplineEditor outputs a fixed/static point (a single anchor that doesn't move), this controls whether audio modulation is applied.

- **True (default)**: Static points will jiggle based on audio
- **False**: Static points pass through unchanged

Useful when you have multiple tracks - some moving paths and some fixed anchors - and you only want the paths to react.

#### static_point_axis
When a static point IS modulated, this controls which direction it wobbles:

- **horizontal**: Left/right movement only
- **vertical**: Up/down movement only  
- **both** (default): Diagonal movement (45°)
- **radial_from_center**: Push toward/away from frame center (512, 512)

The `radial_from_center` option is useful for breathing/pulsing effects where you want the point to expand outward from the middle of the frame.

### Output

| Output | Type | Description |
|--------|------|-------------|
| `track_coords` | STRING | Modulated coordinates in same JSON format |

## Tips

1. **Match your frame count**: SplineEditor's `points_to_sample` should equal your video's total frames. 97 points for a 97-frame video.

2. **Start conservative**: Begin with intensity 30-50 and work up. It's easier to add movement than fix a chaotic mess.

3. **Smoothing prevents jitter**: Raw audio analysis can be spiky. Smoothing 0.3-0.5 usually looks more natural.

4. **Bidirectional for music**: Most musical movement should oscillate around the path, not just push one direction. Enable bidirectional unless you specifically want "punch out and return" behavior.

5. **Bass for rhythm, envelope for everything**: Use `bass` mode when you want movement locked to the kick drum. Use `envelope` when you want it to react to the whole mix.

6. **Preview with WanDrawTracks**: Connect the output to `WanVideoWanDrawWanMoveTracks` to visualize the modulated path on your source frames before running the full generation.

7. **Onsets for percussive hits**: If envelope feels too smooth and you want snappier reaction to drum hits, try `onsets` mode with lower smoothing.

## Example Settings

### Subtle beat-reactive movement
- Mode: `envelope`
- Direction: `perpendicular`
- Intensity: `30`
- Smoothing: `0.5`
- Bidirectional: `False`

### Aggressive bass wobble
- Mode: `bass`
- Direction: `perpendicular`
- Intensity: `100`
- Smoothing: `0.2`
- Bidirectional: `True`

### Breathing/pulsing effect
- Mode: `envelope`
- Direction: `radial`
- Intensity: `50`
- Smoothing: `0.4`
- Bidirectional: `False`

## Compatibility

- **Input**: Any node that outputs SplineEditor-compatible coordinate JSON
- **Output**: WanVideoAddWanMoveTracks, WanMove_native, or any node expecting the same format

## License

MIT - do whatever you want with it.

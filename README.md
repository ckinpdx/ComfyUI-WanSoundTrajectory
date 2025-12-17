[README.md](https://github.com/user-attachments/files/24219534/README.md)
# WanSoundTrajectory

Audio-driven path modulation and trajectory generation for WanMove video generation.

## What it does

A suite of nodes for creating and manipulating WanMove trajectories:

1. **WanSoundTrajectory** - Modulates paths based on audio analysis (beats, bass, envelope)
2. **WanTrajectoryGenerator** - Creates mathematical motion patterns (orbit, spiral, bounce, etc.)
3. **WanTrajectorySaver** - Saves trajectories for reuse
4. **WanTrajectoryLoader** - Loads saved trajectories with transforms (flip, rotate, rescale)
5. **WanPoseToTracks** - Converts DWPose skeleton keypoints to trajectory tracks
6. **WanMove3DZoom** - Creates 3D point clouds from depth maps with rotation, zoom, and background isolation

The result is camera or object movement that can react to music, follow mathematical patterns, or both.

## Installation

### From GitHub

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/ckinpdx/ComfyUI-WanSoundTrajectory.git
```

### Manual

Clone or copy this folder to your ComfyUI `custom_nodes` directory:
```
ComfyUI/custom_nodes/ComfyUI-WanSoundTrajectory/
```

Restart ComfyUI after installation.

No additional dependencies beyond numpy and cv2 (already required by ComfyUI).

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

Use bidirectional when you want symmetrical wobble. Leave it off when you specifically want "punch out and return" behavior.

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
- **local_orbit**: Each point jitters in a small circle around its own home position

The `radial_from_center` option is useful for breathing/pulsing effects where you want the point to expand outward from the middle of the frame.

The `local_orbit` option is specifically designed for use with WanMove3DZoom - each point orbits around its own original position rather than being pushed toward a global center. This creates natural-looking parallax motion where background points stay in the background but wobble in place to the audio.

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

---

## WanMove3DZoom

Creates 3D point clouds from depth maps with camera rotation, zoom animation, and background/foreground isolation.

### What it does

Takes a depth map and generates tracking points in 3D space. The points can be animated with rotation and zoom, then output as coordinates for WanMove. Combined with WanSoundTrajectory's `local_orbit` mode, you can create audio-reactive parallax effects where background points jitter to the beat while staying in place.

### Inputs

| Input | Type | Description |
|-------|------|-------------|
| `images` | IMAGE | Depth map (grayscale - white=near, black=far) |
| `num_points` | INT | Number of tracking points to generate (grid distribution) |
| `depth_scale` | INT | Z-axis range in pixels - separation between near and far |
| `depth_falloff` | FLOAT | Depth curve (0.1=near focus, 0.5=linear, 1.0=far focus) |
| `depth_min` | FLOAT | Minimum depth to include (0=far/black) |
| `depth_max` | FLOAT | Maximum depth to include (1=near/white) |
| `duration` | INT | Number of frames to generate |
| `x_rotation` | FLOAT | Tilt up/down (degrees) |
| `y_rotation` | FLOAT | Orbit left/right (degrees) |
| `z_rotation` | FLOAT | Roll/tilt horizon (degrees) |
| `zoom_amount` | FLOAT | Zoom intensity (positive=in, negative=out, 0=none) |
| `trajectory` | dropdown | Easing curve (Constant, Ease In, Ease Out, Ease In Out) |
| `point_radius` | INT | Preview dot size |
| `export_width` | INT | Output coordinate space width |
| `export_height` | INT | Output coordinate space height |
| `mask` | IMAGE | Optional - white=include, black=exclude points |

### Outputs

| Output | Type | Description |
|--------|------|-------------|
| `preview_images` | IMAGE | Animated preview of the 3D points |
| `coord_tracks` | STRING | Trajectory JSON for WanMove |

### Background Isolation

Use `depth_min` and `depth_max` to select only background points:

- **Background only**: `depth_min=0.0`, `depth_max=0.3` (selects dark/far areas)
- **Foreground only**: `depth_min=0.7`, `depth_max=1.0` (selects bright/near areas)
- **Mid-ground**: `depth_min=0.3`, `depth_max=0.7`

Alternatively, provide a **mask** image where white areas include points and black areas exclude them. This is more flexible for complex scenes where depth alone doesn't cleanly separate subject from background.

### Workflow

**Basic 3D zoom:**
```
[Depth Map] → [WanMove3DZoom] → coord_tracks → [WanMove]
```

**Audio-reactive background parallax:**
```
[Depth Map] → [WanMove3DZoom] → coord_tracks → [WanSoundTrajectory] → track_coords → [WanMove]
     ↑              ↑                                    ↑
   depth       depth_min/max                        local_orbit mode
               for background                         + audio
```

**With mask for subject isolation:**
```
[Image] → [SAM/GroundingDINO] → mask ──┐
                                       ↓
[Depth Map] ────────────────► [WanMove3DZoom] → coord_tracks
```

### Tips

1. **Use local_orbit**: When feeding WanMove3DZoom output to WanSoundTrajectory, use `static_point_axis=local_orbit` so points jitter around their home positions instead of pulling toward frame center.

2. **Preview first**: The node outputs preview frames showing the animated 3D points. Check this before running the full generation.

3. **Depth map quality matters**: Better depth estimation = better 3D separation. Consider using high-quality depth models.

4. **Combine zoom with rotation**: Small amounts of both (e.g., zoom=0.1, y_rotation=15) creates natural-feeling camera movement.

---

## Trajectory Saver & Loader

Save your SplineEditor paths for reuse with different audio or output specs.

### Saver Node

Saves raw coordinates from SplineEditor to a JSON file with metadata.

**Inputs:**
| Input | Type | Description |
|-------|------|-------------|
| `coordinates` | STRING | Raw coordinates from SplineEditor |
| `filename` | STRING | Name for the file (no extension) |
| `width` | INT | Original canvas width these coords were made for |
| `height` | INT | Original canvas height |

**Saves to:** `ComfyUI-WanSoundTrajectory/saved_trajectories/`

### Loader Node

Loads saved trajectories with optional rescaling, resampling, and transforms.

**Inputs:**
| Input | Type | Description |
|-------|------|-------------|
| `trajectory_file` | dropdown | Select from saved files (press R to refresh) |
| `target_width` | INT | New width (0 = use original) |
| `target_height` | INT | New height (0 = use original) |
| `target_frames` | INT | New frame count (0 = use original) |
| `flip_horizontal` | BOOL | Mirror left/right |
| `flip_vertical` | BOOL | Mirror top/bottom |
| `rotate` | dropdown | Rotate 90/180/270 clockwise |
| `reverse_time` | BOOL | Play path backwards |

**Outputs:**
| Output | Type | Description |
|--------|------|-------------|
| `coordinates` | STRING | Loaded (and optionally transformed) coordinates |
| `original_width` | INT | Width from saved metadata |
| `original_height` | INT | Height from saved metadata |
| `original_frames` | INT | Frame count from saved metadata |
| `track_count` | INT | Number of tracks in the file |

### Workflow

**Save once:**
```
[SplineEditor] → coord_str → [WanTrajectorySaver]
                                    ↑
                            filename, width, height
```

**Reuse many times:**
```
[WanTrajectoryLoader] → coordinates → [WanSoundTrajectory] → track_coords → [WanMove]
        ↑                                     ↑
   target_frames,                         new audio
   transforms, etc.
```

This lets you:
- Draw paths once, apply different audio later
- Rescale paths made for 512x512 to 832x480
- Resample 81-frame paths to 101 frames
- Flip/rotate paths for variations
- Reverse paths to play backwards
- Build a library of reusable camera moves

---

## Trajectory Generator

Generate mathematical motion patterns - no drawing required.

### Patterns

| Pattern | Description |
|---------|-------------|
| `oscillate` | Ping-pong back and forth (sine wave motion) |
| `spiral` | Spiral outward or inward from center |
| `orbit` | Circular motion around a center point |
| `diverge` | One origin point, tracks spread outward like explosion |
| `converge` | Multiple origins all meeting at center point |
| `random_walk` | Brownian motion / unpredictable drift |
| `bounce` | Ball bouncing off frame edges |
| `zoom` | Radial motion toward or away from center |
| `wave` | Multiple tracks doing synchronized wave motion |

### Inputs

| Input | Type | Description |
|-------|------|-------------|
| `pattern` | dropdown | Motion pattern to generate |
| `num_frames` | INT | Number of frames (match your video length) |
| `num_tracks` | INT | Number of separate tracks |
| `width` | INT | Canvas width |
| `height` | INT | Canvas height |
| `center_x` | FLOAT | Center X position (0-1 normalized) |
| `center_y` | FLOAT | Center Y position (0-1 normalized) |
| `amplitude` | FLOAT | Movement range (0-1 normalized) |
| `frequency` | FLOAT | Cycles per sequence |
| `phase_offset` | FLOAT | Starting angle in degrees |
| `direction` | dropdown | clockwise/counterclockwise/inward/outward/etc |
| `seed` | INT | Random seed for random_walk and bounce |

### Outputs

| Output | Type | Description |
|--------|------|-------------|
| `coordinates` | STRING | Generated trajectory JSON |
| `width` | INT | Canvas width (passthrough) |
| `height` | INT | Canvas height (passthrough) |
| `num_frames` | INT | Frame count (passthrough) |

### Workflow

**Direct to WanMove:**
```
[WanTrajectoryGenerator] → coordinates → [WanMove]
```

**With audio modulation:**
```
[WanTrajectoryGenerator] → coordinates → [WanSoundTrajectory] → track_coords → [WanMove]
                                                ↑
                                            audio
```

**Save for later:**
```
[WanTrajectoryGenerator] → coordinates → [WanTrajectorySaver]
```

### Example: Orbiting points with audio wobble

1. Generator: `orbit`, 3 tracks, frequency 2.0
2. Sound Trajectory: `bass` mode, perpendicular, intensity 50
3. Result: Points orbit the center while wobbling to the kick drum

### Example: Converging explosion

1. Generator: `diverge`, 5 tracks, amplitude 0.4
2. Loader: `reverse_time` = True (now it's converging)
3. Result: Features from edges all collapse to center

---

## Pose To Tracks

Convert DWPose skeleton keypoints to trajectory tracks. Place tracks at body part positions from a single image.

**Requires:** [ComfyUI ControlNet Aux](https://github.com/Fannovel16/comfyui_controlnet_aux) for DWPose Estimator node.

### Keypoint Selections

| Selection | Body Parts |
|-----------|------------|
| `head` | Nose |
| `shoulders` | Left + right shoulder |
| `elbows` | Left + right elbow |
| `wrists` | Left + right wrist |
| `hips` | Left + right hip |
| `knees` | Left + right knee |
| `ankles` | Left + right ankle |
| `upper_body` | Shoulders, elbows, wrists (6 points) |
| `lower_body` | Hips, knees, ankles (6 points) |
| `torso` | Shoulders + hips (4 points) |
| `limbs` | Wrists + ankles (4 points) |
| `all` | All 18 body keypoints |

### Inputs

| Input | Type | Description |
|-------|------|-------------|
| `pose_keypoint` | POSE_KEYPOINT | Connect from DWPose Estimator |
| `keypoint_selection` | dropdown | Which body parts become tracks |
| `num_frames` | INT | Frame count (positions repeat since single image) |
| `target_width` | INT | Scale coordinates to this width |
| `target_height` | INT | Scale coordinates to this height |
| `existing_coordinates` | STRING | Optional - append pose tracks to existing |
| `min_confidence` | FLOAT | Skip low-confidence detections |

### Outputs

| Output | Type | Description |
|--------|------|-------------|
| `coordinates` | STRING | Trajectory JSON with one track per keypoint |
| `width` | INT | Target width (passthrough) |
| `height` | INT | Target height (passthrough) |

### Workflow

**Basic pose-driven tracks:**
```
[Image] → [DWPose Estimator] → POSE_KEYPOINT → [WanPoseToTracks] → coordinates → [WanMove]
```

**Combined with generator:**
```
[WanTrajectoryGenerator] → coordinates ──┐
                                         ↓
[DWPose] → [WanPoseToTracks] ← existing_coordinates
                  ↓
           combined tracks → [WanSoundTrajectory] → [WanMove]
```

**With audio modulation:**
```
[DWPose] → [WanPoseToTracks] → coordinates → [WanSoundTrajectory] → track_coords → [WanMove]
                                                    ↑
                                                 audio
```

### Tips

1. **Match your resolution**: Set `target_width` and `target_height` to match your actual output resolution.

2. **Check confidence**: If keypoints are missing, lower `min_confidence`. If you're getting phantom points, raise it.

3. **DWPose settings**: In DWPose Estimator, you can disable `detect_hand` and `detect_face` if you only need body keypoints.

4. **Frame inflation bug**: There's a known issue where WanMove sometimes inflates the frame count unexpectedly. Cause unknown - rebuilding the workflow from scratch tends to fix it.

---

## License

MIT - do whatever you want with it.

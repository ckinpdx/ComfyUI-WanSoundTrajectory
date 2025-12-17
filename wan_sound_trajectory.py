"""
WanSoundTrajectory - Audio-driven path modulation for WanMove
Takes coordinates from SplineEditor and modulates them based on audio analysis
"""

import torch
import numpy as np
import json
import math
import gc
import cv2
from typing import Dict, Any, Tuple, List


class WanSoundTrajectory:
    """
    Modulates path coordinates based on audio analysis.
    Takes a base path from SplineEditor and warps it using audio features.
    """

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("track_coords",)
    FUNCTION = "modulate_path"
    CATEGORY = "WanSoundTrajectory"
    DESCRIPTION = """
Modulates path coordinates based on audio analysis for WanMove.

Takes a base path from SplineEditor (or any coordinate source) and 
warps/modulates it based on audio features like amplitude envelope,
bass energy, treble energy, or onset detection.

The coordinate count determines how much audio is analyzed - if you 
provide 81 points at 30fps, it analyzes ~2.7 seconds of audio.

Modulation directions:
- perpendicular: pushes points sideways relative to path direction
- radial: pushes points toward/away from path center
- along_path: speeds up/slows down movement along the path

Connect output to WanVideoAddWanMoveTracks or WanMove_native.
"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO", {"tooltip": "Input audio waveform. The node analyzes this to drive path modulation."}),
                "coordinates": ("STRING", {"forceInput": True, "tooltip": "Path coordinates JSON from SplineEditor, TrajectoryGenerator, or TrajectoryLoader. Format: [{\"x\": n, \"y\": n}, ...]"}),
                "fps": ("INT", {
                    "default": 30,
                    "min": 1,
                    "max": 120,
                    "step": 1,
                    "tooltip": "Video frame rate. Used to calculate how much audio corresponds to each coordinate point."
                }),
                "modulation_mode": ([
                    "envelope",
                    "bass",
                    "treble", 
                    "onsets",
                    "bass_treble_split",
                ], {
                    "default": "envelope",
                    "tooltip": "envelope=overall loudness, bass=low frequencies (kicks), treble=high frequencies (hihats), onsets=transient hits"
                }),
                "modulation_direction": ([
                    "perpendicular",
                    "radial",
                    "along_path",
                ], {
                    "default": "perpendicular",
                    "tooltip": "perpendicular=sideways wobble, radial=breathe in/out from center, along_path=speed variation"
                }),
                "intensity": ("FLOAT", {
                    "default": 50.0,
                    "min": 0.0,
                    "max": 500.0,
                    "step": 1.0,
                    "tooltip": "Maximum displacement in pixels. Start low (30-50), increase for more dramatic movement."
                }),
            },
            "optional": {
                "smoothing": ("FLOAT", {
                    "default": 0.3,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "tooltip": "Temporal smoothing (0=instant/jittery, 0.3=responsive, 0.7+=smooth/laggy). Prevents harsh movements."
                }),
                "normalize": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Scale audio features to 0-1 range. Keeps consistent modulation regardless of source volume."
                }),
                "bidirectional": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "OFF=push outward on loud, return on quiet. ON=oscillate around path center (usually better for music)."
                }),
                "frequency_split_hz": ("INT", {
                    "default": 200,
                    "min": 50,
                    "max": 2000,
                    "step": 10,
                    "tooltip": "Cutoff between bass/treble modes. 150-200 for EDM, 250-300 for rock/acoustic."
                }),
                "modulate_static_points": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Apply modulation to fixed/static points. Disable to keep anchor points stable."
                }),
                "static_point_axis": (["horizontal", "vertical", "both", "radial_from_center", "local_orbit"], {
                    "default": "both",
                    "tooltip": "Direction for static point wobble. radial_from_center pushes toward/away from frame center. local_orbit makes each point jitter around its own home position."
                }),
            }
        }

    def analyze_audio(
        self, 
        waveform: np.ndarray, 
        sample_rate: int, 
        num_frames: int,
        mode: str,
        smoothing: float,
        normalize: bool,
        frequency_split_hz: int
    ) -> np.ndarray:
        """
        Analyze audio and return per-frame modulation values.
        
        Args:
            waveform: Mono audio waveform as numpy array
            sample_rate: Audio sample rate
            num_frames: Number of frames to output
            mode: Analysis mode (envelope, bass, treble, onsets, bass_treble_split)
            smoothing: Temporal smoothing factor
            normalize: Whether to normalize output to 0-1
            frequency_split_hz: Frequency to split bass/treble
            
        Returns:
            Array of shape (num_frames,) or (num_frames, 2) for bass_treble_split
        """
        # Calculate samples per frame
        samples_per_frame = sample_rate / num_frames if num_frames > 0 else sample_rate
        hop_length = max(1, int(len(waveform) / num_frames))
        
        if mode == "envelope":
            # RMS envelope
            values = self._compute_rms(waveform, hop_length, num_frames)
            
        elif mode == "bass":
            # Low frequency energy
            values = self._compute_band_energy(
                waveform, sample_rate, hop_length, num_frames,
                low_hz=20, high_hz=frequency_split_hz
            )
            
        elif mode == "treble":
            # High frequency energy
            values = self._compute_band_energy(
                waveform, sample_rate, hop_length, num_frames,
                low_hz=frequency_split_hz, high_hz=sample_rate // 2
            )
            
        elif mode == "onsets":
            # Onset detection
            values = self._compute_onsets(waveform, sample_rate, hop_length, num_frames)
            
        elif mode == "bass_treble_split":
            # Both bass and treble as separate channels
            bass = self._compute_band_energy(
                waveform, sample_rate, hop_length, num_frames,
                low_hz=20, high_hz=frequency_split_hz
            )
            treble = self._compute_band_energy(
                waveform, sample_rate, hop_length, num_frames,
                low_hz=frequency_split_hz, high_hz=sample_rate // 2
            )
            values = np.stack([bass, treble], axis=1)
        else:
            values = np.ones(num_frames)
        
        # Apply smoothing
        if smoothing > 0 and len(values.shape) == 1:
            values = self._smooth(values, smoothing)
        elif smoothing > 0 and len(values.shape) == 2:
            values[:, 0] = self._smooth(values[:, 0], smoothing)
            values[:, 1] = self._smooth(values[:, 1], smoothing)
        
        # Normalize
        if normalize:
            if len(values.shape) == 1:
                if values.max() > values.min():
                    values = (values - values.min()) / (values.max() - values.min())
                else:
                    values = np.zeros_like(values)
            else:
                for i in range(values.shape[1]):
                    if values[:, i].max() > values[:, i].min():
                        values[:, i] = (values[:, i] - values[:, i].min()) / (values[:, i].max() - values[:, i].min())
                    else:
                        values[:, i] = np.zeros_like(values[:, i])
        
        return values

    def _compute_rms(self, waveform: np.ndarray, hop_length: int, num_frames: int) -> np.ndarray:
        """Compute RMS envelope"""
        rms = []
        for i in range(num_frames):
            start = i * hop_length
            end = min(start + hop_length, len(waveform))
            if start < len(waveform):
                chunk = waveform[start:end]
                rms.append(np.sqrt(np.mean(chunk ** 2)))
            else:
                rms.append(0.0)
        return np.array(rms)

    def _compute_band_energy(
        self, 
        waveform: np.ndarray, 
        sample_rate: int, 
        hop_length: int, 
        num_frames: int,
        low_hz: int,
        high_hz: int
    ) -> np.ndarray:
        """Compute energy in a frequency band using STFT"""
        # Use a window size that gives reasonable frequency resolution
        n_fft = 2048
        
        # Compute STFT
        num_windows = num_frames
        window = np.hanning(n_fft)
        
        energies = []
        for i in range(num_windows):
            start = i * hop_length
            end = start + n_fft
            
            if end <= len(waveform):
                chunk = waveform[start:end] * window
            elif start < len(waveform):
                chunk = np.zeros(n_fft)
                chunk[:len(waveform) - start] = waveform[start:] * window[:len(waveform) - start]
            else:
                chunk = np.zeros(n_fft)
            
            # FFT
            spectrum = np.abs(np.fft.rfft(chunk))
            freqs = np.fft.rfftfreq(n_fft, 1 / sample_rate)
            
            # Get energy in band
            band_mask = (freqs >= low_hz) & (freqs <= high_hz)
            if band_mask.any():
                energy = np.mean(spectrum[band_mask] ** 2)
            else:
                energy = 0.0
            energies.append(energy)
        
        return np.sqrt(np.array(energies))  # Return amplitude, not power

    def _compute_onsets(
        self, 
        waveform: np.ndarray, 
        sample_rate: int, 
        hop_length: int, 
        num_frames: int
    ) -> np.ndarray:
        """Compute onset strength envelope"""
        # Compute spectral flux as onset indicator
        n_fft = 2048
        window = np.hanning(n_fft)
        
        prev_spectrum = None
        onsets = []
        
        for i in range(num_frames):
            start = i * hop_length
            end = start + n_fft
            
            if end <= len(waveform):
                chunk = waveform[start:end] * window
            elif start < len(waveform):
                chunk = np.zeros(n_fft)
                chunk[:len(waveform) - start] = waveform[start:] * window[:len(waveform) - start]
            else:
                chunk = np.zeros(n_fft)
            
            spectrum = np.abs(np.fft.rfft(chunk))
            
            if prev_spectrum is not None:
                # Spectral flux (only positive differences)
                flux = np.sum(np.maximum(0, spectrum - prev_spectrum))
                onsets.append(flux)
            else:
                onsets.append(0.0)
            
            prev_spectrum = spectrum
        
        return np.array(onsets)

    def _smooth(self, values: np.ndarray, factor: float) -> np.ndarray:
        """Apply exponential moving average smoothing"""
        alpha = 1 - factor
        smoothed = np.zeros_like(values)
        smoothed[0] = values[0]
        for i in range(1, len(values)):
            smoothed[i] = alpha * values[i] + factor * smoothed[i - 1]
        return smoothed

    def _compute_path_tangents(self, points: List[Dict]) -> List[Tuple[float, float]]:
        """Compute normalized tangent vectors at each point"""
        tangents = []
        n = len(points)
        
        for i in range(n):
            if i == 0:
                # Forward difference
                dx = points[1]['x'] - points[0]['x']
                dy = points[1]['y'] - points[0]['y']
            elif i == n - 1:
                # Backward difference
                dx = points[n - 1]['x'] - points[n - 2]['x']
                dy = points[n - 1]['y'] - points[n - 2]['y']
            else:
                # Central difference
                dx = points[i + 1]['x'] - points[i - 1]['x']
                dy = points[i + 1]['y'] - points[i - 1]['y']
            
            # Normalize
            length = math.sqrt(dx * dx + dy * dy)
            if length > 0:
                tangents.append((dx / length, dy / length))
            else:
                tangents.append((1.0, 0.0))  # Default to horizontal
        
        return tangents

    def _compute_path_center(self, points: List[Dict]) -> Tuple[float, float]:
        """Compute centroid of path"""
        cx = sum(p['x'] for p in points) / len(points)
        cy = sum(p['y'] for p in points) / len(points)
        return cx, cy

    def _is_static_point(self, points: List[Dict], threshold: float = 1.0) -> bool:
        """
        Detect if a track is a static/fixed point (all coordinates essentially the same).
        
        Args:
            points: List of coordinate dicts
            threshold: Maximum pixel variance to consider static
            
        Returns:
            True if the track is a static point
        """
        if len(points) <= 1:
            return True
            
        xs = [p['x'] for p in points]
        ys = [p['y'] for p in points]
        
        x_range = max(xs) - min(xs)
        y_range = max(ys) - min(ys)
        
        return x_range <= threshold and y_range <= threshold

    def modulate_path(
        self,
        audio: Dict[str, Any],
        coordinates: str,
        fps: int,
        modulation_mode: str,
        modulation_direction: str,
        intensity: float,
        smoothing: float = 0.3,
        normalize: bool = True,
        bidirectional: bool = False,
        frequency_split_hz: int = 200,
        modulate_static_points: bool = True,
        static_point_axis: str = "both"
    ) -> Tuple[str]:
        """
        Modulate path coordinates based on audio analysis.
        """
        print(f"\n{'='*60}")
        print(f"[WanSoundTrajectory] Processing audio-driven path modulation")
        print(f"[WanSoundTrajectory] Mode: {modulation_mode}, Direction: {modulation_direction}")
        print(f"[WanSoundTrajectory] Intensity: {intensity}, Smoothing: {smoothing}")
        print(f"{'='*60}\n")

        # Parse coordinates
        try:
            parsed = json.loads(coordinates.replace("'", '"'))
            
            # Handle empty input
            if not parsed or len(parsed) == 0:
                print(f"[WanSoundTrajectory] ERROR: Empty coordinates received")
                return (coordinates,)
            
            # Handle single track vs multiple tracks
            if isinstance(parsed[0], dict) and 'x' in parsed[0]:
                # Single track
                tracks = [parsed]
            else:
                # Multiple tracks
                tracks = parsed
                
            print(f"[WanSoundTrajectory] Parsed {len(tracks)} track(s)")
            
        except json.JSONDecodeError as e:
            print(f"[WanSoundTrajectory] ERROR: Failed to parse coordinates: {e}")
            return (coordinates,)

        # Get audio data
        waveform = audio['waveform']
        sample_rate = audio['sample_rate']
        
        # Convert to numpy mono
        if isinstance(waveform, torch.Tensor):
            waveform = waveform.cpu().numpy()
        
        if len(waveform.shape) == 3:  # [batch, channels, samples]
            waveform = waveform.squeeze(0)
        if len(waveform.shape) == 2:  # [channels, samples]
            waveform = np.mean(waveform, axis=0)
        
        print(f"[WanSoundTrajectory] Audio: {len(waveform)} samples at {sample_rate}Hz")

        # Process each track
        modulated_tracks = []
        
        # Generate per-point random angles for local_orbit mode (seeded for consistency)
        np.random.seed(42)
        
        for track_idx, track in enumerate(tracks):
            num_frames = len(track)
            print(f"[WanSoundTrajectory] Track {track_idx}: {num_frames} points")
            
            # Calculate how much audio we need
            audio_duration = num_frames / fps
            samples_needed = int(audio_duration * sample_rate)
            
            print(f"[WanSoundTrajectory] Need {audio_duration:.2f}s of audio ({samples_needed} samples)")
            
            # Trim or pad waveform
            if len(waveform) >= samples_needed:
                track_waveform = waveform[:samples_needed]
            else:
                track_waveform = np.pad(waveform, (0, samples_needed - len(waveform)))
                print(f"[WanSoundTrajectory] WARNING: Audio shorter than needed, padding with silence")
            
            # Analyze audio
            mod_values = self.analyze_audio(
                track_waveform, sample_rate, num_frames,
                modulation_mode, smoothing, normalize, frequency_split_hz
            )
            
            print(f"[WanSoundTrajectory] Modulation values range: {mod_values.min():.3f} to {mod_values.max():.3f}")
            
            # Compute path geometry
            tangents = self._compute_path_tangents(track)
            center_x, center_y = self._compute_path_center(track)
            
            # Detect if this is a static/fixed point
            is_static = self._is_static_point(track)
            
            if is_static:
                print(f"[WanSoundTrajectory] Track {track_idx} is a static point")
                if not modulate_static_points:
                    print(f"[WanSoundTrajectory] Skipping modulation (modulate_static_points=False)")
                    modulated_tracks.append(track)
                    continue
            
            # Modulate each point
            modulated_track = []
            
            # Get frame center for radial_from_center mode (assume standard video dimensions)
            # We'll use the first point as reference since we don't have frame dims
            if is_static and static_point_axis == "radial_from_center":
                # Estimate frame center - this is a guess, could be improved with actual dims
                # For now, assume the point isn't at center and use a reasonable frame center
                frame_center_x = 512  # Default assumption
                frame_center_y = 512
            
            # For local_orbit mode, generate a consistent random angle for this track
            if is_static and static_point_axis == "local_orbit":
                # Use track index to seed unique but consistent angle per track
                orbit_base_angle = (track_idx * 137.5) % 360  # Golden angle distribution
                orbit_speed = 2.0  # Rotations over the duration
            
            # Store home position for local_orbit
            if is_static and len(track) > 0:
                home_x = track[0]['x']
                home_y = track[0]['y']
            
            for i, point in enumerate(track):
                x, y = point['x'], point['y']
                
                # Get modulation value for this frame
                if modulation_mode == "bass_treble_split":
                    mod_bass = mod_values[i, 0]
                    mod_treble = mod_values[i, 1]
                    mod = mod_bass  # Use bass for primary, could mix
                else:
                    mod = mod_values[i]
                
                # Apply bidirectional offset
                if bidirectional:
                    mod = mod * 2 - 1  # Remap 0-1 to -1 to 1
                
                # Calculate displacement based on direction
                if is_static:
                    # Static point handling
                    if static_point_axis == "horizontal":
                        dx = mod * intensity
                        dy = 0
                    elif static_point_axis == "vertical":
                        dx = 0
                        dy = mod * intensity
                    elif static_point_axis == "both":
                        # Diagonal movement
                        dx = mod * intensity * 0.707  # cos(45°)
                        dy = mod * intensity * 0.707  # sin(45°)
                    elif static_point_axis == "radial_from_center":
                        # Push away from frame center
                        rx = x - frame_center_x
                        ry = y - frame_center_y
                        dist = math.sqrt(rx * rx + ry * ry)
                        if dist > 0:
                            rx, ry = rx / dist, ry / dist
                        else:
                            rx, ry = 1, 0  # Default direction if at center
                        dx = rx * mod * intensity
                        dy = ry * mod * intensity
                    elif static_point_axis == "local_orbit":
                        # Jitter/orbit around home position
                        # The angle changes over time, amplitude controlled by audio
                        progress = i / max(num_frames - 1, 1)
                        orbit_angle = math.radians(orbit_base_angle + progress * 360 * orbit_speed)
                        
                        # Displacement is a circle around home, radius scaled by audio
                        radius = mod * intensity
                        dx = math.cos(orbit_angle) * radius
                        dy = math.sin(orbit_angle) * radius
                        
                        # Apply from home position, not current position
                        x = home_x
                        y = home_y
                    else:
                        dx, dy = 0, 0
                        
                elif modulation_direction == "perpendicular":
                    # Perpendicular to path tangent
                    tx, ty = tangents[i]
                    # Normal is perpendicular to tangent
                    nx, ny = -ty, tx
                    dx = nx * mod * intensity
                    dy = ny * mod * intensity
                    
                elif modulation_direction == "radial":
                    # Toward/away from path center
                    rx = x - center_x
                    ry = y - center_y
                    dist = math.sqrt(rx * rx + ry * ry)
                    if dist > 0:
                        rx, ry = rx / dist, ry / dist
                    else:
                        rx, ry = 0, 0
                    dx = rx * mod * intensity
                    dy = ry * mod * intensity
                    
                elif modulation_direction == "along_path":
                    # This mode warps timing rather than position
                    # We'll handle this differently - shift point indices
                    # For now, just apply tangent direction
                    tx, ty = tangents[i]
                    dx = tx * mod * intensity
                    dy = ty * mod * intensity
                else:
                    dx, dy = 0, 0
                
                new_x = x + dx
                new_y = y + dy
                
                # Cast to Python float and round for JSON serialization and memory
                modulated_track.append({"x": round(float(new_x), 2), "y": round(float(new_y), 2)})
            
            modulated_tracks.append(modulated_track)
        
        # Output format matches input format
        if len(modulated_tracks) == 1:
            output = modulated_tracks[0]
        else:
            output = modulated_tracks
        
        output_str = json.dumps(output)
        
        print(f"\n{'='*60}")
        print(f"[WanSoundTrajectory] Complete! Output {len(output_str)} chars")
        print(f"{'='*60}\n")
        
        # Cleanup
        gc.collect()
        
        return (output_str,)


class WanTrajectorySaver:
    """
    Saves trajectory coordinates to a JSON file for later reuse.
    Stores metadata about original dimensions and frame count.
    """

    RETURN_TYPES = ()
    OUTPUT_NODE = True
    FUNCTION = "save_trajectory"
    CATEGORY = "WanSoundTrajectory"
    DESCRIPTION = """
Saves SplineEditor coordinates to a JSON file for later reuse.

Stores the raw trajectory along with metadata (width, height, frame count, track count)
so the loader can rescale/resample to different specs.

Files are saved to: ComfyUI-WanSoundTrajectory/saved_trajectories/
"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "coordinates": ("STRING", {"forceInput": True, "tooltip": "Trajectory coordinates to save. From SplineEditor, Generator, or any coordinate source."}),
                "filename": ("STRING", {"default": "my_trajectory", "tooltip": "Name for the saved file (no extension). Will be saved as filename.json"}),
                "width": ("INT", {"default": 512, "min": 64, "max": 4096, "step": 8, "tooltip": "Canvas width these coordinates were designed for. Used for rescaling on load."}),
                "height": ("INT", {"default": 512, "min": 64, "max": 4096, "step": 8, "tooltip": "Canvas height these coordinates were designed for. Used for rescaling on load."}),
            }
        }

    def save_trajectory(self, coordinates: str, filename: str, width: int, height: int):
        import os
        
        # Parse coordinates to extract metadata
        try:
            parsed = json.loads(coordinates.replace("'", '"'))
            
            # Handle single track vs multiple tracks
            if isinstance(parsed[0], dict) and 'x' in parsed[0]:
                tracks = [parsed]
            else:
                tracks = parsed
            
            track_count = len(tracks)
            frame_count = len(tracks[0]) if tracks else 0
            
        except (json.JSONDecodeError, IndexError, KeyError) as e:
            print(f"[WanTrajectorySaver] ERROR: Failed to parse coordinates: {e}")
            return ()
        
        # Build save data with metadata
        save_data = {
            "metadata": {
                "width": width,
                "height": height,
                "frames": frame_count,
                "tracks": track_count
            },
            "coordinates": tracks
        }
        
        # Create save directory
        script_dir = os.path.dirname(os.path.abspath(__file__))
        save_dir = os.path.join(script_dir, "saved_trajectories")
        os.makedirs(save_dir, exist_ok=True)
        
        # Save file
        filepath = os.path.join(save_dir, f"{filename}.json")
        with open(filepath, 'w') as f:
            json.dump(save_data, f, indent=2)
        
        print(f"\n{'='*60}")
        print(f"[WanTrajectorySaver] Saved trajectory to: {filepath}")
        print(f"[WanTrajectorySaver] Metadata: {width}x{height}, {frame_count} frames, {track_count} tracks")
        print(f"{'='*60}\n")
        
        # Cleanup
        gc.collect()
        
        return ()


class WanTrajectoryLoader:
    """
    Loads saved trajectory coordinates with optional rescaling and resampling.
    """

    RETURN_TYPES = ("STRING", "INT", "INT", "INT", "INT")
    RETURN_NAMES = ("coordinates", "original_width", "original_height", "original_frames", "track_count")
    FUNCTION = "load_trajectory"
    CATEGORY = "WanSoundTrajectory"
    DESCRIPTION = """
Loads saved trajectory coordinates from JSON files.

Can rescale coordinates to different dimensions and resample to different frame counts.
If target values match originals (or are set to 0), passes through without modification.

Outputs include original metadata so you can see what was loaded.

Press 'R' to refresh node definitions after saving new trajectories.
"""

    @classmethod
    def INPUT_TYPES(cls):
        # Get list of saved trajectories
        import os
        script_dir = os.path.dirname(os.path.abspath(__file__))
        save_dir = os.path.join(script_dir, "saved_trajectories")
        
        if os.path.exists(save_dir):
            files = [f for f in os.listdir(save_dir) if f.endswith('.json')]
            files.sort()
        else:
            files = []
        
        if not files:
            files = ["no_trajectories_saved"]
        
        return {
            "required": {
                "trajectory_file": (files, {"tooltip": "Select a saved trajectory file. Press R to refresh list after saving new files."}),
            },
            "optional": {
                "target_width": ("INT", {"default": 0, "min": 0, "max": 4096, "step": 8, "tooltip": "Rescale to this width. 0 = keep original dimensions from saved metadata."}),
                "target_height": ("INT", {"default": 0, "min": 0, "max": 4096, "step": 8, "tooltip": "Rescale to this height. 0 = keep original dimensions from saved metadata."}),
                "target_frames": ("INT", {"default": 0, "min": 0, "max": 1000, "step": 1, "tooltip": "Resample to this frame count. 0 = keep original frame count. Uses linear interpolation."}),
                "flip_horizontal": ("BOOLEAN", {"default": False, "tooltip": "Mirror the trajectory left/right."}),
                "flip_vertical": ("BOOLEAN", {"default": False, "tooltip": "Mirror the trajectory top/bottom."}),
                "rotate": (["none", "90", "180", "270"], {"default": "none", "tooltip": "Rotate trajectory clockwise by degrees."}),
                "reverse_time": ("BOOLEAN", {"default": False, "tooltip": "Reverse the trajectory direction - end becomes start."}),
            }
        }

    def _resample_track(self, track: List[Dict], target_frames: int) -> List[Dict]:
        """Resample a single track to a different frame count using linear interpolation."""
        if len(track) == target_frames:
            return track
        
        if len(track) == 0:
            return []
        
        if len(track) == 1:
            # Single point - just repeat it
            return [{"x": track[0]["x"], "y": track[0]["y"]} for _ in range(target_frames)]
        
        resampled = []
        for i in range(target_frames):
            # Map target index to source position
            t = i / (target_frames - 1) if target_frames > 1 else 0
            src_pos = t * (len(track) - 1)
            
            # Get surrounding indices
            idx_low = int(src_pos)
            idx_high = min(idx_low + 1, len(track) - 1)
            
            # Interpolation factor
            frac = src_pos - idx_low
            
            # Linear interpolation
            x = track[idx_low]["x"] + frac * (track[idx_high]["x"] - track[idx_low]["x"])
            y = track[idx_low]["y"] + frac * (track[idx_high]["y"] - track[idx_low]["y"])
            
            resampled.append({"x": x, "y": y})
        
        return resampled

    def _rescale_track(self, track: List[Dict], orig_width: int, orig_height: int, 
                       target_width: int, target_height: int) -> List[Dict]:
        """Rescale track coordinates to different dimensions."""
        if orig_width == target_width and orig_height == target_height:
            return track
        
        scale_x = target_width / orig_width if orig_width > 0 else 1
        scale_y = target_height / orig_height if orig_height > 0 else 1
        
        return [{"x": p["x"] * scale_x, "y": p["y"] * scale_y} for p in track]

    def _flip_horizontal(self, track: List[Dict], width: int) -> List[Dict]:
        """Mirror track left/right."""
        return [{"x": width - p["x"], "y": p["y"]} for p in track]

    def _flip_vertical(self, track: List[Dict], height: int) -> List[Dict]:
        """Mirror track top/bottom."""
        return [{"x": p["x"], "y": height - p["y"]} for p in track]

    def _rotate_track(self, track: List[Dict], width: int, height: int, angle: str) -> List[Dict]:
        """Rotate track clockwise by 90, 180, or 270 degrees."""
        if angle == "90":
            return [{"x": height - p["y"], "y": p["x"]} for p in track]
        elif angle == "180":
            return [{"x": width - p["x"], "y": height - p["y"]} for p in track]
        elif angle == "270":
            return [{"x": p["y"], "y": width - p["x"]} for p in track]
        return track

    def _reverse_time(self, track: List[Dict]) -> List[Dict]:
        """Reverse the temporal order of the track."""
        return track[::-1]

    def load_trajectory(self, trajectory_file: str, target_width: int = 0, 
                        target_height: int = 0, target_frames: int = 0,
                        flip_horizontal: bool = False, flip_vertical: bool = False,
                        rotate: str = "none", reverse_time: bool = False):
        import os
        
        if trajectory_file == "no_trajectories_saved":
            print("[WanTrajectoryLoader] ERROR: No trajectory files found")
            return ("[]", 0, 0, 0, 0)
        
        # Load file
        script_dir = os.path.dirname(os.path.abspath(__file__))
        save_dir = os.path.join(script_dir, "saved_trajectories")
        filepath = os.path.join(save_dir, trajectory_file)
        
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"[WanTrajectoryLoader] ERROR: Failed to load {filepath}: {e}")
            return ("[]", 0, 0, 0, 0)
        
        # Extract metadata and coordinates
        metadata = data.get("metadata", {})
        orig_width = metadata.get("width", 512)
        orig_height = metadata.get("height", 512)
        orig_frames = metadata.get("frames", 0)
        track_count = metadata.get("tracks", 0)
        tracks = data.get("coordinates", [])
        
        print(f"\n{'='*60}")
        print(f"[WanTrajectoryLoader] Loaded: {trajectory_file}")
        print(f"[WanTrajectoryLoader] Original: {orig_width}x{orig_height}, {orig_frames} frames, {track_count} tracks")
        
        # Determine actual targets (0 means use original)
        actual_width = target_width if target_width > 0 else orig_width
        actual_height = target_height if target_height > 0 else orig_height
        actual_frames = target_frames if target_frames > 0 else orig_frames
        
        # Track transforms applied
        transforms_applied = []
        
        # Process tracks
        processed_tracks = []
        for track in tracks:
            # Resample if needed
            if actual_frames != orig_frames:
                track = self._resample_track(track, actual_frames)
                if "resampled" not in transforms_applied:
                    transforms_applied.append("resampled")
            
            # Rescale if needed
            if actual_width != orig_width or actual_height != orig_height:
                track = self._rescale_track(track, orig_width, orig_height, actual_width, actual_height)
                if "rescaled" not in transforms_applied:
                    transforms_applied.append("rescaled")
            
            # Apply transforms (use actual dimensions after rescaling)
            if flip_horizontal:
                track = self._flip_horizontal(track, actual_width)
                if "flip_h" not in transforms_applied:
                    transforms_applied.append("flip_h")
            
            if flip_vertical:
                track = self._flip_vertical(track, actual_height)
                if "flip_v" not in transforms_applied:
                    transforms_applied.append("flip_v")
            
            if rotate != "none":
                track = self._rotate_track(track, actual_width, actual_height, rotate)
                if f"rotate_{rotate}" not in transforms_applied:
                    transforms_applied.append(f"rotate_{rotate}")
            
            if reverse_time:
                track = self._reverse_time(track)
                if "reversed" not in transforms_applied:
                    transforms_applied.append("reversed")
            
            # Ensure Python floats and round for JSON
            track = [{"x": round(float(p["x"]), 2), "y": round(float(p["y"]), 2)} for p in track]
            processed_tracks.append(track)
        
        if transforms_applied:
            print(f"[WanTrajectoryLoader] Transforms: {', '.join(transforms_applied)}")
        else:
            print(f"[WanTrajectoryLoader] No transforms applied, passing through")
        print(f"{'='*60}\n")
        
        # Output format - single track or multi-track
        if len(processed_tracks) == 1:
            output = processed_tracks[0]
        else:
            output = processed_tracks
        
        output_str = json.dumps(output)
        
        # Cleanup
        gc.collect()
        
        return (output_str, orig_width, orig_height, orig_frames, track_count)


class WanTrajectoryGenerator:
    """
    Generates mathematical trajectory patterns for WanMove.
    No drawing required - pure math creates the motion paths.
    """

    RETURN_TYPES = ("STRING", "INT", "INT", "INT")
    RETURN_NAMES = ("coordinates", "width", "height", "num_frames")
    FUNCTION = "generate"
    CATEGORY = "WanSoundTrajectory"
    DESCRIPTION = """
Generates mathematical trajectory patterns for WanMove.

Instead of drawing paths, create them from math:
- oscillate: ping-pong back and forth
- spiral: spiral outward/inward
- orbit: circular motion around center
- diverge: one point splits into many
- converge: many points merge to one
- random_walk: brownian motion drift
- bounce: ball bouncing off edges
- zoom: radial motion toward/away from center
- wave: synchronized wave motion

Output feeds directly to WanMove, WanSoundTrajectory, or WanTrajectorySaver.
"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pattern": ([
                    "oscillate",
                    "spiral",
                    "orbit",
                    "diverge",
                    "converge",
                    "random_walk",
                    "bounce",
                    "zoom",
                    "wave",
                ], {"default": "oscillate", "tooltip": "oscillate=bounce, spiral=spin in/out, orbit=circle, diverge=explode, converge=implode, bounce=ball physics, zoom=radial"}),
                "num_frames": ("INT", {"default": 81, "min": 2, "max": 1000, "step": 1, "tooltip": "Total frames. Should match your video length. Must be valid WanMove count (81, 101, etc)."}),
                "num_tracks": ("INT", {"default": 1, "min": 1, "max": 20, "step": 1, "tooltip": "Number of separate track points. 1=single path, multiple=complex patterns."}),
                "width": ("INT", {"default": 832, "min": 64, "max": 4096, "step": 8, "tooltip": "Canvas width in pixels. Should match your output resolution."}),
                "height": ("INT", {"default": 480, "min": 64, "max": 4096, "step": 8, "tooltip": "Canvas height in pixels. Should match your output resolution."}),
            },
            "optional": {
                "center_x": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Center X position (0=left, 0.5=middle, 1=right). Origin point for most patterns."}),
                "center_y": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Center Y position (0=top, 0.5=middle, 1=bottom). Origin point for most patterns."}),
                "amplitude": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Movement range as fraction of canvas. 0.3 = 30% of canvas dimension."}),
                "frequency": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 10.0, "step": 0.1, "tooltip": "Cycles per video. 1.0=one complete cycle, 2.0=two cycles, 0.5=half cycle."}),
                "phase_offset": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 360.0, "step": 1.0, "tooltip": "Starting angle in degrees. Offsets where motion begins in the cycle."}),
                "direction": (["clockwise", "counterclockwise", "outward", "inward", "horizontal", "vertical", "diagonal"], 
                             {"default": "clockwise", "tooltip": "Movement direction. Available options depend on pattern type."}),
                "seed": ("INT", {"default": 42, "min": 0, "max": 0xffffffff, "step": 1, "tooltip": "Random seed for random_walk and bounce patterns. Same seed = same path."}),
            }
        }

    def _oscillate(self, num_frames: int, num_tracks: int, width: int, height: int,
                   center_x: float, center_y: float, amplitude: float, frequency: float,
                   phase_offset: float, direction: str, **kwargs) -> List[List[Dict]]:
        """Ping-pong oscillation pattern."""
        tracks = []
        cx, cy = center_x * width, center_y * height
        amp_x = amplitude * width
        amp_y = amplitude * height
        
        for t in range(num_tracks):
            track = []
            track_phase = phase_offset + (t * 360 / num_tracks)
            
            for f in range(num_frames):
                progress = f / max(num_frames - 1, 1)
                angle = math.radians(track_phase) + progress * frequency * 2 * math.pi
                
                if direction == "horizontal":
                    x = cx + math.sin(angle) * amp_x
                    y = cy
                elif direction == "vertical":
                    x = cx
                    y = cy + math.sin(angle) * amp_y
                else:  # diagonal
                    x = cx + math.sin(angle) * amp_x
                    y = cy + math.sin(angle) * amp_y
                
                track.append({"x": x, "y": y})
            tracks.append(track)
        
        return tracks

    def _spiral(self, num_frames: int, num_tracks: int, width: int, height: int,
                center_x: float, center_y: float, amplitude: float, frequency: float,
                phase_offset: float, direction: str, **kwargs) -> List[List[Dict]]:
        """Spiral pattern - outward or inward."""
        tracks = []
        cx, cy = center_x * width, center_y * height
        max_radius = amplitude * min(width, height)
        
        for t in range(num_tracks):
            track = []
            track_phase = phase_offset + (t * 360 / num_tracks)
            
            for f in range(num_frames):
                progress = f / max(num_frames - 1, 1)
                angle = math.radians(track_phase) + progress * frequency * 2 * math.pi
                
                if direction == "inward":
                    radius = max_radius * (1 - progress)
                else:  # outward
                    radius = max_radius * progress
                
                x = cx + math.cos(angle) * radius
                y = cy + math.sin(angle) * radius
                
                track.append({"x": x, "y": y})
            tracks.append(track)
        
        return tracks

    def _orbit(self, num_frames: int, num_tracks: int, width: int, height: int,
               center_x: float, center_y: float, amplitude: float, frequency: float,
               phase_offset: float, direction: str, **kwargs) -> List[List[Dict]]:
        """Circular orbit pattern."""
        tracks = []
        cx, cy = center_x * width, center_y * height
        radius = amplitude * min(width, height)
        
        direction_mult = -1 if direction == "counterclockwise" else 1
        
        for t in range(num_tracks):
            track = []
            track_phase = phase_offset + (t * 360 / num_tracks)
            
            for f in range(num_frames):
                progress = f / max(num_frames - 1, 1)
                angle = math.radians(track_phase) + direction_mult * progress * frequency * 2 * math.pi
                
                x = cx + math.cos(angle) * radius
                y = cy + math.sin(angle) * radius
                
                track.append({"x": x, "y": y})
            tracks.append(track)
        
        return tracks

    def _diverge(self, num_frames: int, num_tracks: int, width: int, height: int,
                 center_x: float, center_y: float, amplitude: float, frequency: float,
                 phase_offset: float, direction: str, **kwargs) -> List[List[Dict]]:
        """One origin point, tracks spread outward."""
        tracks = []
        cx, cy = center_x * width, center_y * height
        max_radius = amplitude * min(width, height)
        
        for t in range(num_tracks):
            track = []
            angle = math.radians(phase_offset + (t * 360 / num_tracks))
            
            for f in range(num_frames):
                progress = f / max(num_frames - 1, 1)
                radius = max_radius * progress
                
                x = cx + math.cos(angle) * radius
                y = cy + math.sin(angle) * radius
                
                track.append({"x": x, "y": y})
            tracks.append(track)
        
        return tracks

    def _converge(self, num_frames: int, num_tracks: int, width: int, height: int,
                  center_x: float, center_y: float, amplitude: float, frequency: float,
                  phase_offset: float, direction: str, **kwargs) -> List[List[Dict]]:
        """Multiple origins converging to center."""
        tracks = []
        cx, cy = center_x * width, center_y * height
        max_radius = amplitude * min(width, height)
        
        for t in range(num_tracks):
            track = []
            angle = math.radians(phase_offset + (t * 360 / num_tracks))
            
            for f in range(num_frames):
                progress = f / max(num_frames - 1, 1)
                radius = max_radius * (1 - progress)
                
                x = cx + math.cos(angle) * radius
                y = cy + math.sin(angle) * radius
                
                track.append({"x": x, "y": y})
            tracks.append(track)
        
        return tracks

    def _random_walk(self, num_frames: int, num_tracks: int, width: int, height: int,
                     center_x: float, center_y: float, amplitude: float, frequency: float,
                     phase_offset: float, direction: str, seed: int = 42, **kwargs) -> List[List[Dict]]:
        """Brownian motion / random walk."""
        np.random.seed(seed)
        tracks = []
        step_size = amplitude * min(width, height) * 0.1
        
        for t in range(num_tracks):
            track = []
            # Start positions spread around center
            angle = math.radians(phase_offset + (t * 360 / num_tracks))
            start_offset = amplitude * min(width, height) * 0.3
            x = center_x * width + math.cos(angle) * start_offset
            y = center_y * height + math.sin(angle) * start_offset
            
            for f in range(num_frames):
                track.append({"x": x, "y": y})
                # Random step
                x += np.random.uniform(-step_size, step_size)
                y += np.random.uniform(-step_size, step_size)
                # Clamp to bounds
                x = max(0, min(width, x))
                y = max(0, min(height, y))
            
            tracks.append(track)
        
        return tracks

    def _bounce(self, num_frames: int, num_tracks: int, width: int, height: int,
                center_x: float, center_y: float, amplitude: float, frequency: float,
                phase_offset: float, direction: str, seed: int = 42, **kwargs) -> List[List[Dict]]:
        """Ball bouncing off edges."""
        np.random.seed(seed)
        tracks = []
        speed = amplitude * min(width, height) * 0.05 * frequency
        
        for t in range(num_tracks):
            track = []
            # Start at center with random velocity
            x = center_x * width
            y = center_y * height
            angle = math.radians(phase_offset + (t * 360 / num_tracks))
            vx = math.cos(angle) * speed
            vy = math.sin(angle) * speed
            
            margin = 10
            
            for f in range(num_frames):
                track.append({"x": x, "y": y})
                x += vx
                y += vy
                
                # Bounce off edges
                if x <= margin or x >= width - margin:
                    vx = -vx
                    x = max(margin, min(width - margin, x))
                if y <= margin or y >= height - margin:
                    vy = -vy
                    y = max(margin, min(height - margin, y))
            
            tracks.append(track)
        
        return tracks

    def _zoom(self, num_frames: int, num_tracks: int, width: int, height: int,
              center_x: float, center_y: float, amplitude: float, frequency: float,
              phase_offset: float, direction: str, **kwargs) -> List[List[Dict]]:
        """Radial zoom - toward or away from center."""
        tracks = []
        cx, cy = center_x * width, center_y * height
        
        for t in range(num_tracks):
            track = []
            # Start position at edge
            angle = math.radians(phase_offset + (t * 360 / num_tracks))
            start_radius = amplitude * min(width, height)
            
            for f in range(num_frames):
                progress = f / max(num_frames - 1, 1)
                
                if direction == "inward":
                    radius = start_radius * (1 - progress)
                else:  # outward
                    radius = start_radius * progress
                
                x = cx + math.cos(angle) * radius
                y = cy + math.sin(angle) * radius
                
                track.append({"x": x, "y": y})
            tracks.append(track)
        
        return tracks

    def _wave(self, num_frames: int, num_tracks: int, width: int, height: int,
              center_x: float, center_y: float, amplitude: float, frequency: float,
              phase_offset: float, direction: str, **kwargs) -> List[List[Dict]]:
        """Wave pattern - tracks in a line doing synchronized wave."""
        tracks = []
        amp = amplitude * height * 0.5
        
        for t in range(num_tracks):
            track = []
            # Space tracks horizontally
            base_x = width * (t + 1) / (num_tracks + 1)
            base_y = center_y * height
            
            for f in range(num_frames):
                progress = f / max(num_frames - 1, 1)
                # Phase offset based on track position for wave effect
                wave_phase = phase_offset + (t * 45)
                angle = math.radians(wave_phase) + progress * frequency * 2 * math.pi
                
                if direction == "horizontal":
                    x = base_x + math.sin(angle) * amp
                    y = base_y
                else:  # vertical
                    x = base_x
                    y = base_y + math.sin(angle) * amp
                
                track.append({"x": x, "y": y})
            tracks.append(track)
        
        return tracks

    def generate(self, pattern: str, num_frames: int, num_tracks: int, width: int, height: int,
                 center_x: float = 0.5, center_y: float = 0.5, amplitude: float = 0.3,
                 frequency: float = 1.0, phase_offset: float = 0.0, direction: str = "clockwise",
                 seed: int = 42) -> Tuple[str, int, int, int]:
        """Generate trajectory pattern."""
        
        print(f"\n{'='*60}")
        print(f"[WanTrajectoryGenerator] Generating pattern: {pattern}")
        print(f"[WanTrajectoryGenerator] {num_tracks} tracks, {num_frames} frames, {width}x{height}")
        print(f"{'='*60}\n")
        
        # Get the pattern generator method
        generators = {
            "oscillate": self._oscillate,
            "spiral": self._spiral,
            "orbit": self._orbit,
            "diverge": self._diverge,
            "converge": self._converge,
            "random_walk": self._random_walk,
            "bounce": self._bounce,
            "zoom": self._zoom,
            "wave": self._wave,
        }
        
        generator = generators.get(pattern, self._oscillate)
        
        tracks = generator(
            num_frames=num_frames,
            num_tracks=num_tracks,
            width=width,
            height=height,
            center_x=center_x,
            center_y=center_y,
            amplitude=amplitude,
            frequency=frequency,
            phase_offset=phase_offset,
            direction=direction,
            seed=seed,
        )
        
        # Ensure all values are Python floats and rounded for JSON
        for track in tracks:
            for point in track:
                point["x"] = round(float(point["x"]), 2)
                point["y"] = round(float(point["y"]), 2)
        
        # Output format
        if len(tracks) == 1:
            output = tracks[0]
        else:
            output = tracks
        
        output_str = json.dumps(output)
        
        print(f"[WanTrajectoryGenerator] Generated {len(tracks)} tracks with {num_frames} points each")
        
        # Cleanup
        gc.collect()
        
        return (output_str, width, height, num_frames)


class WanPoseToTracks:
    """
    Converts DWPose skeleton keypoints to WanMove trajectory tracks.
    Places tracks at body part positions from a single image.
    """

    RETURN_TYPES = ("STRING", "INT", "INT")
    RETURN_NAMES = ("coordinates", "width", "height")
    FUNCTION = "convert"
    CATEGORY = "WanSoundTrajectory"
    DESCRIPTION = """
Converts DWPose body keypoints to WanMove trajectory tracks.

Takes pose data from a single image and creates static tracks at body part positions.
These can then be animated with WanTrajectoryGenerator patterns or WanSoundTrajectory audio modulation.

Keypoint selections:
- head: nose position
- shoulders: left + right shoulder
- elbows: left + right elbow
- wrists: left + right wrist
- hips: left + right hip
- knees: left + right knee
- ankles: left + right ankle
- upper_body: shoulders, elbows, wrists
- lower_body: hips, knees, ankles
- torso: shoulders, hips
- limbs: wrists, ankles
- all: all detected body points

Can append to existing coordinates to combine with generator patterns.
"""

    # COCO body keypoint indices
    KEYPOINT_MAP = {
        "nose": [0],
        "neck": [1],
        "shoulders": [2, 5],  # right, left
        "elbows": [3, 6],
        "wrists": [4, 7],
        "hips": [8, 11],
        "knees": [9, 12],
        "ankles": [10, 13],
        "eyes": [14, 15],
        "ears": [16, 17],
        "head": [0],  # just nose
        "upper_body": [2, 5, 3, 6, 4, 7],  # shoulders, elbows, wrists
        "lower_body": [8, 11, 9, 12, 10, 13],  # hips, knees, ankles
        "torso": [2, 5, 8, 11],  # shoulders + hips
        "limbs": [4, 7, 10, 13],  # wrists + ankles
        "all": list(range(18)),
    }

    KEYPOINT_NAMES = [
        "nose", "neck", 
        "right_shoulder", "right_elbow", "right_wrist",
        "left_shoulder", "left_elbow", "left_wrist",
        "right_hip", "right_knee", "right_ankle",
        "left_hip", "left_knee", "left_ankle",
        "right_eye", "left_eye", "right_ear", "left_ear"
    ]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pose_keypoint": ("POSE_KEYPOINT", {"tooltip": "Pose data from DWPose Estimator. Connect the POSE_KEYPOINT output."}),
                "keypoint_selection": ([
                    "head",
                    "shoulders", 
                    "elbows",
                    "wrists",
                    "hips",
                    "knees",
                    "ankles",
                    "upper_body",
                    "lower_body",
                    "torso",
                    "limbs",
                    "all",
                ], {"default": "wrists", "tooltip": "Which body parts become tracks. wrists=hands, limbs=hands+feet, torso=shoulders+hips, all=every point."}),
                "num_frames": ("INT", {"default": 81, "min": 1, "max": 1000, "step": 1, "tooltip": "Frame count. Since pose is from single image, positions repeat for all frames."}),
                "target_width": ("INT", {"default": 832, "min": 64, "max": 4096, "step": 8, "tooltip": "Output width. Coordinates are scaled from pose canvas to this resolution."}),
                "target_height": ("INT", {"default": 480, "min": 64, "max": 4096, "step": 8, "tooltip": "Output height. Coordinates are scaled from pose canvas to this resolution."}),
            },
            "optional": {
                "existing_coordinates": ("STRING", {"forceInput": True, "tooltip": "Append pose tracks to existing coordinates. Useful for combining with Generator patterns."}),
                "min_confidence": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.05, "tooltip": "Skip keypoints below this confidence. 0.3=include most, 0.7=only confident detections."}),
            }
        }

    def convert(self, pose_keypoint, keypoint_selection: str, num_frames: int,
                target_width: int, target_height: int,
                existing_coordinates: str = None, min_confidence: float = 0.3):
        
        print(f"\n{'='*60}")
        print(f"[WanPoseToTracks] Converting pose to tracks")
        print(f"[WanPoseToTracks] Selection: {keypoint_selection}, Frames: {num_frames}")
        print(f"{'='*60}\n")

        # Parse pose data
        if isinstance(pose_keypoint, str):
            pose_data = json.loads(pose_keypoint)
        elif isinstance(pose_keypoint, list):
            pose_data = pose_keypoint
        else:
            print(f"[WanPoseToTracks] ERROR: Unexpected pose_keypoint type: {type(pose_keypoint)}")
            return ("[]", target_width, target_height)

        # Get first frame's data
        if len(pose_data) == 0:
            print("[WanPoseToTracks] ERROR: Empty pose data")
            return ("[]", target_width, target_height)

        frame_data = pose_data[0]
        canvas_width = frame_data.get("canvas_width", target_width)
        canvas_height = frame_data.get("canvas_height", target_height)
        
        # Scale factors
        scale_x = target_width / canvas_width
        scale_y = target_height / canvas_height

        # Get people
        people = frame_data.get("people", [])
        if len(people) == 0:
            print("[WanPoseToTracks] ERROR: No people detected in pose")
            return ("[]", target_width, target_height)

        # Use first person
        person = people[0]
        body_keypoints = person.get("pose_keypoints_2d", [])

        if len(body_keypoints) == 0:
            print("[WanPoseToTracks] ERROR: No body keypoints found")
            return ("[]", target_width, target_height)

        # Parse keypoints (x, y, confidence triplets)
        keypoints = []
        for i in range(0, len(body_keypoints), 3):
            if i + 2 < len(body_keypoints):
                x = body_keypoints[i]
                y = body_keypoints[i + 1]
                conf = body_keypoints[i + 2]
                keypoints.append({"x": x, "y": y, "confidence": conf})

        print(f"[WanPoseToTracks] Parsed {len(keypoints)} body keypoints")

        # Get indices for selection
        indices = self.KEYPOINT_MAP.get(keypoint_selection, [0])
        
        # Build tracks
        new_tracks = []
        for idx in indices:
            if idx >= len(keypoints):
                continue
            
            kp = keypoints[idx]
            
            # Skip low confidence or zero position (not detected)
            if kp["confidence"] < min_confidence or (kp["x"] == 0 and kp["y"] == 0):
                print(f"[WanPoseToTracks] Skipping {self.KEYPOINT_NAMES[idx]}: low confidence or not detected")
                continue

            # Scale to target dimensions
            x = kp["x"] * scale_x
            y = kp["y"] * scale_y

            # Create static track (same position repeated for all frames) - rounded
            rx, ry = round(float(x), 2), round(float(y), 2)
            track = [{"x": rx, "y": ry} for _ in range(num_frames)]
            new_tracks.append(track)
            
            print(f"[WanPoseToTracks] Added track for {self.KEYPOINT_NAMES[idx]}: ({rx}, {ry})")

        # Handle existing coordinates
        if existing_coordinates and existing_coordinates.strip():
            try:
                parsed_existing = json.loads(existing_coordinates.replace("'", '"'))
                
                # Normalize to list of tracks
                if isinstance(parsed_existing[0], dict) and 'x' in parsed_existing[0]:
                    existing_tracks = [parsed_existing]
                else:
                    existing_tracks = parsed_existing
                
                # Check frame count matches
                if existing_tracks and len(existing_tracks[0]) != num_frames:
                    print(f"[WanPoseToTracks] WARNING: Existing tracks have {len(existing_tracks[0])} frames, resampling to {num_frames}")
                    # Resample existing tracks
                    resampled = []
                    for track in existing_tracks:
                        if len(track) == num_frames:
                            resampled.append(track)
                        else:
                            # Simple linear resample
                            new_track = []
                            for i in range(num_frames):
                                t = i / max(num_frames - 1, 1)
                                src_pos = t * (len(track) - 1)
                                idx_low = int(src_pos)
                                idx_high = min(idx_low + 1, len(track) - 1)
                                frac = src_pos - idx_low
                                x = track[idx_low]["x"] + frac * (track[idx_high]["x"] - track[idx_low]["x"])
                                y = track[idx_low]["y"] + frac * (track[idx_high]["y"] - track[idx_low]["y"])
                                new_track.append({"x": round(float(x), 2), "y": round(float(y), 2)})
                            resampled.append(new_track)
                    existing_tracks = resampled

                # Combine
                all_tracks = existing_tracks + new_tracks
                print(f"[WanPoseToTracks] Combined {len(existing_tracks)} existing + {len(new_tracks)} new = {len(all_tracks)} total tracks")
            except (json.JSONDecodeError, IndexError, KeyError) as e:
                print(f"[WanPoseToTracks] WARNING: Could not parse existing coordinates: {e}")
                all_tracks = new_tracks
        else:
            all_tracks = new_tracks

        if len(all_tracks) == 0:
            print("[WanPoseToTracks] WARNING: No valid tracks created")
            return ("[]", target_width, target_height)

        # Output format
        if len(all_tracks) == 1:
            output = all_tracks[0]
        else:
            output = all_tracks

        output_str = json.dumps(output)

        print(f"\n{'='*60}")
        print(f"[WanPoseToTracks] Created {len(all_tracks)} tracks with {num_frames} frames each")
        print(f"{'='*60}\n")
        
        # Cleanup
        gc.collect()

        return (output_str, target_width, target_height)


class WanMove3DZoom:
    """
    Creates 3D point cloud from depth map with rotation and zoom animation.
    Supports background/foreground isolation via depth range and mask input.
    """
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE",),  # Expecting Depth Map (B,H,W,C)
                "num_points": ("INT", {"default": 50, "min": 4, "max": 1000, "step": 1}),
                "depth_scale": ("INT", {"default": 100, "min": 10, "max": 1000, "step": 10, "tooltip": "Depth range in pixels - how much Z separation between near and far"}),
                "depth_falloff": ("FLOAT", {"default": 0.5, "min": 0.01, "max": 1.0, "step": 0.01, "tooltip": "Depth curve - 0.1=near focus, 0.5=linear, 1.0=far focus"}),
                "depth_min": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Minimum depth to include (0=far/black). Use with depth_max to select background only."}),
                "depth_max": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Maximum depth to include (1=near/white). Use with depth_min to select background only."}),
                "duration": ("INT", {"default": 81, "min": 1, "max": 240, "step": 1, "tooltip": "Number of frames to generate"}),
                "x_rotation": ("FLOAT", {"default": 0.0, "min": -360.0, "max": 360.0, "step": 1.0, "tooltip": "Vertical orbit - tilt up/down around subject"}),
                "y_rotation": ("FLOAT", {"default": 30.0, "min": -360.0, "max": 360.0, "step": 1.0, "tooltip": "Horizontal orbit - circle left/right around subject"}),
                "z_rotation": ("FLOAT", {"default": 0.0, "min": -360.0, "max": 360.0, "step": 1.0, "tooltip": "Roll - tilt horizon"}),
                "zoom_amount": ("FLOAT", {"default": 0.0, "min": -2.0, "max": 2.0, "step": 0.05, "tooltip": "Zoom intensity - positive=zoom in, negative=zoom out, 0=no zoom"}),
                "trajectory": (["Constant", "Ease In", "Ease Out", "Ease In Out"],),
                "point_radius": ("INT", {"default": 5, "min": 1, "max": 20, "step": 1, "tooltip": "Preview dot size - doesn't affect output"}),
                "export_width": ("INT", {"default": 512, "min": 64, "max": 4096, "tooltip": "Output coordinate space width"}),
                "export_height": ("INT", {"default": 512, "min": 64, "max": 4096, "tooltip": "Output coordinate space height"}),
            },
            "optional": {
                "mask": ("IMAGE", {"tooltip": "Optional mask - white areas include points, black areas exclude. Use for subject isolation."}),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("preview_images", "coord_tracks")
    FUNCTION = "generate"
    CATEGORY = "WanSoundTrajectory"

    def ease(self, t, mode):
        if mode == "Constant":
            return t
        elif mode == "Ease In":
            return t * t * t
        elif mode == "Ease Out":
            return 1 - pow(1 - t, 3)
        elif mode == "Ease In Out":
            if t < 0.5:
                return 4 * t * t * t
            else:
                return 1 - pow(-2 * t + 2, 3) / 2
        return t

    def get_rotation_matrix(self, rx, ry, rz):
        # Convert degrees to radians
        rx, ry, rz = np.radians(rx), np.radians(ry), np.radians(rz)
        
        # Rotation matrices
        Rx = np.array([
            [1, 0, 0],
            [0, np.cos(rx), -np.sin(rx)],
            [0, np.sin(rx), np.cos(rx)]
        ])
        
        Ry = np.array([
            [np.cos(ry), 0, np.sin(ry)],
            [0, 1, 0],
            [-np.sin(ry), 0, np.cos(ry)]
        ])
        
        Rz = np.array([
            [np.cos(rz), -np.sin(rz), 0],
            [np.sin(rz), np.cos(rz), 0],
            [0, 0, 1]
        ])
        
        # Combined rotation: Rz * Ry * Rx
        return Rz @ Ry @ Rx

    def generate(self, images, num_points, depth_scale, depth_falloff, depth_min, depth_max, 
                 duration, x_rotation, y_rotation, z_rotation, zoom_amount, trajectory, 
                 point_radius, export_width, export_height, mask=None):
        
        # 1. Process Input Image (Take first in batch)
        # ComfyUI images are [B, H, W, C] tensors in range 0-1
        t_img = images[0].permute(2, 0, 1)  # C, H, W
        
        # Resize depth map to export size to ensure accurate sampling
        t_img = torch.nn.functional.interpolate(t_img.unsqueeze(0), size=(export_height, export_width), mode='bilinear', align_corners=False).squeeze(0)
        
        # Convert to numpy for pixel access (H, W, C)
        depth_map = t_img.permute(1, 2, 0).cpu().numpy()
        
        # Collapse to grayscale if RGB
        if depth_map.shape[2] > 1:
            depth_map = cv2.cvtColor(depth_map, cv2.COLOR_RGB2GRAY)
        else:
            depth_map = depth_map[:, :, 0]

        # Process mask if provided
        mask_map = None
        if mask is not None:
            t_mask = mask[0].permute(2, 0, 1)  # C, H, W
            t_mask = torch.nn.functional.interpolate(t_mask.unsqueeze(0), size=(export_height, export_width), mode='bilinear', align_corners=False).squeeze(0)
            mask_map = t_mask.permute(1, 2, 0).cpu().numpy()
            if mask_map.shape[2] > 1:
                mask_map = cv2.cvtColor(mask_map, cv2.COLOR_RGB2GRAY)
            else:
                mask_map = mask_map[:, :, 0]

        # 2. Distribute Key Points (Grid)
        aspect = export_width / export_height
        
        # Calculate grid dimensions that approximate num_points while keeping aspect ratio
        rows = int(math.sqrt(num_points / aspect))
        rows = max(2, rows)
        cols = int(num_points / rows)
        cols = max(2, cols)
        
        # Generate Grid Coordinates
        xs = np.linspace(0, export_width - 1, cols)
        ys = np.linspace(0, export_height - 1, rows)
        xv, yv = np.meshgrid(xs, ys)
        
        points_2d = np.stack([xv.flatten(), yv.flatten()], axis=1).astype(int)
        
        # Safety clamp
        if len(points_2d) > 1000:
            points_2d = points_2d[:1000]

        # 3. Create 3D Point Cloud with Unprojection
        points_3d = []
        center_x = export_width / 2.0
        center_y = export_height / 2.0
        
        focal_length = float(export_width)
        base_camera_z = max(export_width * 1.5, depth_scale * 1.5)
        
        # Calculate Exponent for Depth Distribution
        depth_exponent = 0.5 / max(0.01, depth_falloff)

        for x, y in points_2d:
            # Sample Depth: 0.0 (Far/Black) to 1.0 (Near/White)
            d_raw = depth_map[y, x]
            
            # Check depth range filter
            if d_raw < depth_min or d_raw > depth_max:
                continue
            
            # Check mask if provided (white = include, black = exclude)
            if mask_map is not None:
                mask_val = mask_map[y, x]
                if mask_val < 0.5:  # Below threshold = excluded
                    continue
            
            # Apply Depth Falloff Curve
            d_val = math.pow(d_raw, depth_exponent)
            
            # Map Depth to Z
            z = (1.0 - d_val) * depth_scale - (depth_scale / 2.0)
            
            # BACK PROJECTION:
            dist_from_cam = z + base_camera_z
            inverse_scale = dist_from_cam / focal_length
            
            world_x = (x - center_x) * inverse_scale
            world_y = (y - center_y) * inverse_scale
            
            points_3d.append([world_x, world_y, z])
            
        points_3d = np.array(points_3d)

        # Handle case where all points are filtered out
        if len(points_3d) == 0:
            print(f"[WanMove3DZoom] WARNING: No points passed filters. Check depth_min/depth_max or mask.")
            blank = torch.zeros((duration, export_height, export_width, 3))
            return (blank, "[]")

        print(f"[WanMove3DZoom] {len(points_3d)} points passed depth/mask filters")

        # 4. Animate Frames
        preview_frames = []
        all_tracks_formatted = [[] for _ in range(len(points_3d))]
        
        for f in range(duration):
            # Calculate progress (0.0 to 1.0)
            t = f / max(1, duration - 1)
            t_eased = self.ease(t, trajectory)
            
            # Current Rotation Angles
            cur_x_rot = x_rotation * t_eased
            cur_y_rot = y_rotation * t_eased
            cur_z_rot = z_rotation * t_eased
            
            # Current Zoom (animate camera_z)
            # zoom_amount positive = zoom in (smaller camera_z)
            # zoom_amount negative = zoom out (larger camera_z)
            camera_z = base_camera_z * (1.0 - zoom_amount * t_eased)
            camera_z = max(focal_length * 0.5, camera_z)  # Prevent camera from going too close
            
            # Get Rotation Matrix
            R = self.get_rotation_matrix(cur_x_rot, cur_y_rot, cur_z_rot)
            
            # Rotate Points
            rotated_points = points_3d @ R.T
            
            # Create Preview Canvas (Black)
            canvas = np.zeros((export_height, export_width, 3), dtype=np.uint8)
            
            # Sort points by Z (Painter's algorithm)
            indexed_points = list(enumerate(rotated_points))
            indexed_points.sort(key=lambda p: p[1][2], reverse=True)
            
            for original_idx, p in indexed_points:
                px, py, pz = p
                
                # Perspective Projection with animated camera_z
                dist = pz + camera_z
                if dist < 1:
                    dist = 1
                
                scale = focal_length / dist
                
                screen_x = int((px * scale) + center_x)
                screen_y = int((py * scale) + center_y)
                
                # Store Coordinate (rounded for memory efficiency)
                all_tracks_formatted[original_idx].append({
                    "x": round(float(screen_x), 2),
                    "y": round(float(screen_y), 2)
                })
                
                # Drawing for Preview
                r = max(1, int(point_radius * scale))
                
                norm_z = (pz + depth_scale) / (2 * depth_scale)
                norm_z = np.clip(norm_z, 0, 1)
                
                b = int(255 * norm_z)
                r_col = int(255 * (1 - norm_z))
                g = int(100 * (1 - abs(0.5 - norm_z) * 2))
                
                cv2.circle(canvas, (screen_x, screen_y), r, (b, g, r_col), -1)
            
            # Convert OpenCV (BGR) to Tensor (RGB)
            canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
            tensor_img = torch.from_numpy(canvas).float() / 255.0
            preview_frames.append(tensor_img)

        # Stack frames into batch
        preview_tensor = torch.stack(preview_frames)
        
        # Serialize Coordinates
        json_output = json.dumps(all_tracks_formatted)
        
        print(f"[WanMove3DZoom] Generated {len(points_3d)} tracks, {duration} frames, zoom={zoom_amount}")
        
        return (preview_tensor, json_output)


# ComfyUI registration
NODE_CLASS_MAPPINGS = {
    "WanSoundTrajectory": WanSoundTrajectory,
    "WanTrajectorySaver": WanTrajectorySaver,
    "WanTrajectoryLoader": WanTrajectoryLoader,
    "WanTrajectoryGenerator": WanTrajectoryGenerator,
    "WanPoseToTracks": WanPoseToTracks,
    "WanMove3DZoom": WanMove3DZoom,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "WanSoundTrajectory": "Wan Sound Trajectory",
    "WanTrajectorySaver": "Wan Trajectory Saver",
    "WanTrajectoryLoader": "Wan Trajectory Loader",
    "WanTrajectoryGenerator": "Wan Trajectory Generator",
    "WanPoseToTracks": "Wan Pose To Tracks",
    "WanMove3DZoom": "Wan Move 3D Zoom",
}

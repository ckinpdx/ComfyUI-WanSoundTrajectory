"""
WanSoundTrajectory - Audio-driven path modulation for WanMove
Takes coordinates from SplineEditor and modulates them based on audio analysis
"""

import torch
import numpy as np
import json
import math
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
                "audio": ("AUDIO", {"description": "Input audio"}),
                "coordinates": ("STRING", {"forceInput": True, "description": "Path coordinates from SplineEditor"}),
                "fps": ("INT", {
                    "default": 30,
                    "min": 1,
                    "max": 120,
                    "step": 1,
                    "description": "Frames per second for audio alignment"
                }),
                "modulation_mode": ([
                    "envelope",
                    "bass",
                    "treble", 
                    "onsets",
                    "bass_treble_split",
                ], {
                    "default": "envelope",
                    "description": "Which audio feature drives modulation"
                }),
                "modulation_direction": ([
                    "perpendicular",
                    "radial",
                    "along_path",
                ], {
                    "default": "perpendicular",
                    "description": "How audio affects the path"
                }),
                "intensity": ("FLOAT", {
                    "default": 50.0,
                    "min": 0.0,
                    "max": 500.0,
                    "step": 1.0,
                    "description": "Strength of modulation in pixels"
                }),
            },
            "optional": {
                "smoothing": ("FLOAT", {
                    "default": 0.3,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "description": "Temporal smoothing of audio features"
                }),
                "normalize": ("BOOLEAN", {
                    "default": True,
                    "description": "Normalize audio features to 0-1 range"
                }),
                "bidirectional": ("BOOLEAN", {
                    "default": False,
                    "description": "Allow negative displacement (oscillate around path)"
                }),
                "frequency_split_hz": ("INT", {
                    "default": 200,
                    "min": 50,
                    "max": 2000,
                    "step": 10,
                    "description": "Frequency cutoff between bass and treble (Hz)"
                }),
                "modulate_static_points": ("BOOLEAN", {
                    "default": True,
                    "description": "Apply modulation to static/fixed points"
                }),
                "static_point_axis": (["horizontal", "vertical", "both", "radial_from_center"], {
                    "default": "both",
                    "description": "Which axis to wobble static points on"
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
                
                # Cast to Python float for JSON serialization
                modulated_track.append({"x": float(new_x), "y": float(new_y)})
            
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
        
        return (output_str,)


# ComfyUI registration
NODE_CLASS_MAPPINGS = {
    "WanSoundTrajectory": WanSoundTrajectory,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "WanSoundTrajectory": "Wan Sound Trajectory",
}

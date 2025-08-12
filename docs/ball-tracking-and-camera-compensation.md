# Ball Tracking and Camera Compensation

This document provides detailed information on ball tracking techniques, camera motion compensation, and recommendations for scaling the system to full matches.

## Current Implementation Analysis

### Ball Tracking

The current system uses a simple but effective approach:

1. **Detection**: YOLO model detects the ball in each frame
2. **Tracking**: ByteTrack maintains ball identity across frames
3. **Interpolation**: Linear interpolation fills missing detections
4. **Position Assignment**: Ball center coordinates are extracted

**Current Limitations**:
- Linear interpolation in image space (not accounting for ball physics)
- No trajectory modeling or prediction
- Fixed detection confidence threshold
- No occlusion handling beyond interpolation

### Camera Motion Compensation

The system estimates camera movement using:

1. **Feature Detection**: Good features to track (corners, edges) on sidelines
2. **Optical Flow**: Lucas-Kanade method tracks features between frames
3. **Motion Estimation**: Maximum displacement indicates camera movement
4. **Position Adjustment**: Object positions are adjusted by camera motion

**Current Limitations**:
- Single maximum displacement can be noisy
- No distinction between pan, tilt, and zoom
- Feature tracking fails during rapid camera movements
- No global motion model validation

## Recommended Improvements

### 1. Enhanced Ball Tracking

#### Kalman Filter for Trajectory Modeling

```python
class BallTracker:
    def __init__(self):
        # State: [x, y, vx, vy, ax, ay]
        self.kalman = cv2.KalmanFilter(6, 2)
        self.setup_kalman()
    
    def setup_kalman(self):
        # State transition matrix (constant velocity + acceleration)
        self.kalman.transitionMatrix = np.array([
            [1, 0, 1, 0, 0.5, 0],    # x = x + vx + 0.5*ax
            [0, 1, 0, 1, 0, 0.5],    # y = y + vy + 0.5*ay
            [0, 0, 1, 0, 1, 0],      # vx = vx + ax
            [0, 0, 0, 1, 0, 1],      # vy = vy + ay
            [0, 0, 0, 0, 0.9, 0],    # ax = 0.9*ax (decay)
            [0, 0, 0, 0, 0, 0.9]     # ay = 0.9*ay (decay)
        ], np.float32)
        
        # Measurement matrix (only x, y observed)
        self.kalman.measurementMatrix = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0]
        ], np.float32)
        
        # Process noise (adjust based on ball dynamics)
        self.kalman.processNoiseCov = np.eye(6, dtype=np.float32) * 0.01
        
        # Measurement noise (adjust based on detection confidence)
        self.kalman.measurementNoiseCov = np.eye(2, dtype=np.float32) * 1.0
```

#### 1€ Filter for Real-time Smoothing

Alternative to Kalman filter for low-latency applications:

```python
class OneEuroFilter:
    def __init__(self, min_cutoff=1.0, beta=0.0, d_cutoff=1.0):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.x_filter = LowPassFilter(min_cutoff, beta)
        self.dx_filter = LowPassFilter(d_cutoff, beta)
        self.last_time = None
    
    def filter(self, x, time):
        if self.last_time is None:
            self.last_time = time
            return x
        
        dt = time - self.last_time
        if dt <= 0:
            return x
        
        # Smooth the derivative
        dx = (x - self.x_filter.last_value) / dt
        dx_smooth = self.dx_filter.filter(dx, time)
        
        # Adaptive cutoff based on derivative
        cutoff = self.min_cutoff + self.beta * abs(dx_smooth)
        
        # Update the main filter
        x_smooth = self.x_filter.filter(x, time, cutoff)
        self.last_time = time
        
        return x_smooth
```

#### Occlusion Handling

```python
def handle_ball_occlusion(self, ball_tracks, frame_window=10):
    """Handle ball occlusion using motion prediction and context"""
    for frame_idx in range(len(ball_tracks)):
        if not ball_tracks[frame_idx]:  # Ball not detected
            # Predict position using previous motion
            predicted_pos = self.predict_ball_position(
                ball_tracks, frame_idx, frame_window
            )
            
            # Validate prediction using player context
            if self.validate_prediction(predicted_pos, frame_idx):
                ball_tracks[frame_idx] = {
                    1: {"bbox": self.pos_to_bbox(predicted_pos)}
                }
    
    return ball_tracks

def predict_ball_position(self, ball_tracks, frame_idx, window):
    """Predict ball position using polynomial fitting"""
    # Collect recent ball positions
    positions = []
    times = []
    
    for i in range(max(0, frame_idx - window), frame_idx):
        if ball_tracks[i] and 1 in ball_tracks[i]:
            pos = ball_tracks[i][1]["position"]
            positions.append(pos)
            times.append(i)
    
    if len(positions) < 3:
        return None
    
    # Fit polynomial to recent trajectory
    positions = np.array(positions)
    times = np.array(times)
    
    # Fit x and y separately
    x_coeffs = np.polyfit(times, positions[:, 0], 2)
    y_coeffs = np.polyfit(times, positions[:, 1], 2)
    
    # Predict at current frame
    x_pred = np.polyval(x_coeffs, frame_idx)
    y_pred = np.polyval(y_coeffs, frame_idx)
    
    return (x_pred, y_pred)
```

### 2. Robust Camera Motion Estimation

#### Feature Matching + RANSAC Homography

```python
class RobustCameraMotionEstimator:
    def __init__(self):
        self.orb = cv2.ORB_create(nfeatures=1000)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    
    def estimate_camera_motion(self, frame1, frame2):
        """Estimate camera motion using feature matching and RANSAC"""
        # Detect and describe features
        kp1, des1 = self.orb.detectAndCompute(frame1, None)
        kp2, des2 = self.orb.detectAndCompute(frame2, None)
        
        if des1 is None or des2 is None:
            return [0, 0]
        
        # Match features
        matches = self.matcher.match(des1, des2)
        matches = sorted(matches, key=lambda x: x.distance)
        
        # Extract matched points
        src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        
        if len(src_pts) < 4:
            return [0, 0]
        
        # Estimate homography with RANSAC
        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        
        if H is None:
            return [0, 0]
        
        # Extract translation from homography
        translation = self.extract_translation(H)
        
        return translation
    
    def extract_translation(self, H):
        """Extract translation vector from homography matrix"""
        # For small camera movements, translation ≈ H[0:2, 2]
        tx, ty = H[0, 2], H[1, 2]
        
        # Apply scale factor based on image dimensions
        scale_factor = 1.0  # Adjust based on field size
        
        return [tx * scale_factor, ty * scale_factor]
```

#### Multi-scale Motion Estimation

```python
def estimate_multi_scale_motion(self, frame1, frame2):
    """Estimate motion at multiple scales for robustness"""
    motions = []
    
    # Estimate at original resolution
    motion_full = self.estimate_camera_motion(frame1, frame2)
    motions.append(motion_full)
    
    # Estimate at reduced resolutions
    for scale in [0.5, 0.25]:
        h, w = frame1.shape[:2]
        new_h, new_w = int(h * scale), int(w * scale)
        
        frame1_scaled = cv2.resize(frame1, (new_w, new_h))
        frame2_scaled = cv2.resize(frame2, (new_w, new_h))
        
        motion_scaled = self.estimate_camera_motion(frame1_scaled, frame2_scaled)
        # Scale back to original resolution
        motion_scaled = [m / scale for m in motion_scaled]
        motions.append(motion_scaled)
    
    # Combine estimates (median for robustness)
    motions = np.array(motions)
    final_motion = np.median(motions, axis=0)
    
    return final_motion.tolist()
```

### 3. Adaptive Player-Ball Association

#### Field-scale Distance Gating

```python
class AdaptiveBallAssigner:
    def __init__(self):
        self.base_threshold_meters = 2.0  # 2 meters base threshold
        self.ball_speed_factor = 0.5      # Adjust threshold based on ball speed
    
    def assign_ball_to_player(self, players, ball_bbox, ball_speed=None):
        """Assign ball using field-scale distance and adaptive threshold"""
        ball_position = get_center_of_bbox(ball_bbox)
        
        # Transform ball position to field coordinates
        ball_transformed = self.view_transformer.transform_point(ball_position)
        if ball_transformed is None:
            return -1
        
        # Adaptive threshold based on ball speed
        threshold = self.base_threshold_meters
        if ball_speed is not None:
            # Faster ball = larger threshold (more uncertainty)
            threshold += ball_speed * self.ball_speed_factor
        
        min_distance = float('inf')
        assigned_player = -1
        
        for player_id, player in players.items():
            if 'position_transformed' not in player:
                continue
            
            player_pos = player['position_transformed']
            if player_pos is None:
                continue
            
            # Calculate distance in meters
            distance = measure_distance(ball_transformed, player_pos)
            
            if distance < threshold and distance < min_distance:
                min_distance = distance
                assigned_player = player_id
        
        return assigned_player
```

#### Possession Smoothing

```python
def smooth_possession(self, possession_sequence, window_size=5):
    """Smooth ball possession using temporal majority voting"""
    smoothed = []
    
    for i in range(len(possession_sequence)):
        # Get window of possession values
        start_idx = max(0, i - window_size // 2)
        end_idx = min(len(possession_sequence), i + window_size // 2 + 1)
        window = possession_sequence[start_idx:end_idx]
        
        # Count occurrences of each player
        counts = {}
        for pos in window:
            if pos != -1:  # Valid player ID
                counts[pos] = counts.get(pos, 0) + 1
        
        if counts:
            # Select most frequent player in window
            most_frequent = max(counts, key=counts.get)
            smoothed.append(most_frequent)
        else:
            smoothed.append(-1)
    
    return smoothed
```

## Full-Match Scalability

### 1. Streaming Pipeline Design

#### Chunked Processing

```python
class StreamingPipeline:
    def __init__(self, chunk_size=500, overlap=50):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.frame_buffer = []
        self.results_buffer = []
    
    def process_video_stream(self, video_path):
        """Process video in chunks for memory efficiency"""
        cap = cv2.VideoCapture(video_path)
        frame_count = 0
        
        while True:
            # Read chunk of frames
            chunk_frames = []
            for _ in range(self.chunk_size):
                ret, frame = cap.read()
                if not ret:
                    break
                chunk_frames.append(frame)
                frame_count += 1
            
            if not chunk_frames:
                break
            
            # Process chunk
            chunk_results = self.process_chunk(chunk_frames, frame_count)
            
            # Save results incrementally
            self.save_chunk_results(chunk_results, frame_count)
            
            # Keep overlap frames for next chunk
            if len(chunk_frames) > self.overlap:
                self.frame_buffer = chunk_frames[-self.overlap:]
        
        cap.release()
        return self.merge_results()
    
    def process_chunk(self, frames, start_frame):
        """Process a chunk of frames"""
        # Initialize tracker with overlap frames if available
        if self.frame_buffer:
            frames = self.frame_buffer + frames
        
        # Run detection and tracking
        tracks = self.tracker.get_object_tracks(frames)
        
        # Process pipeline stages
        tracks = self.camera_movement_estimator.process(tracks)
        tracks = self.view_transformer.process(tracks)
        tracks = self.speed_estimator.process(tracks)
        
        return tracks
```

### 2. Memory Management

#### Incremental Processing

```python
class MemoryEfficientProcessor:
    def __init__(self, max_frames_in_memory=1000):
        self.max_frames = max_frames_in_memory
        self.frame_cache = {}
        self.result_cache = {}
    
    def process_with_memory_management(self, video_path):
        """Process video with controlled memory usage"""
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Process in memory-managed chunks
        for start_frame in range(0, total_frames, self.max_frames):
            end_frame = min(start_frame + self.max_frames, total_frames)
            
            # Load chunk
            chunk_frames = self.load_frame_chunk(cap, start_frame, end_frame)
            
            # Process chunk
            chunk_results = self.process_chunk(chunk_frames)
            
            # Save results and clear memory
            self.save_chunk_results(chunk_results, start_frame, end_frame)
            self.clear_memory()
        
        cap.release()
        return self.merge_all_results()
    
    def clear_memory(self):
        """Clear frame and result caches"""
        self.frame_cache.clear()
        self.result_cache.clear()
        gc.collect()  # Force garbage collection
```

### 3. Fault Tolerance and Recovery

#### Checkpointing and Recovery

```python
class FaultTolerantPipeline:
    def __init__(self, checkpoint_interval=100):
        self.checkpoint_interval = checkpoint_interval
        self.checkpoint_dir = "checkpoints"
        os.makedirs(self.checkpoint_dir, exist_ok=True)
    
    def save_checkpoint(self, frame_idx, tracks, metadata):
        """Save processing checkpoint"""
        checkpoint_data = {
            'frame_idx': frame_idx,
            'tracks': tracks,
            'metadata': metadata,
            'timestamp': time.time()
        }
        
        checkpoint_path = os.path.join(
            self.checkpoint_dir, 
            f"checkpoint_{frame_idx:06d}.pkl"
        )
        
        with open(checkpoint_path, 'wb') as f:
            pickle.dump(checkpoint_data, f)
    
    def load_latest_checkpoint(self):
        """Load the most recent checkpoint"""
        checkpoint_files = glob.glob(os.path.join(self.checkpoint_dir, "checkpoint_*.pkl"))
        
        if not checkpoint_files:
            return None
        
        # Find latest checkpoint
        latest_checkpoint = max(checkpoint_files, key=os.path.getctime)
        
        with open(latest_checkpoint, 'rb') as f:
            return pickle.load(f)
    
    def resume_processing(self, video_path):
        """Resume processing from last checkpoint"""
        checkpoint = self.load_latest_checkpoint()
        
        if checkpoint is None:
            # Start from beginning
            return self.process_video(video_path)
        
        # Resume from checkpoint
        start_frame = checkpoint['frame_idx']
        print(f"Resuming from frame {start_frame}")
        
        # Continue processing from checkpoint
        return self.process_video_from_frame(video_path, start_frame, checkpoint)
```

### 4. Performance Optimization

#### GPU/CPU Toggle

```python
class AdaptiveProcessor:
    def __init__(self):
        self.use_gpu = torch.cuda.is_available()
        self.batch_size = 32 if self.use_gpu else 20
    
    def optimize_for_hardware(self):
        """Optimize processing based on available hardware"""
        if self.use_gpu:
            # GPU optimizations
            self.model = self.model.cuda()
            self.batch_size = 32
            self.enable_mixed_precision()
        else:
            # CPU optimizations
            self.model = self.model.cpu()
            self.batch_size = 20
            self.enable_cpu_optimizations()
    
    def enable_mixed_precision(self):
        """Enable mixed precision for GPU processing"""
        if hasattr(torch, 'autocast'):
            self.autocast = torch.autocast(device_type='cuda')
    
    def enable_cpu_optimizations(self):
        """Enable CPU-specific optimizations"""
        torch.set_num_threads(os.cpu_count())
        if hasattr(torch, 'set_num_interop_threads'):
            torch.set_num_interop_threads(1)
```

## Implementation Priority

### Phase 1: Core Improvements (Immediate)
1. **Ball tracking**: Implement Kalman filter or 1€ filter
2. **Camera motion**: Add RANSAC homography estimation
3. **Ball assignment**: Use field-scale distance gating

### Phase 2: Scalability (Short-term)
1. **Streaming**: Implement chunked processing
2. **Memory**: Add incremental result saving
3. **Checkpointing**: Add fault tolerance

### Phase 3: Advanced Features (Medium-term)
1. **Occlusion handling**: Motion prediction and validation
2. **Multi-scale**: Robust motion estimation
3. **Performance**: GPU/CPU optimization

## References

- **ByteTrack**: [arXiv:2110.06864](https://arxiv.org/abs/2110.06864) - Multi-object tracking
- **TrackNet**: [arXiv:1907.03698](https://arxiv.org/abs/1907.03698) - Ball tracking with temporal heatmaps
- **1€ Filter**: [hal-00670496](https://hal.inria.fr/hal-00670496/document) - Real-time smoothing
- **Kalman Filter**: [Wikipedia](https://en.wikipedia.org/wiki/Kalman_filter) - State estimation
- **Homography**: [Roboflow guide](https://blog.roboflow.com/camera-calibration-sports-computer-vision/) - Field registration

## Next Steps

For the complete execution pipeline overview, see [Execution Pipeline](execution-pipeline.md).

To implement these improvements, start with Phase 1 core improvements, then move to scalability features based on your specific requirements and hardware constraints.

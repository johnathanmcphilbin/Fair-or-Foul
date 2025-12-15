// Pose Detection Module for Live Camera Feed

class PoseDetector {
    constructor() {
        this.video = document.getElementById('video');
        this.canvas = document.getElementById('poseCanvas');
        this.ctx = this.canvas.getContext('2d');
        this.net = null;
        this.isRunning = false;
        this.crossingCount = 0;
        this.sessionData = [];
        this.frameCount = 0;
        this.startTime = null;
        this.confidenceSum = 0;
        this.previousPoses = [];
        this.scaleFactor = 1;
        this.overlayMode = 'both'; // 'skeleton', 'ellipses', 'both'
        this.debugMode = false; // set true to enable coordinate logs
        this.minConfidence = 0.2; // lower threshold so weaker keypoints are visible
        this.calibrationOffset = { x: 0, y: 0 };
        this.calibrationMode = false;
        this.lastPose = null;
        this.autoCalibrate = true; // try automatic calibration on first frames
        this.autoCalibrated = false;
        this._autoCalibBuffer = [];
        this._autoCalibSamples = 6;

        // Limb indices for skeleton
        this.limbConnections = [
            [5, 6],   // shoulders
            [5, 7], [7, 9], // left arm
            [6, 8], [8, 10], // right arm
            [5, 11], [6, 12], // torso
            [11, 13], [13, 15], // left leg
            [12, 14], [14, 16] // right leg
        ];

        // Joint names for labeling
        this.jointNames = [
            'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
            'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
            'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
            'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
        ];

        // initialize ellipse smoothing history now that limbConnections exists
        this.ellipseHistory = this.limbConnections.map(() => null);
        this.keypointHistory = Array(17).fill(null);
        this.smoothedKeypoints = Array(17).fill(null);
        this.glowColor = 'rgba(0,255,180,0.9)';
        this.trailAlpha = 0.6;

        this.init();
    }

    setOverlayMode(mode) {
        if (['skeleton', 'ellipses', 'both'].includes(mode)) {
            this.overlayMode = mode;
            this.updateStatus('Overlay mode: ' + mode);
        }
    }

    async init() {
        // Robust dynamic script loader with timeout
        const loadScript = (src, timeout = 15000) => {
            return new Promise((resolve, reject) => {
                const s = document.createElement('script');
                let timer = setTimeout(() => {
                    s.onerror = s.onload = null;
                    reject(new Error('Timeout loading ' + src));
                }, timeout);

                s.src = src;
                s.onload = () => {
                    clearTimeout(timer);
                    resolve();
                };
                s.onerror = (e) => {
                    clearTimeout(timer);
                    reject(new Error('Error loading script ' + src));
                };
                document.head.appendChild(s);
            });
        };

        // Start loading and keep a promise reference so start() can wait for it
        this.modelReadyPromise = (async () => {
            try {
                this.updateStatus('Loading TensorFlow.js...');
                await loadScript('https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4');

                this.updateStatus('Loading PoseNet model script...');
                await loadScript('https://cdn.jsdelivr.net/npm/@tensorflow-models/posenet@2');

                this.updateStatus('Initializing PoseNet model...');
                this.net = await posenet.load();
                this.updateStatus('Model loaded. Ready to start!');
            } catch (err) {
                this.net = null;
                this.updateStatus('Error loading model: ' + (err && err.message ? err.message : err));
                console.error('Model load error:', err);
                throw err;
            }
        })();
    }

    async start() {
        // Start camera immediately; model may still be loading in background.
        console.log('PoseDetector.start() called');
        this.updateStatus('Starting camera; model may still be loading...');
        if (!this.net) {
            console.log('Model not yet available, continuing to start camera');
        }

        try {
            this.isRunning = true;
            this.crossingCount = 0;
            this.sessionData = [];
            this.frameCount = 0;
            this.startTime = Date.now();
            this.confidenceSum = 0;
            this.previousPoses = [];

            // Request camera access
            console.log('Requesting getUserMedia');
            this.updateStatus('Requesting camera permission...');
            const stream = await navigator.mediaDevices.getUserMedia({
                video: { width: 1280, height: 720 },
                audio: false
            });

            console.log('getUserMedia succeeded, attaching stream to video element');
            this.video.srcObject = stream;
            // Unhide video so user can see the raw feed for debugging
            try { this.video.style.display = 'block'; } catch (e) {}

            this.video.onloadedmetadata = () => {
                console.log('video.onloadedmetadata fired');
                this.video.play();
                this.setupCanvas();
                // Visual cue that camera is active
                try { this.canvas.style.border = '3px solid #28a745'; } catch(e){}
                this.detectPose();
            };

            // Also handle the stream immediately in case metadata is delayed
            if (this.video.readyState >= 2) {
                try {
                    console.log('video readyState indicates metadata available');
                    this.video.play();
                    this.setupCanvas();
                    try { this.canvas.style.border = '3px solid #28a745'; } catch(e){}
                    this.detectPose();
                } catch (e) {
                    console.warn('Error in immediate video setup', e);
                }
            }

            // Fallback: if metadata event doesn't fire, try after short delay
            setTimeout(() => {
                if (!this.isRunning) return;
                if (!this.displayWidth || !this.displayHeight) {
                    try {
                        this.setupCanvas();
                        this.detectPose();
                    } catch (e) {
                        // ignore
                    }
                }
            }, 800);

            document.getElementById('startBtn').style.display = 'none';
            document.getElementById('stopBtn').style.display = 'inline-block';
            this.updateStatus('Camera started. Detecting poses...');

        } catch (err) {
            this.updateStatus('Error accessing camera: ' + (err && err.message ? err.message : err));
            console.error('Error accessing camera:', err);
            try { this.canvas.style.border = '3px solid #dc3545'; } catch(e){}
        }
    }

    setupCanvas() {
        // Set canvas size to match video
        // Use video natural dimensions and account for devicePixelRatio so
        // drawn overlays align exactly with the displayed video.
        const vw = this.video.videoWidth || 640;
        const vh = this.video.videoHeight || 480;
        const dpr = window.devicePixelRatio || 1;

        // Match CSS display size to video pixel size, but set backing store size
        this.canvas.width = Math.round(vw * dpr);
        this.canvas.height = Math.round(vh * dpr);
        this.canvas.style.width = vw + 'px';
        this.canvas.style.height = vh + 'px';

        // Scale drawing operations so 1 unit in canvas space == 1 video pixel
        this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

        this.displayWidth = vw;
        this.displayHeight = vh;
    }

    // Map PoseNet keypoint position to canvas coordinates (defensive)
    mapToCanvas(pos) {
        if (!pos || typeof pos.x !== 'number' || typeof pos.y !== 'number') return null;
        // Compute mapping from PoseNet positions to canvas/display size.
        // PoseNet positions are in the source image/video pixel space (video.videoWidth/Height).
        const videoW = this.video.videoWidth || this.displayWidth || 640;
        const videoH = this.video.videoHeight || this.displayHeight || 480;

        const dispW = this.displayWidth || videoW;
        const dispH = this.displayHeight || videoH;

        const scaleX = dispW / videoW;
        const scaleY = dispH / videoH;

        const x = pos.x * scaleX;
        const y = pos.y * scaleY;
        if (!isFinite(x) || !isFinite(y)) return null;

        // If display sizes are known, reject obviously out-of-range points
        if (this.displayWidth && this.displayHeight) {
            if (x < -50 || y < -50 || x > this.displayWidth + 50 || y > this.displayHeight + 50) {
                return null;
            }
        }

        const mapped = { x, y };
        if (this.debugMode) {
            console.log('mapToCanvas:', { pos, videoW, videoH, dispW, dispH, mapped });
        }
        return mapped;
    }

    async detectPose() {
        if (!this.isRunning) return;

        try {
            if (this.net) {
                // Get pose from current video frame
                const poses = await this.net.estimateSinglePose(this.video, {
                    flipHorizontal: false
                });

                if (this.debugMode) console.log('pose result', poses);

                // keep last pose for calibration
                this.lastPose = poses;

                // Shoulder-based auto-calibration (preferred over centroid)
                if (this.autoCalibrate && !this.autoCalibrated) {
                    const left = poses.keypoints[5];
                    const right = poses.keypoints[6];
                    const minC = this.minConfidence;
                    if (left && right && left.score > minC && right.score > minC) {
                        const mid = { x: (left.position.x + right.position.x) / 2, y: (left.position.y + right.position.y) / 2 };
                        const mappedMid = this.mapToCanvas(mid);
                        if (mappedMid) {
                            const desiredX = (this.displayWidth || this.canvas.width) / 2;
                            const desiredY = (this.displayHeight || this.canvas.height) * 0.45;
                            this.calibrationOffset = { x: desiredX - mappedMid.x, y: desiredY - mappedMid.y };
                            this.autoCalibrated = true;
                            this.updateStatus('Auto-calibrated to shoulders.');
                        }
                    }
                }

                // Smooth keypoints into smoothedKeypoints array for nicer visuals
                poses.keypoints.forEach((kp, idx) => {
                    const mapped = this.mapToCanvas(kp.position);
                    if (!mapped) {
                        this.smoothedKeypoints[idx] = null;
                        return;
                    }
                    const prev = this.smoothedKeypoints[idx];
                    const alpha = 0.35; // smoothing
                    if (!prev) {
                        this.smoothedKeypoints[idx] = { x: mapped.x, y: mapped.y, score: kp.score };
                    } else {
                        this.smoothedKeypoints[idx] = {
                            x: prev.x * (1 - alpha) + mapped.x * alpha,
                            y: prev.y * (1 - alpha) + mapped.y * alpha,
                            score: kp.score
                        };
                    }
                });

                // Automatic calibration: collect a few centroids then set offset so centroid maps to canvas center
                if (this.autoCalibrate && !this.autoCalibrated) {
                    const pts = poses.keypoints.filter(kp => kp.score > this.minConfidence).map(kp => kp.position);
                    if (pts.length >= 2) {
                        const avg = pts.reduce((acc, p) => ({ x: acc.x + p.x, y: acc.y + p.y }), { x: 0, y: 0 });
                        avg.x /= pts.length; avg.y /= pts.length;
                        this._autoCalibBuffer.push(avg);
                    }

                    if (this._autoCalibBuffer.length >= this._autoCalibSamples) {
                        // average the collected centroids
                        const a = this._autoCalibBuffer.reduce((acc, p) => ({ x: acc.x + p.x, y: acc.y + p.y }), { x: 0, y: 0 });
                        a.x /= this._autoCalibBuffer.length; a.y /= this._autoCalibBuffer.length;

                        const mappedAvg = this.mapToCanvas({ x: a.x, y: a.y });
                        if (mappedAvg) {
                            const desiredX = (this.displayWidth || (this.canvas.width)) / 2;
                            const desiredY = (this.displayHeight || (this.canvas.height)) * 0.45;
                            this.calibrationOffset = { x: desiredX - mappedAvg.x, y: desiredY - mappedAvg.y };
                            this.autoCalibrated = true;
                            this.updateStatus('Auto-calibrated overlays.');
                        }
                        this._autoCalibBuffer = [];
                    }
                }

                // Draw frame
                this.drawFrame(poses);

                // Check for limb crossings
                this.detectCrossings(poses);

                // Record session data
                this.recordFrameData(poses);

                this.frameCount++;
            } else {
                // Model not ready yet: draw video so overlays still show
                // Clear canvas and draw video at display size
                try {
                    this.ctx.clearRect(0, 0, this.displayWidth, this.displayHeight);
                    this.ctx.drawImage(this.video, 0, 0, this.displayWidth, this.displayHeight);
                } catch (err) {
                    // drawImage may fail early; ignore and continue
                }

                // If model is loading, show status and attach one-time handlers
                if (this.modelReadyPromise && !this._modelReadyHandled) {
                    this._modelReadyHandled = true;
                    this.modelReadyPromise.then(() => {
                        this.updateStatus('Model loaded. Resuming detection.');
                    }).catch(err => {
                        console.error('Model load failed:', err);
                        this.updateStatus('Model failed to load. See console.');
                    });
                }
            }

            // Continue detecting
            requestAnimationFrame(() => this.detectPose());
        } catch (err) {
            console.error('Error in pose detection:', err);
            if (this.isRunning) {
                requestAnimationFrame(() => this.detectPose());
            }
        }
    }

    drawFrame(poses) {
        // Draw video frame at display size (canvas backing store is scaled by dpr)
        try {
            this.ctx.clearRect(0, 0, this.displayWidth || this.canvas.width, this.displayHeight || this.canvas.height);
            this.ctx.drawImage(this.video, 0, 0, this.displayWidth || this.canvas.width, this.displayHeight || this.canvas.height);
        } catch (err) {
            // If drawImage fails (video not ready), skip drawing for this frame
        }

        if (!poses || poses.keypoints.length === 0) return;

        // Draw keypoints and overlays based on selected mode
        // Always draw keypoints for reference
        this.drawKeypoints(poses.keypoints);

        if (this.overlayMode === 'skeleton') {
            this.drawSkeleton(poses.keypoints);
        } else if (this.overlayMode === 'ellipses') {
            this.drawLimbEllipses(poses.keypoints);
        } else { // both
            this.drawSkeleton(poses.keypoints);
            this.drawLimbEllipses(poses.keypoints);
        }

        // Update confidence average
        const confidence = poses.score || 0;
        this.confidenceSum += confidence;
    }

    drawKeypoints(keypoints) {
        const minConfidence = this.minConfidence;

        // Neon joint particles
        this.ctx.fillStyle = this.glowColor;
        this.ctx.strokeStyle = 'rgba(0,150,120,0.9)';
        this.ctx.lineWidth = 2;
        this.ctx.shadowColor = this.glowColor;
        this.ctx.shadowBlur = 12;

        keypoints.forEach((keypoint, idx) => {
            const sk = this.smoothedKeypoints[idx];
            if (!sk || keypoint.score <= minConfidence) return;
            const radius = 5 + Math.max(0, Math.min(6, (keypoint.score - minConfidence) * 20));
            // small pulsating effect
            const pulse = 1 + 0.8 * Math.sin(Date.now() / 200 + idx);
            this.ctx.beginPath();
            this.ctx.arc(sk.x, sk.y, radius * pulse, 0, 2 * Math.PI);
            this.ctx.fill();
            this.ctx.stroke();

            // tiny highlight
            this.ctx.fillStyle = 'rgba(255,255,255,0.6)';
            this.ctx.beginPath();
            this.ctx.arc(sk.x, sk.y, Math.max(1, radius * 0.25), 0, 2 * Math.PI);
            this.ctx.fill();
            this.ctx.fillStyle = this.glowColor;
        });

        // reset shadow
        this.ctx.shadowBlur = 0;
    }

    drawSkeleton(keypoints) {
        const minConfidence = this.minConfidence;
        this.ctx.lineWidth = 6;
        this.ctx.strokeStyle = this.glowColor;
        this.ctx.shadowColor = this.glowColor;
        this.ctx.shadowBlur = 24;

        this.limbConnections.forEach(([start, end], idx) => {
            const s = this.smoothedKeypoints[start];
            const e = this.smoothedKeypoints[end];
            if (!s || !e) return;
            if (s.score <= minConfidence || e.score <= minConfidence) return;

            // gradient along limb
            const grad = this.ctx.createLinearGradient(s.x, s.y, e.x, e.y);
            grad.addColorStop(0, 'rgba(0,255,180,0.95)');
            grad.addColorStop(1, 'rgba(0,150,255,0.85)');
            this.ctx.strokeStyle = grad;

            this.ctx.beginPath();
            this.ctx.moveTo(s.x, s.y);
            this.ctx.lineTo(e.x, e.y);
            this.ctx.stroke();
        });

        this.ctx.shadowBlur = 0;
    }

    drawLimbEllipses(keypoints) {
        const minConfidence = this.minConfidence;
        // Estimate a base limb width from shoulder distance for scale-invariance
        let limbWidth = 15;
        const leftShoulder = keypoints[5];
        const rightShoulder = keypoints[6];
        if (leftShoulder && rightShoulder && leftShoulder.score > minConfidence && rightShoulder.score > minConfidence) {
            const sd = Math.hypot(leftShoulder.position.x - rightShoulder.position.x,
                                  leftShoulder.position.y - rightShoulder.position.y);
            limbWidth = Math.max(8, sd * 0.18);
        }

        this.limbConnections.forEach(([start, end], index) => {
            const startKp = keypoints[start];
            const endKp = keypoints[end];

            if (startKp.score > minConfidence && endKp.score > minConfidence) {
                const sPos = this.mapToCanvas(startKp.position);
                const ePos = this.mapToCanvas(endKp.position);
                if (!sPos || !ePos) return;

                const startPos = sPos;
                const endPos = ePos;

                // Compute center, length and angle in canvas space
                const cx = (startPos.x + endPos.x) / 2;
                const cy = (startPos.y + endPos.y) / 2;
                const dx = endPos.x - startPos.x;
                const dy = endPos.y - startPos.y;
                const length = Math.hypot(dx, dy);
                const angle = Math.atan2(dy, dx);

                // Radius along limb: slightly shorter than half length to avoid overlap
                const radiusX = Math.max(2, length / 2 * 0.95);

                // Radius perpendicular to limb: based on limbWidth computed from shoulders
                const radiusY = Math.max(3, limbWidth / 2);

                // Smooth parameters across frames to reduce jitter
                const prev = this.ellipseHistory[index];
                const alpha = 0.45; // smoothing factor (0 - new only, 1 - old only)
                let smoothed = { cx, cy, radiusX, radiusY, angle };
                if (prev) {
                    smoothed = {
                        cx: prev.cx * alpha + cx * (1 - alpha),
                        cy: prev.cy * alpha + cy * (1 - alpha),
                        radiusX: prev.radiusX * alpha + radiusX * (1 - alpha),
                        radiusY: prev.radiusY * alpha + radiusY * (1 - alpha),
                        angle: prev.angle * alpha + angle * (1 - alpha)
                    };
                }

                this.ellipseHistory[index] = smoothed;

                // Color per limb
                const color = `hsl(${index * 25}, 85%, 50%)`;

                this.drawEllipse(smoothed.cx, smoothed.cy, smoothed.radiusX, smoothed.radiusY, smoothed.angle, color);
            }
        });
    }

    drawEllipse(centerX, centerY, radiusX, radiusY, rotation, color) {
        this.ctx.save();
        this.ctx.translate(centerX, centerY);
        this.ctx.rotate(rotation);

        this.ctx.strokeStyle = color;
        this.ctx.lineWidth = 2;
        this.ctx.globalAlpha = 0.6;

        this.ctx.beginPath();
        this.ctx.ellipse(0, 0, radiusX, radiusY, 0, 0, 2 * Math.PI);
        this.ctx.stroke();

        this.ctx.restore();
    }

    detectCrossings(poses) {
        if (!poses || poses.keypoints.length === 0) return;

        const keypoints = poses.keypoints;
        const minConfidence = 0.5;

        // Build current limb list
        const currentLimbs = [];
        this.limbConnections.forEach(([start, end]) => {
            const startKp = keypoints[start];
            const endKp = keypoints[end];

            if (startKp.score > minConfidence && endKp.score > minConfidence) {
                currentLimbs.push({
                    start: startKp.position,
                    end: endKp.position,
                    startIdx: start,
                    endIdx: end
                });
            }
        });

        // Check for crossings between limbs
        if (this.previousPoses.length > 0) {
            const previousLimbs = this.previousPoses;

            for (let i = 0; i < currentLimbs.length; i++) {
                for (let j = i + 1; j < currentLimbs.length; j++) {
                    const limb1 = currentLimbs[i];
                    const limb2 = currentLimbs[j];

                    // Check if same joint is not involved
                    if (limb1.startIdx === limb2.startIdx || 
                        limb1.startIdx === limb2.endIdx ||
                        limb1.endIdx === limb2.startIdx ||
                        limb1.endIdx === limb2.endIdx) {
                        continue;
                    }

                    if (this.linesCross(limb1.start, limb1.end, limb2.start, limb2.end)) {
                        this.crossingCount++;
                        this.updateCounterDisplay();
                    }
                }
            }
        }

        this.previousPoses = currentLimbs;
    }

    linesCross(p1, p2, p3, p4) {
        // Check if line segment p1-p2 crosses p3-p4
        return this.ccw(p1, p3, p4) !== this.ccw(p2, p3, p4) &&
               this.ccw(p1, p2, p3) !== this.ccw(p1, p2, p4);
    }

    ccw(A, B, C) {
        return (C.y - A.y) * (B.x - A.x) > (B.y - A.y) * (C.x - A.x);
    }

    recordFrameData(poses) {
        const timestamp = ((Date.now() - this.startTime) / 1000).toFixed(2);
        const confidence = poses.score || 0;

        const frameData = {
            timestamp: timestamp,
            frame_number: this.frameCount,
            crossings_so_far: this.crossingCount,
            confidence: confidence.toFixed(3),
            joints_detected: poses.keypoints.filter(kp => kp.score > 0.5).length
        };

        // Add individual joint positions
        poses.keypoints.forEach((kp, idx) => {
            if (this.jointNames[idx]) {
                frameData[`${this.jointNames[idx]}_x`] = kp.position.x.toFixed(2);
                frameData[`${this.jointNames[idx]}_y`] = kp.position.y.toFixed(2);
                frameData[`${this.jointNames[idx]}_confidence`] = kp.score.toFixed(3);
            }
        });

        this.sessionData.push(frameData);
    }

    updateCounterDisplay() {
        document.getElementById('crossingCount').textContent = this.crossingCount;
        document.getElementById('totalCrossings').textContent = this.crossingCount;
    }

    async stop() {
        this.isRunning = false;

        // Stop video stream
        if (this.video.srcObject) {
            this.video.srcObject.getTracks().forEach(track => track.stop());
        }

        document.getElementById('startBtn').style.display = 'inline-block';
        document.getElementById('stopBtn').style.display = 'none';

        // Update stats
        this.updateStats();

        // Save session data
        await this.saveSessionData();

        this.updateStatus('Session stopped and saved to data/');
    }

    updateStats() {
        const duration = Math.floor((Date.now() - this.startTime) / 1000);
        const avgConfidence = this.frameCount > 0 ? 
            ((this.confidenceSum / this.frameCount) * 100).toFixed(1) : 0;

        document.getElementById('sessionDuration').textContent = duration;
        document.getElementById('framesProcessed').textContent = this.frameCount;
        document.getElementById('avgConfidence').textContent = avgConfidence + '%';
    }

    async saveSessionData() {
        try {
            const response = await fetch('/save_pose_session', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    session_data: this.sessionData
                })
            });

            const result = await response.json();

            if (result.success) {
                this.updateStatus('Session saved: ' + result.filename);
                await this.loadSavedSessions();
            } else {
                this.updateStatus('Error saving session: ' + result.error);
            }
        } catch (err) {
            this.updateStatus('Error saving session: ' + err.message);
            console.error('Error saving session:', err);
        }
    }

    async loadSavedSessions() {
        try {
            const response = await fetch('/get_saved_sessions');
            const sessions = await response.json();

            const container = document.getElementById('savedSessions');
            if (!sessions || sessions.length === 0) {
                container.innerHTML = '<p class="text-muted">No sessions saved yet.</p>';
                return;
            }

            container.innerHTML = sessions.map(session => `
                <div class="col-md-6 col-lg-4">
                    <div class="card">
                        <div class="card-body">
                            <h6 class="card-title">${session.filename}</h6>
                            <small class="text-muted d-block">
                                ${new Date(session.created).toLocaleString()}
                            </small>
                            <small class="text-muted d-block">
                                ${(session.size / 1024).toFixed(2)} KB
                            </small>
                            <a href="/download_session/${session.filename}" 
                               class="btn btn-sm btn-primary mt-2" download>
                                <i class="fas fa-download me-1"></i>Download CSV
                            </a>
                        </div>
                    </div>
                </div>
            `).join('');
        } catch (err) {
            console.error('Error loading sessions:', err);
        }
    }

    reset() {
        this.stop();
        this.crossingCount = 0;
        this.sessionData = [];
        this.frameCount = 0;
        this.confidenceSum = 0;
        this.previousPoses = [];

        document.getElementById('crossingCount').textContent = '0';
        document.getElementById('totalCrossings').textContent = '0';
        document.getElementById('sessionDuration').textContent = '0';
        document.getElementById('framesProcessed').textContent = '0';
        document.getElementById('avgConfidence').textContent = '0%';

        // Clear canvas
        this.ctx.fillStyle = '#000000';
        this.ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);

        this.updateStatus('Reset complete. Ready to start again.');
    }

    updateStatus(message) {
        document.getElementById('statusText').textContent = message;
    }
}

// Global instance
let poseDetector = null;

// Initialize when page loads
document.addEventListener('DOMContentLoaded', function() {
    poseDetector = new PoseDetector();
});

// Global functions for HTML onclick handlers
function startPoseDetection() {
    if (poseDetector) {
        poseDetector.start();
    }
}

function stopPoseDetection() {
    if (poseDetector) {
        poseDetector.stop();
    }
}

function resetDetection() {
    if (poseDetector) {
        poseDetector.reset();
    }
}

function setOverlayMode(mode) {
    if (poseDetector && typeof poseDetector.setOverlayMode === 'function') {
        poseDetector.setOverlayMode(mode);
    }
}

// Expose calibration wrappers for inline onclick handlers
function startCalibration() {
    if (typeof poseDetector !== 'undefined' && poseDetector) {
        // If instance provides the method, use it
        if (typeof poseDetector.startCalibration === 'function') {
            return poseDetector.startCalibration();
        }

        // Fallback: implement calibration inline using poseDetector state
        if (!poseDetector.isRunning) {
            console.error('Pose detector not running. Start camera first.');
            return;
        }

        poseDetector.calibrationMode = true;
        poseDetector.updateStatus('Calibration: click on your shoulder on the video canvas.');

        const handler = (e) => {
            if (!poseDetector.calibrationMode) return;
            const rect = poseDetector.canvas.getBoundingClientRect();
            const clickX = e.clientX - rect.left;
            const clickY = e.clientY - rect.top;

            const lastPose = poseDetector.lastPose;
            if (!lastPose || !lastPose.keypoints) {
                poseDetector.updateStatus('No pose available to calibrate. Try again.');
                poseDetector.calibrationMode = false;
                poseDetector.canvas.removeEventListener('click', handler);
                return;
            }

            const pts = lastPose.keypoints.filter(kp => kp.score > poseDetector.minConfidence).map(kp => kp.position);
            if (pts.length === 0) {
                poseDetector.updateStatus('No reliable keypoints for calibration. Try again.');
                poseDetector.calibrationMode = false;
                poseDetector.canvas.removeEventListener('click', handler);
                return;
            }

            const avg = pts.reduce((acc, p) => ({ x: acc.x + p.x, y: acc.y + p.y }), { x: 0, y: 0 });
            avg.x /= pts.length; avg.y /= pts.length;

            const mappedAvg = poseDetector.mapToCanvas({ x: avg.x, y: avg.y });
            if (!mappedAvg) {
                poseDetector.updateStatus('Could not map detected centroid. Try again.');
                poseDetector.calibrationMode = false;
                poseDetector.canvas.removeEventListener('click', handler);
                return;
            }

            const offsetX = clickX - mappedAvg.x;
            const offsetY = clickY - mappedAvg.y;

            poseDetector.calibrationOffset = { x: offsetX, y: offsetY };
            poseDetector.calibrationMode = false;
            poseDetector.canvas.removeEventListener('click', handler);

            poseDetector.updateStatus('Calibration applied. Offset: x=' + offsetX.toFixed(1) + ', y=' + offsetY.toFixed(1));
        };

        poseDetector.canvas.addEventListener('click', handler);
        return;
    }

    console.error('Pose detector not ready for calibration');
}

function resetCalibration() {
    if (typeof poseDetector !== 'undefined' && poseDetector) {
        if (typeof poseDetector.resetCalibration === 'function') {
            return poseDetector.resetCalibration();
        }
        poseDetector.calibrationOffset = { x: 0, y: 0 };
        poseDetector.calibrationMode = false;
        poseDetector.updateStatus('Calibration reset.');
        return;
    }
    console.error('Pose detector not ready for calibration reset');
}

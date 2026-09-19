/**
 * BODY2FIT HYPERFRAMES COMPOSITION CONTROLLER
 * Complies with Hyperframes deterministic seek protocol: window.__hf.seek(t)
 * Timeline Duration: 20.0 seconds | 30 FPS | 1920x1080
 */

(function () {
  const DURATION = 20.0;
  const FPS = 30;
  let currentTime = 0;
  let isPlaying = true;
  let lastTimestamp = null;
  let isScrubbing = false;

  // DOM Elements
  let stage, clockEl, progressBar, scrubber, btnPlay, btnRestart, sceneBtns, audioToggle;
  let valWaist, valHip, valChest;
  let whtrBar, whrBar, briBar;
  let iouBar, chamferBar, verdictBanner, outroOverlay;
  let animElements = [];

  const scenes = [
    { id: 'scene-1', start: 0.0, end: 3.5 },
    { id: 'scene-2', start: 3.5, end: 8.5 },
    { id: 'scene-3', start: 8.5, end: 13.5 },
    { id: 'scene-4', start: 13.5, end: 20.0 }
  ];

  // =========================================================================
  // RESPONSIVE SCALING
  // =========================================================================
  function resizeStage() {
    if (!stage) return;
    const windowWidth = window.innerWidth;
    const windowHeight = window.innerHeight - 100; // Account for player controls toolbar
    const scale = Math.min(windowWidth / 1920, windowHeight / 1080, 1.0);
    stage.style.transform = `scale(${Math.max(0.2, scale * 0.95)})`;
  }

  // =========================================================================
  // WEB AUDIO PROCEDURAL SYNTHESIZER
  // =========================================================================
  let audioCtx = null;
  function initAudio() {
    if (!audioCtx && audioToggle && audioToggle.checked) {
      const AudioCtxClass = window.AudioContext || window.webkitAudioContext;
      if (AudioCtxClass) {
        audioCtx = new AudioCtxClass();
      }
    }
    if (audioCtx && audioCtx.state === 'suspended') {
      audioCtx.resume();
    }
  }

  function playTone(freq, type, duration, gainLevel, startTime = 0) {
    if (!audioCtx || !audioToggle || !audioToggle.checked) return;
    try {
      const osc = audioCtx.createOscillator();
      const gain = audioCtx.createGain();
      osc.type = type;
      osc.frequency.setValueAtTime(freq, audioCtx.currentTime + startTime);
      gain.gain.setValueAtTime(gainLevel, audioCtx.currentTime + startTime);
      gain.gain.exponentialRampToValueAtTime(0.0001, audioCtx.currentTime + startTime + duration);
      osc.connect(gain);
      gain.connect(audioCtx.destination);
      osc.start(audioCtx.currentTime + startTime);
      osc.stop(audioCtx.currentTime + startTime + duration);
    } catch (e) {}
  }

  function playSubDrop() {
    if (!audioCtx || !audioToggle || !audioToggle.checked) return;
    try {
      const osc = audioCtx.createOscillator();
      const gain = audioCtx.createGain();
      osc.type = 'sine';
      osc.frequency.setValueAtTime(130, audioCtx.currentTime);
      osc.frequency.exponentialRampToValueAtTime(35, audioCtx.currentTime + 0.8);
      gain.gain.setValueAtTime(0.3, audioCtx.currentTime);
      gain.gain.exponentialRampToValueAtTime(0.001, audioCtx.currentTime + 0.8);
      osc.connect(gain);
      gain.connect(audioCtx.destination);
      osc.start();
      osc.stop(audioCtx.currentTime + 0.8);
    } catch (e) {}
  }

  function playBeat() {
    playTone(65, 'triangle', 0.12, 0.22);
    playTone(180, 'sine', 0.04, 0.12);
  }

  function playChime() {
    playTone(523.25, 'sine', 0.7, 0.2, 0.0);
    playTone(659.25, 'sine', 0.7, 0.2, 0.1);
    playTone(783.99, 'sine', 1.0, 0.25, 0.2);
    playTone(1046.50, 'sine', 1.4, 0.3, 0.3);
  }

  let lastBeatInterval = -1;

  // =========================================================================
  // DETERMINISTIC TIMELINE RENDERER (SEEK-SAFE)
  // =========================================================================
  function formatClock(seconds) {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    const hundredths = Math.floor((seconds % 1) * 100);
    return `${String(mins).padStart(2, '0')}:${String(secs).padStart(2, '0')}.${String(hundredths).padStart(2, '0')}`;
  }

  function updateTimeline(time) {
    // 1. Update Clock, Progress Bar, and Scrubber
    if (clockEl) clockEl.textContent = `${formatClock(time)} / 00:20.00`;
    const progressPercent = Math.min((time / DURATION) * 100, 100);
    if (progressBar) progressBar.style.width = `${progressPercent}%`;
    if (scrubber && !isScrubbing) scrubber.value = time.toFixed(2);

    // 2. Audio rhythm & milestone triggers (only during real-time interactive playback)
    if (isPlaying && !isScrubbing && audioCtx && audioToggle && audioToggle.checked) {
      if (time >= 3.5 && time < 17.0) {
        const beatInterval = Math.floor(time * 2);
        if (beatInterval !== lastBeatInterval) {
          lastBeatInterval = beatInterval;
          playBeat();
        }
      }
      if (Math.abs(time - 0.3) < 0.04 && lastBeatInterval !== -99) {
        playSubDrop();
        lastBeatInterval = -99;
      }
      if (Math.abs(time - 15.8) < 0.04 && lastBeatInterval !== -88) {
        playChime();
        lastBeatInterval = -88;
      }
    }

    // 3. Deterministic Scene Visibility
    scenes.forEach((scene, idx) => {
      const el = document.getElementById(scene.id);
      const btn = sceneBtns && sceneBtns[idx];
      const isActive = time >= scene.start && time < scene.end;
      if (el) {
        if (isActive) {
          el.classList.add('active');
          el.style.opacity = '1';
          el.style.visibility = 'visible';
          el.style.pointerEvents = 'auto';
        } else {
          el.classList.remove('active');
          el.style.opacity = '0';
          el.style.visibility = 'hidden';
          el.style.pointerEvents = 'none';
        }
      }
      if (btn) {
        btn.classList.toggle('active', isActive);
      }
    });

    // 4. Deterministic data-anim & data-delay Element Interpolation
    animElements.forEach((el) => {
      const delay = parseFloat(el.dataset.delay) || 0;
      const animType = el.dataset.anim;
      const animDuration = 0.45; // 450ms entrance window
      const delta = time - delay;
      const progress = Math.max(0, Math.min(1, delta / animDuration));

      if (progress <= 0) {
        el.style.opacity = '0';
        el.style.pointerEvents = 'none';
        if (animType === 'slam') el.style.transform = 'scale(1.35)';
        else if (animType === 'zoom-in') el.style.transform = 'scale(0.75)';
        else if (animType === 'slide-left') el.style.transform = 'translateX(-35px)';
        else if (animType === 'slide-right') el.style.transform = 'translateX(35px)';
        else if (animType === 'fade-up') el.style.transform = 'translateY(25px)';
        else if (animType === 'card-drop') el.style.transform = 'translateY(-25px)';
        else if (animType === 'fade-scale') el.style.transform = 'scale(0.92)';
      } else if (progress < 1) {
        // easeOutCubic: 1 - (1 - x)^3
        const ease = 1 - Math.pow(1 - progress, 3);
        el.style.opacity = ease.toFixed(3);
        el.style.pointerEvents = 'auto';

        if (animType === 'slam') {
          const s = (1.35 - ease * 0.35).toFixed(3);
          el.style.transform = `scale(${s})`;
        } else if (animType === 'zoom-in') {
          const s = (0.75 + ease * 0.25).toFixed(3);
          el.style.transform = `scale(${s})`;
        } else if (animType === 'slide-left') {
          const x = ((1 - ease) * -35).toFixed(1);
          el.style.transform = `translateX(${x}px)`;
        } else if (animType === 'slide-right') {
          const x = ((1 - ease) * 35).toFixed(1);
          el.style.transform = `translateX(${x}px)`;
        } else if (animType === 'fade-up') {
          const y = ((1 - ease) * 25).toFixed(1);
          el.style.transform = `translateY(${y}px)`;
        } else if (animType === 'card-drop') {
          const y = ((1 - ease) * -25).toFixed(1);
          el.style.transform = `translateY(${y}px)`;
        } else if (animType === 'fade-scale') {
          const s = (0.92 + ease * 0.08).toFixed(3);
          el.style.transform = `scale(${s})`;
        } else {
          el.style.transform = 'none';
        }
      } else {
        el.style.opacity = '1';
        el.style.transform = 'none';
        el.style.pointerEvents = 'auto';
      }
    });

    // 5. Scene 3 Numeric Counter Roll-ups (Deterministic)
    if (time < 8.7) {
      if (valWaist) valWaist.textContent = "00.00";
      if (valHip) valHip.textContent = "00.00";
      if (valChest) valChest.textContent = "00.00";
    } else {
      // Waist: 8.7s to 10.5s
      const tW = Math.max(0, Math.min(1, (time - 8.7) / 1.8));
      const easeW = 1 - Math.pow(1 - tW, 3);
      if (valWaist) valWaist.textContent = (easeW * 91.65).toFixed(2);

      // Hip: 9.0s to 10.8s
      const tH = Math.max(0, Math.min(1, (time - 9.0) / 1.8));
      const easeH = 1 - Math.pow(1 - tH, 3);
      if (valHip) valHip.textContent = (easeH * 106.68).toFixed(2);

      // Chest: 9.3s to 11.1s
      const tC = Math.max(0, Math.min(1, (time - 9.3) / 1.8));
      const easeC = 1 - Math.pow(1 - tC, 3);
      if (valChest) valChest.textContent = (easeC * 99.79).toFixed(2);
    }

    // 6. Scene 3 Risk Gauges Width Interpolation (Deterministic)
    if (whtrBar) {
      const p = Math.max(0, Math.min(1, (time - 10.2) / 1.2));
      const ease = 1 - Math.pow(1 - p, 3);
      whtrBar.style.width = `${(ease * 52.4).toFixed(1)}%`;
    }
    if (whrBar) {
      const p = Math.max(0, Math.min(1, (time - 10.5) / 1.2));
      const ease = 1 - Math.pow(1 - p, 3);
      whrBar.style.width = `${(ease * 85.9).toFixed(1)}%`;
    }
    if (briBar) {
      const p = Math.max(0, Math.min(1, (time - 10.8) / 1.2));
      const ease = 1 - Math.pow(1 - p, 3);
      briBar.style.width = `${(ease * 38.1).toFixed(1)}%`;
    }

    // 7. Scene 4 Scorecard Bars & Verdict Banner (Deterministic)
    if (iouBar) {
      const p = Math.max(0, Math.min(1, (time - 14.2) / 1.0));
      const ease = 1 - Math.pow(1 - p, 3);
      iouBar.style.width = `${(ease * 76.6).toFixed(1)}%`;
    }
    if (chamferBar) {
      const p = Math.max(0, Math.min(1, (time - 14.5) / 1.0));
      const ease = 1 - Math.pow(1 - p, 3);
      chamferBar.style.width = `${(ease * 80.8).toFixed(1)}%`;
    }

    // 8. Outro Overlay Reveal at 17.2s
    if (outroOverlay) {
      if (time >= 17.2) {
        outroOverlay.classList.add('visible');
        outroOverlay.style.opacity = '1';
        outroOverlay.style.pointerEvents = 'auto';
      } else {
        outroOverlay.classList.remove('visible');
        outroOverlay.style.opacity = '0';
        outroOverlay.style.pointerEvents = 'none';
      }
    }
  }

  // =========================================================================
  // HYPERFRAMES DETERMINISTIC SEEK PROTOCOL CONTRACT
  // =========================================================================
  window.__hf = {
    duration: DURATION,
    fps: FPS,
    seek: function (timeSeconds) {
      isPlaying = false;
      if (btnPlay) btnPlay.textContent = '▶ Play';
      currentTime = Math.max(0, Math.min(DURATION, parseFloat(timeSeconds) || 0));
      updateTimeline(currentTime);
    }
  };
  window.seekTo = window.__hf.seek;
  window.renderFrame = window.__hf.seek;

  window.__timelines = window.__timelines || {};
  window.__timelines['bodyfit-launch'] = window.__hf;

  // =========================================================================
  // INTERACTIVE ANIMATION LOOP (FOR IN-BROWSER PREVIEWS)
  // =========================================================================
  function frame(timestamp) {
    if (!lastTimestamp) lastTimestamp = timestamp;
    const delta = (timestamp - lastTimestamp) / 1000;
    lastTimestamp = timestamp;

    if (isPlaying && !isScrubbing) {
      currentTime += delta;
      if (currentTime >= DURATION) {
        currentTime = DURATION;
        isPlaying = false;
        if (btnPlay) btnPlay.textContent = '▶ Play';
      }
      updateTimeline(currentTime);
    }

    requestAnimationFrame(frame);
  }

  // =========================================================================
  // INITIALIZATION ON DOM READY
  // =========================================================================
  document.addEventListener('DOMContentLoaded', () => {
    stage = document.getElementById('stage');
    clockEl = document.getElementById('telemetry-clock');
    progressBar = document.getElementById('master-progress');
    scrubber = document.getElementById('time-scrubber');
    btnPlay = document.getElementById('btn-play');
    btnRestart = document.getElementById('btn-restart');
    sceneBtns = document.querySelectorAll('.scene-btn');
    audioToggle = document.getElementById('audio-toggle');

    valWaist = document.getElementById('val-waist');
    valHip = document.getElementById('val-hip');
    valChest = document.getElementById('val-chest');

    // Gauges & Bars
    whtrBar = document.querySelector('.gauge-item:nth-child(1) .gauge-fill');
    whrBar = document.querySelector('.gauge-item:nth-child(2) .gauge-fill');
    briBar = document.querySelector('.gauge-item:nth-child(3) .gauge-fill');

    iouBar = document.querySelector('.score-row:nth-child(2) + .score-bar .score-bar-fill') || 
             document.querySelectorAll('.score-bar-fill')[0];
    chamferBar = document.querySelector('.score-row:nth-child(4) + .score-bar .score-bar-fill') || 
                 document.querySelectorAll('.score-bar-fill')[1];

    verdictBanner = document.querySelector('.verdict-banner');
    outroOverlay = document.querySelector('.outro-overlay');

    animElements = Array.from(document.querySelectorAll('[data-anim]'));

    // Resize handling
    window.addEventListener('resize', resizeStage);
    resizeStage();

    // Toolbar Controls
    if (btnPlay) {
      btnPlay.addEventListener('click', () => {
        initAudio();
        if (currentTime >= DURATION) {
          currentTime = 0;
        }
        isPlaying = !isPlaying;
        btnPlay.textContent = isPlaying ? '⏸ Pause' : '▶ Play';
      });
    }

    if (btnRestart) {
      btnRestart.addEventListener('click', () => {
        initAudio();
        currentTime = 0;
        isPlaying = true;
        if (btnPlay) btnPlay.textContent = '⏸ Pause';
        updateTimeline(0);
      });
    }

    if (sceneBtns) {
      sceneBtns.forEach((btn) => {
        btn.addEventListener('click', () => {
          initAudio();
          const seekTime = parseFloat(btn.dataset.seek);
          window.__hf.seek(seekTime);
        });
      });
    }

    if (scrubber) {
      scrubber.addEventListener('input', (e) => {
        isScrubbing = true;
        const seekTime = parseFloat(e.target.value);
        window.__hf.seek(seekTime);
      });
      scrubber.addEventListener('change', () => {
        isScrubbing = false;
      });
    }

    // Auto audio resume on user gesture
    document.body.addEventListener('click', () => {
      initAudio();
    }, { once: true });

    // Initial render at time = 0
    updateTimeline(0);
    requestAnimationFrame(frame);
  });
})();

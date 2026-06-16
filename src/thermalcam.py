#!/usr/bin/env python3
"""
thermalcam.py - Universal Linux viewer for InfiRay-based USB thermal cameras.

Originally written by Les Wright for the Topdon TC001 (see tc001v4.2.py), this
version is a camera-agnostic rewrite that also works with newer Topdon/InfiRay
models such as the TC002C Duo.

These cameras pack a frame as a stack of horizontal bands: a high-res thermal
*image* band (8-bit AGC, neutral chroma) plus a 16-bit *temperature* band (raw
radiometric data, which shows up with extreme chroma when mis-read as YUYV). The
band geometry differs per model:

  * TC001        256x384 : image 256x192  + temp 256x192 (16-bit, /64 Kelvin)
  * TC002C Duo   512x484 : temp 256x192 (16-bit, /16 Kelvin) + image 512x384

This viewer auto-detects the camera, picks a mode that carries real temperature
data, decodes it, and shows real degrees C. If no radiometric mode is available
it falls back to a clearly-labelled relative scale. See docs/TC002C-DUO.md.
"""

import argparse
import fcntl
import glob
import os
import re
import struct
import sys
import time

try:
    import cv2
    import numpy as np
except ImportError as exc:  # pragma: no cover - environment guard
    sys.exit(
        f"Missing dependency: {exc.name}. Install with:\n"
        "  sudo apt-get install python3-opencv   (Debian/Ubuntu/Raspberry Pi)\n"
        "  pip install opencv-python numpy        (everything else)"
    )

# TC002C Duo true-temperature decode (its own self-calibrating stream + bundled
# calibration LUTs). Optional: the viewer still runs without it on other cameras.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from radiometry import duo as _duo
except Exception:  # pragma: no cover - radiometry is optional
    _duo = None

# USB vendor IDs known to ship InfiRay-based thermal cameras. 0x2BDF is Topdon.
KNOWN_THERMAL_VIDS = {0x2BDF, 0x0BDA, 0x1514, 0x3474}

KELVIN = 273.15  # temp_C = raw/scale - 273.15

# Exact band geometry per capture resolution, keyed by (width, height):
#   temp_rows, image_rows, sensor (w,h), scale (raw units per Kelvin), byte order.
# Fixed geometry is far more robust than per-frame chroma detection, which breaks
# on the occasional desynced/metadata-header frame these cameras emit.
KNOWN_LAYOUTS = {
    (256, 384): ((192, 384), (0, 192), (256, 192), 64.0, "le"),   # TC001 / P2 family
    (512, 484): ((0, 96), (98, 482), (256, 192), 16.0, "le"),     # TC002C Duo
}
# The TC002C Duo's true-°C mode: the camera advertises it as the bogus-looking
# "8x12578" (= 100624 u16 = 201248 B), a header+temperature+image frame the firmware
# self-calibrates. Decoded by radiometry/duo.py (real °C, no per-unit calibration).
DUO_MODE = (8, 12578)
# Resolutions to try for genuine temperature data, in order of preference. The Duo's
# self-calibrating mode (real absolute °C) is preferred over the 512x484 linear band.
RADIOMETRIC_MODES = ([DUO_MODE] if _duo is not None else []) + [(256, 384), (512, 484)]
# Plain image modes (no temperature), used only if no radiometric mode works.
IMAGE_MODES = [(512, 384), (256, 392), (256, 192)]

# (cv2 colormap or None for the grayscale white/black-hot pseudo-palettes), name.
# white-hot is first (the default): on flat/noisy scenes a colour map turns sensor
# noise into person-shaped blobs that fool detectors, while grayscale stays clean.
COLORMAPS = [
    (None, "white_hot"),
    (None, "black_hot"),
    (cv2.COLORMAP_INFERNO, "inferno"),
    (cv2.COLORMAP_JET, "jet"),
    (cv2.COLORMAP_HOT, "ironbow"),
    (cv2.COLORMAP_MAGMA, "magma"),
    (cv2.COLORMAP_VIRIDIS, "viridis"),
    (cv2.COLORMAP_BONE, "bone"),
]


def apply_palette(luma, idx):
    """Colormap an 8-bit luma image; returns (bgr, name). Handles the grayscale
    white-hot / black-hot pseudo-palettes (no cv2 colormap)."""
    cmap, name = COLORMAPS[idx]
    if cmap is None:
        if name == "black_hot":
            luma = 255 - luma
        return cv2.cvtColor(luma, cv2.COLOR_GRAY2BGR), name
    return cv2.applyColorMap(luma, cmap), name


def is_raspberrypi():
    try:
        with open("/sys/firmware/devicetree/base/model", "r") as m:
            return "raspberry pi" in m.read().lower()
    except OSError:
        return False


def _sysfs_usb_ids(video_node):
    """Return (vid, pid, name) for a /dev/videoN node by walking sysfs, or None."""
    name = os.path.basename(video_node)
    base = f"/sys/class/video4linux/{name}"
    try:
        modalias = open(os.path.join(base, "device", "modalias")).read().strip()
        match = re.search(r"v([0-9A-Fa-f]{4})p([0-9A-Fa-f]{4})", modalias)
        if not match:
            return None
        vid, pid = int(match.group(1), 16), int(match.group(2), 16)
    except OSError:
        return None
    try:
        product = open(os.path.join(base, "name")).read().strip()
    except OSError:
        product = ""
    return vid, pid, product


def find_thermal_device():
    """Locate the thermal camera's /dev/videoN, preferring known thermal VIDs."""
    nodes = sorted(glob.glob("/dev/video*"),
                   key=lambda p: int(p[10:]) if p[10:].isdigit() else 999)
    fallback = None
    for node in nodes:
        ids = _sysfs_usb_ids(node)
        if ids is None:
            continue
        vid, _pid, product = ids
        if vid in KNOWN_THERMAL_VIDS and _can_capture(node):
            return node
        if fallback is None and "USB Camera" in product and _can_capture(node):
            fallback = node
    return fallback


def supported_resolutions(device):
    """Set of (w, h) YUYV modes the device advertises, via VIDIOC_ENUM_FRAMESIZES.

    Used to avoid even *opening* an unsupported resolution: doing so desyncs these
    cameras' subsequent streams. Returns None if enumeration fails.
    """
    yuyv = ord("Y") | (ord("U") << 8) | (ord("Y") << 16) | (ord("V") << 24)
    # _IOWR('V', 74, sizeof(struct v4l2_frmsizeenum)==44)
    vidioc_enum_framesizes = (3 << 30) | (44 << 16) | (ord("V") << 8) | 74
    try:
        fd = os.open(device, os.O_RDWR)
    except OSError:
        return None
    sizes = set()
    try:
        for idx in range(128):
            buf = struct.pack("III", idx, yuyv, 0) + b"\x00" * 32
            try:
                res = fcntl.ioctl(fd, vidioc_enum_framesizes, buf)
            except OSError:
                break
            _i, _fmt, ftype, w, h = struct.unpack("IIIII", res[:20])
            if ftype != 1:  # 1 == V4L2_FRMSIZE_TYPE_DISCRETE
                break
            sizes.add((w, h))
    finally:
        os.close(fd)
    return sizes or None


def _can_capture(node):
    cap = cv2.VideoCapture(node, cv2.CAP_V4L2)
    try:
        if not cap.isOpened():
            return False
        cap.set(cv2.CAP_PROP_CONVERT_RGB, 0)
        ok, _ = cap.read()
        return ok
    finally:
        cap.release()


# -- frame parsing --------------------------------------------------------
def _largest_run(mask):
    """Return (start, end) of the longest contiguous True run in a 1-D mask."""
    best_len = best = 0
    start = None
    for i, v in enumerate(mask):
        if v and start is None:
            start = i
        elif not v and start is not None:
            if i - start > best_len:
                best_len, best = i - start, (start, i)
            start = None
    if start is not None and len(mask) - start > best_len:
        best = (start, len(mask))
    return best or None


def classify_bands(raw, swap=False):
    """Split a raw YUYV frame into (image_band, temp_band) row ranges.

    The image band reads as a real YUYV picture (chroma near the neutral 128);
    the temperature band is 16-bit data that reads as extreme chroma. Telemetry
    rows (luma pinned to 0/255) are excluded. Either range may be None.
    """
    chroma = raw[..., 1].mean(axis=1)
    luma = raw[..., 0].mean(axis=1)
    telemetry = (luma < 8) | (luma > 248)
    is_image = (np.abs(chroma - 128.0) < 40) & ~telemetry
    is_temp = ~is_image & ~telemetry
    image_band = _largest_run(is_image)
    temp_band = _largest_run(is_temp)
    if swap:
        image_band, temp_band = temp_band, image_band
    return image_band, temp_band


class Frame:
    """One parsed frame: a display image and (optionally) a temperature map,
    both as float32 arrays at the image band's native resolution.

    `valid` is False when a known radiometric layout produced an implausible
    temperature field - i.e. the camera emitted a desynced/garbage frame that the
    caller should skip rather than display.
    """

    def __init__(self, image, temp, is_real, valid=True):
        self.image = image            # (H, W) luma
        self.temp = temp              # (H, W) degrees C, or None
        self.is_real = is_real        # True when temp is genuine radiometric
        self.valid = valid
        self.h, self.w = image.shape


def _plausible(temp):
    """True for a physically sensible temperature field. Rejects desynced frames,
    whose shifted image/telemetry bytes decode to a big chunk of absurd values,
    while still allowing a small genuinely-hot region (a soldering iron, a flame)."""
    median = float(np.median(temp))
    if not -25.0 <= median <= 160.0:
        return False
    garbage = float(np.mean((temp < -40.0) | (temp > 300.0)))
    return garbage < 0.02


def _image_aligned(luma):
    """False if the image band looks horizontally tiled - the signature of a
    desynced frame whose line stride is wrong (content repeats at width/4 or /2).

    Tested by mean-absolute-difference against a shifted copy: tiled panels are
    near-identical (diff ~0), whereas a normal scene - even a smooth left-to-right
    temperature gradient - differs substantially when shifted. Flat/near-uniform
    scenes are exempt."""
    s = float(luma.std())
    if s < 8.0:                          # too uniform to judge; treat as fine
        return True
    w = luma.shape[1]
    for period in (w // 4, w // 2):
        if period < 2:
            continue
        diff = float(np.abs(luma[:, period:] - luma[:, :-period]).mean())
        if diff < max(3.0, 0.12 * s):    # panels ~identical at this period => tiled
            return False
    return True


def _decode_temp(raw, temp_rows, sensor, scale, order):
    """Decode a temperature band into a sensor-resolution °C array, or None."""
    t0, t1 = temp_rows
    sw, sh = sensor
    band = np.ascontiguousarray(raw[t0:t1]).reshape(-1)
    lo, hi = band[0::2].astype(np.uint16), band[1::2].astype(np.uint16)
    u16 = (lo + (hi << 8)) if order == "le" else (hi + (lo << 8))
    if u16.size < sw * sh:
        return None
    temp = u16[:sw * sh].astype(np.float32) / float(scale) - KELVIN
    temp = temp.reshape(sh, sw)
    return temp if _plausible(temp) else None


def _parse_duo(raw):
    """Decode a TC002C Duo radiometric frame (the 8x12578 mode) into a Frame.

    The temperature plane carries the thermal scene (real °C from the firmware's
    self-calibrating decode); the raw display-image plane is unpopulated over plain
    UVC, so the temperature field doubles as the display source.
    """
    u16 = np.frombuffer(np.ascontiguousarray(raw).tobytes(), dtype="<u2")
    _header, temp_raw, _image = _duo.split_frame(u16)
    temp = _duo.apparent_celsius(temp_raw)            # real apparent °C, auto Vtemp index
    if not _plausible(temp):
        return Frame(temp.astype(np.float32), None, False, valid=False)
    return Frame(temp.astype(np.float32), temp, True)


def parse_frame(raw, scale=None, order="le", swap=False):
    """Decode a raw YUYV frame into a Frame (image + optional temperature)."""
    if _duo is not None and raw.size >= _duo.FRAME_U16 * 2 and _duo.is_duo_frame(
            np.frombuffer(np.ascontiguousarray(raw).tobytes(), dtype="<u2")):
        return _parse_duo(raw)
    h, w, _ = raw.shape
    layout = KNOWN_LAYOUTS.get((w, h))
    if layout is not None:
        temp_rows, image_rows, sensor, def_scale, def_order = layout
        if swap:
            temp_rows, image_rows = image_rows, temp_rows
        i0, i1 = image_rows
        image = raw[i0:i1, :, 0].astype(np.float32)
        if not _image_aligned(image):
            return Frame(image, None, False, valid=False)
        temp = _decode_temp(raw, temp_rows, sensor, scale or def_scale, order or def_order)
        if temp is None:
            # Known radiometric layout but bad data this frame -> skip it.
            return Frame(image, None, False, valid=False)
        if temp.shape != image.shape:
            temp = cv2.resize(temp, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
        return Frame(image, temp, True)

    # Unknown resolution: fall back to chroma-based band detection (best effort).
    image_band, temp_band = classify_bands(raw, swap)
    if image_band is None:
        return Frame(raw[..., 0].astype(np.float32), None, False)
    i0, i1 = image_band
    image = raw[i0:i1, :, 0].astype(np.float32)
    if temp_band is None or not scale:
        return Frame(image, None, False)
    ih = i1 - i0
    npix = (temp_band[1] - temp_band[0]) * w
    sensor = (w, ih) if ih * w <= npix else (w // 2, ih // 2)
    temp = _decode_temp(raw, temp_band, sensor, scale, order)
    if temp is None:
        return Frame(image, None, False)
    if temp.shape != image.shape:
        temp = cv2.resize(temp, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
    return Frame(image, temp, True)


class ThermalApp:
    def __init__(self, args):
        self.args = args
        self.is_pi = is_raspberrypi()

        self.alpha = args.contrast
        self.colormap = 0
        self.blur = 0
        self.threshold = 2.0
        self.hud = True
        self.recording = False
        self.video_out = None
        self.rec_start = 0.0
        self.elapsed = "00:00:00"
        self.snaptime = "None"
        self.fullscreen = False
        self.swap = args.swap_halves
        self.smooth = args.smooth
        self._ema = None
        self.rotate = args.rotate     # 0/90/180/270
        self.flip = args.flip         # none/h/v

        self.temp_scale = None        # raw-units-per-Kelvin, or None for relative
        self.temp_order = args.temp_order
        self.temp_offset = args.temp_offset
        self.radiometric = False
        self.is_duo = False

        self.cap = None
        self.native_w = self.native_h = 0
        self.scale = args.scale       # finalised once native size is known

    # -- camera lifecycle -------------------------------------------------
    @staticmethod
    def _open_at(device, w, h):
        cap = cv2.VideoCapture(device, cv2.CAP_V4L2)
        if not cap.isOpened():
            return None, None
        cap.set(cv2.CAP_PROP_CONVERT_RGB, 0)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"YUYV"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        # Without this OpenCV hands back stale buffered frames that are desynced
        # mid-frame (the band layout shifts and the parse fails).
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        # Read a few frames: the first ones after a mode change can be warm-up
        # garbage that wouldn't classify correctly.
        raw = None
        for _ in range(5):
            ok, frame = cap.read()
            raw = ThermalApp._as_raw(frame) if ok else None
        if raw is None or raw.shape[1] != w or raw.shape[0] != h:
            cap.release()
            return None, None
        return cap, raw

    def _probe_radiometric(self, cap, raw, scale):
        """True if this mode yields genuine temperature data. Checks the probe
        frame plus a few live ones, since the occasional frame desyncs."""
        if parse_frame(raw, scale, self.temp_order, self.swap).is_real:
            return True
        for _ in range(10):
            ok, frame = cap.read()
            r = self._as_raw(frame) if ok else None
            if r is not None and parse_frame(r, scale, self.temp_order, self.swap).is_real:
                return True
        return False

    def open_camera(self):
        device = self.args.device or find_thermal_device()
        if device is None:
            sys.exit(
                "No thermal camera found. Plug it in and check it appears in\n"
                "  v4l2-ctl --list-devices\n"
                "then pass it explicitly, e.g.  --device /dev/video2"
            )
        if device.isdigit():
            device = "/dev/video" + device

        forced = self.args.temp_scale != "auto"
        forced_scale = float(self.args.temp_scale) if forced else None

        # Only probe modes the camera actually advertises - opening an
        # unsupported resolution desyncs the stream that follows it.
        supported = supported_resolutions(device)
        ok = (lambda wh: supported is None or wh in supported)

        pick = None  # (cap, frame, scale, order)
        if self.args.resolution:
            w, h = (int(x) for x in self.args.resolution.lower().split("x"))
            cap, raw = self._open_at(device, w, h)
            if cap is not None:
                pick = (cap, raw, forced_scale, self.temp_order)
        else:
            # Prefer a mode that yields genuine temperature data. Probe a few
            # frames since the occasional one desyncs.
            for w, h in RADIOMETRIC_MODES:
                if not ok((w, h)):
                    continue
                cap, raw = self._open_at(device, w, h)
                if cap is None:
                    continue
                if (w, h) == DUO_MODE:
                    scale = "duo"                       # LUT-based decode, real °C
                else:
                    scale = forced_scale if forced else KNOWN_LAYOUTS[(w, h)][3]
                if self._probe_radiometric(cap, raw, scale):
                    pick = (cap, raw, scale, self.temp_order)
                    break
                cap.release()
            if pick is None:
                for w, h in IMAGE_MODES:
                    if not ok((w, h)):
                        continue
                    cap, raw = self._open_at(device, w, h)
                    if cap is not None:
                        pick = (cap, raw, forced_scale, self.temp_order)
                        break

        if pick is None:
            sys.exit(f"{device}: no usable YUYV mode. Try --resolution 256x192.")

        self.cap, raw, self.temp_scale, self.temp_order = pick
        self._device = device
        self._wh = (raw.shape[1], raw.shape[0])   # for reopen() after a desync
        frame = parse_frame(raw, self.temp_scale, self.temp_order, self.swap)
        self.is_duo = self.temp_scale == "duo"
        self.radiometric = self.temp_scale is not None or frame.is_real
        if not self.radiometric:
            self.temp_scale = None
        self.native_h, self.native_w = frame.h, frame.w
        if not self.args.scale:                       # auto: aim for ~768px wide
            self.scale = max(1, min(5, round(768 / self.native_w)))
        print(f"Opened {device}: frame {raw.shape[1]}x{raw.shape[0]}, "
              f"image {self.native_w}x{self.native_h}, "
              f"temperature={'REAL °C' if self.radiometric else 'relative (uncalibrated)'}")
        if self.is_duo:
            print(f"  radiometric: TC002C Duo self-calibrating decode (real °C), "
                  f"offset={self.temp_offset:+.1f} °C (adjust with [ and ])")
        elif self.radiometric:
            print(f"  radiometric: scale=1/{self.temp_scale:g} K, order={self.temp_order}, "
                  f"offset={self.temp_offset:+.1f} °C (adjust with [ and ])")
        else:
            print("  no 16-bit data in this stream - see docs/TC002C-DUO.md")

    def reopen(self):
        """Reopen the camera at the same resolution to recover from a stuck
        desync (the stream can wedge into a misaligned state)."""
        try:
            self.cap.release()
        except Exception:
            pass
        cap, _raw = self._open_at(self._device, *self._wh)
        if cap is not None:
            self.cap = cap
            return True
        return False

    @staticmethod
    def _as_raw(frame):
        if frame is None:
            return None
        if frame.ndim == 3 and frame.shape[2] == 2:
            return frame
        if frame.ndim == 2 and frame.shape[1] % 2 == 0:
            return frame.reshape(frame.shape[0], frame.shape[1] // 2, 2)
        return None

    # -- per-frame processing --------------------------------------------
    @staticmethod
    def _destripe(luma):
        """Remove per-row and per-column fixed-pattern offsets while keeping
        smooth gradients (uncorrected previews carry 2D FPN that a stretch
        turns into stripes; we can't run the camera's shutter NUC over UVC)."""
        row = np.median(luma, axis=1)
        luma = luma - (row - cv2.blur(row.reshape(-1, 1), (1, 9)).ravel())[:, None]
        col = np.median(luma, axis=0)
        return luma - (col - cv2.blur(col.reshape(1, -1), (9, 1)).ravel())[None, :]

    def _display_luma(self, image):
        """Turn the raw image band into a stretched, denoised 8-bit luma."""
        luma = image.copy()
        if self.smooth > 0:
            if self._ema is None or self._ema.shape != luma.shape:
                self._ema = luma
            else:
                self._ema = self.smooth * self._ema + (1.0 - self.smooth) * luma
            luma = self._ema
        if not self.args.no_destripe:
            luma = self._destripe(luma)
        if not self.args.no_stretch:
            lo, hi = np.percentile(luma, [1, 99])
            span = max(hi - lo, 45.0)
            luma = np.clip((luma - lo) * (255.0 / span), 0, 255)
        return luma.astype(np.uint8)

    def _orient(self, arr):
        """Apply the current rotation/flip to an image or temperature array."""
        if arr is None:
            return None
        if self.rotate == 90:
            arr = cv2.rotate(arr, cv2.ROTATE_90_CLOCKWISE)
        elif self.rotate == 180:
            arr = cv2.rotate(arr, cv2.ROTATE_180)
        elif self.rotate == 270:
            arr = cv2.rotate(arr, cv2.ROTATE_90_COUNTERCLOCKWISE)
        if self.flip == "h":
            arr = cv2.flip(arr, 1)
        elif self.flip == "v":
            arr = cv2.flip(arr, 0)
        return arr

    def render(self, raw):
        frame = parse_frame(raw, self.temp_scale, self.temp_order, self.swap)
        if not frame.valid:
            return None      # desynced/garbage frame - caller reuses the last one
        is_real = frame.is_real
        image = self._orient(frame.image)
        if is_real:
            temp_map = self._orient(frame.temp) + self.temp_offset
            unit = "C"
        else:
            temp_map = self.args.rel_gain * image + self.args.rel_offset
            unit = "lvl"

        h, w = image.shape
        center = temp_map[h // 2, w // 2]
        inner = temp_map[2:-2, 2:-2]
        max_pos = tuple(p + 2 for p in np.unravel_index(np.argmax(inner), inner.shape))
        min_pos = tuple(p + 2 for p in np.unravel_index(np.argmin(inner), inner.shape))
        tmax, tmin = float(temp_map[max_pos]), float(temp_map[min_pos])
        tavg = float(temp_map.mean())

        luma = self._display_luma(image)
        luma = cv2.convertScaleAbs(luma, alpha=self.alpha)
        new_w, new_h = w * self.scale, h * self.scale
        luma = cv2.resize(luma, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        if self.blur > 0:
            luma = cv2.blur(luma, (self.blur, self.blur))

        heatmap, cmap_name = apply_palette(luma, self.colormap)

        self._draw_crosshair(heatmap, new_w, new_h, center, unit)
        self._draw_markers(heatmap, max_pos, min_pos, tmax, tmin, tavg, unit, w, h, new_w, new_h)
        if self.hud:
            self._draw_hud(heatmap, cmap_name, tavg, unit, is_real)
        return heatmap

    def colormap_frame(self, raw):
        """A clean colormapped BGR frame (oriented, no crosshair/HUD/markers),
        for feeding other apps. Returns None for a desynced frame."""
        frame = parse_frame(raw, self.temp_scale, self.temp_order, self.swap)
        if not frame.valid:
            return None
        image = self._orient(frame.image)
        luma = self._display_luma(image)
        luma = cv2.convertScaleAbs(luma, alpha=self.alpha)
        nw, nh = image.shape[1] * self.scale, image.shape[0] * self.scale
        luma = cv2.resize(luma, (nw, nh), interpolation=cv2.INTER_CUBIC)
        if self.blur > 0:
            luma = cv2.blur(luma, (self.blur, self.blur))
        return apply_palette(luma, self.colormap)[0]

    def _draw_crosshair(self, img, w, h, center_temp, unit):
        cx, cy = w // 2, h // 2
        for color, thick in (((255, 255, 255), 2), ((0, 0, 0), 1)):
            cv2.line(img, (cx, cy + 20), (cx, cy - 20), color, thick)
            cv2.line(img, (cx + 20, cy), (cx - 20, cy), color, thick)
        label = f"{center_temp:.1f} {unit}"
        cv2.putText(img, label, (cx + 10, cy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(img, label, (cx + 10, cy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)

    def _draw_markers(self, img, max_pos, min_pos, tmax, tmin, tavg, unit, w, h, nw, nh):
        sx, sy = nw / w, nh / h
        for (row, col), col_bgr, value in ((max_pos, (0, 0, 255), tmax), (min_pos, (255, 0, 0), tmin)):
            if abs(value - tavg) < self.threshold:
                continue
            x, y = int(col * sx), int(row * sy)
            cv2.circle(img, (x, y), 5, (0, 0, 0), 2)
            cv2.circle(img, (x, y), 5, col_bgr, -1)
            text = f"{value:.1f} {unit}"
            cv2.putText(img, text, (x + 10, y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(img, text, (x + 10, y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)

    def _draw_hud(self, img, cmap_name, tavg, unit, is_real):
        cv2.rectangle(img, (0, 0), (180, 130), (0, 0, 0), -1)
        rows = [
            f"Avg: {tavg:.1f} {unit}",
            f"Threshold: {self.threshold:.0f}",
            f"Colormap: {cmap_name}",
            f"Blur: {self.blur}   Scale: {self.scale}",
            f"Contrast: {self.alpha:.1f}",
            f"Snapshot: {self.snaptime}",
        ]
        for i, text in enumerate(rows):
            cv2.putText(img, text, (8, 14 + i * 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1, cv2.LINE_AA)
        mode = "RADIOMETRIC" if is_real else "RELATIVE (uncal)"
        mode_color = (0, 255, 0) if is_real else (0, 165, 255)
        cv2.putText(img, mode, (8, 14 + len(rows) * 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, mode_color, 1, cv2.LINE_AA)
        rec_color = (40, 40, 255) if self.recording else (200, 200, 200)
        cv2.putText(img, f"REC {self.elapsed}", (8, 14 + (len(rows) + 1) * 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, rec_color, 1, cv2.LINE_AA)

    # -- recording / snapshots -------------------------------------------
    def start_recording(self, frame_size):
        now = time.strftime("%Y%m%d--%H%M%S")
        self.video_out = cv2.VideoWriter(now + "-output.avi", cv2.VideoWriter_fourcc(*"XVID"), 25, frame_size)
        self.recording = True
        self.rec_start = time.time()

    def stop_recording(self):
        self.recording = False
        self.elapsed = "00:00:00"
        if self.video_out is not None:
            self.video_out.release()
            self.video_out = None

    def snapshot(self, heatmap):
        now = time.strftime("%Y%m%d-%H%M%S")
        cv2.imwrite(f"thermal-{now}.png", heatmap)
        self.snaptime = time.strftime("%H:%M:%S")
        print(f"Saved thermal-{now}.png")

    # -- main loop --------------------------------------------------------
    def run(self):
        self.open_camera()
        win = "Thermal"
        cv2.namedWindow(win, cv2.WINDOW_GUI_NORMAL)
        cv2.resizeWindow(win, self.native_w * self.scale, self.native_h * self.scale)
        self._print_keys()
        while True:
            ok, frame = self.cap.read()
            if not ok or frame is None:
                time.sleep(0.05)
                continue
            raw = self._as_raw(frame)
            if raw is None:
                continue
            heatmap = self.render(raw)
            if heatmap is None:        # desynced frame; show the previous one
                continue
            cv2.imshow(win, heatmap)
            if self.recording and self.video_out is not None:
                self.elapsed = time.strftime("%H:%M:%S", time.gmtime(time.time() - self.rec_start))
                self.video_out.write(heatmap)
            if self.handle_key(cv2.waitKey(1) & 0xFF, heatmap):
                break
        self.cleanup()

    def handle_key(self, key, heatmap):
        if key in (ord("q"), 27):
            return True
        elif key == ord("a"):
            self.blur += 1
        elif key == ord("z"):
            self.blur = max(0, self.blur - 1)
        elif key == ord("s"):
            self.threshold += 1
        elif key == ord("x"):
            self.threshold = max(0, self.threshold - 1)
        elif key == ord("d"):
            self.scale = min(5, self.scale + 1)
            self._resize_window()
        elif key == ord("c"):
            self.scale = max(1, self.scale - 1)
            self._resize_window()
        elif key == ord("f"):
            self.alpha = min(3.0, round(self.alpha + 0.1, 1))
        elif key == ord("v"):
            self.alpha = max(0.0, round(self.alpha - 0.1, 1))
        elif key == ord("m"):
            self.colormap = (self.colormap + 1) % len(COLORMAPS)
        elif key == ord("h"):
            self.hud = not self.hud
        elif key == ord("n"):
            self.swap = not self.swap
        elif key == ord("g"):
            self.smooth = 0.0 if self.smooth > 0 else (self.args.smooth or 0.5)
        elif key == ord("o"):  # cycle rotation 0->90->180->270
            self.rotate = (self.rotate + 90) % 360
        elif key == ord("w"):
            self._set_fullscreen(False)
        elif key == ord("e"):
            self._set_fullscreen(True)
        elif key == ord("p"):
            self.snapshot(heatmap)
        elif key == ord("r") and not self.recording:
            self.start_recording((heatmap.shape[1], heatmap.shape[0]))
        elif key == ord("t"):
            self.stop_recording()
        elif key == ord("]"):  # calibration: nudge temperature offset / relative gain
            if self.radiometric:
                self.temp_offset = round(self.temp_offset + 0.5, 1)
            else:
                self.args.rel_gain = round(self.args.rel_gain + 0.05, 3)
        elif key == ord("["):
            if self.radiometric:
                self.temp_offset = round(self.temp_offset - 0.5, 1)
            else:
                self.args.rel_gain = round(self.args.rel_gain - 0.05, 3)
        return False

    def _resize_window(self):
        if not self.fullscreen and not self.is_pi:
            cv2.resizeWindow("Thermal", self.native_w * self.scale, self.native_h * self.scale)

    def _set_fullscreen(self, on):
        self.fullscreen = on
        cv2.setWindowProperty("Thermal", cv2.WND_PROP_FULLSCREEN,
                              cv2.WINDOW_FULLSCREEN if on else cv2.WINDOW_NORMAL)
        if not on:
            self._resize_window()

    def cleanup(self):
        if self.recording:
            self.stop_recording()
        if self.cap is not None:
            self.cap.release()
        cv2.destroyAllWindows()

    def _print_keys(self):
        print(
            "\nKey bindings:\n"
            "  a/z  blur +/-        s/x  min-max label threshold +/-\n"
            "  d/c  scale +/-       f/v  contrast +/-\n"
            "  m    cycle colormap  h    toggle HUD\n"
            "  n    swap bands      g    toggle temporal smoothing\n"
            "  o    rotate 90 deg   e/w  fullscreen on/off\n"
            "  r/t  record / stop   p    snapshot\n"
            "  [/]  temperature offset (or relative gain) calibration\n"
            "  q/ESC quit\n"
        )

    # -- headless self-test ----------------------------------------------
    def selftest(self, frames):
        self.open_camera()
        last = None
        for i in range(frames):
            ok, frame = self.cap.read()
            if not ok or frame is None:
                continue
            raw = self._as_raw(frame)
            if raw is None:
                continue
            f = parse_frame(raw, self.temp_scale, self.temp_order, self.swap)
            rendered = self.render(raw)
            if rendered is not None:
                last = rendered
            if i == frames - 1 and f.valid:
                if f.is_real:
                    t = f.temp + self.temp_offset
                    print(f"frame {i}: REAL temps center={t[f.h//2, f.w//2]:.1f}C "
                          f"min={t.min():.1f} max={t.max():.1f}")
                else:
                    print(f"frame {i}: relative (no radiometric data)")
        self.cap.release()
        if last is None:
            print("SELFTEST FAILED: no frames decoded")
            return 1
        cv2.imwrite("selftest-thermal.png", last)
        print(f"SELFTEST OK: wrote selftest-thermal.png ({last.shape[1]}x{last.shape[0]})")
        return 0


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Universal Linux thermal camera viewer (TC001, TC002C Duo, InfiRay).")
    p.add_argument("--device", help="Video device, e.g. /dev/video2 or 2. Default: auto-detect.")
    p.add_argument("--resolution", help="Force capture WxH, e.g. 512x484. Default: best radiometric mode.")
    p.add_argument("--scale", type=int, default=0, help="Display upscaling 1-5 (0 = auto from image size).")
    p.add_argument("--contrast", type=float, default=1.0, help="Initial contrast (0.0-3.0).")
    p.add_argument("--temp-scale", default="auto",
                   help="Radiometric raw-units-per-Kelvin divisor: 'auto', 64 (TC001), or 16 (Duo).")
    p.add_argument("--temp-order", default="le", choices=["le", "be"], help="16-bit byte order.")
    p.add_argument("--temp-offset", type=float, default=0.0,
                   help="°C added to decoded temperatures (calibrate against a known reference; live keys [ ]).")
    p.add_argument("--rel-gain", type=float, default=0.2, help="Relative-mode level->pseudo-temp gain.")
    p.add_argument("--rel-offset", type=float, default=0.0, help="Relative-mode offset.")
    p.add_argument("--no-stretch", action="store_true", help="Disable the display contrast stretch.")
    p.add_argument("--no-destripe", action="store_true", help="Disable fixed-pattern-noise removal.")
    p.add_argument("--smooth", type=float, default=0.5, help="Temporal smoothing 0.0-0.9 (0 disables). Live: 'g'.")
    p.add_argument("--swap-halves", action="store_true", help="Swap which band is image vs data. Live: 'n'.")
    p.add_argument("--rotate", type=int, default=0, choices=[0, 90, 180, 270],
                   help="Rotate the image clockwise. Live: cycle with 'o'.")
    p.add_argument("--flip", default="none", choices=["none", "h", "v"], help="Mirror the image horizontally/vertically.")
    p.add_argument("--selftest", type=int, metavar="N", help="Headless: grab N frames, save a snapshot, exit.")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    app = ThermalApp(args)
    if args.selftest:
        return app.selftest(args.selftest)
    try:
        app.run()
    except KeyboardInterrupt:
        app.cleanup()
        print("\nBye.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Topdon TC002C Duo radiometric decode: raw stream frame -> temperature (°C).

The Duo's true-temperature mode is the UVC resolution the camera advertises as the
bogus-looking ``8x12578`` (= 100624 "pixels"; as YUYV that's 201248 bytes). Each
frame is 100624 little-endian u16:

    [0:2320]        header / telemetry (magic 0x70827773, dims, scene config, and
                    the app's own min/max/avg of the scene as float32 at byte 144)
    [2320:51472]    temperature plane, 256x192, raw sensor counts
    [51472:100624]  display image, 256x192 (unpopulated over plain UVC)

In the indoor operating range the firmware's conversion is linear in the raw counts:

    temp_C = raw / 64 - K

The gain (1/64 °C per count) is validated **exactly** against the official app: for a
captured frame whose header carries the app's own min/max/avg (22.6 / 34.7 / 27.8 °C),
``raw_min/64 - 50 = 22.62`` and ``raw_max/64 - 50 = 34.69`` -- a 0.1 °C match on all
three. (The camera's full kt/bt + LUT pipeline, reverse-engineered from
libadvirtempac020.so, reduces to this line in-range; the LUT only matters at
temperature extremes outside indoor monitoring.)

``K`` is a per-frame offset that tracks the sensor's Vtemp / focal-plane temperature
(it drifts as the sensor warms). The firmware reads Vtemp from an internal sensor we
can't see over plain UVC, so `estimate_offset` pins a robust cold-baseline percentile
of the frame to an assumed ambient (`ambient_bg`); this auto-tracks Vtemp drift and is
exact for temperature *differences* (e.g. person-vs-background, the reflection gate),
while the absolute level can be fine-tuned with thermalcam's ``--temp-offset`` / ``[``
``]`` keys. Emissivity / reflected-temperature correction is applied on top by
`radiometry.corrector.RadiometricCorrector`.
"""

import numpy as np

GAIN_DIV = 64.0                          # °C per raw count = 1/64 (validated vs the app)
FRAME_U16 = 100624                       # 201248 bytes / 2
FRAME_MAGIC = 0x70827773                 # header[0:2] as a little-endian u32
HEADER_U16 = 2320
SENSOR_W, SENSOR_H = 256, 192
_PLANE = SENSOR_W * SENSOR_H             # 49152
TEMP_OFFSET = HEADER_U16                 # 2320
IMAGE_OFFSET = HEADER_U16 + _PLANE       # 51472

# `estimate_offset` pins this low percentile of the raw plane (the coldest stable
# background, ~ambient) to `AMBIENT_BG` °C. The captured calibration frame's cold
# baseline sat at ~23 °C, which reproduces its app min/max/avg; rooms differ, so this
# is the absolute anchor to nudge via --temp-offset (the gain above is fixed/exact).
BG_PERCENTILE = 2.0
AMBIENT_BG = 22.0


def is_duo_frame(u16):
    """True if a flat u16 buffer looks like a Duo radiometric frame (size + magic)."""
    u = np.asarray(u16).reshape(-1)
    return u.size >= FRAME_U16 and (int(u[0]) | (int(u[1]) << 16)) == FRAME_MAGIC


def split_frame(u16):
    """Split a flat u16 frame into (header, temp_raw[192,256], image[192,256] u8)."""
    u = np.asarray(u16).reshape(-1)[:FRAME_U16]
    header = u[:HEADER_U16]
    temp = u[TEMP_OFFSET:TEMP_OFFSET + _PLANE].reshape(SENSOR_H, SENSOR_W)
    image = (u[IMAGE_OFFSET:IMAGE_OFFSET + _PLANE] & 0xFF).astype(np.uint8).reshape(SENSOR_H, SENSOR_W)
    return header, temp, image


def estimate_offset(temp_raw, ambient_bg=AMBIENT_BG):
    """Per-frame offset K so the cold background sits at ``ambient_bg`` °C.

    Tracks the sensor's Vtemp drift (the whole plane shifts with focal-plane
    temperature) by anchoring a robust low percentile to ambient.
    """
    return float(np.percentile(temp_raw, BG_PERCENTILE)) / GAIN_DIV - ambient_bg


def apparent_celsius(temp_raw, offset=None, ambient_bg=AMBIENT_BG):
    """Raw temperature-plane counts -> temperature (°C): ``raw/64 - K``.

    ``offset`` (K) is auto-estimated from the frame's cold baseline when ``None``.
    Returns a float32 array (apparent temperature, before emissivity correction).
    """
    raw = np.asarray(temp_raw, dtype=np.float32)
    if offset is None:
        offset = estimate_offset(raw, ambient_bg)
    return raw / GAIN_DIV - np.float32(offset)

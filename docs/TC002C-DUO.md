# Topdon TC002C Duo (and other InfiRay UVC cameras) on Linux

Notes from getting `src/thermalcam.py` working with a **Topdon TC002C Duo**, and a
reverse-engineering log of the radiometric (true-temperature) protocol pulled
out of the Android app. Written so the work can be continued.

The TC002C Duo is a dual-sensor (thermal + visible) camera built on the InfiRay
`iruvc` engine — the same family as the TC001 this repo originally targeted, but
it behaves differently over USB.

USB id: `2bdf:0102` (`Topdon`). It enumerates as a standard UVC camera
(`/dev/videoN`), so `thermalcam.py` auto-detects it.

## Frame geometry

`v4l2-ctl -d /dev/video2 --list-formats-ext` reports a partly-garbage table (the
firmware exposes bogus descriptors like `4x12305`, `60x3299`). The real, usable
YUYV modes are:

| Resolution | What it contains |
|-----------|------------------|
| `256x192` | thermal preview only (8-bit) |
| `256x392` | stacked: two 8-bit thermal planes (**no** 16-bit data) |
| `512x384` | 2× thermal preview, 8-bit only |
| `512x484` | **16-bit temperature band (top) + 512x384 thermal image** ← use this |
| `644x384`, `520x192` | other banded/fused layouts |

**The `512x484` mode is the one to use.** It is a stack of horizontal bands:

```
rows   0 - 95 : 16-bit temperature data  (256x192, little-endian, /16 Kelvin),
                packed two sensor rows per 512-wide line  -> 96*512*2 = 256*192*2 bytes
rows  96 - 97 : telemetry
rows  98 -481 : 512x384 thermal AGC image (8-bit, what we display)
rows 482 -483 : telemetry
```

Read naively as one YUYV image the temperature band looks like a green/torn
stripe (its high byte is mis-read as chroma) — which is exactly why it was first
mistaken for garbage. Decoded as 16-bit it is a clean radiometric frame.

Two gotchas cost real debugging time and are handled in `thermalcam.py`:

* **OpenCV buffering** hands back stale, mid-frame-desynced buffers. Set
  `CAP_PROP_BUFFERSIZE = 1`. (`v4l2-ctl --stream-mmap` never desyncs.)
* **Opening an unsupported resolution desyncs the next stream.** Enumerate the
  advertised modes (`VIDIOC_ENUM_FRAMESIZES`) and never probe one the camera
  doesn't list. The occasional desynced frame is also detected (its shifted
  bytes decode to absurd temperatures) and skipped.

## Temperature

Decode the band as little-endian 16-bit and apply the standard InfiRay TPD
formula:

```
temp_C = raw / 16 - 273.15
```

This gives a stable, spatially-correct field (the warm/cool structure matches the
scene exactly). The **absolute** value can be offset by several °C because we are
not running the camera's shutter-based NUC or ambient/emissivity correction — use
`--temp-offset` / the `[` `]` keys to dial it in against a known reference (e.g.
your own skin ~34 °C). The original TC001 uses the same scheme at `256x384` but
with a `/64` divisor; `thermalcam.py` knows both via `KNOWN_LAYOUTS`.

> Earlier notes in this file claimed the Duo exposes *no* 16-bit data over UVC.
> That was wrong — it was based on only testing `256x392`. The `512x484` mode
> carries full radiometric data with no vendor command required.

## Vendor command protocol (still useful for image quality)

True temperatures need no command, but the **image** still shows the sensor's
uncorrected non-uniformity (blotchy on flat scenes) because we can't trigger the
camera's shutter NUC over plain UVC. That — and finer temperature calibration —
is what the vendor command protocol below is for. It was reverse-engineered from
the Android app's native libraries before the `512x484` mode was found.

## The vendor command protocol (reverse-engineered)

The Android APK's Kotlin/Java only holds the InfiRay SDK *wrapper*
(`com.energy.iruvc.*`); the real protocol is in the native `.so` libraries,
which live in the **`split_config.arm64_v8a.apk`** split (pull from the phone via
`adb`, see below). The relevant libs:

- `libircmd.so` — the `ir_cmd` command set + temperature math
- `libUSBUVCCamera.so` — a libuvc + libusb fork; `UVCCamera::sendCommand` is the transport
- `libUSBDualCamera.so` — the Duo dual-sensor path
- `libadvirtemp.so`, `libdualcalibration.so` — Y16→°C + Duo calibration

### Transport: `UVCCamera::sendCommand`

Commands are **USB vendor control transfers** (not UVC-class), i.e. plain
`libusb_control_transfer`:

```
sendCommand(this, bmRequestType, bRequest, wValue, wIndex, data, wLength)
              ->  libusb_control_transfer(handle, bmRequestType, bRequest,
                                          wValue, wIndex, data, wLength, timeout=1000)
```

| Operation | bmRequestType | bRequest | wValue |
|-----------|--------------|----------|--------|
| register WRITE | `0x41` (OUT, vendor, interface) | `0x45` (`'E'`) | `0x0078` |
| register READ  | `0xc1` (IN, vendor, interface)  | `0x44` (`'D'`) | `0x0078` |

`wIndex` is a **device register**:

| Register | Meaning |
|----------|---------|
| `0x9d00` | command descriptor (opcode + arg length) |
| `0x9d08` | command payload (bulk data) |
| `0x1d00`, `0x1d08`, `0x1d40` | command parameters |
| `0x1d04`, `0x1d08` | result read-back |
| `0x200`  | status register (poll: bit0 = busy, retry; error if >3) |

### Worked example: `preview_start`

Decoded from `Java_com_energy_iruvc_ircmd_LibIRCMD_preview_1start` in `libircmd.so`:

```
1. WRITE 0x9d00 <- 0f c1 00 00 00 00 00 08           # opcode 0xc10f, 8-byte arg block
2. WRITE 0x1d08 <- [fps,
                    (source==1 ? 0x80 : 0x00),       # source: 0=SENSOR, 1=FIX_PATTERN
                    width_hi, width_lo,              # big-endian
                    height_hi, height_lo,            # big-endian
                    mode,                            # StartPreviewMode (0=VOC_DVP_MODE)
                    path]                            # PreviewPathChannel
3. POLL  0x200  (read 1 byte) until (status & 1) == 0; abort if status > 3
```

`thermal_protocol.py` in this directory implements this transport and command.
**It is experimental and unverified on hardware** — see the safety notes there.

## Optional next steps (nice-to-have, not required)

Basic temperatures and a clean hi-res image already work from the `512x484`
stream with no commands. The vendor protocol would let us go further:

* **Shutter NUC** — periodically trigger the sensor's flat-field correction
  (`shutter_manual_switch` / `shutter_sta_set`) to remove the slowly-drifting
  blotchy non-uniformity that's visible on flat scenes.
* **Calibrated °C** — apply the camera's emissivity / ambient / NUC correction
  (`libadvirtemp.so` + calibration assets `nuc_table_high.bin`, `tau*.bin`,
  `kt_high.bin`, `bt_high.bin`, `dual_calibration_parameters2.bin`) instead of the
  raw linear formula, removing the absolute offset.
* **Visible / fused image** — the `644x384` and other modes appear to carry the
  Duo's visible camera for thermal+visible fusion (`libUSBDualCamera.so`).

To pursue these, either continue the static RE the way `preview_start` was
decoded, or capture the real app's USB traffic (phone with `usbmon`+root, or
Waydroid on Linux with the camera passed through). None of it is needed for the
working radiometric viewer.

Once the stream delivers genuine 16-bit data, `thermalcam.py` already
auto-detects it (`auto_detect_radiometric`) and shows real °C — or force it with
`--temp-scale`.

## Pulling the native libs from the phone

```bash
adb shell pm path com.topdon.topInfrared       # find base + split APKs
adb pull <.../split_config.arm64_v8a.apk> /tmp/split.apk
unzip -o /tmp/split.apk 'lib/*' -d /tmp/topdon_libs
ls /tmp/topdon_libs/lib/arm64-v8a/             # *.so
```

The libs are ARM64 Android (won't run on an x86_64 laptop) but disassemble fine
(Ghidra/radare2), which is how the protocol above was recovered.

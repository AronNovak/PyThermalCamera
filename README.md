# PyThermalcam
Python Software to use the Topdon TC001 Thermal Camera on Linux and the Raspberry Pi. It **may** work with other similar cameras! Please feed back if it does!

Huge kudos to LeoDJ on the EEVBlog forum for reverse engineering the image format from these kind of cameras (InfiRay P2 Pro) to get the raw temperature data!
https://www.eevblog.com/forum/thermal-imaging/infiray-and-their-p2-pro-discussion/200/
Check out Leo's Github here: https://github.com/LeoDJ/P2Pro-Viewer/tree/main


## Universal viewer — `src/thermalcam.py`

The original program (`src/tc001v4.2.py`, below) is hard-coded for the **Topdon
TC001** frame format. `src/thermalcam.py` is a camera-agnostic rewrite that also
works with newer Topdon / InfiRay models such as the **TC002C Duo**, which expose
a different USB frame geometry.

What it adds:

- **Auto-detects** the thermal camera on the USB bus — no `--device` needed.
- **Auto-detects the frame geometry** instead of assuming `256x384`, so it
  handles the TC001 (`256x384`) and the TC002C Duo family (`256x392`, …).
- **Real error handling** (the original has none).
- **Real temperatures.** It auto-detects the camera's radiometric mode and
  decodes true °C — the TC001 at `256x384` and the **TC002C Duo at `512x484`**
  (which also gives a crisp 512×384 image). Cameras with no 16-bit mode fall back
  to a clearly-labelled *relative* scale. The absolute reading can be offset-
  calibrated with `--temp-offset` / the `[` `]` keys. See `docs/TC002C-DUO.md`.
- Same niceties: colormaps, HUD, recording, snapshots, scaling, blur, contrast,
  hot/cold spot tracking — plus fixed-pattern-noise removal and a contrast stretch
  for cameras that don't pre-AGC their preview.

### Install

```bash
sudo apt-get install python3-opencv        # Debian/Ubuntu/Raspberry Pi
# or:  pip install -r requirements.txt
```

### Run

```bash
python3 src/thermalcam.py                   # auto-detect everything
python3 src/thermalcam.py --device /dev/video2
python3 src/thermalcam.py --resolution 256x384   # e.g. force TC001 radiometric mode
python3 src/thermalcam.py --selftest 20     # headless: save a snapshot + diagnostics, no GUI
```

Useful options: `--temp-offset N` to correct the absolute °C, `--temp-scale
{auto,64,16}` / `--temp-order {le,be}` to force a radiometric decode,
`--no-stretch` / `--no-destripe` / `--smooth` for the image cleanups.

### Keys

`a/z` blur · `s/x` min-max threshold · `d/c` scale · `f/v` contrast · `m` colormap
· `h` HUD · `n` swap bands · `g` temporal smoothing · `e/w` fullscreen on/off ·
`r/t` record/stop · `p` snapshot · `[`/`]` temperature-offset calibration · `q`/ESC quit

### Camera support

| Camera | Image | Temperature |
|--------|-------|-------------|
| Topdon TC001 | ✅ | ✅ real °C (`256x384`, /64) |
| Topdon TC002C Duo | ✅ hi-res 512×384 | ✅ real °C (`512x484`, /16; offset-calibratable) |
| Other InfiRay UVC | ✅ likely | auto-detected if a radiometric mode is present |


## Introduction

This is a quick and dirty Python implimentation of Thermal Camera software for the Topdon TC001!
(https://www.amazon.co.uk/dp/B0BBRBMZ58)
No commands are sent the the camera, instead, we take the raw video feed, do some openCV magic, and display a nice heatmap along with relevant temperature points highlighted.

![Screenshot](media/TC00120230701-131032.png)

This program, and associated information is Open Source (see Licence), but if you have gotten value from these kinds of projects and think they are worth something, please consider donating: https://paypal.me/leslaboratory?locale.x=en_GB 

This readme is accompanied by youtube videos. Visit my Youtube Channel at: https://www.youtube.com/leslaboratory

The video is here: https://youtu.be/PiVwZoQ8_jQ



## Features


Tested on Debian all features are working correctly This has been tested on the Pi However a number of workarounds are implemented! Seemingly there are bugs in the compiled version of openCV that ships with the Pi!!

The following features have been implemented:

<img align="right" src="media/colormaps.png">

- Bicubic interpolation to scale the small 256*192 image to something more presentable! Available scaling multiplier range from 1-5 (Note: This will not auto change the window size on the Pi (openCV needs recompiling), however you can manually resize). Optional blur can be applied if you want to smooth out the pixels. 
- Fullscreen / Windowed mode (Note going back to windowed  from fullscreen does not seem to work on the Pi! OpenCV probably needs recompiling!).
- False coloring of the video image is provided. the avilable colormaps are listed on the right.
- Variable Contrast.
- Average Scene Temperature.
- Center of scene temperature monitoring (Crosshairs).
- Floating Maximum and Minimum temperature values within the scene, with variable threshold.
- Video recording is implemented (saved as AVI in the working directory).
- Snapshot images are implemented (saved as PNG in the working directory).

The current settings are displayed in a box at the top left of the screen (The HUD):

- Avg Temperature of the scene
- Label threshold (temperature threshold at which to display floating min max values)
- Colormap
- Blur (blur radius)
- Scaling multiplier
- Contrast value
- Time of the last snapshot image
- Recording status




## Dependencies

Python3 OpenCV Must be installed:


Run: **sudo apt-get install python3-opencv**



## Running the Program

In src you will find two programs:

**tc001-RAW.py** Just demonstrates how to grab raw frames from the Thermal Camera, a starting point if you want to code your own app.


**tc001v4.2.py** The main program!

To run it plug in the thermal camera and run: **v4l2-ctl --list-devices** to list the devices on the system. You will need its device number.

Assuming the device number is 0 simply issue: **python3 tc001v4.2.py --device 0**

**Note**
This is in Alpha. No error checking has been implemented yet! So if the program tries to start, then quits, either a camera is not connected, or you have entered the wrong device number.

Error checking will be implemented after I refactor and optimize the code!



## Key Bindings


- a z: Increase/Decrease Blur

- s x: Floating High and Low Temp Label Threshold'

- d c: Change Interpolated scale.(Note: This will not change the window size on the Pi!)

- f v: Contrast

- q w: Fullscreen Windowed. (Note: Going back to windowed does not seem to work on the Pi!)

- r t: Record and Stop

- m : Cycle through ColorMaps
  
- h : Toggle HUD



## TODO:

- No Error checking is implemented!
- No attempt has been made to refactor the code (Yet!)!
- The code would benefit from threading especially on low speed but multicore architectures like the Pi!
- I might add a graph.
- I may add the ability to arbitrarily measure points.


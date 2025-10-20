# 🚀 Quick Start Guide - Universal Live Translator v3.1

## First Time Setup (2 minutes)

### Step 1: Install Dependencies
```bash
python app.py
```
The app will automatically detect and install missing packages. Press `y` when prompted.

### Step 2: Check GPU Status
Look at the top of the window:
- 🟢 **"🚀 CUDA (NVIDIA...)"** = GPU detected! ✅
- 🟡 **"🚀 CPU"** = No GPU (still works, just slower)

### Step 3: Configure (Optional but Recommended)
1. Click **⚙️ Settings** button
2. Go to **"GPU & Models"** tab
3. ✅ Check **"Enable GPU Acceleration"** (if available)
4. Select **"base"** for Whisper model (good balance)
5. Click **💾 Save**

### Step 4: Download Models (Optional)
1. Click **📥 Models** button
2. Go to **"Whisper Models"** tab
3. Click **📥 Download** next to **"Base (~150 MB)"**
4. Wait for download to complete

## Daily Use (30 seconds)

### Quick Translation

1. **Select Languages:**
   - From: `Auto` (or choose language)
   - To: `French` (or your target language)

2. **Choose Engine:**
   - Engine: `Whisper` (best quality with GPU)
   - Mode: `Continuous` (never stops listening)

3. **Start Listening:**
   - Click **🎤 Start Continuous Listening**
   - The button turns red when active

4. **Speak Naturally:**
   - Just talk normally - no need to pause!
   - Translation appears in real-time
   - Overlay shows translation on screen

5. **Stop When Done:**
   - Click **⏹️ Stop Continuous Listening**

## Settings You Can Change

### Click ⚙️ Settings to access:

**General Tab:**
- 🎨 Theme (Dark/Light)
- 📏 Font size
- 🌈 Text color
- ⏱️ Animation speed

**Audio Tab:**
- 🔊 Speech rate
- 🎚️ Volume
- 🎤 Microphone selection

**GPU & Models Tab:**
- 🚀 GPU on/off
- 🤖 Model size
- 📊 GPU status

**Cache Tab:**
- 💾 Enable caching
- 📅 Cache expiry
- 📁 Data locations

## Model Downloads

### Click 📥 Models to access:

**Whisper Models:**
- Tiny (75 MB) - Fastest
- **Base (150 MB) - Recommended** ⭐
- Small (500 MB) - Better
- Medium (1.5 GB) - Great
- Large (3 GB) - Best

**Vosk Models (Offline):**
- English, French, Spanish, German
- ~50 MB each
- Works without internet

## Performance Tips

### For Best Speed:
1. ✅ Enable GPU (if available)
2. Use **base** model
3. Select Whisper engine
4. Expected: 150-300ms per phrase

### For Best Accuracy:
1. ✅ Enable GPU
2. Use **small** or **medium** model
3. Select Whisper engine
4. Expected: 300-800ms per phrase

### For Offline Use:
1. Download Vosk models
2. Select Vosk engine
3. No internet required!

### CPU Only:
1. Use **tiny** model
2. Or use Google engine (needs internet)
3. Expected: 500-1000ms per phrase

## Common Tasks

### Change Translation Speed
Settings → Audio → Speech Rate → Adjust slider

### Change Volume
Settings → Audio → Volume → Adjust slider

### Make Text Bigger
Settings → General → Font Size → Increase value

### Switch to Light Theme
Settings → General → Theme → Select "Light"

### Test Different Microphone
Settings → Audio → Input Device → Select device

### Enable/Disable GPU
Settings → GPU & Models → Toggle checkbox

## Keyboard Shortcuts

- **F1** - Help
- **Ctrl+L** - Start/Stop listening
- **Ctrl+T** - Translate manually
- **Ctrl+S** - Speak translation
- **Ctrl+O** - Show/hide overlay
- **Ctrl+D** - Toggle dark/light theme
- **Ctrl+Q** - Quit

## Troubleshooting

### "No GPU detected"
1. Check if you have NVIDIA GPU or Apple Silicon
2. Install CUDA Toolkit (NVIDIA)
3. Reinstall PyTorch: `pip install torch --index-url https://download.pytorch.org/whl/cu118`

### "Transcription failed"
1. Check microphone permissions
2. Settings → Audio → Try different device
3. Download model: Models → Download base

### "Audio not recognized"
1. Speak clearly near microphone
2. Check input volume (system settings)
3. Try different recognition engine

### "Translation is slow"
1. Enable GPU if available
2. Use smaller model (tiny/base)
3. Check performance: Status bar shows ms

### "Settings not saving"
1. Check file permissions
2. Look for config: `~/.universal_translator/translator_config.json`
3. Restart app

## Status Bar Guide

**Performance Monitor:**
```
⚡ 150ms | Q:0/0 | cuda
   ^^^    ^^^^^   ^^^^
   │      │       └─ Active device
   │      └───────── Queue sizes (recognition/translation)
   └──────────────── Processing time (lower = faster)
```

**Good Performance:**
- < 300ms with GPU
- < 1000ms with CPU
- Q: 0-2 (low queue = responsive)

**Poor Performance:**
- > 2000ms
- Q: 8-10 (full queue = lagging)
- Try smaller model or enable GPU

## Data Locations

All app data stored in: `~/.universal_translator/`

**Files:**
- `translator_config.json` - Your settings
- `translator_history.db` - Translation history
- `audio_cache/` - Temporary audio files
- `vosk_models/` - Downloaded Vosk models
- `translator.log` - Debug logs

**Models:**
- Whisper: `~/.cache/whisper/`
- Vosk: `~/.universal_translator/vosk_models/`

## Next Steps

1. ✅ **Configure settings** to your preference
2. ✅ **Download models** you'll use
3. ✅ **Test different engines** to find your favorite
4. ✅ **Customize appearance** (theme, font, colors)
5. ✅ **Set up shortcuts** you'll use often

## Need Help?

- Press **F1** in the app
- Check `translator.log` for errors
- Review `UPDATE_SUMMARY.md` for details
- Check `IMPROVEMENTS.md` for technical info

## Enjoy! 🎉

You now have a professional live translator with:
- 🎯 Real-time translation
- 🚀 GPU acceleration
- 📱 Beautiful overlay
- ⚙️ Full customization
- 📥 Easy model management

**Happy translating!** 🌍

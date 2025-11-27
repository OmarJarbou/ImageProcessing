## Posterized Noir Webcam Demo

`part1.py` captures webcam frames, applies grayscale sharpening, posterizes the tones, and darkens the image corners with a vignette. Follow the steps below to run it locally.

### Requirements
- Python 3.9+ (works cross-platform; instructions below use PowerShell syntax)
- A webcam accessible by OpenCV
- Packages: `opencv-python`, `numpy`

### Setup
1. Open a terminal and move into the project folder:
   ```
   cd C:\Users\User\Documents\MyProjects\ImageProcessing
   ```
2. (Optional) Create and activate a virtual environment:
   ```
   python -m venv venv
   venv\Scripts\activate
   ```
3. Install dependencies:
   ```
   pip install opencv-python numpy
   ```

### Run
1. Ensure no other application is using the webcam.
2. Ensure you are on: "(venv) PS C:\Users\User\Documents\MyProjects\ImageProcessing>"
3. Launch the script:
   ```
   python part1.py
   ```
4. A window titled “Posterized Noir” will open showing the live effect.

### Controls
- `q`: quit
- `+`: increase posterization (fewer intensity bands, more stylized)
- `-`: decrease posterization (more intensity bands)

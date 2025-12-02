## Posterized Noir Webcam Demo

`part1.py` captures webcam frames, applies grayscale sharpening, posterizes the tones, and darkens the image corners with a vignette. Follow the steps below to run it locally.

### Requirements
- Python 3.9+ (works cross-platform; instructions below use PowerShell syntax)
- A webcam accessible by OpenCV
- Packages: `opencv-python`, `numpy` , `scikit-learn`

### Setup
0. INSURE YOU HAVE THIS PYTHON VERSION: python 3.13.5
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
   python -m pip install --upgrade pip
   pip install opencv-python numpy scikit-learn
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


## Image Compression & Redundancy Analysis Demo

`part2.py` analyzes a grayscale image to detect redundancy and applies an appropriate **lossless compression** method (RLE, Huffman, or DPCM + Huffman). It then displays compression results including original & compressed size, compression ratio, and redundancy percentage.


### Requirements
- Python 3.12+  
- Packages: `opencv-python`, `numpy`  


### Setup
1. Open a terminal and move into the project folder:
    ```
   cd C:\Users\User\Documents\MyProjects\ImageProcessing
   ```
2. Install dependencies:
   ```
   pip install opencv-python numpy
   ```
### Run
1. Ensure the input image exists and no other application is locking the file.  
2. Open a terminal and navigate to the project folder:

### Output
Image Analysis:<br>
- Entropy (coding redundancy)<br>
- Spatial similarity score<br>
- Chosen compression method<br>

Results:<br>
- Original size (bytes)<br>
- Estimated compressed size (bytes)<br>
- Compression ratio<br>
- Redundancy percentage<br>



   


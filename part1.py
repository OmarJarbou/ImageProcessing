import cv2
import numpy as np

def create_vignette_mask(width, height):
    """
    Create a 2D radial mask (bright center → dark corners) used for the vignette effect.

    Steps:
    1. Generate X and Y coordinate ranges from -1 to 1.
       These ranges represent normalized horizontal and vertical axes.
       For example (ASSUMING IMAGE SIZE IS 5*5):
       X = -1 -0.5 0(center) 0.5 1
       Y = -1
           -0.5
           0(center)
           0.5
           1
    2. Use meshgrid() to convert the 1D coordinate arrays into 2D matrices
       so every pixel gets an (x, y) coordinate.
       Result for the above example:
       ( -1,  -1) (-0.5,  -1) (0,  -1) (0.5,  -1) (1,  -1)
       ( -1,-0.5) (-0.5,-0.5) (0,-0.5) (0.5,-0.5) (1,-0.5)
       ( -1,   0) (-0.5,   0) (0,   0) (0.5,   0) (1,   0)
       ( -1, 0.5) (-0.5, 0.5) (0, 0.5) (0.5, 0.5) (1, 0.5)
       ( -1,   1) (-0.5,   1) (0,   1) (0.5,   1) (1,   1)
    3. Compute the "radius" = distance of each pixel from the image center.
       The factors (2/3) slightly compress the circle to cover image corners.

       Result for the above example:
       √2       √1.25   √1      √1.25   √2      =>  1.4  1.1   1   1.1  1.4
       √1.25    √0.5    √0.25   √0.5    √1.25   =>  1.1  0.7  0.5  0.7  1.1
       √1       √0.25   0       √0.25   √1      =>   1   0.5   0   0.5   1 
       √1.25    √0.5    √0.25   √0.5    √1.25   =>  1.1  0.7  0.5  0.7  1.1
       √2       √1.25   √1      √1.25   √2      =>  1.4  1.1   1   1.1  1.4

       RESULT AFTER ADDING 2/3 FACTORS TO THE EXPERIMENT:
       0.94  0.75  0.44  0.75  0.94
       0.75  0.47  0.33  0.47  0.75
       0.66  0.33   0    0.33  0.66
       0.75  0.47  0.33  0.47  0.75
       0.94  0.75  0.44  0.75  0.94

    4. Convert radius values into a mask:
         - radius near 0   → center → mask = 1   (fully bright)
         - radius near 1   → edges  → mask = 0   (fully dark)
        To do so, we use (1 - clip):
        Clip (limit) the values in an array.
        Given an interval, values outside the interval are clipped to the interval edges. 
        For example, if an interval of [0, 1] is specified, values smaller than 0 become 0, 
        and values larger than 1 become 1.

        IN OUR EXAMPLE CASE, NOTHING WILL CHANGE AFTER APPLYING THE CLIP BECAUSE ALL NUMBERS BETWEEN 0-1:
        SO MASK = 1 - RADIUS =
        0.06  0.25  0.56  0.25  0.06
        0.25  0.53  0.67  0.53  0.25
        0.34  0.67   1    0.67  0.34
        0.25  0.53  0.67  0.53  0.25
        0.06  0.25  0.56  0.25  0.06

    5. Return a smooth gradient mask used to darken the image edges (BY MULTIPLYING THE MASK WITH IT).
    """
    X = np.linspace(-1, 1, width)
    Y = np.linspace(-1, 1, height)
    xv, yv = np.meshgrid(X, Y)

    # distance from center map (normalized)
    radius = np.sqrt((xv*2/3)**2 + (yv*2/3)**2)

    mask = 1 - np.clip(radius, 0, 1)
    return mask

def posterize(img, levels=6):
    """
    Reduce the grayscale image to a limited number of intensity levels
    (also called quantization or banding).

    Explanation:
    - The posterization effect works by forcing pixel values into 'levels'
      equally spaced gray bands.
    - Example: if levels = 6, the resulting intensities become:
        0, 51, 102, 153, 204, 255
    - This gives a stylized "cartoon" or noir look by removing smooth gradients.

    Steps:
    1. Compute the interval size between allowed gray levels.
    2. Integer-divide the image by this interval to find each pixel's band.
    3. Multiply back to stretch the band index to the full 0–255 range.
    4. Convert result to uint8 (OpenCV image format).
    """
    # we do a floor operation "//" to achive that
    factor = 255 // (levels - 1) # Step size between allowed gray values (ex: if levels == 6: factor = step size = 51)
    poster = (img // factor) * factor
    return poster.astype(np.uint8)

cap = cv2.VideoCapture(0)
ret, frame = cap.read()
h, w = frame.shape[:2]
vignette = create_vignette_mask(w, h)
lvls = [4, 6, 8, 10, 16, 18, 52, 86, 256]
k = 0

print("Press q to quit, + to increase posterize level, - to decrease posterize level")
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 1. Convert to Grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 2. Gaussian blur
    blurred = cv2.GaussianBlur(gray, (5,5), 1)

    # 3. Unsharp masking DOES sharpen, but the name comes from subtracting a blurred (unsharp) copy,
    # so it needs a blurred version of the image to work, otherwise no changes happen
    # Formula: sharp = original + amount*(original - blurred)
    sharp = cv2.addWeighted(gray, 1.4, blurred, -0.4, 0)

    # 4. Posterization (based on gray level scaling)
    poster = posterize(sharp, levels=lvls[k])

    # 5. Apply vignette (dark gradient from image corners)
    noir = poster * vignette
    noir = noir.astype(np.uint8)

    # To show final image result
    cv2.imshow("Posterized Noir", noir)

    # The while loop won't terminate until we press 'q'
    # Increase posturization (make image more like carton) we press '+', to decrease we press '-'
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('+'):
        k = max(k - 1, 0)
        print(f"k: {k} Intensity levels: {lvls[k]}")
    elif key == ord('-'):
        k = min(k + 1, len(lvls) - 1)
        print(f"k: {k} Intensity levels: {lvls[k]}")

cap.release()
cv2.destroyAllWindows()
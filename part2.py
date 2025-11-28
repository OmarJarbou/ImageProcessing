import cv2
import numpy as np
from collections import Counter

# ============================================================
#   1) Redundancy Analysis Functions (from the course)
# ============================================================

def compute_entropy(img):
    """
    Compute image entropy:
    - Measures true information content in the image.
    - Low entropy → high coding redundancy.
    - Used to decide if Huffman coding is suitable.
    """
    # Calculate histogram (frequency of pixel values 0–255)
    hist = cv2.calcHist([img], [0], None, [256], [0,256]).flatten()

    # Convert frequencies to probabilities P(rk)
    prob = hist / np.sum(hist)

    # Remove zero probabilities (log2(0) is undefined)
    prob = prob[prob > 0]

    # Shannon entropy formula: H = Σ p * log2(p)
    entropy = -np.sum(prob * np.log2(prob))
    return float(entropy)


def compute_spatial_redundancy(img):
    """
    Compute Spatial Redundancy:
    - Measures similarity between neighboring pixels.
    - If differences are small → many repeated values → RLE is effective.
    """
    # Compute absolute difference between neighboring pixels
    diff = np.abs(np.diff(img.astype(np.int16), axis=1))

    # Return average difference value
    return float(np.mean(diff))


# ============================================================
#   2) RLE – Run-Length Encoding
# ============================================================

def rle_encode(data):
    """
    Run-Length Encoding:
    - If a pixel repeats 10 times, store (value, 10)
      instead of writing the value 10 times.
    """
    encoded = []
    prev = data[0]   # first pixel value
    count = 1        # repetition counter

    # Loop through all pixels starting from the second one
    for x in data[1:]:

        # If same value → increase counter
        if x == prev:
            count += 1

        # If value changes → save previous run
        else:
            encoded.append((prev, count))
            prev = x
            count = 1

    # Store the last run
    encoded.append((prev, count))
    return encoded


# ============================================================
#   3) Huffman Coding
# ============================================================

def build_huffman_codes(freq):
    """
    Build Huffman codes using pixel frequencies:
    - More frequent symbols → shorter binary codes.
    """

    # Define a tree node
    class Node:
        def __init__(self, symbol=None, freq=0, left=None, right=None):
            self.symbol = symbol
            self.freq = freq
            self.left = left
            self.right = right

        # Comparison function for priority queue
        def __lt__(self, other):
            return self.freq < other.freq

    import heapq
    heap = []

    # Insert all symbols into priority queue
    for sym, f in freq.items():
        heapq.heappush(heap, Node(sym, f))

    # Special case: image has only one unique value
    if len(heap) == 1:
        return {list(freq.keys())[0]: "0"}

    # Build Huffman tree
    while len(heap) > 1:
        n1 = heapq.heappop(heap)
        n2 = heapq.heappop(heap)
        merged = Node(None, n1.freq + n2.freq, n1, n2)
        heapq.heappush(heap, merged)

    # Final root node
    root = heapq.heappop(heap)
    codes = {}

    # Recursively walk the tree to create codes
    def walk(node, code):
        if node.symbol is not None:
            codes[node.symbol] = code
        else:
            walk(node.left, code + "0")
            walk(node.right, code + "1")

    walk(root, "")
    return codes


def huffman_estimate_bits(data):
    """
    Estimate compressed size using Huffman coding:
    total bits = Σ freq(symbol) × length(Huffman code)
    """
    freq = Counter(data)                      # count each pixel value
    codes = build_huffman_codes(freq)         # generate Huffman codes
    total_bits = sum(freq[sym] * len(code) for sym, code in codes.items())
    return total_bits


# ============================================================
#   4) DPCM (Predictive Coding)
# ============================================================

def dpcm_residuals(data):
    """
    DPCM (Differential Pulse-Code Modulation):
    - Uses prediction: each pixel ≈ previous pixel.
    - Stores differences instead of actual pixel values.
    - Differences are small → very compressible by Huffman.
    """
    res = np.zeros_like(data, dtype=np.int16)  # allow negative values
    res[0] = int(data[0])
    for i in range(1, len(data)):
        res[i] = int(data[i]) - int(data[i - 1])

    return res


# ============================================================
#   5) Main Compression Program
# ============================================================

def compress_image(path):

    # Load image in grayscale mode
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    flat = img.flatten()  # convert to 1D array

    print("\n=== Image Analysis ===")

    # Compute both types of redundancy
    entropy = compute_entropy(img)
    spatial = compute_spatial_redundancy(img)

    print(f"Entropy (coding redundancy): {entropy:.3f} bits")
    print(f"Spatial similarity score:   {spatial:.3f}")

    # Choose compression method based on heuristics
    if spatial < 1.0:
        method = "RLE"
    elif entropy > 7.0:
        method = "DPCM+HUFFMAN"
    else:
        method = "HUFFMAN"

    print(f"Chosen Compression Method: {method}")

    # Original image size in bits (each pixel = 8 bits)
    original_bits = len(flat) * 8

    # Apply selected compression method
    if method == "RLE":
        rle = rle_encode(flat)
        compressed_bits = len(rle) * 24   # 8-bit value + 16-bit count

    elif method == "HUFFMAN":
        compressed_bits = huffman_estimate_bits(flat)

    else:  # DPCM + Huffman
        res = dpcm_residuals(flat)
        shifted = res - res.min()          # make values non-negative
        compressed_bits = huffman_estimate_bits(shifted)


    compressed_bytes = compressed_bits / 8
    cr = original_bits / compressed_bits             # compression ratio
    redundancy = 1 - (1/cr)                          # redundancy %

    print("\n=== Results ===")
    print(f"Original size:    {original_bits/8:.0f} bytes")
    print(f"Compressed size:  {compressed_bytes:.0f} bytes")
    print(f"Compression ratio: {cr:.2f}")
    print(f"Redundancy:        {redundancy*100:.2f}%")


compress_image("image.png")

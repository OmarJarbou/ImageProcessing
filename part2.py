#!/usr/bin/env python3
"""
part2.py
Image redundancy analysis and compression (RLE, Huffman, DPCM+Huffman).

Usage:
    python part2.py --input path/to/image.png --out compressed_output.npz

Outputs:
 - prints analysis (spatial score, entropy, correlation)
 - chosen compression method
 - original size (bytes), compressed size (bytes), compression ratio, redundancy %
 - saves compressed data to .npz
"""

import argparse
import numpy as np
import cv2
import os
from collections import Counter, defaultdict
import heapq
import math
import sys

# ----------------------------
# Analysis Functions
# ----------------------------
def compute_spatial_score(img_gray):
    # mean absolute difference between horizontal neighbors
    diff = np.abs(np.diff(img_gray.astype(np.int16), axis=1))
    return float(np.mean(diff))

def compute_entropy(img_gray):
    hist = cv2.calcHist([img_gray], [0], None, [256], [0,256]).flatten()
    prob = hist / np.sum(hist)
    prob = prob[prob > 0]
    entropy = -np.sum(prob * np.log2(prob))
    return float(entropy)

def compute_correlation(img_gray):
    flat = img_gray.flatten().astype(np.float64)
    if flat.size < 2:
        return 0.0
    x1 = flat[:-1]
    x2 = flat[1:]
    # avoid degenerate case
    if x1.std() == 0 or x2.std() == 0:
        return 0.0
    corr = np.corrcoef(x1, x2)[0,1]
    return float(corr)

# ----------------------------
# RLE Compression (simple)
# ----------------------------
def rle_encode(flat):
    if flat.size == 0:
        return []
    encoding = []
    prev = int(flat[0])
    count = 1
    for v in flat[1:]:
        v = int(v)
        if v == prev and count < 65535:  # keep count within 16-bit for storage simplicity
            count += 1
        else:
            encoding.append((prev, count))
            prev = v
            count = 1
    encoding.append((prev, count))
    return encoding

def rle_estimated_bits(rle_encoded):
    # store value as 8 bits, count as 16 bits => 24 bits per pair
    return len(rle_encoded) * (8 + 16)

# ----------------------------
# Huffman Coding
# ----------------------------
class HuffmanNode:
    def __init__(self, freq, symbol=None, left=None, right=None):
        self.freq = freq
        self.symbol = symbol
        self.left = left
        self.right = right
    def __lt__(self, other):
        return self.freq < other.freq

def build_huffman_codes(freq_dict):
    heap = []
    for sym, f in freq_dict.items():
        heapq.heappush(heap, HuffmanNode(f, symbol=sym))
    if len(heap) == 0:
        return {}
    # edge case: single symbol
    if len(heap) == 1:
        node = heapq.heappop(heap)
        return {node.symbol: "0"}

    while len(heap) > 1:
        n1 = heapq.heappop(heap)
        n2 = heapq.heappop(heap)
        merged = HuffmanNode(n1.freq + n2.freq, left=n1, right=n2)
        heapq.heappush(heap, merged)

    root = heapq.heappop(heap)
    codes = {}
    def walk(node, prefix):
        if node.symbol is not None:
            codes[node.symbol] = prefix
            return
        walk(node.left, prefix + "0")
        walk(node.right, prefix + "1")
    walk(root, "")
    return codes

def huffman_bits(freq_dict, codes):
    # total bits = sum(freq(symbol) * len(code(symbol)))
    total = 0
    for sym, f in freq_dict.items():
        total += f * len(codes[sym])
    return total

# ----------------------------
# DPCM (predictive) + Huffman on residuals
# ----------------------------
def dpcm_residuals(flat):
    # simple predictor: previous pixel (causal)
    if flat.size == 0:
        return np.array([], dtype=np.int16)
    res = np.empty_like(flat.astype(np.int16))
    prev = int(flat[0])
    res[0] = prev  # store first pixel raw
    for i in range(1, flat.size):
        cur = int(flat[i])
        pred = prev
        r = cur - pred  # signed residual
        res[i] = r
        prev = cur
    return res

# ----------------------------
# Utility: bytes size of original file (if available)
# ----------------------------
def file_size_bytes(path):
    try:
        return os.path.getsize(path)
    except Exception:
        return None

# ----------------------------
# Main pipeline
# ----------------------------
def analyze_and_compress(input_path, output_path):
    # Read image as grayscale
    img_bgr = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)
    if img_bgr is None:
        print("Error: couldn't read image:", input_path)
        sys.exit(1)

    # If image has alpha or is colored, convert to grayscale
    if len(img_bgr.shape) == 3:
        img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    else:
        img_gray = img_bgr.copy()

    h, w = img_gray.shape
    print(f"Image: {input_path}  ({w} x {h})")

    # Analysis
    spatial = compute_spatial_score(img_gray)
    entropy = compute_entropy(img_gray)
    corr = compute_correlation(img_gray)

    print(f"Spatial score (mean abs horiz diff): {spatial:.4f}")
    print(f"Entropy (bits per pixel): {entropy:.4f}")
    print(f"Inter-pixel correlation (adjacent): {corr:.4f}")

    # thresholds (heuristic)
    # lower spatial score -> more smoothness -> good for RLE
    # lower entropy -> more coding redundancy -> good for Huffman
    # higher correlation -> predictive coding helps
    choose = None

    if spatial < 5.0:
        choose = "RLE"
    elif entropy < 6.0:
        choose = "HUFFMAN"
    elif corr > 0.85:
        choose = "DPCM+HUFFMAN"
    else:
        # default fallback
        choose = "HUFFMAN"

    print("Chosen compression method:", choose)

    flat = img_gray.flatten()

    original_bits = img_gray.size * 8  # raw bitmap in memory (uncompressed) - bits
    original_file_bytes = file_size_bytes(input_path)
    if original_file_bytes is not None:
        print(f"Original file size on disk: {original_file_bytes} bytes")
    print(f"Original (raw) size: {original_bits} bits")

    compressed_bits = None
    metadata = {"method": choose, "width": w, "height": h}

    if choose == "RLE":
        rle = rle_encode(flat)
        compressed_bits = rle_estimated_bits(rle)
        # store as arrays for later verification
        vals = np.array([v for v,c in rle], dtype=np.uint8)
        counts = np.array([c for v,c in rle], dtype=np.uint16)
        np.savez_compressed(output_path, method="RLE", values=vals, counts=counts, width=w, height=h)
        print(f"RLE pairs: {len(rle)}")

    elif choose == "HUFFMAN":
        freq = Counter(map(int, flat))
        codes = build_huffman_codes(freq)
        compressed_bits = huffman_bits(freq, codes)
        # save codebook and data frequencies (we save frequencies to reconstruct)
        keys = np.array(list(freq.keys()), dtype=np.uint8)
        vals = np.array([freq[k] for k in keys], dtype=np.int32)
        np.savez_compressed(output_path, method="HUFFMAN", symbols=keys, freqs=vals, width=w, height=h)
        print(f"Huffman symbol count: {len(keys)}")

    elif choose == "DPCM+HUFFMAN":
        res = dpcm_residuals(flat)  # signed residuals, first is raw pixel
        # shift residuals to make them non-negative for simple frequency counting
        # find min to shift
        minr = int(res.min())
        shift = 0
        if minr < 0:
            shift = -minr
        shifted = (res + shift).astype(np.int32)
        freq = Counter(map(int, shifted))
        codes = build_huffman_codes(freq)
        compressed_bits = huffman_bits(freq, codes)
        # plus we need to store shift (assume 32-bit) and first pixel raw (8-bit)
        # accounted for in metadata in saved file
        keys = np.array(list(freq.keys()), dtype=np.int32)
        vals = np.array([freq[k] for k in keys], dtype=np.int32)
        np.savez_compressed(output_path, method="DPCM+HUFFMAN", symbols=keys, freqs=vals, shift=shift, width=w, height=h)
        print(f"DPCM residuals unique symbols: {len(keys)}")

    else:
        raise RuntimeError("Unknown method chosen")

    # compute compressed bytes estimate
    compressed_bytes_est = math.ceil(compressed_bits / 8) if compressed_bits is not None else None

    print("\n--- Results ---")
    if original_file_bytes is not None:
        orig_bytes = original_file_bytes
    else:
        orig_bytes = math.ceil(original_bits / 8)
    print(f"Original size (bytes): {orig_bytes}")
    print(f"Estimated compressed size (bytes): {compressed_bytes_est}")
    if compressed_bytes_est and compressed_bytes_est > 0:
        cr = orig_bytes / compressed_bytes_est
        redundancy = 1.0 - (1.0 / cr) if cr != 0 else 0.0
        print(f"Compression Ratio (orig / compressed): {cr:.4f}")
        print(f"Redundancy percentage: {redundancy * 100:.2f}%")
    else:
        print("Couldn't compute compressed size properly.")

    print(f"Compressed data saved to: {output_path}")
    return {
        "spatial": spatial,
        "entropy": entropy,
        "correlation": corr,
        "method": choose,
        "original_bytes": orig_bytes,
        "compressed_bytes_est": compressed_bytes_est
    }

# ----------------------------
# CLI
# ----------------------------
def main():
    parser = argparse.ArgumentParser(description="Image redundancy analysis & compression.")
    parser.add_argument("--input", "-i", required=True, help="Input image path")
    parser.add_argument("--out", "-o", default="compressed_output.npz", help="Output compressed .npz path")
    args = parser.parse_args()
    analyze_and_compress(args.input, args.out)

if __name__ == "__main__":
    main()

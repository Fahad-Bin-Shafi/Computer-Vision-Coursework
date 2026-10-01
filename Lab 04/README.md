# Lab 04 — Skin Lesion Boundary Detection Using Canny Edge Detection

**Course:** Computer Vision  
**Student:** Fahad Bin Shafi  
**Registration No.:** FA23-BAI-033  

---

## 1. Problem Statement

The aim of this lab is to develop a simple computer vision system that detects the boundary of a skin lesion from a skin image using image filtering and Canny edge detection.

The main objective is to determine how effectively edge detection can separate a lesion from the surrounding skin and then use the detected boundary to estimate lesion area and perimeter.

---

## 2. Dataset Used

The implementation uses images from the **HAM10000 Skin Cancer dataset** downloaded through KaggleHub.

Five images were selected in the notebook:

1. `ISIC_0028933.jpg`
2. `ISIC_0028394.jpg`
3. `ISIC_0027799.jpg`
4. `ISIC_0028100.jpg`
5. `ISIC_0027960.jpg`

---

# Lab Tasks

## Task 1 — Load the Image

The skin-lesion images were loaded using OpenCV.

The notebook:

- downloaded the HAM10000 dataset,
- located `.jpg` images,
- selected five images,
- loaded them with `cv2.imread()`,
- converted them from BGR to RGB for correct visualization.

Example implementation:

```python
img = cv2.imread(img_path)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
```

The original images were then displayed as part of the complete processing pipeline.

---

## Task 2 — Preprocess the Image

Each image was converted to grayscale before edge detection.

```python
grayscale = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
```

A **Gaussian filter** with a `5 × 5` kernel was then applied:

```python
filtered = cv2.GaussianBlur(grayscale, (5, 5), 0)
```

The notebook also implemented Median and Average filters for comparison.

### Why preprocessing was required

Preprocessing reduces unwanted image noise and small intensity variations before edge detection. This makes the final lesion edge cleaner and reduces false edges caused by skin texture, hair, or image noise.

The required visualization sequence was implemented as:

**Original → Grayscale → Gaussian Filter → Canny → Lesion Boundary**

---

## Task 3 — Apply Canny Edge Detection

Canny edge detection was tested using the three threshold settings required in the assignment:

- `50–100`
- `100–200`
- `150–250`

Implementation:

```python
edges = cv2.Canny(image, threshold1, threshold2)
```

The notebook generated a separate edge map for each threshold pair to compare their effects.

---

## Task 4 — Select the Best Result

The selected Canny threshold was:

## **100–200**

This threshold gave the best overall balance in the experiment.

### Comparison

| Threshold | Observation |
|---|---|
| **50–100** | Very sensitive. More weak edges and unwanted noise were detected. |
| **100–200** | Best balance. The main lesion boundary was clearer with less unwanted noise. |
| **150–250** | Too strict in many cases. Some weaker lesion boundary sections could disappear. |

Therefore, `100–200` was used in the final processing pipeline.

---

## Task 5 — Detect the Lesion Boundary

After Canny edge detection, morphological closing was applied to connect broken edge segments.

```python
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
closed = cv2.morphologyEx(edges_image, cv2.MORPH_CLOSE, kernel, iterations=2)
```

Contours were then detected:

```python
contours, _ = cv2.findContours(
    closed,
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE
)
```

The **largest contour** was selected as the lesion boundary:

```python
largest_contour = max(contours, key=cv2.contourArea)
```

Finally, the detected contour was drawn on the original image using:

```python
cv2.drawContours(boundary_image, [largest_contour], 0, (0, 255, 0), 2)
```

This produced the final lesion-boundary visualization.

---

## Task 6 — Calculate Lesion Area and Perimeter

The lesion area and perimeter were calculated from the selected contour.

```python
area = cv2.contourArea(contour)
perimeter = cv2.arcLength(contour, True)
```

### Experimental Results

| Image | Filename | Best Filter | Edge Method | Area (pixels) | Perimeter (pixels) |
|---|---|---|---|---:|---:|
| Image 1 | ISIC_0028933.jpg | Gaussian | Canny (100–200) | 2232.00 | 1342.79 |
| Image 2 | ISIC_0028394.jpg | Gaussian | Canny (100–200) | 0.00 | 0.00 |
| Image 3 | ISIC_0027799.jpg | Gaussian | Canny (100–200) | 265.50 | 197.46 |
| Image 4 | ISIC_0028100.jpg | Gaussian | Canny (100–200) | 121.50 | 67.98 |
| Image 5 | ISIC_0027960.jpg | Gaussian | Canny (100–200) | 0.00 | 0.00 |

### Important Observation

For **Image 2** and **Image 5**, the pipeline returned an area and perimeter of `0`. This means the selected processing configuration did not obtain a usable lesion contour for those images. This is an important limitation of the current method rather than a valid lesion measurement.

The full visualization pipeline successfully completed for three images in the notebook.

---

# Filter and Edge-Detection Comparison

The notebook also compared Gaussian, Median, and Average filters using both Canny and Sobel edge detection on the first image.

| Filter | Method | Edge Pixels | Edge Quality | Area | Perimeter | Performance |
|---|---|---:|---:|---:|---:|---|
| Gaussian | Canny | 2677 | 0.99% | 2232 | 1343 | Good |
| Gaussian | Sobel | 248254 | 91.95% | 268951 | 2096 | Good |
| Median | Canny | 3468 | 1.28% | 2858 | 1451 | Good |
| Median | Sobel | 235295 | 87.15% | 268822 | 2107 | Good |
| Average | Canny | 358 | 0.13% | 142 | 567 | Poor |
| Average | Sobel | 239525 | 88.71% | 268719 | 2132 | Good |

These results show that different preprocessing filters greatly affect the edge map and the contour selected by the algorithm.

---

# Final Comparison Table

| Method | Noise Handling | Edge Quality | Boundary Detection | Overall Performance |
|---|---|---|---|---|
| Original + Sobel | None | Good | Excellent | Excellent |
| Original + Canny | None | Excellent | Excellent | Excellent |
| Average + Sobel | Fair | Good | Excellent | Excellent |
| Average + Canny | Fair | Excellent | Excellent | Poor |
| Gaussian + Sobel | Good | Good | Excellent | Excellent |
| Gaussian + Canny | Good | Excellent | Excellent | Excellent |
| Median + Sobel | Good | Good | Excellent | Excellent |
| Median + Canny | Good | Excellent | Excellent | Excellent |

> **Note:** These labels are the evaluation values produced by the notebook's rule-based comparison function. The main pipeline selected **Gaussian + Canny (100–200)** for the final lesion-boundary experiment.

---

# Questions to Answer

## Q1. Why is Gaussian filtering applied before Canny detection?

Gaussian filtering is applied before Canny detection mainly to **reduce noise**.

Canny detects changes in image intensity, so small noise variations, skin texture, hair, or sensor noise may also be interpreted as edges. Gaussian filtering smooths these small variations before Canny is applied.

Its benefits in this experiment are:

- reduces random noise,
- smooths small pixel-level variations,
- reduces false edges,
- makes the main lesion boundary easier to identify,
- gives Canny a cleaner input image.

Therefore, Gaussian filtering improves the reliability of lesion-boundary detection.

---

## Q2. How did the three Canny threshold settings affect the result?

The three threshold settings changed the sensitivity of Canny edge detection.

### `50–100`

This was the most sensitive setting. It detected weaker intensity changes, so it could capture small details but also produced more unwanted edges and noise.

### `100–200`

This provided the best balance between sensitivity and noise suppression in the notebook. The lesion boundary was clearer and the number of irrelevant edges was reduced.

### `150–250`

This was the strictest setting. It mainly detected strong edges. The output could be cleaner, but weak portions of the lesion boundary could be lost, making the contour incomplete.

In summary:

**Lower thresholds → more edges and more noise**  
**Higher thresholds → fewer edges but possible loss of important boundary information**

---

## Q3. Which threshold produced the best lesion boundary?

The best threshold in this experiment was:

## **Canny 100–200**

It was selected because it provided a better balance between detecting the lesion outline and suppressing unwanted edges.

Compared with the other two settings:

- `50–100` was more likely to over-detect weak/noisy edges.
- `150–250` could remove useful weak boundary sections.
- `100–200` produced the clearest practical boundary for the implemented pipeline.

---

## Q4. Why are edges useful for detecting skin lesions?

Edges represent locations where image intensity changes significantly.

A skin lesion often has different pigmentation or texture from the surrounding normal skin. Because of this difference, the transition between the lesion and normal skin can form an edge.

Edge detection is useful because it can help:

- locate the lesion border,
- isolate the lesion from surrounding skin,
- determine lesion shape,
- calculate lesion area,
- calculate lesion perimeter,
- support later shape analysis.

Once a closed lesion boundary is detected, measurements can be obtained from its contour.

---

## Q5. What problems did you observe in detecting the lesion boundary?

Several problems were observed or revealed by the experiment:

1. **Skin texture can create false edges.**  
   Pores and natural variations in skin intensity may be detected by Canny.

2. **Hair can interfere with the boundary.**  
   Hair produces strong edges that may connect with or cross the lesion contour.

3. **Uneven lighting can create unwanted gradients.**  
   Shadows or highlights may be detected as edges.

4. **Some lesion boundaries are gradual.**  
   If the lesion smoothly blends into the skin, there may not be a strong enough intensity transition for Canny.

5. **Threshold selection affects the result strongly.**  
   Low thresholds detect noise, while high thresholds can miss useful boundary sections.

6. **Contours may be fragmented.**  
   Broken Canny edges can prevent formation of a complete lesion boundary.

7. **Largest-contour selection is not always reliable.**  
   The largest detected contour may sometimes represent another structure instead of the lesion.

8. **The method failed on some tested images.**  
   Image 2 and Image 5 produced zero area and perimeter with the final configuration, showing that a fixed traditional pipeline does not work equally well for every lesion image.

---

## Q6. How could your method be improved?

The method could be improved in several ways.

### Better preprocessing

- Apply dedicated **hair-removal** preprocessing.
- Use **CLAHE** or illumination normalization for uneven lighting.
- Try bilateral filtering to smooth noise while preserving important edges.
- Use color spaces such as HSV or LAB instead of relying only on grayscale intensity.

### Better edge and threshold selection

- Automatically select Canny thresholds for each image.
- Use adaptive thresholding instead of one fixed threshold pair.
- Combine edges detected at multiple scales.

### Better contour processing

- Improve morphological closing and opening.
- Fill holes inside the lesion mask.
- Reject contours using area, location, shape, or circularity constraints instead of always selecting only the largest contour.

### More advanced segmentation

A stronger improvement would be to use a segmentation model such as:

- U-Net
- DeepLab
- other CNN-based medical-image segmentation models

These methods can learn lesion appearance from labeled data and are generally more flexible than fixed edge thresholds.

### Validation

The detected lesion masks should also be compared with expert-annotated ground-truth masks using metrics such as:

- IoU / Jaccard Index
- Dice coefficient
- precision
- recall

This would provide a more objective measure of segmentation quality.

---

# Overall Conclusion

This lab implemented a complete traditional computer-vision pipeline for skin-lesion boundary detection:

**HAM10000 Image → Grayscale → Gaussian Filtering → Canny Edge Detection → Morphological Closing → Contour Detection → Area and Perimeter**

Among the tested Canny threshold values, **100–200** was selected for the final pipeline because it provided the best practical balance between lesion-edge detection and noise suppression.

The experiment also demonstrated an important limitation: the fixed edge-based method did not successfully produce a usable contour for every image. This shows why preprocessing, adaptive parameters, stronger contour-selection logic, and modern segmentation approaches can improve robustness.

For the current lab implementation, **Gaussian filtering + Canny edge detection (100–200)** was used as the final configuration.


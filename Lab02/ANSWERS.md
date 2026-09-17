# Lab 02: Answers

Haider Mushtaq, FA23-BAI-044

All numbers come from `results/comparison_table.csv` and `results/delta_vs_baseline.csv`. Test set is 381 images, held out by lesion so no image of a test lesion appears in training.

---

**1. Which three pretrained models performed best in Lab Activity 1?**

Ranked by test accuracy on the ISIC 9-class set:

| Rank | Model | Accuracy | Macro-F1 | AUC |
|---|---|---|---|---|
| 1 | ResNet50 | 61.86% | 60.90% | 91.39% |
| 2 | DenseNet121 | 59.32% | 60.09% | 90.83% |
| 3 | ResNet101 | 58.47% | 57.22% | 92.24% |

ResNet50 led on accuracy and F1. DenseNet121 was close behind with the highest precision of the eight and a third of the parameters. ResNet101 had the best AUC but weaker argmax accuracy under the same training budget. These three are used for everything in this lab.

**2. How does filtering affect each of the three models?**

The three models don't react the same way.

ResNet50 (baseline Macro-F1 76.41) is the only one that benefits from smoothing. Gaussian pushed it to 77.90 (+1.49 pp) and Average to 76.71 (+0.30). Median hurt it (72.78, -3.63) and Sharpening cost about a point. Its best run, Gaussian, is also the best single result in the whole study on accuracy (76.90%) and AUC (95.63%).

DenseNet121 (baseline 75.89) lost ground under every filter. The mildest was Sharpening at -1.02 pp, then Median -2.33, Gaussian -2.81, Average -3.02. Nothing improved it. The unfiltered baseline was its best configuration by a clear margin.

ResNet101 (baseline 76.29) is the odd one out. Median, which hurt both other models, gave it its best result: 77.61 Macro-F1 (+1.32 pp) and 76.90% accuracy, tying ResNet50-Gaussian for top accuracy. Gaussian and Sharpening cost it under a point, Average cost 1.58.

Sobel destroyed all three. Roughly 20 points of accuracy and 22 to 24 points of Macro-F1 gone in every case.

**3. Which filter produces the greatest change compared with the unfiltered baseline?**

Sobel, by a mile. Mean Δ Macro-F1 across the three models is -22.51 pp, mean Δ accuracy -19.51 pp, mean Δ AUC -7.85 pp. Every other filter moved Macro-F1 by less than 1.6 pp on average. Sobel is an order of magnitude more disruptive than any smoothing or sharpening filter, and the direction is always down.

**4. Does the effect of a filter remain consistent across all three models?**

Only for two of the five.

Sharpening was consistently slightly negative (-0.97, -1.02, -0.87 pp Macro-F1). Sobel was consistently catastrophic. Both are the same sign for all three models.

Average, Gaussian and Median all flipped sign depending on the model. Gaussian helped ResNet50 and hurt the other two. Median hurt ResNet50 and DenseNet121 but helped ResNet101. Average helped ResNet50 by a hair and hurt the rest.

My read on why: for the smoothing filters the effect sizes are small, one to three points, and 381 test images means roughly 0.26% per image. A three-point swing is about 11 images. Some of that is real, some is seed noise. The one pattern that does look real is DenseNet121 being hurt by everything. Its dense connectivity reuses early-layer features throughout the network, so fine texture removed at the input never gets recovered downstream. The ResNets have skip connections but not that level of feature reuse, and they have a bit more room to adapt.

**5. Does filtering improve or decrease macro-F1 and balanced accuracy?**

On average it decreases both. Across all 15 filtered runs the mean change is -5.44 pp Macro-F1 and -4.85 pp balanced accuracy, but that's dragged down by Sobel. Excluding Sobel, the mean is -1.17 pp Macro-F1 and -0.76 pp balanced accuracy. So the smoothing and sharpening filters are mildly negative on average.

Three individual runs improved both metrics over their baseline: ResNet50-Gaussian (+1.49 F1, +1.45 bal acc), ResNet50-Average (+0.30, +1.68), ResNet101-Median (+1.32, +0.88). Twelve of fifteen runs got worse. The honest summary is that filtering is a coin flip with bad odds unless the filter and model happen to match, and there's no way to know which pairing works without running the experiment.

**6. Which lesion classes are most affected by filtering?**

From `class_sensitivity.csv`, mean |Δ F1| over the three models:

| Class | Average | Gaussian | Median | Sharpening | Sobel | mean abs |
|---|---|---|---|---|---|---|
| df | -8.32 | -4.63 | -5.49 | -5.05 | -38.84 | 12.47 |
| vasc | -0.84 | -2.06 | -4.26 | +0.56 | -35.20 | 8.58 |
| nv | -4.03 | -0.22 | -5.60 | -4.12 | -18.78 | 6.55 |
| bkl | +5.65 | +4.50 | +3.93 | +2.90 | -15.53 | 6.50 |
| mel | -2.12 | -3.51 | -3.05 | -0.14 | -20.34 | 5.83 |
| bcc | +0.08 | +0.60 | +2.41 | -0.41 | -15.27 | 3.75 |
| akiec | -0.44 | +0.06 | +1.27 | -0.41 | -13.60 | 3.16 |

Dermatofibroma (df) is the most sensitive. It's the smallest class (22 test images) and it's diagnosed largely on a central white scar-like patch with fine surrounding pigment. Smoothing softens that patch, and Sobel wipes it entirely, hence the -38.84 pp under Sobel.

Vascular lesions (vasc) come second and the reason is obvious: they're identified by colour. Red and purple lacunae against skin. Sobel throws colour away and the class F1 drops 35 points. The smoothing filters barely touch it though, because the colour survives blur.

Benign keratosis (bkl) is the interesting one. It's the only class that improved under all four non-Sobel filters, +2.9 to +5.7 pp. bkl images often carry hair, scale and surface roughness that acts as noise. Smoothing removes it and the model sees the lesion more clearly. Sharpening helping too is harder to explain and might be noise.

Melanoma (mel) lost 2 to 3.5 points under smoothing. Melanoma diagnosis depends on border irregularity and fine pigment network, exactly what a blur removes. Clinically this is the class where you'd least want to lose recall, so it's a good argument against smoothing in a real pipeline.

---

**7. Why might smoothing remove useful lesion texture or morphological information?**

Smoothing is low-pass filtering. Every pixel becomes a weighted average of its neighbours, so high-frequency content, which is exactly where fine texture lives, is attenuated or removed. Dermoscopic diagnosis leans on high-frequency cues: the pigment network in a nevus, dots and globules, streaks at the border of a melanoma, the milia-like cysts of a seborrheic keratosis, arborising vessels in a basal cell carcinoma. These are a few pixels wide at 224 × 224. A 5 × 5 average filter has a support wider than most of them, so it blurs them into the surrounding skin tone. Border sharpness is also a diagnostic feature (an abrupt border is a melanoma warning sign), and smoothing turns an abrupt border into a gradual one.

The three smoothing filters degrade differently. The average filter treats all neighbours equally and also smears edges. The Gaussian filter weights the centre more, so it preserves slightly more structure at the same kernel size. The median filter is non-linear: it removes impulsive noise and hair artefacts while keeping step edges, so it is usually the least destructive of the three, but it still flattens texture.

The results back this up on the melanoma row: every smoothing filter cost mel between 2 and 3.5 points of F1.

**8. Why might sharpening or edge detection help or hurt classification?**

Sharpening adds a scaled high-pass version of the image back to itself. It amplifies borders, vessels and texture, which could help a model that under-weights those cues. It also amplifies noise, JPEG artefacts, hair and ruler markings, and it changes the pixel-intensity distribution the pretrained network was calibrated on. With a mild kernel the net effect is often close to zero; with a strong kernel it usually hurts. That matches what happened here: sharpening was consistently negative but only by about one point on every model.

Sobel is more drastic. It keeps only gradient magnitude and discards colour and absolute intensity. Colour is one of the strongest signals in dermoscopy: blue-white veil, red vascular lesions, brown versus black pigment. Throwing it away removes information the model relies on. The edge map also looks nothing like a natural photograph, so the ImageNet-pretrained early layers, which are tuned to colour and texture statistics of natural images, are now operating out of distribution. Some accuracy can be recovered during fine-tuning, but the expectation is a clear drop. The measured drop was 20 points of accuracy on every model, and the class that lost the most was vasc, the colour-defined class. Edge detection can help in narrow cases where shape or border irregularity alone is the discriminating feature, but for a 7-class dermoscopy problem it removes more than it adds.

**9. What is the difference between convolution and correlation?**

Both slide a kernel over an image and compute a weighted sum at each position. The difference is the orientation of the kernel.

Cross-correlation: `(I ⋆ K)(x, y) = Σ_i Σ_j I(x+i, y+j) · K(i, j)`

Convolution: `(I ∗ K)(x, y) = Σ_i Σ_j I(x−i, y−j) · K(i, j)`

Convolution flips the kernel both horizontally and vertically before applying it. For a symmetric kernel (average, Gaussian, the sharpening kernel used here) the two operations give identical results. For an asymmetric kernel like Sobel they differ by a sign flip in the output.

Convolution is the mathematically "proper" operation: it is commutative and associative, which is what makes the convolution theorem and the frequency-domain view of filtering work. `cv2.filter2D` actually computes correlation, which is why the OpenCV documentation says to flip the kernel yourself if you need true convolution. Deep learning frameworks also compute correlation in their "convolution" layers. It does not matter there because the kernels are learned, so the network simply learns the flipped version.

**10. Based on your results, explain the relationship between classical image processing and deep-learning-based feature extraction.**

A classical filter is a fixed, hand-designed convolution kernel. The first convolutional layer of a CNN is a bank of learned kernels of the same kind. Visualising the first layer of any ImageNet-trained network shows Gaussian-like blobs, oriented edge detectors that look like Sobel or Gabor filters, and colour-opponent filters. The network rediscovers classical image processing on its own, then stacks further layers to combine those primitives into texture, shape and part detectors.

That framing explains the pattern in my results. Applying a fixed filter before the network is equivalent to forcing an extra, non-learnable layer in front of it. If that layer removes information the network cannot get it back. Sobel removed colour and every model lost 20 points. If the layer only re-weights information the network already extracts, the network compensates during fine-tuning and the change is small, which is what happened with sharpening and the three smoothing filters: all within about 3 points of baseline, sometimes up, mostly down.

The unfiltered baseline was the best or within 1.5 points of the best for all three models. The learned pipeline is more flexible than any single hand-picked kernel, because the network can learn to blur where blurring helps and keep detail where it doesn't, per feature map, per layer. A global 5x5 Gaussian can't make that distinction.

Where classical preprocessing still earns its place is in removing things the network should not learn: hair, vignetting, colour casts from different dermatoscopes, scale differences. Those are normalisation problems, not feature-extraction problems. The bkl result hints at this: it's the class with the most surface artefacts and the only one that consistently improved under smoothing. The lesson is that preprocessing should target artefacts, not features, and that any filter which changes the appearance of the lesion itself should be treated as a risk rather than a default.

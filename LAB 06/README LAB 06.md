# Landmark Recognition Using the SIFT Algorithm

**Course:** Computer Vision – Lab 06
**Student:** Fahad Bin Shafi (FA23-BAI-033)
**Tools:** Python, OpenCV (SIFT), NumPy, Pandas, Matplotlib, Google Colab

## 1. Overview

This project recognizes famous landmarks from a query photograph using the **Scale-Invariant Feature Transform (SIFT)**. The system keeps a database of reference images for five landmarks. When a new image is given, it extracts SIFT features, matches them against the database, verifies the match geometrically, and either reports the landmark (and draws its location) or rejects the image as unknown.

**Landmarks in the database:** Eiffel Tower, Taj Mahal, Great Wall of China, Statue of Liberty, Colosseum.
**Unknown landmark used for rejection testing:** Great Pyramid of Giza.

## 2. Pipeline

| Step | Description |
|---|---|
| 1. Database | 6 reference images per landmark, stored in one folder per landmark (30 images total). 2 extra held-out images per landmark are kept aside as queries. |
| 2. Feature extraction | Grayscale conversion → SIFT keypoints → 128-D descriptors (max 2000 keypoints per image). Descriptors are saved with their labels in `sift_database.pkl`. |
| 3. Query processing | Same extraction applied to the query image; keypoints are displayed. |
| 4. Matching | Brute-Force matcher (L2 / Euclidean) → kNN with k = 2 → Lowe's ratio test (0.75) → reliable matches counted per landmark → candidates ranked. |
| 5. Localization | `cv2.findHomography` with RANSAC → reference-image corners projected onto the query → polygon drawn. If the transformation is unreliable, a message is shown and no polygon is drawn. |
| 6. Evaluation | Each landmark is tested under original, rotated, resized, low-brightness and different-viewpoint conditions, plus an unknown landmark. |

### Decision rule (not based on match count alone)

A landmark is **recognized** only if all of the following hold, otherwise the query is **rejected**:

1. Reliable matches ≥ `MIN_GOOD_MATCHES` (20)
2. A valid (convex, sensibly sized) homography is found and RANSAC inliers ≥ `MIN_INLIERS` (12)
3. Inlier ratio ≥ `MIN_INLIER_RATIO` (0.20)
4. The best landmark has at least `MARGIN` (1.3×) the inliers of the runner-up, which rejects ambiguous matches

Candidates are ranked by RANSAC inliers first, then by reliable-match count.

## 3. Project structure

```
landmark_project/
├── landmark_db/              # reference images (one folder per landmark)
│   ├── Eiffel Tower/
│   ├── Taj Mahal/
│   ├── Great Wall of China/
│   ├── Statue of Liberty/
│   └── Colosseum/
├── test_images/              # held-out query images (not in the database)
│   ├── <landmark>/
│   └── _unknown/
├── sift_database.pkl         # stored descriptors + labels
└── experimental_results.csv  # Task 6 results table
CV_LAB_06_Landmark_Recognition_SIFT.ipynb
README.md
```

## 4. How to run

1. Open `CV_LAB_06_Landmark_Recognition_SIFT.ipynb` in Google Colab.
2. Choose **Runtime → Run all**. The dataset is downloaded automatically from Wikimedia Commons (internet access required).
3. To use your own images instead, set `USE_MY_OWN_ZIP = True` in Task 1 and upload a zip containing `landmark_db/` and `test_images/` in the structure above.
4. The last cells let you upload any photo for testing and download a zip of all outputs.

No extra installation is needed on Colab (`opencv-python`, `numpy`, `pandas`, `matplotlib` are preinstalled).

## 5. Tunable parameters

| Parameter | Value | Meaning |
|---|---|---|
| `RATIO` | 0.75 | Lowe's ratio test threshold |
| `MIN_GOOD_MATCHES` | 20 | Minimum reliable matches |
| `MIN_HOM_MATCHES` | 8 | Minimum matches to attempt a homography |
| `MIN_INLIERS` | 12 | Minimum RANSAC inliers |
| `MIN_INLIER_RATIO` | 0.20 | Inliers / reliable matches |
| `MARGIN` | 1.3 | Required lead over the runner-up landmark |
| `RANSAC_THRESH` | 5.0 px | Reprojection tolerance |

## 6. Experimental results

Test conditions: original, rotated 45°, resized 50%, brightness reduced to 40%, different viewpoint (second held-out photo), and a landmark not in the database. Full per-image output is in `experimental_results.csv`.

| Ground truth | Test condition | Predicted | Keypoints | Reliable matches | RANSAC inliers | Decision | Time (s) | Correct |
|---|---|---|---|---|---|---|---|---|
| Colosseum | Original | Unknown | 2001 | 18 | 0 | Rejected | 2.08 | No |
| Colosseum | Rotated 45° | Unknown | 2000 | 35 | 0 | Rejected | 2.34 | No |
| Colosseum | Resized 50% | Unknown | 977 | 7 | 0 | Rejected | 0.94 | No |
| Colosseum | Low brightness | Unknown | 1166 | 14 | 0 | Rejected | 1.18 | No |
| Colosseum | Different viewpoint | Unknown | 2000 | 55 | 0 | Rejected | 3.00 | No |
| Eiffel Tower | Original | Unknown | 2000 | 18 | 0 | Rejected | 2.28 | No |
| Eiffel Tower | Rotated 45° | Unknown | 2000 | 13 | 4 | Rejected | 2.28 | No |
| Eiffel Tower | Resized 50% | Unknown | 2000 | 18 | 0 | Rejected | 1.96 | No |
| Eiffel Tower | Low brightness | Unknown | 2000 | 11 | 4 | Rejected | 2.06 | No |
| Eiffel Tower | Different viewpoint | Unknown | 2000 | 11 | 4 | Rejected | 2.08 | No |
| Great Wall of China | Original | Unknown | 2000 | 9 | 0 | Rejected | 2.49 | No |
| Great Wall of China | Rotated 45° | Unknown | 2000 | 8 | 5 | Rejected | 2.47 | No |
| Great Wall of China | Resized 50% | Unknown | 975 | 12 | 0 | Rejected | 0.95 | No |
| Great Wall of China | Low brightness | Unknown | 1303 | 6 | 0 | Rejected | 1.29 | No |
| Great Wall of China | Different viewpoint | Unknown | 2000 | 14 | 0 | Rejected | 1.97 | No |
| Statue of Liberty | Original | Unknown | 540 | 21 | 0 | Rejected | 0.70 | No |
| Statue of Liberty | Rotated 45° | Unknown | 709 | 25 | 6 | Rejected | 1.24 | No |
| Statue of Liberty | Resized 50% | Unknown | 298 | 11 | 6 | Rejected | 0.34 | No |
| Statue of Liberty | Low brightness | Unknown | 240 | 9 | 0 | Rejected | 0.32 | No |
| Statue of Liberty | Different viewpoint | Unknown | 1905 | 13 | 5 | Rejected | 2.57 | No |
| Taj Mahal | Original | Taj Mahal | 1660 | 1007 | 973 | Recognized | 1.99 | Yes |
| Taj Mahal | Rotated 45° | Taj Mahal | 2000 | 957 | 891 | Recognized | 2.80 | Yes |
| Taj Mahal | Resized 50% | Taj Mahal | 590 | 380 | 365 | Recognized | 0.59 | Yes |
| Taj Mahal | Low brightness | Taj Mahal | 577 | 418 | 406 | Recognized | 0.62 | Yes |
| Taj Mahal | Different viewpoint | Taj Mahal | 1753 | 828 | 780 | Recognized | 2.12 | Yes |
| Unknown (Pyramid) | Not in database | Unknown | 2000 | 11 | 4 | Rejected | 2.53 | Yes |
| Unknown (Pyramid) | Not in database | Unknown | 696 | 21 | 0 | Rejected | 0.81 | Yes |

**Summary**

| Test condition | Correct / Tests |
|---|---|
| Original | 1 / 5 |
| Rotated 45° | 1 / 5 |
| Resized 50% | 1 / 5 |
| Low brightness | 1 / 5 |
| Different viewpoint | 1 / 5 |
| Not in database | 2 / 2 |

Overall accuracy: **25.9 % (7 / 27)**. Average processing time: **1.70 s per query** (CPU, 30 reference images).

## 7. Discussion

**What worked.**
- Both unknown-landmark (Pyramid) queries were correctly rejected. The ratio test, RANSAC verification and thresholds stopped weak, ambiguous matches from being accepted.
- The Taj Mahal was recognized in all five conditions with hundreds of reliable matches and very high inlier ratios (about 90–97 %). This shows that once good correspondences exist, SIFT handles **rotation, scale change and reduced brightness** well, and the localization polygon is reliable.

**What did not work, and why.**
- For the other four landmarks, every query was rejected, including the unmodified "Original" image. The held-out query photos are different photographs from the 6 database images, not copies of them. Only the Taj Mahal test photo had a near-duplicate view in its database folder, which explains its 1000+ matches. For the others, only 6–55 matches survived the ratio test and RANSAC found almost no consistent geometry, so the strict thresholds correctly (by design) rejected them.
- So the low accuracy mainly reflects **viewpoint and appearance gaps between different photos of the same landmark and an under-sized database**, not a failure of SIFT's rotation, scale or illumination invariance. This is why the "Original", "Rotated", "Resized" and "Low brightness" accuracies are identical per landmark: they all derive from the same held-out photo.

**Limitations.**
1. SIFT is only partially invariant to viewpoint change; a planar homography is a weak model for 3-D landmarks such as the Eiffel Tower and Statue of Liberty.
2. Repetitive or low-texture structures (Colosseum arches, Great Wall bricks, sky, smooth walls) give ambiguous descriptors.
3. Only 6 reference images per landmark cover few viewpoints, seasons and lighting conditions.
4. Thresholds (`MIN_GOOD_MATCHES`, `MIN_INLIERS`, `MARGIN`) trade false rejects against false accepts and need tuning on a larger validation set.
5. Brute-force matching is O(N) in the database size (about 1.7 s per query on 30 images) and will not scale to thousands of landmarks.
6. Occlusion, crowds, weather and day/night changes reduce match counts.

**Possible improvements.**
- Add more reference images per landmark, covering varied viewpoints and lighting, and include a reference image close to each test view.
- Relax the thresholds moderately (for example `RATIO` 0.8, `MIN_GOOD_MATCHES` ~10, `MIN_INLIERS` ~8) and re-validate against the unknown-landmark rejection rate.
- Use FLANN or a vocabulary tree for faster search, RootSIFT for better descriptors, multi-image voting, and deep-learning retrieval (e.g. NetVLAD) followed by SIFT-based geometric verification.

## 8. Conclusion

SIFT with Lowe's ratio test and RANSAC verification provides a transparent, training-free way to recognize and localize landmarks, and it rejects unknown landmarks reliably. It is robust to rotation, scaling and brightness changes when sufficient matching views exist (as shown by the Taj Mahal results), but its accuracy depends heavily on the coverage of the reference database and on viewpoint similarity, which limits it as a standalone solution for a full tourism information system.

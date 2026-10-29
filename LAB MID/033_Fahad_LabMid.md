# Lab Mid Exam: HOG vs SIFT on Blood-Cell Images

**Name:** Fahad Bin Shafi  
**Registration number:** 033

This report presents the saved results and answers from the executed Google Colab notebook, `033__Fahad_LabMid.ipynb`. The code is in that notebook. No model was rerun while preparing this Markdown report.

## Personal and fixed settings

| Setting | Value used |
| --- | --- |
| Random seed | 33, from the last two registration digits |
| HOG cell size | 16 x 16, because the last digit is odd |
| SIFT grid | 4 x 4, because the last digit is between 0 and 4 |
| Dataset | BloodMNIST, size 64 |
| Classes | Eosinophil (1), monocyte (5), neutrophil (6), platelet (7) |
| Image preparation | Grayscale, resized to 128 x 128, scaled to [0, 1] |
| HOG orientations | 9 |
| HOG cells per block | 2 x 2 |
| Classifier | StandardScaler followed by LinearSVC |
| Values of C tested | 0.1, 1, 10 |
| Rotation | 20 degrees counterclockwise |
| Gaussian noise | Mean 0, sigma 0.05 |

## Q1. Data setup

Images were selected without replacement from the official training, validation and test splits using the registration-based seed. Training and validation data were loaded first. Test data were loaded only in Q4; their recorded counts are also included here for the complete split summary.

|  | Split | Images | eosinophil | monocyte | neutrophil | platelet |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | Train | 400 | 100 | 100 | 100 | 100 |
| 1 | Validation | 120 | 30 | 30 | 30 | 30 |
| 2 | Test | 120 | 30 | 30 | 30 | 30 |

| Split | Original RGB shape | Prepared grayscale shape | Label shape |
| --- | --- | --- | --- |
| Training | (400, 64, 64, 3) | (400, 128, 128) | (400,) |
| Validation | (120, 64, 64, 3) | (120, 128, 128) | (120,) |
| Test | (120, 64, 64, 3) | (120, 128, 128) | (120,) |

### Four example images per class

![Q1: four training examples for each of the four classes](033_Fahad_LabMid_files/figure_01.png)

## Q2. HOG + SVM

### Feature length calculated by hand

Each 128 x 128 image has 128 / 16 = 8 cells along each axis. A 2 x 2 block moves one cell at a time, giving 8 - 2 + 1 = 7 blocks along each axis. Each block contains 2 x 2 x 9 = 36 values.

**HOG feature length = 7 x 7 x 36 = 1,764.**

The saved feature shapes are **(400, 1764)** for training and **(120, 1764)** for validation, matching the calculation. HOG uses L2-Hys block normalisation.

### HOG picture for one image from each class

![Q2: grayscale images and their HOG visualisations](033_Fahad_LabMid_files/figure_02.png)

### Choosing C and validation results

The scaler and SVM were fitted on training features only. C was chosen by the highest validation accuracy, then higher macro-F1, then smaller C if still tied. The selected model was kept without refitting on validation data.

|  | C | Validation accuracy | Validation macro-F1 |
| --- | --- | --- | --- |
| 0 | 0.1 | 0.8 | 0.79734 |
| 1 | 1.0 | 0.8 | 0.79734 |
| 2 | 10.0 | 0.8 | 0.79734 |

**Selected C: 0.1. Validation accuracy: 0.8000 (80.00%). Validation macro-F1: 0.7973.**

### Validation confusion matrix

Rows are true classes and columns are predicted classes.

|  | Pred eosinophil | Pred monocyte | Pred neutrophil | Pred platelet |
| --- | --- | --- | --- | --- |
| True eosinophil | 20 | 5 | 3 | 2 |
| True monocyte | 3 | 25 | 2 | 0 |
| True neutrophil | 6 | 3 | 21 | 0 |
| True platelet | 0 | 0 | 0 | 30 |

![Q2: HOG validation confusion matrix](033_Fahad_LabMid_files/figure_03.png)

## Q3. SIFT + SVM

### Normal SIFT detection

Normal SIFT detection was run on all 400 selected training images using `detectAndCompute`.

|  | Images | Average keypoints | Minimum keypoints | Maximum keypoints | Images with zero keypoints |
| --- | --- | --- | --- | --- | --- |
| 0 | 400 | 71.3925 | 8 | 188 | 0 |

![Q3: normal SIFT keypoints on one training image](033_Fahad_LabMid_files/figure_04.png)

An SVM needs the same number of features for every image, but joining normal SIFT descriptors gives different lengths when the number of keypoints varies.

### Fixed-grid SIFT and feature length

A regular 4 x 4 grid was placed over each image, with coordinates evenly spaced from 16 to 112 pixels on each axis. Each keypoint had size 32 and angle 0 degrees. Descriptors were calculated at the supplied points and joined from left to right, then top to bottom.

There are 4 x 4 = 16 grid points, and each descriptor has 128 values.

**SIFT feature length = 16 x 128 = 2,048.**

The saved feature shapes are **(400, 2048)** for training and **(120, 2048)** for validation. This fixed-angle grid method does not provide full rotation invariance.

![Q3: the fixed 4 by 4 SIFT grid](033_Fahad_LabMid_files/figure_05.png)

### Choosing C and validation results

The same training-only scaling and validation selection rule used for HOG was applied to SIFT features.

|  | C | Validation accuracy | Validation macro-F1 |
| --- | --- | --- | --- |
| 0 | 0.1 | 0.75 | 0.747563 |
| 1 | 1.0 | 0.75 | 0.747563 |
| 2 | 10.0 | 0.75 | 0.747563 |

**Selected C: 0.1. Validation accuracy: 0.7500 (75.00%). Validation macro-F1: 0.7476.**

### Validation confusion matrix

Rows are true classes and columns are predicted classes.

|  | Pred eosinophil | Pred monocyte | Pred neutrophil | Pred platelet |
| --- | --- | --- | --- | --- |
| True eosinophil | 16 | 9 | 3 | 2 |
| True monocyte | 4 | 24 | 2 | 0 |
| True neutrophil | 10 | 0 | 20 | 0 |
| True platelet | 0 | 0 | 0 | 30 |

![Q3: SIFT validation confusion matrix](033_Fahad_LabMid_files/figure_06.png)

## Q4. Robustness

### Prediction recorded in the notebook before the code cells

I predict that HOG will lose more accuracy under rotation because its orientation bins and cell positions are fixed, while SIFT pools gradients over local regions.
I predict that HOG will lose more accuracy under Gaussian noise because small noisy gradients may disturb its cell histograms more than SIFT's local aggregation.
These are hypotheses, not results: the fixed-grid SIFT implementation uses a fixed angle and spatial order, so it is not fully rotation invariant.

### Test conditions

Both selected pipelines, including their scalers, were frozen. The same 120 test images were used for both models under three separate conditions: clean, rotated by 20 degrees, and noisy. Each change was applied before feature extraction.

Rotation used bilinear interpolation and reflection padding on the original 128 x 128 canvas. Noise had mean 0 and sigma 0.05; images were then clipped to [0, 1]. The saved standard deviation of the generated noise before clipping was **0.05001**. SIFT converted images to uint8 after corruption.

![Q4: clean, rotated and noisy test examples for each class](033_Fahad_LabMid_files/figure_07.png)

### Required comparison table

Accuracies are shown as proportions from 0 to 1.

| Model | Clean accuracy | Rotated accuracy | Noisy accuracy |
| --- | --- | --- | --- |
| HOG + SVM | 0.6500 | 0.7417 | 0.2500 |
| SIFT + SVM | 0.7917 | 0.7417 | 0.7833 |

### Accuracy changes

Loss is clean accuracy minus corrupted-image accuracy, expressed in percentage points. A negative loss means accuracy improved.

| Model | Rotation loss (percentage points) | Noise loss (percentage points) |
| --- | --- | --- |
| HOG + SVM | -9.17 | 40.00 |
| SIFT + SVM | 5.00 | 0.83 |

The saved output confirms that both models and their scalers remained unchanged during the test evaluations.

### Was the prediction correct? Three-sentence answer

The rotation prediction was incorrect because HOG accuracy increased from 65.00% to 74.17%, while SIFT accuracy decreased from 79.17% to 74.17%.
The noise prediction was correct because HOG lost 40.00 percentage points, compared with 0.83 percentage points for SIFT.
On these 120 test images, SIFT had higher clean and noisy accuracy, while both models had the same rotated accuracy.

## Q5. Short answers and AI log

### (a) Why does HOG use block normalisation?

HOG normalises neighbouring cells together to reduce the effect of lighting and contrast changes on gradient strength. This helps the descriptor represent the relative pattern of edges rather than mainly their brightness.

The notebook also multiplied one training image by 0.6 and compared its normalised HOG descriptor with the original. The relative change printed as **0.000000** to six decimal places; this supports the explanation for that example but does not prove invariance to every lighting change.

### (b) Why is it wrong to choose C using the test set?

Choosing C using test results would allow the test data to influence model selection, making test accuracy an optimistic estimate of independent performance. Validation data should be used to choose C, and the test set should be kept for the final comparison of the frozen models.

The saved checks show that each scaler was fitted on **400 training rows**, and both selected models used **C = 0.1**.

### (c) Which two classes were most confused, and why might grayscale make this harder?

Both models most often confused **eosinophils and monocytes** on the clean test set. HOG made **21** errors between this pair and SIFT made **13**, counting both directions, for **34 prediction errors** across the two models.

Grayscale removes colour differences that may help distinguish cells with similar shapes but different staining colours. This is a possible explanation rather than a tested cause, because no RGB classifier was compared.

#### HOG class-pair error counts

|  | Class pair | Errors in both directions |
| --- | --- | --- |
| 0 | eosinophil / monocyte | 21 |
| 1 | eosinophil / neutrophil | 16 |
| 3 | monocyte / neutrophil | 4 |
| 2 | eosinophil / platelet | 1 |
| 4 | monocyte / platelet | 0 |
| 5 | neutrophil / platelet | 0 |

#### SIFT class-pair error counts

|  | Class pair | Errors in both directions |
| --- | --- | --- |
| 0 | eosinophil / monocyte | 13 |
| 1 | eosinophil / neutrophil | 8 |
| 3 | monocyte / neutrophil | 2 |
| 4 | monocyte / platelet | 1 |
| 5 | neutrophil / platelet | 1 |
| 2 | eosinophil / platelet | 0 |

#### Combined class-pair error counts

|  | Class pair | Errors in both directions |
| --- | --- | --- |
| 0 | eosinophil / monocyte | 34 |
| 1 | eosinophil / neutrophil | 24 |
| 3 | monocyte / neutrophil | 6 |
| 2 | eosinophil / platelet | 1 |
| 4 | monocyte / platelet | 1 |
| 5 | neutrophil / platelet | 1 |

The combined table counts model predictions; an image misclassified by both models can contribute twice.

![Q5: clean-test confusion matrices for both models](033_Fahad_LabMid_files/figure_08.png)

![Q5: RGB and grayscale training examples](033_Fahad_LabMid_files/figure_09.png)

### (d) AI log

| Prompt | What changed | How correctness was checked |
| --- | --- | --- |
| Analyse the attached HOG vs SIFT brief and provide all code for Google Colab. | Registration number 033 and name Fahad Bin Shafi were entered, and the notebook was executed in Colab. The code cells match the supplied notebook. | The run completed without saved error outputs. Automated checks passed for class counts, feature lengths, training-only scaling and unchanged frozen models. |
| Check the executed notebook and prepare the Markdown file requested by the brief. | Saved results, answers and figures were organised into Q1-Q5; code remains in the executed notebook. | During export review, validation accuracies were checked against confusion-matrix totals, clean-test accuracies against error totals, and all nine figure files against their saved notebook images. |

This log records actions supported by the uploaded notebook and checks carried out during export. It does not claim additional personal checks that were not reported. The AI-log cell in the executed notebook still contains placeholders and should be updated to match the final log before submission.

## Recorded software versions

|  | Package | Version |
| --- | --- | --- |
| 0 | Python | 3.13.18 |
| 1 | NumPy | 2.1.3 |
| 2 | OpenCV | 5.0.0 |
| 3 | scikit-image | 0.25.2 |
| 4 | scikit-learn | 1.6.1 |
| 5 | MedMNIST | 3.0.2 |

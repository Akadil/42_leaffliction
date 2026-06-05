# 42_leaffliction

Computer vision project for analyzing, augmenting, and transforming leaf images
before building a disease classification model.

## Setup

Activate the virtual environment before running the scripts:

```bash
source .venv/bin/activate
```

Install the dependencies if the environment is not ready yet:

```bash
pip install -r requirements.txt
```

Once the virtual environment is active, the scripts can be executed directly:

```bash
./Distribution.py ./Apple
```

Without activating the virtual environment, use the project Python explicitly:

```bash
./.venv/bin/python Distribution.py ./Apple
```

## Part 1 - Distribution

Script:

```bash
./Distribution.py ./Apple
```

Goal:

- browse a directory containing class subdirectories;
- count the images in each class;
- display a pie chart and a bar chart;
- use the directory names automatically to label the charts.

Examples:

```bash
./Distribution.py ./Apple
./Distribution.py ./Grape
./Distribution.py ./images
```

The subject says: "Your program must work with all the directory in the data
set." This means the script must not be hard-coded for `Apple` only. It should
work with every dataset directory that follows the same structure: one root
directory containing class subdirectories.

## Part 2 - Data Augmentation

Script:

```bash
./Augmentation.py "./Apple/Apple_healthy/image (1).JPG"
```

Goal in single-image mode:

- display 6 augmentations;
- save the 6 augmented images in the same directory as the source image
- name each output file with the original filename followed by the augmentation
  type.

Augmentations used:

- `Flip`
- `Rotate`
- `Scaling`
- `Illumination`
- `Contrast`
- `Projective`

Example output files:

```text
image (1)_Flip.JPG
image (1)_Rotate.JPG
image (1)_Scaling.JPG
image (1)_Illumination.JPG
image (1)_Contrast.JPG
image (1)_Projective.JPG
```

Paths containing spaces or parentheses must be quoted:

```bash
./Augmentation.py "./Apple/Apple_healthy/image (1).JPG"
```

Goal in directory mode:

- copy the dataset into `augmented_directory`;
- balance the classes by adding only the missing number of images;
- keep the original images in the augmented dataset.

Commands:

```bash
./Augmentation.py ./Apple
./Augmentation.py ./Grape
```

Outputs:

```text
augmented_directory/Apple
augmented_directory/Grape
```

With the current dataset, the expected balanced counts are:

```text
Apple: 1640 images per class
Grape: 1382 images per class
```

## Part 3 - Image Transformation

Script:

```bash
./Transformation.py "./Apple/Apple_healthy/image (1).JPG"
```

Goal in single-image mode:

- display the original image;
- display at least 6 feature-extraction transformations.

Main transformations, aligned with the subject example:

- `GaussianBlur`: reduces noise;
- `Mask`: isolates the leaf;
- `RoiObjects`: displays the contour and region of interest;
- `AnalyzeObject`: displays shape measurements and annotations;
- `Pseudolandmarks`: displays morphological landmark points;
- `ColorHistogram`: displays the leaf color distribution.

Directory mode:

```bash
./Transformation.py -src ./Apple/Apple_healthy -dst ./dst_directory -mask
```

This mode processes every image in the source directory and saves the
transformations in the destination directory.

Example output files:

```text
image (1)_GaussianBlur.JPG
image (1)_Mask.JPG
image (1)_RoiObjects.JPG
image (1)_AnalyzeObject.JPG
image (1)_Pseudolandmarks.JPG
image (1)_ColorHistogram.JPG
```

## Quick Check

Check that all scripts compile:

```bash
python -m py_compile Distribution.py Augmentation.py Transformation.py
```

Check the Python norm with `flake8`:

```bash
flake8 Distribution.py Augmentation.py Transformation.py
```

Check the distributions:

```bash
./Distribution.py ./Apple
./Distribution.py ./Grape
```

Regenerate the augmented datasets:

```bash
./Augmentation.py ./Apple
./Augmentation.py ./Grape
```

Test transformations on one image:

```bash
./Transformation.py "./Apple/Apple_healthy/image (1).JPG"
```

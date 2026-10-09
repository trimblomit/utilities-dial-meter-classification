READING WATER METER DIALS: EfficientNet with Coordinate Attention vs ResNet-18

--- DESCRIPTION ---
This project classifies cropped water meter dial images into 30 dial-position
classes using the WMeter5K dataset (Zhang et al. 2025). It compares
EfficientNet-B0 with a Coordinate Attention module against a ResNet-18
benchmark, with a comparison against EfficientNet-B0 without Coordinate
Attention.

--- PROJECT STRUCTURE ---
a2913987_uoml_assessment2/
    a2913987_uoml_assessment2.ipynb     main notebook (outputs saved)
    readme.txt
    resnet18_best.pth                   trained ResNet-18 weights
    effnetb0_ca_best.pth                trained EfficientNet-B0 + CA weights
    effnetb0_no_ca_best.pth             trained EfficientNet-B0 no-CA weights
    resnet18_history.csv                per-epoch training log
    effnetb0_ca_history.csv             per-epoch training log
    effnetb0_no_ca_history.csv          per-epoch training log
    WMeter5K/
        WMeter5K/
            images/                     source meter images
            label.txt                   dataset pointer labels

The notebook creates WMeter5K/WMeter5K/pointers_A2/ automatically and writes
the cropped train, validation, and test images into it.

Over 27k image files, so be careful before running the notebook. May take a
minute to delete it all at the end.

--- REQUIREMENTS ---
Python 3.x

Libraries:
- torch, torchvision     for the models and training
- opencv-python (cv2)    for cropping, rotating, and resizing
- mlxtend                for McNemar's test
- numpy, pandas, matplotlib, tqdm, Pillow

Install:
pip install torch torchvision opencv-python mlxtend numpy pandas matplotlib tqdm Pillow

--- HOW TO RUN ---
1. Extract all of WMeter5K to this folder.
2. Run the notebook.

--- EXAMPLE INPUT ---
Each line of label.txt holds an image name, its pointer annotations, and the
full meter reading, separated by tabs.

Each pointer entry is a set of coordinates and a dial class joined by a slash,
with multiple pointers separated by semicolons.

--- REPRODUCIBILITY NOTES ---
torch.manual_seed(42) is used, but many areas of the code still carry
randomness.

Small run-to-run variations will remain during training even with seeds built
into more areas of the code, simply due to the nature of convolutions.

The three .pth checkpoints and _history.csv logs from this run are included so
results can be checked if needed: the .pth files hold the trained weights and
the CSV files hold the per-epoch training and validation history.

--- AUTHOR ---
Thomas Brown (a2913987 / thomas.brown01)

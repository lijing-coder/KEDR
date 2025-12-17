# Knowledge-enhanced Explicitly Disentangled Representation with Missing Modality for Medical Image Diagnosis
## Overview
This repository contains the code for "Knowledge-enhanced Explicitly Disentangled Representation with Missing Modality for Medical Image Diagnosis"!
![Formula](/img/overall.png)
## Datasets
**Skin Lesion:** Seven-Point Checklist [https://derm.cs.sfu.ca/]

**AMD Diseases:** MMC-AMD [https://github.com/li-xirong/mmc-amd]

**Glaucoma Diseases:** Harvard30k Glaucoma [https://yutianyt.com/projects/fairvision30k/]
## Dependencies

The following Python packages are required:

```bash
torch==1.13.0
torchvision==0.14.0
tensorflow==2.2.0
protobuf==3.20.3
torchcontrib==0.0.2
numpy==1.19.5
pandas==1.2.0
pillow==10.4.0
opencv-python==4.5.5.64
opencv-python-headless==4.5.3.56
matplotlib==3.6.3
scikit-image==0.18.1
albumentations==0.5.2
scikit-learn==0.24.1
seaborn==0.11.2
transformers==4.44.2
```
## Usage
Run the main program using Python:

```bash
python main.py
```

## Acknowledgement
This repository is built upon [FusionM4Net](https://github.com/pixixiaonaogou/MLSDR), [MMDynamic](https://github.com/TencentAILabHealthcare/mmdynamics), [MOTCat](https://github.com/Innse/MOTCat). Thanks again for their great works!

## Contact
For any questions, feel free to contact: Jing.Li2@liverpool.ac.uk

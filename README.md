# QRS Detector

This repository contains a QRS detector for ECG signals, implemented as
**Analysis of electrocardiographic (ECG) signals** assignment for the *Biomedical
Signal and Image Processing* course at the master's study at the Faculty of
Computer and Information Science Ljubljana in 2023/24. It is based on the
*Robust Detection of Heart Beats using Dynamic Thresholds and Moving Windows*
paper by Vollmer.

## Environment

The project was tested on Python 3.11.5 using the packages listed in
`requirements.txt`. To create a virtual environment and install the required
packages, run:

```bash
python --version
> Python 3.11.5
python -m venv venv
. venv/bin/activate
pip install -r requirements.txt
```

## Usage

Script `detector.py` takes a single argument, the path to the ECG record
without the file extension. For help and examples, run the script without
arguments.

Script `eval_all.sh` runs the detector on all database records and saves the
annotations in the `annotations` directory.

## Evaluation

Evaluation is done using the `evaluation.ipynb` Jupyter notebook on the `set-p`
dataset, which was also used in the original paper.

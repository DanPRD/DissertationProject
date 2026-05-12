## Requirements
 - Python libraries required are located in the requirements.txt file, and can be installed using `pip install -r requirements.txt`
 - The Dataset, CIC-IDS2017-Improved, can be downloaded from https://intrusion-detection.distrinet-research.be/CNS2022/Datasets/ and extracted into a directory called `dataset` in the repository root. e.g. the directory structure should look like this:
 ```
.
├── model.py
└── dataset/
    ├── monday.csv
    ├── tuesday.csv
    ├── wednesday.csv
    ├── thursday.csv
    └── friday.csv
 ```

## Notes
- Experiments were run using pandas `3.0.1`, numpy `2.4.2`, scikit-learn `1.8.0`, and python `3.14.4`, although it should work with newer versions as well.
 - Atleast 16GB RAM is advised as it has not been tested on anything lower than 16GB

## Running
From the repository root run `python model.py`

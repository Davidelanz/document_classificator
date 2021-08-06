![](https://img.shields.io/badge/Python-3.8-yellow)
![](https://img.shields.io/badge/Conda-yes-green)
![](https://img.shields.io/badge/PyTorch-1.9-red)

# Document classificator

--- 

- [Document classificator](#document-classificator)
  - [Install](#install)
  - [Architecture](#architecture)
  - [Results](#results)

---

In this repository, I built a classifier that takes a document image and/or text and
predicts a certain class. 

The model is trained on a small toy dataset based on the [RVL-CDIP Dataset](http://www.cs.cmu.edu/~aharley/rvl-cdip/). The dataset contains 100 documents of 4 classes: 
- “resumee”,
- “invoice”, 
- “letter”, 
- “email.

For each document, bot image and OCR  data is provided.


## Install

The code comes with a ready-to-use conda enviroment:
```
git clone https://github.com/Davidelanz/document_classificator
cd document_classificator
conda env create -f environment.yml 
conda activate document_classificator
jupyter-lab
```

With the previous commands, you should be able to navigate easily the notebooks provided in the repository at http://localhost:8888/lab.

## Architecture

(...)

## Results 

(...)
# Challenger Deep
A tool to create concordance matrices from labels. The concordance maps input labels onto HSCPC categories; HSCPC is classification formed by the combining the UN's HS and CPC classifications (commodities and services). HSCPC contains 6357 unique categories.

## Setup

### Installing tensor flow on MacOS
Follow the instructions here for install tensorflow on MacOS: https://developer.apple.com/metal/tensorflow-plugin/

### Environment variables

```bash
WORK_DIR=/Users/quoll/challenger_deep/ 
```

## Running
To make a prediction, run:
```
predict_hscpc_labels.py
```
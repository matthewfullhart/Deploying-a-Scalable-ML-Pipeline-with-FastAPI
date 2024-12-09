# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
This model was created as part of a Machine Learning Devops course from Udacity.
I used a Random Forest classifier.

## Intended Use
This model predicts whether a given individual has a salary higher or lower than $50,000.00 (this is in the "income" field of the training data) based on the census data provided.

## Training Data
Census bureau data from 1994, created by Barry Becker
More information here:
https://archive.ics.uci.edu/dataset/20/census+income
I split this starting data into 80% for training...

## Evaluation Data
... and 20% for testing.

## Metrics

Precision: 0.7436
Recall:    0.6110
F1:        0.6708

## Ethical Considerations

This sample has a large imbalance in a couple fields ("race" and "native_country") and care should be taken not to draw too general a conclusion based on the outcome of the results. Nor can we say definitively that any specific factors here are causitive rather than merely correlative.

## Caveats and Recommendations

Further investigation of the data and refinement of the model based on that investigation is warranted. "sex" and "relationship" are highly correlated (I suspect because relationship split out "husband" and "wife" into separate categories. I would recommend combining "husband" and "wife" into "spouse" to avoid overindexing on that variable), and "education" and "education_num" appear to be 1-to-1 matches for each other, so dropping one of those columns should increase model performance.
If desired, every new census, this data model could be updated by running newer census data into it, though the cadence of census data in the United States is only every 10 years. Depending on business requirements, that may be extraneous.
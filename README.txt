Constructed a recommender system to predict item ratings for users using collaborative filtering

Included files:

main_code.py //code file
train.txt //sample training file
output.txt //output file

Format of input file: Each line contains 3 space seperated parameters: user #, item #, rating given (out of 5)

Command to run code : python3 main_code.py

Following in-built python libraries have been used:

pandas
numpy
sklearn (train_test_split)
surprise (Dataset, Reader, KNNWithMeans, KNNBasic, KNNWithZScore, SVD, SVDPP, GridSearchCV, cross_validate)
math

Kindly install necessary libraries (from list above) if absent to run code successfully.

References used for coding: 

https://bmanohar16.github.io/blog/recsys-evaluation-in-surprise
https://surprise.readthedocs.io/en/stable/model_selection.html
http://surpriselib.com/
https://surprise.readthedocs.io/en/stable/model_selection.html
http://surprise.readthedocs.io/en/stable/getting_started.html
https://surprise.readthedocs.io/en/stable/FAQ.html
https://en.wikipedia.org/wiki/Matrix_factorization_(recommender_systems)#SVD++

Note: 

1. I have used in-built surprise python library to develop recommender system.
2. Output file may change depending on the best fit model that we get for each run. From analysis, SVDPP turns out to be best fit, but the threshold range values may change, which results in different prediction values.   



# sgd-screening
Matlab code to reproduce the results of the paper


There are three folders
	online 
	finite sum
here 
- "online fold" contains the source codes of the synthetic example in our paper, and "online_toyexample.m" is the main file to run. In this code, $n$ is the dimension of the problem, it is set to 10^3.
- "finite sum" fold contains the course codes of LASSO and sparse logistic regression (SLR), "runme_lasso.m" and "runme_slr.m" are the main files to run. In the data folder, 8 datasets are included for tests. Set the "i_file" value to choose the dataset. 
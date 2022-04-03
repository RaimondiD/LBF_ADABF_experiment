# ADAPTIVE LEARNED BLOOM FILTER (ADA-BF): EFFICIENT UTILIZATION OF THE CLASSIFIER

The python files include the implementation of the Bloom filter, learned Bloom filter, Ada Bloom filter, and train and analysis of some classifier (linear SVM, random forest and fast-forward NN).
classifier.py is used to train and save result of classifier; main.py is used to train and save Bloom filter, learned Bloom filter and Ada Bloom filter.

**Classifier options**:
- Use \models\classifier_conf.json to set the invariant hyperparameter of the classifier. 
 *Syntax*:  `{name_classifier_1 : {name_param_1 : value, name_param2 : value...}, name_classifier_2 : {...} ... }`.


- Use \models\params_grid_search.json to set the grid of parameters tested in Model selection.
*Syntax*: `{name_classifier_1 : {name_param1 :Grid, name_param2 : Grid_2 ...}...}`

Grid:
 - `[start,end,"range"]` ->  all the integer from start to end ([1,4,"range"] -> 1,2,3,4)
 - `[start,end,n_el]` -> try n_el elements in the log space form 10^(start) to 10^(end) ([-3,1,5] -> 10e-3, 10e-2, 10e-1, 10e0. 10e1)
 
 there is an example of classifier_conf.json and params_grid_search.json in \models\

**Input arguments (classifer.py and main.py)**: 
- `--data_path`: a csv file includes the items and labels; 
- `--nfoldsCV` : number of fold used in CV; default is 5.
- `--classifier_list`: list of classifiers that will be used in the learned filter (RF -> random forest, SVM -> linear support vector machine, FFNN -> multi-strate perceptron).
- `--trees` : specifies the number of trees of the RF
- `--layers` : specifies the number of neurons in the hidden layers of the FFNN. es: --layers 50 50 50  specifies 3 hidden layers with 
50 neurons each.

The following arguments are used to handle dataset's ratio in different parts of the script: 
- `--pos_ratio`, `--neg_ratio`: given the initial dataset, specify the ratio of positive and negative samples to be used in the filter's training phase. The filter is trained by trying different values for the classifier threshold (chosen according to `thresholds_q`), and then choosing the value that results in the filter having the lowest fpr value. 
- `--pos_ratio_clc`, `--neg_ratio_clc`: given the dataset obtained according to `pos_ratio` and `neg_ratio`, specify the ratio of positive and negative samples to be used in the training phase of the classifiers. In particular, the resulting dataset will be used to perform a nested cross validation, with which the best configuration of hyperparameters is chosen.

**Input arguments (only main.py)**:
- `--size_of_filter`: size of the entire structure (classifier + filter);
- `--force_train`: force training of all classifiers specified in classifier list. if the argument isn't provided only classifier without a saved model and score are trained.
- `--type_filter` : specify the type of filter (learned bloom filter, sandwiched learned bloom filter o Ada-BF).
- (For LBF and SLBF) `--thresholds_q`: for these types of filters, the thresholds to be tested correspond to the q-order quantiles of the dataset used for training the filter, the one that generates the structure with the lowest number of false positives is chosen. This argument specifies the order q of the quantiles. For example, if thresholds_q is set to 10, all quantiles of order 10 will be tested as thresholds.
- (For Ada-BF and disjoint Ada-BF) `--num_group_min` and `--num_group_max` give the range of number of groups to divide (range of *g*
); `--c_min` and `--c_max` provide the range of *c* where *c=m_j/m_{j+1}*
-  `--negTest_ratio`: specifies the ratio of negative samples used for the testing query of the trained filter. The samples are extracted from the unsued part of the initial dataset. default is 1.
- `--test_path` : optional argument that specifies the path of a dataset for the testing query of the trained filter. if negTest_ratio is specified (!= 1) it specifies the ratio of negative samples extracted from the specified dataset

**Results**
Results are saved in \results\params_path where params_path is obtained by concatenation of the name of used Dataset, value of seed and params relative to dataset Ratio. For the classifiers are saved metrics on all classifier trained (results\params_path\name_classifer_result.csv), a file \results\params_path\total_result.csv with the results of all classifier regruped and a file \results\params_path\info_dataset.csv with the dimension of the used dataset.
For the filters the results are saved in results\params_path\name_filter; at the end of the execution of main.py is printed the path of the results of the filter tested in that run.

**Commands**:
- Run Bloom filter: 

`python Bloom_filter.py --data_path ./Datasets/URL_data_features_all.csv --size_of_BF 200000 --pos_ratio 0.7 --neg_ratio 0.7 --negTest_ratio 1.0`

- Run learned Bloom filter on all classifier: 

`python main.py --data_path ./Datasets/URL_data_features_all.csv --size_of_filter 500000 --thresholds_q 10 --type_filter learned_Bloom_filter --classifier_list RF SVM FFNN --pos_ratio 0.7 --neg_ratio 0.7 --pos_ratio_clc 0.7 --neg_ratio_clc 0.7 --negTest_ratio 1.0`  
- Run sandwiched learned Bloom filter on all classifier: 

`python main.py --data_path ./Datasets/URL_data_features_all.csv --size_of_filter 500000 --thresholds_q 50 --type_filter sandwiched_learned_Bloom_filter --classifier_list RF SVM FFNN  --pos_ratio 0.7 --neg_ratio 0.7 --pos_ratio_clc 0.7 --neg_ratio_clc 0.7 --negTest_ratio 1.0` 
- Run Ada-BF on all classifier: 

`python main.py --data_path ./Datasets/URL_data_features_all.csv --size_of_filter 500000  --num_group_min 8  --num_group_max 12  --c_min 1.6  --c_max 2.5 --type_filter Ada-BF --classifier_list RF SVM FFNN  --pos_ratio 0.7 --neg_ratio 0.7 --pos_ratio_clc 0.7 --neg_ratio_clc 0.7 --negTest_ratio 1.0`

- Run disjoint Ada-BF (not tested): 

`python disjoint_Ada-BF.py --data_path ./Datasets/URL_data.csv --size_of_Ada_BF 200000  --num_group_min 8  --num_group_max 12  --c_min 1.6  --c_max 2.5`

- Run only analysis on classifiers:

 `python classifier.py --data_path ./Datasets/URL_data_features_all.csv --classifier_list RF SVM FFNN --nfoldCV 5  --pos_ratio 0.7 --neg_ratio 0.7 --pos_ratio_clc 0.7 --neg_ratio_clc 0.7`



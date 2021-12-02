# ADAPTIVE LEARNED BLOOM FILTER (ADA-BF): EFFICIENT UTILIZATION OF THE CLASSIFIER

The python files include the implementation of the Bloom filter, learned Bloom filter, Ada-BF and disjoint Ada-BF, and print the size of False Positives of the corresponding algorithm.
 
**Classifier options**:
use \models\classifier_conf.json to set the invariant hyperparameter of the classifier. 
Syntax : {name_classifier_1 : {name_param_1 : value, name_param2 : value...},
name_classifier_2 : {...} ... }


use \models\params_grid_search,json to set the grid of parameters tested in Model selection.
Syntax:
{name_classifier_1 : {name_param1 :grid, name_param2 : grid_2 ...}...}
grid:
-[start,end,"range"] ->  all the integer from start to end ([1,4,"range"] -> 1,2,3,4)
-[start,end,n_el] -> try n_el elements in the log space form 10^(start) to 10^(end) ([-3,1,5] -> 10e-3, 10e-2, 10e-1, 10e0. 10e1)

 

**Input arguments**: 
- `--data_path`: a csv file includes the items and labels; `--size_of_filter`: size of the entire structure (classifier + filter);
- `--classifier_list`: list of classifier that will be used in the learned filter (RF -> random forest, SVM -> linear support vector machine, FFNN -> multi-strate perceptron).
- `--force_train`: force training of all classifiers specified in classifier list. if the argument isn't provided only classifier without a saved model and score are trained.
- `type_filter` : specify the type of filter (learned bloom filter, sandwiched learned bloom filter o Ada-BF)
- (for LBF and SLBF)`--threshold_min` and `--threshold_max`, provide the range of the score threshold (between `threshold_min` and `threshold_max`). Items with score larger than the threshold are identified as keys;
- (for Ada-BF and disjoint Ada-BF) `--num_group_min` and `--num_group_max` give the range of number of groups to divide (range of *g*
); `--c_min` and `--c_max` provide the range of *c* where *c=m_j/m_{j+1}*



**Commands**:
- Run Bloom filter: `python Bloom_filter.py --data_path ./Datasets/URL_data.csv --size_of_BF 200000`
- Run learned Bloom filter on all classifier: `python main.py --data_path ./Datasets/URL_data.csv --size_of_LBF 500000  --threshold_min 0.5   --threshold_max 0.95 --type_filter learned Bloom filter --classifier_list RF SVM FFNN`  
- Run sandwiched learned Bloom filter on all classifier: `python main.py --data_path ./Datasets/URL_data.csv --size_of_LBF 500000  --threshold_min 0.3   --threshold_max 0.8 --type_filter sandwiched learned Bloom filter --classifier_list RF SVM FFNN`  
- Run Ada-BF on all classifier: `python Ada-BF.py --data_path ./Datasets/URL_data.csv --size_of_Ada_BF 500000  --num_group_min 8  --num_group_max 12  --c_min 1.6  --c_max 2.5 --type_filter Ada-BF --classifier_list RF SVM FFNN`
- disjoint Ada-BF (not tested): `python disjoint_Ada-BF.py --data_path ./Datasets/URL_data.csv --size_of_Ada_BF 200000  --num_group_min 8  --num_group_max 12  --c_min 1.6  --c_max 2.5`




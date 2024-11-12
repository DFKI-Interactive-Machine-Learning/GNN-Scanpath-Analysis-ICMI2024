# Important Notes 
- The eye tracking data and data loader can be downloaded from the [gazeRE](https://github.com/DFKI-Interactive-Machine-Learning/gazeRE-dataset) repository.
- The data folder from gazeRE was renamed into _Eye_Tracking_Data_.
- The functions to compute the saccade and area features, i.e., _compute_saccade_aoi_features_, _compute_area_aoi_features_ are not uploaded to this repository.

# Repository Contents

### GNN
- The graph generation codes for both gREL and GoogleNQ.
- The stimuli coordinates acquired from the gazeRE dataset.
- The generated graph data.
- The reported GNN results.
- The graph classification code for gREL.
- The node classification code for GoogleNQ.

### CNN
- The CNN stimului generation codes.
- The generated CNN data.
- The CNN classification codes using VGG19.
- The reported CNN results.
  
### Traditional ML
- The traditional Machine Learning approach in a Jupyter Notebook.
- The generated feature data.
- The reported Traditional ML results.

# GNN & CNN Model Training and Evaluation Using Nested Cross-validation Pseudocode

```python
# Input: Scanpath Data
# Output: Average Test Data Balanced Accuracy & Model Configurations

# For each fold i in {1, 2, 3, 4, 5}
for i in range(5):
    # Split input data into training/validation set D_train_val_i and test set D_test_i

    # For Optuna trial t in {1, 2, ..., n_trials}
    for t in range(n_trials):
        # Pick model configuration c_t_i

        # For each fold j in {1, 2, 3, 4, 5}
        for j in range(5):
            # Split D_train_val_i into training set D_train_ij and validation set D_val_ij
            # Train Model m_ct_ij using configuration c_t_i and training set D_train_ij
            # Evaluate Model m_ct_ij on validation set D_val_ij to get the Balanced Accuracy BA_val_ct_ij
            # Store model configuration c_t_i, and Balanced Accuracy BA_val_ct_ij

        # Compute Average Validation Balanced Accuracy BA_val_ct_i for configuration c_t_i

    # Determine the best configuration c_best_i with maximum BA_val_ct_i
    # Train the best model m_best_i using c_best_i and D_train_val_i
    # Test m_best_i on D_test_i to get test Balanced Accuracy BA_test_i
    # Store c_best_i, and BA_test_i

# Return Average Test Balanced Accuracy BA_test and & The Best Model Configurations [c_best_i for i in range(5)]
```

# Citation
If you find this work useful to your research, please consider citing our publications (This is a temporary citation).

```
@inproceedings{10.1145/3678957.3685736,
author = {Mohamed Selim, Abdulrahman and Bhatti, Omair Shahzad and Barz, Michael and Sonntag, Daniel},
title = {Perceived Text Relevance Estimation Using Scanpaths and GNNs},
year = {2024},
isbn = {9798400704628},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3678957.3685736},
doi = {10.1145/3678957.3685736},
abstract = {A scanpath is an important concept in eye tracking that represents a person’s eye movements in a graph-like structure. Passive gaze-based interfaces, in which users do not consciously interact using their eyes, typically interpret users’ scanpaths to enable adaptive and personalised interaction. Despite the benefits of graph neural networks (GNNs) in graph processing, this technology has not been considered for that purpose. An example application is perceived relevance estimation, which still suffers from low classification performance. In this work, we investigate how and whether GNNs can be used to analyse scanpaths for readers’ perceived relevance estimation using the gazeRE dataset. This dataset contains eye tracking data from 24 participants, who rated the relevance of 12 short and 12 long documents in relation to a given query. The relevance was assigned either to an entire short document or to each paragraph within a long document, which allowed us to investigate two different GNN tasks. For comparison, we reproduced the gazeRE baseline using Random Forest and Support Vector classifiers, and an additional Convolutional Neural Network (CNN) from the literature. All models were evaluated using leave-users-out cross-validation. For short documents, the GNNs surpassed the baseline methods, with certain experiments showing an absolute balanced accuracy improvement of 7.6\% and 14.3\% over the CNN and gazeRE baselines, respectively. However, similar improvements were not observed in long documents. This work investigates and discusses the future potential of using GNNs as a scanpath analysis method for passive gaze-based applications, such as implicit relevance estimation.},
booktitle = {Proceedings of the 26th International Conference on Multimodal Interaction},
pages = {418–427},
numpages = {10},
keywords = {Eye Tracking, GNN, Passive Gaze-based Application, Scanpath},
location = {San Jose, Costa Rica},
series = {ICMI '24}
}
```

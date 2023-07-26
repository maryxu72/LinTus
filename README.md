# LinTus
LinTus: Learned Index based on Transformer Model for Unordered String Data Sets on GPU
train 20M.py and train 83M.py: train model for 20 million  and 82 million  Wiki string entries, respectively.

inference all data2npy.py: project all 82 Wiki string entries to 64-float embeddings

inference_20230305_make_DataLink.py: construct linked lists

inference_throughput_analysis.py: analysize the throught via 128 rounds of inferences

Ablation_experiment2000w.py and Ablation_experiment_83M.py : ablation experiments on 20 million and 82 million Wiki string entries and analyze the effect of different dimension lists

statistics of linked list characteristics.py: calculate characteristics of data link lists, such as the number of collions, the number of effective list elements, etc.

range index test.py: calculate cosine similarities of range queries

semantics similarity.py: calculate the semantics similarity of modified preposition and keywords on the original data 

view some data.py: view the details of linked list elements.

chars_encode.csv: character encoding table

model_demo20M.pth and model_demo82M.pth: models from LinTus for 20 million and 80 million Wiki string entries, respectively.

CUDA_VISIBLE_DEVICES=2 python -m training.train_nn --config training/config_dnn_full.yaml --device cuda 
CUDA_VISIBLE_DEVICES=2 python -m training.train_nn --config training/config_dnn_test.yaml --device cuda 


python -m training.train_enet --config training/config_enet_test.yaml
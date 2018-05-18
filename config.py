

class Configs(object):
    start_epoch = 0  # start epoch to train
    test_epoch = 0   # checkpoints to use for test
    epochs = 300     # training epochs
    dataset_name = "cityscapes"   
    saved_path = "checkpoints"
    sample_path = "sample"
    use_gpu = True
    batch_size = 1
    decay_epoch = 100  # learning_rate decay,but i did not use
    learning_rate = 0.0002
    threads = 8   # threads to use for loading datasets
    width = 256
    height = 256
    channels = 3
    n_sample = 1000     # sample images interval
    n_checkpoint = 50   # save checkpoints interval
    beta1 = 0.5
    beta2 = 0.999
    Lambda = 100  # balance L1 loss and cGAN loss
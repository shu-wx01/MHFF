import os


class args:

    dataset = "gambling"  #'gambling' 't-finance' , 'Amazon', 'chameleon', 'squirrel', 'film', 'twitch-e', 'polblogs', 'etg_syn_hom'
    sub_dataset = ""  # 'DE', "1.0", "0.9", "0.8", "0.7", "0.6", "0.5", "0.4", "0.3", "0.2", "0.1", "0"
    DATAPATH = os.getcwd() + "/data/"
    model_path = os.getcwd() + "/model/"

    # t-finance
    # seed = 30

    seed = 30

    hidden_channels = 8
    weight_decay = 6e-6
    lr = 0.01
    dropout = 0.1
    epoch = 100
    early = 100

    cpu = False

    gamma = 1
    C = 1
    K = 0
    n_class = 2
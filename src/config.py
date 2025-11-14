import os


class Config:
    # image
    IMAGE_MODEL_NAME = 'tf_efficientnet_b0'
    IMAGE_MODEL_UNFREEZE = 'blocks.6|conv_head|bn2'

    # mlp
    MLP_IN_FEATURES = 551
    MLP_HIDDEN_SIZE = 1024
    MLP_OUT_FEATURES = 512

    # projection
    PROJ_SIZE = 256
    

    # hyperparameters
    BATCH_SIZE = 128
    NUM_WORKERS = 8
    IMAGE_LR = 1e-3
    MLP_LR = 1e-3
    REGRESSOR_LR = 1e-3
    EPOCHS = 100
    
    # paths
    IMAGE_PATH = 'data/images'
    DISHES_PATH = 'data/dish_fixed.csv'
    INGREDIENTS_PATH = 'data/ingredients_fixed.csv'
    SAVE_PATH = 'models'

    @classmethod
    def check_save_path(cls):
        if not os.path.isdir(cls.SAVE_PATH):
            os.mkdir(cls.SAVE_PATH)
        else:
            for f in os.listdir(cls.SAVE_PATH):
                os.remove(os.path.join(cls.SAVE_PATH, f))
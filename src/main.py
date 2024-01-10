from preprocess import PreProcessor, IMAGE_SIZE
from model import CNN

DATASET_TRAINING_PATH = "/home/white/uni_workspace/ecm3401-dissertation/data/example_ad_dataset/train"
DATASET_TEST_PATH = "/home/white/uni_workspace/ecm3401-dissertation/data/example_ad_dataset/test"


CLASS_NAMES = ['MildDementia', 'ModerateDementia',
               'NonDementia', 'VeryMildDementia']


if __name__ == "__main__":

    preProcess = PreProcessor(DATASET_TRAINING_PATH,
                              DATASET_TEST_PATH, CLASS_NAMES)

    model = CNN(IMAGE_SIZE, len(CLASS_NAMES))

    print("COMPILING...")
    model.compile()
    print("TRAINING...")
    model.train(preProcess.train_ds, preProcess.val_ds)
    print("TESTING...")
    model.test(preProcess.test_ds)

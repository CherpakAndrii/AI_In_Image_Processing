from get_data import *
from nn_model import AlexNet, compile_model

CLASS_NAMES = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

if __name__ == '__main__':
    train_ds, test_ds, validation_ds = get_data()
    visualize_data(train_ds, CLASS_NAMES)

    train_ds_size = get_ds_size(train_ds)
    test_ds_size = get_ds_size(test_ds)
    validation_ds_size = get_ds_size(validation_ds)

    print("Training data size:", train_ds_size)
    print("Test data size:", test_ds_size)
    print("Validation data size:", validation_ds_size)

    train_ds = process_ds(train_ds)
    test_ds = process_ds(test_ds)
    validation_ds = process_ds(validation_ds)

    mdl = AlexNet(len(CLASS_NAMES))
    model = compile_model(mdl)

    model.fit(train_ds, epochs=10, validation_data=validation_ds, validation_freq=1)
    model.evaluate(test_ds)

    model.save('alexNet.h5')

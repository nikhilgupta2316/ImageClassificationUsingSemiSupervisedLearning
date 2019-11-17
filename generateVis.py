import json
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import params

if __name__ == "__main__":
    args = params.parse_args()
    filename = args.filename
    # filename = "resnet-49k_values.log"
    modelName = filename[:filename.find("_")]
    modelName = modelName[0].capitalize() + modelName[1:]

    with open(filename, 'r') as f:
        data = json.load(f)

        test_accuracy = np.array(data['test_accuracy'])[:,1]
        val_accuracy = np.array(data['val_accuracy_per_epoch'])[:, 1]

        plt.title(modelName+" Accuracy", fontdict={'fontsize': 15})
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")
        plt.plot(val_accuracy, '-', linewidth=1.5)
        plt.plot(test_accuracy, 'g-', linewidth=1.5)
        plt.legend(["Validation Accuracy", "Test Accuracy"])
        plt.grid()
        plt.savefig("visualization/"+modelName+" Accuracy.svg")
        # plt.show()
        plt.clf()
        test_loss = np.array(data['test_loss'])[:, 1]
        val_loss = np.array(data['val_loss_per_epoch'])[:, 1]

        plt.title(modelName + " Loss", fontdict={'fontsize': 15})
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.plot(val_loss, '-', linewidth=1.5)
        plt.plot(test_loss, 'g-', linewidth=1.5)
        plt.legend(["Validation Loss", "Test Loss"])
        plt.grid()
        plt.savefig("visualization/"+modelName + " Loss.svg")
        # plt.show()


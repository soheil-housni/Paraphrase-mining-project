import matplotlib.pyplot as plt

def plotting(epochs, avg_batch_train_loss, epoch_train_acc,
             avg_batch_val_loss, epoch_val_acc, epoch_val_f1,
             cosines_train_set=None, thresholds_train_set=None):

    epochs_list = [i for i in range(epochs)]


    fig, ax = plt.subplots(2, 1, figsize=(10, 12))

    ax[0].plot(epochs_list, avg_batch_train_loss, linewidth=2, marker="o")
    ax[0].set_title("Training Loss per Epoch", fontsize=14)
    ax[0].set_xlabel("Epoch", fontsize=12)
    ax[0].set_ylabel("Loss", fontsize=12)
    ax[0].grid(True, linestyle="--", alpha=0.4)

    ax[1].plot(epochs_list, epoch_train_acc, linewidth=2, marker="o")
    ax[1].set_title("Training Accuracy per Epoch", fontsize=14)
    ax[1].set_xlabel("Epoch", fontsize=12)
    ax[1].set_ylabel("Accuracy", fontsize=12)
    ax[1].grid(True, linestyle="--", alpha=0.4)

    fig.suptitle("Training Metrics", fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.97])


    fig, ax = plt.subplots(3, 1, figsize=(10, 15))

    ax[0].plot(epochs_list, avg_batch_val_loss, linewidth=2, marker="o")
    ax[0].set_title("Validation Loss per Epoch", fontsize=14)
    ax[0].set_xlabel("Epoch", fontsize=12)
    ax[0].set_ylabel("Loss", fontsize=12)
    ax[0].grid(True, linestyle="--", alpha=0.4)

    ax[1].plot(epochs_list, epoch_val_acc, linewidth=2, marker="o")
    ax[1].set_title("Validation Accuracy per Epoch", fontsize=14)
    ax[1].set_xlabel("Epoch", fontsize=12)
    ax[1].set_ylabel("Accuracy", fontsize=12)
    ax[1].grid(True, linestyle="--", alpha=0.4)

    ax[2].plot(epochs_list, epoch_val_f1, linewidth=2, marker="o")
    ax[2].set_title("Validation F1 per Epoch", fontsize=14)
    ax[2].set_xlabel("Epoch", fontsize=12)
    ax[2].set_ylabel("F1 Score", fontsize=12)
    ax[2].grid(True, linestyle="--", alpha=0.4)

    fig.suptitle("Validation Metrics", fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.97])



    n_rows = int(cosines_train_set is not None) + int(thresholds_train_set is not None)

    if n_rows > 0:
        fig, ax = plt.subplots(n_rows, 1, figsize=(10, 12))

        if n_rows == 1:
            ax = [ax]

        row = 0

        if cosines_train_set is not None:
            ax[row].hist(cosines_train_set.clone().detach().numpy(), density=True, edgecolor="black", alpha=0.7)
            ax[row].set_title("Cosine Similarity Distribution", fontsize=14)
            ax[row].set_xlabel("Cosine Similarity", fontsize=12)
            ax[row].grid(True, linestyle="--", alpha=0.4)
            row += 1

        if thresholds_train_set is not None:
            ax[row].hist(thresholds_train_set.clone().detach().numpy(), density=True, edgecolor="black", alpha=0.7)
            ax[row].set_title("Threshold Distribution", fontsize=14)
            ax[row].set_xlabel("Threshold", fontsize=12)
            ax[row].grid(True, linestyle="--", alpha=0.4)

        fig.suptitle("Training Distributions", fontsize=16)
        fig.tight_layout(rect=[0, 0, 1, 0.97])

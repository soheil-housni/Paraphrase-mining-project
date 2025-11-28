import matplotlib.pyplot as plt

import matplotlib.pyplot as plt

def plotting_metrics(df, CrossAttention=True):
    fig, ax = plt.subplots(3, 1, figsize=(15, 15))
    plt.style.use('seaborn-darkgrid')

    ax[0].plot(df["epoch"], df["avg_batch_train_loss"], label="Train Loss", color='#d62728' , linestyle='-', marker='o')
    ax[0].plot(df["epoch"], df["avg_batch_val_loss"], label="Validation Loss", color='#ff7f0e', linestyle='--', marker='x')
    ax[0].set_title("Training and Validation Loss across Epochs", fontsize=14, fontweight='bold')
    ax[0].set_xlabel("Epochs", fontsize=12)
    ax[0].set_ylabel("Loss", fontsize=12)
    
    y_min = min(df["avg_batch_train_loss"].min(), df["avg_batch_val_loss"].min()) * 0.95
    y_max = max(df["avg_batch_train_loss"].max(), df["avg_batch_val_loss"].max()) * 1.05
    ax[0].set_ylim(y_min, y_max)
    
    ax[0].legend(loc='upper right', frameon=True, framealpha=0.9)
    ax[0].grid(True, linestyle='--', alpha=0.7)

    ax[1].plot(df["epoch"], df["epoch_train_acc"], label="Train Accuracy",color='#1f77b4' , linestyle='-', marker='o')
    ax[1].plot(df["epoch"], df["epoch_val_acc"], label="Validation Accuracy",color='#2ca02c' , linestyle='--', marker='x')
    ax[1].set_title("Training and Validation Accuracy across Epochs", fontsize=14, fontweight='bold')
    ax[1].set_xlabel("Epochs", fontsize=12)
    ax[1].set_ylabel("Accuracy", fontsize=12)
    ax[1].legend(loc='lower right', frameon=True, framealpha=0.9)
    ax[1].grid(True, linestyle='--', alpha=0.7)

    ax[2].plot(df["epoch"], df["epoch_val_f1"], label="Validation F1 Score", color='#9467bd', linestyle='--', marker='s')
    ax[2].set_title("Validation F1 Scores across Epochs", fontsize=14, fontweight='bold')
    ax[2].set_xlabel("Epochs", fontsize=12)
    ax[2].set_ylabel("F1 Score", fontsize=12)
    ax[2].legend(loc='lower right', frameon=True, framealpha=0.9)
    ax[2].grid(True, linestyle='--', alpha=0.7)

    if CrossAttention:
        fig.suptitle("Metrics for the best CrossAttention model", fontsize=16, fontweight='bold')
    else:
        fig.suptitle("Metrics for the best Classic Classifier model", fontsize=16, fontweight='bold')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()


def plotting_distribution_hyperparams(df, CrossAttention=True):
    fig, ax = plt.subplots(3, 2, figsize=(15, 15))
    plt.style.use('seaborn-darkgrid') 

    fig.patch.set_facecolor('white')
    for a_row in ax:
        for a in a_row:
            a.set_facecolor('#f5f5f5')

    ax[0,0].hist(df["lr"], bins=10, color='#1f77b4', alpha=0.7, edgecolor='black')
    ax[0,0].set_title("Distribution of Learning Rates", fontsize=12, fontweight='bold')
    ax[0,0].set_xlabel("Learning rate")
    ax[0,0].set_ylabel("Count")

    ax[0,1].hist(df["weight_decay"], bins=10, color='#ff7f0e', alpha=0.7, edgecolor='black')
    ax[0,1].set_title("Distribution of Weight Decay Rates", fontsize=12, fontweight='bold')
    ax[0,1].set_xlabel("Weight decay rate")
    ax[0,1].set_ylabel("Count")

    ax[1,0].hist(df["dropout_p"], bins=10, color='#2ca02c', alpha=0.7, edgecolor='black')
    ax[1,0].set_title("Distribution of Dropout Proportions", fontsize=12, fontweight='bold')
    ax[1,0].set_xlabel("Dropout proportion")
    ax[1,0].set_ylabel("Count")

    ax[1,1].hist(df["n_freeze"], bins=range(int(df["n_freeze"].min()), int(df["n_freeze"].max())+2),
                  color='#d62728', alpha=0.7, edgecolor='black')
    ax[1,1].set_title("Distribution of the Number of Frozen Layers", fontsize=12, fontweight='bold')
    ax[1,1].set_xlabel("Number of frozen layers")
    ax[1,1].set_ylabel("Count")

    ax[2,0].hist(df["use_n_layers"], bins=10, color='#9467bd', alpha=0.7, edgecolor='black')
    ax[2,0].set_title("Distribution of the Number of Layers Used", fontsize=12, fontweight='bold')
    ax[2,0].set_xlabel("use_n_layers")
    ax[2,0].set_ylabel("Count")

    ax[2,1].hist(df["fc1"], bins=10, color='#8c564b', alpha=0.7, edgecolor='black')
    ax[2,1].set_title("Distribution of the Number of Neurons for the First Layer", fontsize=12, fontweight='bold')
    ax[2,1].set_xlabel("Number of neurons for the first layer")
    ax[2,1].set_ylabel("Count")

    if CrossAttention:
        fig.suptitle("Hyperparameter Distributions for the Best CrossAttention Model", fontsize=16, fontweight='bold')
    else:
        fig.suptitle("Hyperparameter Distributions for the Best Classic Classifier Model", fontsize=16, fontweight='bold')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

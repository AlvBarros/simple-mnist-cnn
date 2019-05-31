import matplotlib.pyplot as plt
import json

print('Plotting our charts')

def plot(file_path):
    f = open(file_path, 'r')
    history_dict = json.loads(f.read())

    loss_values = history_dict['loss']
    val_loss_values = history_dict['val_loss']
    acc_values = history_dict['acc']
    val_acc_values = history_dict['val_acc']
    epochs = range(1, len(loss_values) + 1)

    plt.figure(1)
    plt.subplot(211)
    plt.plot([0] + loss_values, marker='o', label='Loss Values')
    plt.plot([0] + val_loss_values, marker='o', label='Test loss Values')
    plt.xlabel('Epochs', fontsize=14, color='black')
    plt.ylabel('Loss values', fontsize=14, color='black')
    plt.legend()
    plt.subplot(212)
    plt.plot([0] + acc_values, marker='o', label='Accuracy values')
    plt.plot([0] + val_acc_values, marker='o', label='Test accuracy values')
    plt.xlabel('Epochs', fontsize=14, color='black')
    plt.ylabel('Accuracy values', fontsize=14, color='black')
    plt.show()
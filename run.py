import mnist
import sys
import os

epochs = 0
name = ''
plot = False

for num in range(1, len(sys.argv)):
    arg = sys.argv[num]
    if arg == '--epochs' and len(sys.argv) > num + 1 and epochs == 0:
        if int(sys.argv[num+1]) > 0:
            epochs = int(sys.argv[num+1])
    if arg == '--name' and len(sys.argv) > num + 1 and name == '':
        name = sys.argv[num+1]
    if arg == '--graphs':
        plot = True

history_dict = mnist.run(epochs, name)

with open(name + '-dict.json', 'w') as f: 
    print(history_dict, file=f) 

if plot:
    print('Plotting our charts')
    import matplotlib.pyplot as plot

    loss_values = history_dict['loss']
    val_loss_values = history_dict['val_loss']
    acc_values = history_dict['acc']
    val_acc_values = history_dict['val_acc']
    epochs = range(1, len(loss_values) + 1)


    plot.subplot(2, 1, 1)
    x = range(0, epochs)
    y1 = val_loss_values
    y2 = loss_values
    #plt.setp(line1, linewidth=2.0, marker='+', markersize=10.0)
    #plt.setp(line2, linewidth=2.0, marker='4', markersize=10.0)
    plot.xlabel('Epochs')
    plot.ylabel('Loss')
    plot.grid(True)
    plot.legend()

    plot.subplot(2, 1, 2)
    y3 = val_acc_values
    y4 = acc_values
    #plt.setp(line3, linewidth=2.0, marker='+', markersize=10.0)
    #plt.setp(line4, linewidth=2.0, marker='4', markersize=10.0)
    plt.show()
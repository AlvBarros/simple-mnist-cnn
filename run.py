import mnist
import plotter
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
history_dict_filepath = os.getcwd() + '/dictionaries/' + name + '.json'
with open('dictionaries/' + name + '.json', 'x') as f: 
    print(str(history_dict).replace('\'', '"'), file=f) 
    print('Dictionary saved at ' + history_dict_filepath)

if plot:
    plotter.plot(history_dict_filepath)
import mnist
import sys
import os

epochs = 0
name = ''

for num in range(1, len(sys.argv)):
    arg = sys.argv[num]
    if arg == '--epochs' and len(sys.argv) > num + 1 and epochs == 0:
        epochs = sys.argv[num+1]
    if arg == '--name' and len(sys.argv) > num + 1 and name == '':
        name = sys.argv[num+1]

history = mnist.run(epochs, name)

with open(name + '-dict.json', 'w') as f: 
    print(history, file=f) 
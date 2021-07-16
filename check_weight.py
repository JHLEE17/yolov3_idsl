import torch, torchvision
import torch.nn as nn
import argparse


parser = argparse.ArgumentParser(description='Prune Yolact')

parser.add_argument('--path', default='weights/', help='Directory for load weight.')
args = parser.parse_args()

path = args.path
print('path :', path)

    
weight = torch.load(path)

for key in list(weight.keys()):
  #print(key,'\t\t',weight[key].shape)
  print(key)
  #print(weight[key])


print('Proccess finished')

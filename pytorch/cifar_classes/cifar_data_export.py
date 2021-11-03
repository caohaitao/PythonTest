import torchvision as tv
import torch as t

#训练集
trainset=tv.datasets.CIFAR10(root='./cifar/test/',
                             train=True,
                             download=False)

trainloader=t.utils.data.DataLoader(trainset,
                                    batch_size=1,
                                    shuffle=False,
                                    num_workers=4)

classes=('plane','car','bird','cat','deer','dog','frog','horse','ship','truck')


def extract_cifar_pics():
    count = 0
    for d in trainset:
        data = d[0]
        label = d[1]
        jpg_path = format("imgs_test\\%d_%d_%s.jpg"%(count,label,classes[label]))
        data.save(jpg_path)
        count += 1


if __name__=="__main__":
    extract_cifar_pics()
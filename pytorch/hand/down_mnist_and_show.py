import tensorflow_datasets as tfds
import tensorflow as tf
import cv2
import os
import shutil

tf.enable_eager_execution()

print(tfds.list_builders())



def save_minist_pics(dest_dir):
    ds_train, ds_test = tfds.load(name='mnist', split=["train", "test"])

    ds_train = ds_train.shuffle(1000).batch(128).prefetch(10)
    ds_test = ds_test.shuffle(1000).batch(128).prefetch(10)

    def save_one_dataset(ds,d):
        index = 0
        for features in tfds.as_numpy(ds):
            image,label = features["image"],features["label"]
            for i in range(image.shape[0]):
                img = image[i]
                l = label[i]
                dest_file_name = format("%s\\%d_%d.jpg"%(
                    d,
                    index,
                    l
                ))
                cv2.imwrite(dest_file_name,img)
                index += 1

    train_dir = os.path.join(dest_dir,'train')
    if not os.path.exists(train_dir):
        os.mkdir(train_dir)
    test_dir = os.path.join(dest_dir,'test')
    if not os.path.exists(test_dir):
        os.mkdir(test_dir)

    save_one_dataset(ds_train,train_dir)
    save_one_dataset(ds_test,test_dir)


if __name__=='__main__':
    save_minist_pics(r"E:\tensorflow_datas\minist")
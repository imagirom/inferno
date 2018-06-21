from inferno.io.box.posetrack import *
from matplotlib import pyplot as plt

path = '/export/home/rremme/Datasets/PoseTrack/posetrack_data'
#PoseTrack(path, split='val')
train_loader, validate_loader = get_posetrack_loaders(path, input_res=512, output_res=128)
#valid_batch = validate_loader.__iter__().__next__()
iterator = train_loader.__iter__()
for i in range(10):
    train_batch = iterator.__next__()
    print(train_batch[0][0].numpy().shape)
    plt.imshow(np.moveaxis(train_batch[0][0].numpy(), source=0, destination=-1)*255)
    plt.show()

#plt.imshow(valid_batch[0][0])
#plt.show()

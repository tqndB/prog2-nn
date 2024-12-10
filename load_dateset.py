import matplotlib.pyplot as plt
from torchvision import datesets
import torchvision.transforms.v2 as transforms


ds_train = datesets.FashionMNIST(
    root='data',
    train=True,
    download=True,
)

print(f'numbers of datasets:{len(ds_train)}')

image, target = ds_train[0]
print(type(image),target)

plt.imshow(image)
plt.title(target)
plt.show()


image_tensor = transforms.functional.to_image(image)
print(image_tensor.shape, image_tensor.dtype)
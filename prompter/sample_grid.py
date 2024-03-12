import torch
import matplotlib.pyplot as plt
import numpy as np

def main():
    # Generate random float numbers between 0 and 255
    space = 8
    w = 256
    h = 256
    bs = 4

    anchors = np.stack(
        np.meshgrid(
            np.arange(np.ceil(w / space)),
            np.arange(np.ceil(h / space))),
        -1) * space

    origin_coord = np.array([w % space or space, h % space or space]) / 2
    anchors += origin_coord

    random_floats_x = 2.9 * (torch.rand(bs, 32, 32) - 0.5)
    random_floats_y = 2.9 * (torch.rand(bs, 32, 32) - 0.5)

    random_floats_x = random_floats_x.unsqueeze(-1)
    random_floats_y = random_floats_y.unsqueeze(-1)

    tensor = torch.stack([random_floats_x, random_floats_y], 3).squeeze()
    anchors = torch.from_numpy(anchors).float()
    anchors = anchors.repeat(bs, 1, 1, 1)

    print('!!!!')
    print(anchors.shape)
    dupa = anchors.clone()
    dupa = dupa.reshape(bs, 16, 2, 16, 2, 2)
    dupa = dupa.permute(0, 1, 3, 2, 4, 5)
    dupa = dupa.reshape(bs, 16, 16, 4, 2)
    print(dupa[0, 0, 0, 1, :])

    #anchors += tensor
    #print(anchors)

    # Flatten the tensor
    flattened_tensor = anchors.reshape(bs, 32 * 32, 2)

    # Extract x and y coordinates
    x_coords = flattened_tensor[0, :, 0]
    y_coords = flattened_tensor[0, :, 1]

    # Scatter plot
    plt.figure(figsize=(8, 8))
    plt.scatter(x_coords, y_coords, s=10, c='blue', alpha=0.5)
    plt.title('Scatter Plot of the Tensor')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()
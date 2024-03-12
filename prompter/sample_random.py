import torch
import matplotlib.pyplot as plt

def main():
    # Generate random float numbers between 0 and 255
    random_floats_x = torch.rand(32, 32) * 255.0
    random_floats_y = torch.rand(32, 32) * 255.0

    random_floats_x = random_floats_x.unsqueeze(-1)
    random_floats_y = random_floats_y.unsqueeze(-1)

    # Reshape the tensor to have a third dimension of size 2
    tensor = torch.stack([random_floats_x, random_floats_y], 2).squeeze()

    print(tensor.shape)

    # Flatten the tensor
    flattened_tensor = tensor.reshape(32 * 32, 2)

    # Extract x and y coordinates
    x_coords = flattened_tensor[:, 0]
    y_coords = flattened_tensor[:, 1]

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
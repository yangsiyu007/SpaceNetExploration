from skimage import io
import torch
import matplotlib.pyplot as plt
import os

def segment_and_visualize_image(model, image_path, show_image=True, save_to_dir=None, result_image_name='result.png'):
    """
    Utility to apply a model on one image and save the result.
    """
    image = io.imread(image_path)
    image_tensor = image.transpose((2, 0, 1))
    image_tensor = image_tensor.reshape((1, image_tensor.shape[0], image_tensor.shape[1], image_tensor.shape[2]))
    image_tensor = torch.from_numpy(image_tensor).type(torch.float32)
    print(image_tensor.shape)

    model.eval()  # set model to evaluation mode
    with torch.no_grad():
        scores = model(image_tensor)
        _, prediction = scores.max(1)

    prediction = prediction.transpose((1, 2, 0))
    fig = plt.figure()
    plt.imshow(image, interpolation='none')
    plt.imshow(prediction, cmap='grey', interpolation='none', alpha=0.5)

    if save_to_dir is not None:
        fig.savefig(os.path.join(save_to_dir, result_image_name))

    if show_image:
        plt.show()

    # for testing
    # prediction = np.zeros((10, 10))
    # prediction[3:-3, 3:-3] = 255
    # image = np.ones((10, 10))
    # image = 120 * image
    # image[0:2, 0:2] = 170

#segment_and_visualize_image(None, None, save_to_dir='./')

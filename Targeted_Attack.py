import numpy as np
import pickle
import matplotlib.pyplot as plt

def generate_image_for_class(model, target_class):
    """
    This function Generates a random image that will be classified
    as class target_class by the neural network.
    Parameters:
    ------------------------------------
    model: neural network model object
    target_class: integer, target_class to which the network classifies the image
    alpha: each pixel in the image is initialized by sampling from
        uniform distribution over (-alpha, alpha)
    """
    
    max_iterations=784
    alpha=0.1
    learning_rate = 0.1
    
    # Generate a random image with size 28x28
    x0 = np.random.uniform(-alpha, alpha, size=(784,))

    # Perform gradient descent
    for iteration in range(max_iterations):
        # Forward to get model predictions
        logits = model.forward(x0)

        # Calculate the loss
        loss = -logits[0][target_class]

        # Colculate loss of the gradient
        grad_x0 = model.grad_wrt_input(x0.reshape(1, -1), np.array([target_class]))

        # turn gradient into a 1D array
        grad_x0 = grad_x0.ravel()

        # Change x0 with gradient descent
        x0 -= learning_rate * grad_x0

        # Check if x0 is classified as target_class by the model
        predicted_class = np.argmax(logits)
        if predicted_class == target_class:
            # Save generated image with a suitable filename
            filename = f"targeted_random_img_class_{target_class}.png"
            plt.imshow(x0.reshape(28, 28), cmap='gray') 
            plt.axis('off')
            plt.savefig(filename)
            print(f"Generated image saved as '{filename}'")
            print("image classified as digit: {}".format(predicted_class))
            break

def main():
    model = None
    with open('trained_model.pkl', 'rb') as fid:
        model = pickle.load(fid)

    for c in range(10):
        generate_image_for_class(model, c)

if __name__ == "__main__":
    main()

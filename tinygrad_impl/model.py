from tinygrad.nn import Conv2d
from tinygrad.tensor import Tensor
from tinygrad.nn.state import get_parameters

class SRCNN:
    def __init__(self, upscale_factor=3, color_channels=1):
        # Determine c, n1, n2, f1, f2, f3 based on paper and upscale_factor
        c = color_channels
        if color_channels == 1:  # Y channel of YCbCr
             n1, n2 = 64, 32
             if upscale_factor == 3:  # Default settings from the paper.
                f1, f2, f3 = 9, 5, 5 #9-5-5 configuration
             else:
                raise NotImplementedError("Only upscale factor of 3 is supported for this example. You'd need to choose appropriate filter sizes for other factors.")

        elif color_channels == 3:  # RGB
            n1, n2 = 64, 32
            if upscale_factor == 3:
                f1, f2, f3 = 9, 5, 5 #9-5-5 configuration
            else:
                raise NotImplementedError("Only upscale factor of 3 is supported for this example.")

        else:
             raise ValueError("Invalid number of color channels. Must be 1 (grayscale/Y) or 3 (RGB).")

        self.conv1 = Conv2d(c, n1, kernel_size=(f1, f1), bias=True)
        self.conv2 = Conv2d(n1, n2, kernel_size=(f2, f2), bias=True)
        self.conv3 = Conv2d(n2, c, kernel_size=(f3, f3), bias=True)

    def __call__(self, x: Tensor) -> Tensor:
        x = self.conv1(x).relu() # Correct ReLU application
        x = self.conv2(x).relu() # Correct ReLU application
        x = self.conv3(x)  # No ReLU after the final layer
        return x

    def parameters(self):
        return get_parameters(self)

if __name__ == '__main__':
    #Example usage
    model = SRCNN()
    print(model)

    #dummy input
    x = Tensor.randn(1, 1, 224, 224)
    output = model(x)
    print(f"Output shape: {output.shape}")
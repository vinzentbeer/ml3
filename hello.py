import torch
import torch_directml



def main():
    dml = torch_directml.device()
    print(torch.cuda.is_available())
    print("Hello from ml3!")
    # Create and add two tensors on DirectML device
    tensor1 = torch.tensor([1]).to(dml)
    tensor2 = torch.tensor([2]).to(dml)
    result = tensor1 + tensor2

    # Print result
    print(result.item())  # Should output 3

if __name__ == "__main__":
    main()

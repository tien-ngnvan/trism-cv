# ⚡ TRISM Inference Client

Perform **batch inference** on models deployed in a **Triton Inference
Server** using a Python client.

------------------------------------------------------------------------

## Introduction

`TritonModel` supports: - Connecting to Triton Server (gRPC or HTTP)
- **Multi-input** and **multi-output** models
- Automatic **batching** and **data stacking**
- Auto-generating `config.pbtxt` based on model metadata from Triton

------------------------------------------------------------------------

## Input Data Structure

The input passed to the `run()` function is a **dict**:

``` python
data = {
    "input_name_1": [array1, array2, ...],
    "input_name_2": [array1, array2, ...],
}
```

-   **Key**: must match the input name defined on the Triton Server.
-   **Value**: a list of tensors with the same shape (already padded but
    not stacked).
    The client will automatically stack, split into batches, and send to
    the server.

------------------------------------------------------------------------

## Example Usage

``` python
from trism_cv import TritonModel
import numpy as np

# Initialize client & run inference
model = TritonModel(model="model_name", version=1, url="# Triton server address", grpc=True)

# --- Case 1: Batched input ---
# data is a list of images with the same shape (HxWx3)
batched_data = [np.random.rand(640,640,3).astype(np.uint8)] * 3
outputs_batched = model.run({"INPUT": batched_data}, auto_config=True, batch_size=2) #batch_size default=2, can be customized
print("Batched output:", outputs_batched)


# --- Case 2: No batch ---
single_data = [np.random.rand(640, 640, 3).astype(np.float32)]  
outputs_single = model.run({"INPUT": single_data}, auto_config=True, batch_size=1) 
print("Single output:", outputs_single)
```

------------------------------------------------------------------------

## Output Format

The `TritonModel.run()` method can return 4 different formats depending
on the model:

| Output Type   | Batched                        | Non-batched                  |
|----------------|--------------------------------|------------------------------|
| Single output  | `list[np.ndarray]`             | `np.ndarray`                 |
| Multi-output   | `dict[str, list[np.ndarray]]`  | `dict[str, np.ndarray]`      |



Example of how to handle the output:

``` python
if isinstance(outputs, dict):
    # Multi-output model
    for name, out in outputs.items():
        for i in range(len(out) if isinstance(out, list) else 1):
            result = out[i] if isinstance(out, list) else out
            print(f"{name}: {result.shape}")
else:
    # Single-output model
    for out in (outputs if isinstance(outputs, list) else [outputs]):
        print(out.shape)
```

------------------------------------------------------------------------

## Auto Configuration (`config.pbtxt`)

`TritonModel` can automatically generate the `config.pbtxt` file by
reading metadata from the Triton Server. This helps users:

-   Avoid writing input/output configuration manually
-   Ensure compatibility with the server's model definition

------------------------------------------------------------------------

## Environment Requirements

-   Python \>= 3.9
-   opencv-python
-   numpy
-   tqdm
-   tritonclient
-   trism_cv (custom module)

Quick installation:

``` bash
pip install opencv-python numpy tqdm tritonclient[grpc]
```

------------------------------------------------------------------------

## Notes

-   All inputs must have the same shape for batching.
-   Input keys must match those defined on Triton.
-   The model must be in **READY** state on the server.
-   Multiple batches can be processed in parallel to increase
    throughput.

------------------------------------------------------------------------

## License
[GNU AGPL v3.0](LICENSE).<br>
Copyright &copy; 2025 [Tien Nguyen Van](https://github.com/tien-ngnvan). All rights reserved.

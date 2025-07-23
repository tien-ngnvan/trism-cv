# Python TRISM
The Triton Inference Server Model is a lightweight Python package designed to simplify the process of performing inference using the Triton Inference Server.
```bash
pip install trism-cv
# Or
pip install https://github.com/tien-ngnvan/trism-cv
```
## How to use
### 1. For Standard Models

```python
from trism import TritonModel

# Create triton model.
model = TritonModel(
  model="my_model",     # Model name.
  version=0,            # Model version.
  url="localhost:8001", # Triton Server URL.
  grpc=True             # Use gRPC or Http.
)

# View metadata.
for inp in model.inputs:
  print(f"name: {inp.name}, shape: {inp.shape}, datatype: {inp.dtype}\n")
for out in model.outputs:
  print(f"name: {out.name}, shape: {out.shape}, datatype: {out.dtype}\n")


## License
[GNU AGPL v3.0](LICENSE).<br>
Copyright &copy; 2025 [Tien Nguyen Van](https://github.com/tien-ngnvan). All rights reserved.



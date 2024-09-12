import numpy as np
from trism import types
import tritonclient.grpc as grpcclient
import tritonclient.http as httpclient


class InferMeta:
  """
  Keep metadata about inference input/output.
  """

  @property
  def name(self) -> str:
    return self.__name
  
  @property
  def shape(self) -> tuple:
    return self.__shape
  
  @property
  def dtype(self) -> np.dtype:
    return self.__dtype

  def __init__(self, name: str, shape: tuple, dtype: str, *args, **kwds):
    self.__name = name
    self.__shape = shape
    self.__dtype = types.triton_to_np(dtype)

  def make_input(self, client, data: np.array) -> grpcclient.InferInput | httpclient.InferInput:
    return client.InferInput(
      name=self.name,
      shape=self.shape,
      datatype=types.np_to_triton(self.dtype)
    ).set_data_from_numpy(data)
  
  def make_output(self, client) -> grpcclient.InferRequestedOutput | httpclient.InferRequestedOutput:
    return client.InferRequestOutput(name=self.name)

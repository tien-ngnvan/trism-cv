from trism import client
import numpy as np


class TritonModel:
  """
  Triton inference server model.
  """

  @property
  def model(self) -> str:
    return self._model
  
  @property
  def version(self) -> str:
    return self._version
  
  @property
  def url(self) -> str:
    return self._url
  
  @property
  def grpc(self) -> bool:
    return self._grpc
  
  @property
  def inputs(self):
    return self._inputs
  
  @property
  def outputs(self):
    return self._outputs

  def __init__(
    self,
    model: str,
    version: int,
    url: str,
    grpc: bool = True
  ) -> None:
    self._url = url
    self._grpc = grpc
    self._model = model
    self._version = str(version) if version > 0 else ""
    self._proto = client.protoclient(grpc)
    self._client = client.serverclient(url, grpc)
    self._inputs, self._outputs = client.metadata(self._client, model, version)
  
  def run(self, data: list[np.array]):
    inputs = [self._inputs[i].make_input(self._proto, data[i]) for i in range(len(self._inputs))]
    outputs = [output.make_output(self._proto) for output in self._outputs]
    results = self._client.infer(self._model, inputs, self._version, outputs)
    return {output.name: results.as_numpy(output.name) for output in self._outputs}

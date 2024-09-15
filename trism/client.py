from typing import Any
from trism.meta import Metadata
import tritonclient.grpc as grpcclient
import tritonclient.http as httpclient
from tritonclient.grpc import InferenceServerClient as GrpcClient
from tritonclient.http import InferenceServerClient as HttpClient
# NOTE: attrdict broken in python 3.10 and not maintained.
# https://github.com/wallento/wavedrompy/issues/32#issuecomment-1306701776
try:
  from attrdict import AttrDict
except ImportError:
  # Monkey patch collections
  import collections
  import collections.abc
  for type_name in collections.abc.__all__:
    setattr(collections, type_name, getattr(collections.abc, type_name))
  from attrdict import AttrDict


def protoclient(grpc: bool = True) -> Any:
  return grpcclient if grpc else httpclient


def serverclient(
    url: str, 
    grpc: bool = True, 
    concurrency: int = 10, 
    *args, 
    **kwds
) -> GrpcClient | HttpClient:
  return GrpcClient(url=url, *args, **kwds) if grpc else \
    HttpClient(url=url, concurrency=concurrency, *args, **kwds)


def metadata(
  client: GrpcClient | HttpClient,
  model: str,
  version: str = "",
  *args,
  **kwds    
) -> Any:
  meta = client.get_model_metadata(model, version, *args, **kwds)
  conf = client.get_model_config(model, version, *args, **kwds)
  if isinstance(client, GrpcClient):
    conf = conf.config
  else:
    meta, conf = AttrDict(meta), AttrDict(conf)
  inputs = [Metadata(
    name=inp.name,
    shape=inp.shape,
    dtype=inp.datatype
  ) for inp in meta.inputs]
  outputs = [Metadata(
    name=out.name,
    shape=out.shape,
    dtype=out.datatype
  ) for out in meta.outputs]
  return inputs, outputs

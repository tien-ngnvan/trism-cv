from typing import Any
from trism.meta import InferMeta
import tritonclient.grpc as grpcclient
import tritonclient.http as httpclient
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


def client(grpc: bool = True) -> Any:
  """
  Get client package.
  :param grpc:  grpc or http.
  """
  return grpcclient if grpc else httpclient


def serverclient(
    url: str, 
    grpc: bool = True, 
    concurrency: int = 10, 
    *args, 
    **kwds
) -> grpcclient.InferenceServerClient | httpclient.InferenceServerClient:
  return grpcclient.InferenceServerClient(url=url, *args, **kwds) if grpc else \
    httpclient.InferenceServerClient(url=url, concurrency=concurrency, *args, **kwds)


def infermeta(
  client: grpcclient.InferenceServerClient | httpclient.InferenceServerClient,
  model: str,
  version: str = "",
  *args,
  **kwds    
) -> Any:
  meta = client.get_model_metadata(model, version, *args, **kwds)
  conf = client.get_model_config(model, version, *args, **kwds)
  if isinstance(client, grpcclient.InferenceServerClient):
    conf = conf.config
  else:
    meta, conf = AttrDict(meta), AttrDict(conf)
  inputs = [InferMeta(
    name=inp.name,
    shape=inp.shape,
    dtype=inp.datatype
  ) for inp in meta.inputs]
  outputs = [InferMeta(
    name=out.name,
    shape=out.shape,
    dtype=out.datatype
  ) for out in meta.outputs]
  return inputs, outputs

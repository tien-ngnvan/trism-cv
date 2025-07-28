import inspect
from typing import Any, Union

import tritonclient.grpc as grpcclient
import tritonclient.http as httpclient
import tritonclient.grpc.aio as grpcclient_aio

from tritonclient.grpc import InferenceServerClient as GrpcClient
from tritonclient.grpc.aio import InferenceServerClient as GrpcClientAsync
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

from trism_cv.inout import Inout



def protoclient(grpc: bool = True) -> Any:
    return grpcclient if grpc else httpclient


def protoclient_async(grpc: bool = True) -> Any:
    return grpcclient_aio if grpc else httpclient


def serverclient(
    url: str,
    grpc: bool = True,
    concurrency: int = 10,
    *args,
    **kwargs
) -> Union[GrpcClient, HttpClient]:
    """Return synchronous client instance."""
    return GrpcClient(url=url, *args, **kwargs) if grpc else \
        HttpClient(url=url, concurrency=concurrency, *args, **kwargs)

def serverclient_async(
    url: str,
    grpc: bool = True,
    concurrency: int = 10,
    async_mode: bool = False,
    *args,
    **kwargs
) -> Union[GrpcClient, GrpcClientAsync, HttpClient]:
    """Return asynchronous or synchronous client based on mode."""
    if grpc:
        return GrpcClientAsync(url=url, *args, **kwargs) if async_mode \
            else GrpcClient(url=url, *args, **kwargs)
    return HttpClient(url=url, concurrency=concurrency, *args, **kwargs)


def get_inout(
    meta: Any,
    conf: Any
) -> tuple[list[Inout], list[Inout]]:
    """Extract input/output definitions from model metadata/config."""
    if hasattr(conf, "config"):
        conf = conf.config

    inputs = [
        Inout(name=inp.name, shape=inp.shape, dtype=inp.datatype)
        for inp in meta.inputs
    ]
    outputs = [
        Inout(name=out.name, shape=out.shape, dtype=out.datatype)
        for out in meta.outputs
    ]
    return inputs, outputs


def inout(
    serverclient: Union[GrpcClient, HttpClient],
    model: str,
    version: str = "",
    *args,
    **kwargs
) -> tuple[list[Inout], list[Inout]]:
    """Fetch model metadata and config, return input/output specs."""
    meta = serverclient.get_model_metadata(model, version, *args, **kwargs)
    conf = serverclient.get_model_config(model, version, *args, **kwargs)

    if not isinstance(serverclient, GrpcClient):
        meta, conf = AttrDict(meta), AttrDict(conf)

    return get_inout(meta, conf)


async def iout_async(
    serverclient_async: Any,
    model: str,
    version: str = "",
    *args,
    **kwargs
) -> tuple[list[Inout], list[Inout]]:
    """Async version of `inout()` with coroutine-aware handling."""
    if inspect.iscoroutinefunction(serverclient_async.get_model_metadata):
        meta = await serverclient_async.get_model_metadata(model, version, *args, **kwargs)
        conf = await serverclient_async.get_model_config(model, version, *args, **kwargs)
    else:
        meta = serverclient_async.get_model_metadata(model, version, *args, **kwargs)
        conf = serverclient_async.get_model_config(model, version, *args, **kwargs)

    return get_inout(meta, conf)

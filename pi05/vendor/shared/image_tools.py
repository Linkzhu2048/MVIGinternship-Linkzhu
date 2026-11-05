"""保留官方 JAX 和 PIL 图像等比例缩放补边，用于模型预处理和数据适配。"""

import functools
import jax
import jax.numpy as jnp
import openpi.shared.array_typing as at
import numpy as np
from PIL import Image


@functools.partial(jax.jit, static_argnums=(1, 2, 3))
@at.typecheck
def resize_with_pad(
    images: at.UInt8[at.Array, "*b h w c"] | at.Float[at.Array, "*b h w c"],
    height: int,
    width: int,
    method: jax.image.ResizeMethod = jax.image.ResizeMethod.LINEAR,
) -> at.UInt8[at.Array, "*b {height} {width} c"] | at.Float[at.Array, "*b {height} {width} c"]:
    """使用 JAX 等比例缩放并居中补边。

    输入：images、height、width、method。
    输出：保持输入批次与数值范围的目标尺寸图像。"""
    has_batch_dim = images.ndim == 4
    if not has_batch_dim:
        images = images[None]
    cur_height, cur_width = images.shape[1:3]
    ratio = max(cur_width / width, cur_height / height)
    resized_height = int(cur_height / ratio)
    resized_width = int(cur_width / ratio)
    resized_images = jax.image.resize(
        images, (images.shape[0], resized_height, resized_width, images.shape[3]), method=method
    )
    if images.dtype == jnp.uint8:
        resized_images = jnp.round(resized_images).clip(0, 255).astype(jnp.uint8)
    elif images.dtype == jnp.float32:
        resized_images = resized_images.clip(-1.0, 1.0)
    else:
        raise ValueError(f"Unsupported image dtype: {images.dtype}")
    pad_h0, remainder_h = divmod(height - resized_height, 2)
    pad_h1 = pad_h0 + remainder_h
    pad_w0, remainder_w = divmod(width - resized_width, 2)
    pad_w1 = pad_w0 + remainder_w
    padded_images = jnp.pad(
        resized_images,
        ((0, 0), (pad_h0, pad_h1), (pad_w0, pad_w1), (0, 0)),
        constant_values=0 if images.dtype == jnp.uint8 else -1.0,
    )
    if not has_batch_dim:
        padded_images = padded_images[0]
    return padded_images


def resize_with_pad_numpy(images: np.ndarray, height: int, width: int, method=Image.BILINEAR) -> np.ndarray:
    """使用 PIL 双线性缩放并居中补边。

    输入：images、height、width、method。
    输出：目标尺寸的 NumPy 图像。"""
    if images.shape[-3:-1] == (height, width):
        return images
    original_shape = images.shape
    images = images.reshape(-1, *original_shape[-3:])
    resized = np.stack(
        [_resize_with_pad_pil(Image.fromarray(im), height, width, method=method) for im in images]
    )
    return resized.reshape(*original_shape[:-3], *resized.shape[-3:])


def _resize_with_pad_pil(image: Image.Image, height: int, width: int, method: int) -> Image.Image:
    """对单张 PIL 图像等比例缩放后黑色补边。

    输入：image、height、width、method。
    输出：目标尺寸的 PIL 图像。"""
    cur_width, cur_height = image.size
    if cur_width == width and cur_height == height:
        return image
    ratio = max(cur_width / width, cur_height / height)
    resized_height = int(cur_height / ratio)
    resized_width = int(cur_width / ratio)
    resized_image = image.resize((resized_width, resized_height), resample=method)
    zero_image = Image.new(resized_image.mode, (width, height), 0)
    pad_height = max(0, int((height - resized_height) / 2))
    pad_width = max(0, int((width - resized_width) / 2))
    zero_image.paste(resized_image, (pad_width, pad_height))
    assert zero_image.size == (width, height)
    return zero_image

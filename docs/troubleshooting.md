## Try `--disable-fa`

By default, **stable-diffusion.cpp** uses Flash Attention to improve generation speed and optimize GPU memory usage. However, on some backends, Flash Attention may cause unexpected issues, such as generating completely black images. In such cases, you can try disabling Flash Attention by using `--disable-fa`.
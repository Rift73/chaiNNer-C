# zlib 1.3.2

Upstream: https://zlib.net/, tarball `https://zlib.net/zlib-1.3.2.tar.gz`, SHA-256
`bb329a0a2cd0274d05519d61c667c062e06990d72e125ee2dfa8de64f0119d16`, verified against Mark Adler's signature
(`.asc`, key 5ED4 6A67 21D3 6558 7791 E2AA 783F CD8E 58BC AFBA).

The files here are unmodified copies of the tarball's: the deflate and inflate sources and headers the static
library needs (no gz* file I/O), `LICENSE` and `README`. OpenCV 5.0.0's `3rdparty/zlib` is the same release plus
two patches: an initialised checksum in `inflate.c` (no effect on a valid stream) and warning fixes; this copy keeps
upstream's file. Classic zlib, not zlib-ng: OpenCV's PNG and TIFF encoders deflate with it, and zlib-ng (Pillow's,
CPython's) writes other streams.

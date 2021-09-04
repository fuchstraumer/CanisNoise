# CanisNoise

Based on my old [CUDA_Noise](https://github.com/fuchstraumer/CUDA_Noise) project, CanisNoise aims to recreate the successes of that project but using Vulkan instead. So no Vendor locking, and because I'm not entirely sold on CUDA as an API (to say the least).

It's WIP as hell right now and being torn down nearly completely but the end goal is:

1. Have a backend that can be linked into other projects as a DLL, allowing for noise generation and handling the dependencies/chaining of dispatches between modules (plus some memory work, too!)
2. Have a frontend demo using ImGui + some nodegraph magic, so that users can experiment with the right layout/settings for their noise chains. Ideally, I'll let these be serialized to/from JSON so that a setup created in the GUI can be used with the library ^^

Currently: developing a nodegraph system, and using this to help refine and develop my interface and API for [Petrichor](https://github.com/fuchstraumer/Petrichor). First steps on making something good - make something bad, first. Namely, a bad nodegraph system :)

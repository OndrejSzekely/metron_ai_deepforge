# Project Installation

## Installation Steps

::::{admonition} Attention
:class: warning

```{eval-rst}
|:exclamation:| Docker is required. |:exclamation:|
```

::::

Follow the instruction steps.

1. Download the repository using

    ```shell
    git clone --recurse-submodules https://github.com/OndrejSzekely/metron_ai_deepforge.git
    ```

2. Log in into *NVIDIA NGC* *Docker* registries using this [guide](https://ngc.nvidia.com/setup/api-key).
3. Go into repository root folder and build all supported images using:

    ```shell
    ./build_images.sh
    ```

    It will build following docker images:
        - `metron_ai/deepforge_pt`: *TensorFlow* based Docker image.
        - `metron_ai/deepforge_tf`: *PyTorch* based Docker image.
  
4. Then run a container using following command:

    ```shell
     ./run_container.sh -b=<BACKEND_TYPE>  
    ```

    where `<BACKEND_TYPE>` can be of following options:
        - `pt` for `metron_ai/deepforge_pt` image
        - `tf` for `metron_ai/deepforge_tf` image
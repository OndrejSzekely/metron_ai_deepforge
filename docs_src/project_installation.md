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
3. Go into repository root folder and run
    ```shell
    ./build_container.sh
    ```

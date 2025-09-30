# Project Structure

```shell
├── config # Contains component's config files in Hydra framework and DeepForge's Pythons setting files.
├── config_schema # Contains component's Structured Config schema files (See https://hydra.cc/docs/tutorials/structured_config/schema/) used for config files type validation.
├── docs_src # Contains documentation source files.
├── forge # Contains custom reusable building blocks.
├── hydra_plugin # Stores Hydra plugins, for instance plugin for multiple config search paths.
├── tests # Tests implementation folder.
```

## New Use Case Skeleton

Use case represents a new programme/initiative/project/..., i.e. a high level project structure which could organized into smaller, more granular sub-projects. It is located in `usecases` folder.

Therefore, to create a new use case, we must create a new folder in `usecases` with with sub-project folders if needed.

In next sections, it is not distinguished between project or sub-project level.

### Required Structure

```shell
├── train.py
├── config.yaml
```

#### *train.py*

Represents a training business logic of the project.

:::{attention}
Training script **must** use *Hydra* configuration framework and script's "main" function must be decorated with following decorator
with the **same** values - `config_path="."` and `config_name="config"`.

```shell 
@hydra.main(version_base="1.3", config_path=".", config_name="config")
```

:::

#### *config.yaml*

Stores project's configuration. It shall be loaded by all project's Python's scripts. It compose configuration from component's configuration files stored in `config` folder.
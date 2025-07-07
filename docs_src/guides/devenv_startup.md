# Devcontainer Startup Guide

*DeepForge* requires following variables available in your host environment to run devcontainer:

- `DATASTORE_PATH`: A path to a dataset folder on the host machine. Pointed folder is mounted to the devcontainer on
path `/mnt/datastore/`. 

:::{attention}
There is no general *VSCODE* environmental variables management. Variables must be available in the shell which starts
a *VSCODE* instance.
:::

It is not recommended to hardcode variables into *shell* configuration file (e.g. `./basrc`). Instead, use **direnv**,
a *dotenv* management tool, which searches for `.envrc` and `.env` files in given folder and loads the varaiables.

:::{tip}
It is recommended to store required host machine environmental variables in `.env` file on the root level of
*DeepForge*. `.env` file is not tracked by Git.
:::

### *direnv* installation steps

1. Build from the latest main and install via following command:
    ```shell
    curl -sfL https://direnv.net/install.sh | bash
    ```
2. Hook *direnv* into the shell [using this guide](https://direnv.net/docs/hook.html).
3. Get expected *direnv* config location (`DIRENV_CONFIG`) via command:
    ```shell
    direnv status
    ```
4. Create a `direnv` folder expected on the `DIRENV_CONFIG` path.
5. Inside the folder create a `direnv.toml` file with following context:
    ```toml
    [global]
    load_dotenv = true
    ```
6. Sing-out & sign-in or restart the machine.

**Do not run VSCODE from desktop via icon!** *You must open a shell, go to the root DeepForge folder and inside the
folder run* `code .`.
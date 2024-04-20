Release Page
============

.. mermaid::
    :align: center

    gitGraph
        commit id: "init"
        branch develop
        checkout develop
        commit id: "0.1.0 dev"
        checkout main
        merge develop tag: "v0.1.0"

v0.1.0
******

* Devcontainer development setup.
* *TensorFlow* and *PyTorch* support.

# Development Notes

## Start Development Environment

The project is designed fot *Visual Studio Code* development environment.

1. Start *VSCODE*.

2. Press *Ctrl + Shift + P* and search for *Dev Containers: Open Folder in Container...* and click on it.

3. Select dev env, either PyTorch or Tensorflow.

## Programming Principles

### Parameters Validation

It is assumed that all functions and class methods follow
[precondition principle](https://en.wikipedia.org/wiki/Precondition), specifically demanding approach. Demanding
approach states, that part of function or class method logic should be a pre-section which does the validation of all
input parameters and prevents code to be injected by wrong arguments. This also offloads the responsibility from user to
the function / class method itself.

## Python Guidelines

**Max Line Length**: It is set to 150 characters.

### Commits Validation

Every commit has to pass through and be compliant with following tools:

- *Ruff* (linter & formatter)
- *PyRight* (type check)
- *Bandit* (vulnerabilities check)

The default configuration/standards of the tools is slightly modified generally and occasionally, there is a particular
PEP message disabling on a line level in the code.

Tools could be installed manually by a user or **using *pre-commit*** tool which integrates all the tools.
**This is also the preferred way.** See [Project Installation](../docs/project_installation.md) for more details about
*pre-commit* installation and location of modified configuration files.

### Docstring

It is used Google format.

To improve docstring readability, following highlight types are assumed:

- `<>`: Used to highlight `<variable>`, `<class>`, `<class.method()>`, `<class.attribute>`, `<function()>`, etc...
- `**`: Used to highlight Metron's modules (e.g. `*Video Streamer*`) and terms to emphasize that wording represents a
  specific solution's piece.
- ``` `` ```: Used to highlight source code blocks and commands (
  e.g. `` `from metron_conduit.source_connector import *` ``), values (e.g. `` `-1` ``, `` `1.44` ``
  or `` `"string_value"` ``) or paths (e.g. `` `docs/images` ``).

## Python Dependencies

[*uv* package manager](https://astral.sh/blog/uv) is used.

## YAML Configs

### Comments

YAML comments convention should follow [Python docstring](#docstring) convention.

### Config Values Validation

Metron does not use any main all YAML config parameters validation routine. Hydra does validation of expected config
parameters and their type. In the code, configuration variable is accessed whenever is needed using specific decorator,
applied on a function or class method. It passes the configuration variable as the first parameter in the function /
class method. Therefore, the same principle as in [Parameters Validation](#parameters-validation) is followed. The
function / class method has to validate required config parameter values and if the requirements are not met,
raise `ValueError` with custom message referring to the particular YAML config parameter.

## Diagram Drawings

All Metron diagrams are drawn using [diagrams.net](https://www.diagrams.net) tool. Diagram projects are found in
`docs/diagram_projects`.

## Design

### Colour Palette

The colour palette is given by following colours:

- orange: #F9895C
- blue: #65BCD8
- yellow: #F8D643
- dark grey: #545A5E

### Fonts

As a theme
font, [QUANTUM](https://www.behance.net/gallery/63174797/QUANTUM-FREE-FONT?tracking_source=project_owner_other_projects) ([download link](https://www.dafont.com/quantum-4.font))
font is used. This font should be used for visually appealing diagrams or headlines.

*Tahoma* is used as a standard font everywhere else.
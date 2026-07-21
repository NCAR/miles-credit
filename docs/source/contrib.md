# Contributing to CREDIT

## Introduction

Your interest in contributing to CREDIT is a *credit* to your (user) name\! CREDIT is intended to be a community platform for ML Earth System Prediction, so community contributions are highly encouraged. We appreciate all forms of support, including running CREDIT on your own system, reporting bugs or performance issues, requesting additional or new functionality, improving our documentation, promoting CREDIT to your community, or contributing your own code, models, or datasets. 

If you have any questions, please feel free to reach out to us on [GitHub Issues](https://github.com/NCAR/miles-credit/issues) (for bugs or feature requests) or [Github Discussions](https://github.com/NCAR/miles-credit/discussions) (for more open-ended discussions about CREDIT features or strategic directions). You can also reach us by email at [miles@ucar.edu](mailto:miles@ucar.edu).

## Where to start

If you are interested in contributing to the core CREDIT repository, first reach out to the MILES team at [miles@ucar.edu](mailto:miles@ucar.edu) to set up an introductory meeting and discuss your interests and what would be a good focus area. 

Once we have agreed on a focus area, please start issues on the [CREDIT Issues](https://github.com/NCAR/miles-credit/issues) page so that we can track progress on our internal project board and version Milestone tracker. 

If you are interested in working on an existing issue, please comment on the issue to let us know you are working on it. This will help us avoid duplicate work.

The code for CREDIT is hosted on GitHub. If you do not have one, you will need to create a [free GitHub account](https://github.com/signup). The [GitHub Quickstart Guide](https://docs.github.com/en/get-started/start-your-journey) is a great place to get started with git and GitHub.

## Reporting bugs

Something not working as expected? We would love to hear about it! Please report any bugs you find by opening an issue on GitHub. 

When reporting a bug, please include as much information as possible. This will help us reproduce the bug and fix it efficiently. For more information on how to write a good bug report, see this Stack Overflow post on [how to make a good bug report](https://stackoverflow.com/help/minimal-reproducible-example).  
To increase the helpfulness of the bug report, please include the following information:

* Compute platform (e.g., Casper, Derecho, Macbook Pro, AWS VM, etc.)  
* Whether issue is on CPU or GPU and what kind of GPU is being used.  
* Version of PyTorch, CUDA toolkit, NCCL (if applicable), other relevant libraries.  
* What stage of ML process (processing, training, inference).

## Requesting new features

Have an idea for a new feature? Please let us know in the [Issues](https://github.com/NCAR/miles-credit/issues) section. We will review and prioritize feature requests based on available bandwidth.

## Improving Documentation

We are always looking for ways to improve our documentation. If you find something that is unclear or confusing, please let us know by [opening an issue](https://github.com/NCAR/miles-credit/issues/new/choose). To contribute to our documentation yourself, see the [Documentation](#documentation) section of this guide.

## Development workflow overview

This is a brief overview of the development workflow we use for CREDIT. A more detailed description of each step is provided in the following sections.

**Get set up to develop on your local machine or the HPC systems**

1. Clone the repository.  
2. Create a development environment.  
3. Create a branch for your changes.  
4. Install pre-commit hooks.

**Make your changes**

1. Understanding the codebase.  
2. Write and run tests.  
3. [Generate](#generate-the-documentation-locally) and [check](#check-the-documentation) the documentation.

**Contribute your code**

1. Push your changes to your branch.  
2. Open a pull request.  
3. Address feedback.  
4. Wait for your pull request to be merged.  
5. Delete your branch.

## Get set up to develop on your local machine

### Internal developers and close collaborators: Clone and create a branch

CREDIT developers at NCAR and close collaborators (e.g., interns, visitors, partners on funded research projects) should manage their code contributions through branches in the `NCAR/miles-credit` Github repository. 

### External contributors: Fork and clone the repository

Get started by forking the `NCAR/miles-credit` repository on GitHub. To do this, find the "Fork" button near the top of the page and click it. This will create a copy of the project under your personal github account.

Next, clone your forked copy to your local machine.

```bash  
git clone https://github.com/your-user-name/miles-credit.git  
```

Enter the project folder and set the upstream remote to the NCAR/miles-credit repository. This will allow you to keep your fork up to date with the main repository.

```bash  
cd miles-credit  
git remote add upstream https://github.com/NCAR/miles-credit.git  
```

For more information, see the [GitHub quickstart section on forking a repository](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/fork-a-repo).

### Create a development environment

To run and test any changes you make in CREDIT, you will need to create a development environment. We recommend installing and using [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html) and/or [mamba](https://mamba.readthedocs.io/en/latest/installation/mamba-installation.html).

Use the following commands to create a new conda environment to develop CREDIT in.

```bash  
# Create a new conda environment  
conda create -c conda-forge -n credit python=3.13

# Activate your new environment  
conda activate credit

# Install your local copy of miles-credit in interactive mode  
pip install -e ".[develop]"  
```

To test your new install, open a python session and try importing `credit`. You can also try printing the version number, which should be unique to the latest commit on your fork.

```python  
import credit  
print(credit.__version__) 
# '2026.2.0'  
```

### Create a branch for your changes

We highly recommend creating a new branch on your fork for each new feature or bug that you work on.

To create and check out a new branch, use the following command:

```bash  
git checkout -b <branch-name>  
```

Track upstream changes for `git pull` and `git push`

```bash  
git branch --set-upstream-to=origin/<branch-name>\ <branch-name>  
```

You can see a list of all branches in your local repository by running:

```bash  
git branch  
```

For more information on branching, check out this [learn git branching](https://learngitbranching.js.org/) interactive tool.

### Install pre-commit hooks

CREDIT uses pre-commit hooks to ensure a standardized base-level code formatting and style. We strive to follow PEP8 Python standards.

The `pre-commit` package is installed as part of the `develop` set of optional dependencies. To set up the pre-commit hooks to run before every git commit, run the following command from the root of the repository:

```bash  
pre-commit install  
```

Now, whenever you commit changes, the pre-commit hooks will run and may make small modifications to your code. If the pre-commit hooks make any changes, you will need to re-add the files and commit them again in order to successfully make the commit.

To manually run the pre-commit hooks, use the following command:

```bash  
pre-commit run --all-files  
```

You can skip the pre-commit hooks by adding the `--no-verify` flag to your commit command like this:

```bash  
git commit -m "your commit message" --no-verify  
```

For more information on pre-commit hooks, see the [pre-commit documentation](https://pre-commit.com/).

Pre-commit also runs through github actions as a continuous integration check through [pre-commit.ci](http://pre-commit.ci). This pre-commit will perform auto syntax checking and safe reformatting and will fail if safe reformats are not feasible. 

## Understanding the CREDIT Codebase

Please read the CREDIT User Guide to understand the structure of the code in more detail, especially sections related to what you want to work on.

## Feature Integration Process

CREDIT Gen 2 now allows users to create and integrate new datasets, pre and post blocks, models, and losses as custom objects without inserting them into the CREDIT codebase. Before asking to include your new code into CREDIT, first register it as a custom object in your config file. 

If tests with the custom object are successful, and the feature has broader applicability beyond your individual use case, please consider adding it to the main CREDIT codebase as an option.

1. Add your new module under the appropriate CREDIT section folder.   
2. In the `__init__.py` file for each section, edit the `REGISTRY` and `CLASS_SOURCE` dictionaries to add the path to your module and the relevant class in that module.  
3. Add a test file and relevant tests to the tests directory.  
4. Open a PR.

In some cases, new features may be considered as the default option for future CREDIT releases. Before the default is changed, feature flags will be used to optionally change defaults in new releases prior to defaults officially being changed in following releases.

## Make your changes

After you're all set up to develop CREDIT, you can start making your changes. This section describes where, how, and what to change to add your contributions to the CREDIT codebase.

### LLM Coding Assistant Usage

The CREDIT development team recognizes the power of LLMs for aiding in software development and is increasingly incorporating them into our development practices. We are also aware of the limitations of these tools and encourage you to follow these guidelines:

1. Generate your own [AGENTS.md](http://AGENTS.md) or [CLAUDE.md](http://CLAUDE.md) file for the repository. Review the file and add instructions related to which conda environment to use. To avoid LLM claiming authorship of commits since LLMs cannot take responsibility for code, edit your guidance file as follows:   
   ```

   ## Commit Guidelines

   - Do NOT include any AI authorship, attribution, or co-author lines.

   - Never append "Co-Authored-By: Claude" or "Generated by Claude" to messages.

    ```

2. Before engaging with an LLM, write a clear, specific goal and specification for the coding task. Include specific details about what the expected functionality should be and how the functionality is to be tested.   
3. Make sure your code is committed and pushed and in a separate branch to provide backups against accidental deletions. If the code is accessing data files, make sure they are read-only from the LLM’s perspective or that you have a protected backup.   
4. Tag relevant files in the repository using the @ symbol to provide the model style and functionality context. Add instructions to “ask clarifying questions” in the event you do not provide enough context.  
5. Have some real data or other code to test against. Encourage the LLM to also make plots for visualization if appropriate.   
6. LLMs have a tendency to add local private functions to a module rather than importing from elsewhere in the repository, so double check that suggested additions are not already implemented elsewhere in the code.  
7. Give the LLM high standards for passing and have it iterate on running code against tests. Make sure the tests cover a broad range of use cases.   
8. When trying to minimize usage costs, use a larger LLM for planning (e.g., Claude Opus 4.8) and a smaller LLM for execution (e.g., Claude Sonnet 5). You can even instruct the larger LLMs to out source their code generation to sub agents, including local open weight LLMs.   
9. Acknowledge LLM usage in a PR, and list which LLMs were used and their versions.  
10. Ask for a summary of changes in markdown format that you can copy into your PR.  
11. Go ahead and submit a PR for the individual session rather than stringing many sessions together into one giant PR.

### Write and run tests

`miles-credit` uses \[pytest\](https://docs.pytest.org/en/stable/) for unit tests, so we encourage you to write new tests using `pytest` as well. 

A good unit test runs a function or method and validates that the code produces the expected set of outputs given a certain set of inputs. At the very least, test functions should cover a commonly expected set of inputs as well as some expected edge cases and calls that have resulted in bugs or prior issues. Unit tests should run quickly, so providing the test with the minimum viable amount of data or small models. 

To run the tests locally, use the following command from the root of the repository:

```bash  
pytest  
```

To run a specific test, use the following command:

```bash  
pytest tests/test_mod.py::test_func  
```

These tests will also run automatically when you open a pull request using GitHub Actions and the `.github/workflows/ci.yml` file.

While unit tests cover individual functions and classes, some errors arise from the closely integrated nature of auto-regressive emulators and errors that grow through time. Some errors also only occur on GPUs or through multi-node communication. We perform integration tests through the NSF NCAR CIRRUS on-premises cloud. Multi-GPU tests have to be performed manually on Derecho or other large HPC systems. See the tests/manual directory for examples of these manual multi-node tests and how to write your own if necessary.

See the [pytest documentation](https://docs.pytest.org/en/stable/) for more information about writing and running tests.

## Documentation

`miles-credit` uses [sphinx](https://www.sphinx-doc.org/en/master/) and [ReadTheDocs](https://docs.readthedocs.io/en/stable/) to build and host the documentation.

### Docstrings

The most common situation in which you will need to add to the documentation is through docstrings.

`miles-credit` uses [Google](https://google.github.io/styleguide/pyguide.html\#38-comments-and-docstrings)-style docstrings. Please follow the format of short-summary, longer description, short code example, and list of arguments with associated types. When possible, also use types for input arguments to make consistency checking easier. 

### Editing other documentation files

We welcome changes and improvements to all parts of our documentation (including this guide)! You can find these files in the `docs` directory.

These files are mainly written in [reStructuredText](https://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html), but additional file types such as `.md` and `.ipynb` are also used.

Important documentation files to know about include:

* `docs/source/index.rst`: This file is the main page of the documentation. Files added to `toctree` blocks in this file will be included in the documentation as top-level subpages.

* `docs/source/contrib.md`: This file is the source for this guide\!

* `docs/source/conf.py`: This file contains the configuration for building the documentation.

See the [sphinx documentation](https://www.sphinx-doc.org/en/master/) for more information about writing sphinx documentation.

### Generate the documentation locally

To generate the documentation locally, follow the steps below.

1. Activate the credit environment.  
2. Enter the `docs` directory.  
3. Run `make html` or to build the documentation.  
4. Open `docs/_build/html/index.html` in your browser to view the documentation.

### Check the documentation

As well as checking local documentation generation, you should also check the preview documentation generated as part of a PR. To do this, scroll down to the "checks" section of the PR and click on the "Details" link next to the "docs/readthedocs.org:miles-credit" check. This will take you to the corresponding build on ReadTheDocs, where you can view the documentation built from your PR and see any warnings or errors on your build. ReadTheDocs will highlight tracked changes. 

## Contribute your code

Once you have prepared your changes and are ready for them to be reviewed by the CREDIT development team, you can open a pull request. This section describes how to open a pull request and what to expect after you open it.

### Push your changes to your branch or fork

Once you have made your changes locally, you will need to push them to your branch on your fork on GitHub. To do this, use the following command:

```bash  
git push  
```

From here, you can request that your changes be merged into the main repository in the form of a pull request.

### Open a pull request

GitHub has extensive [pull request guides and documentation](https://docs.github.com/en/pull-requests) that we recommend. This section describes the basics for our workflow.

From your branch on your fork, open the "Pull requests" tab and click the "New pull request" button. Make sure the "base repository" is "NCAR/miles-credit" and the "base" branch is set to "main", with the "compare" branch set to your prepared branch, respectively.

From this page, you can see a view of the changes you have made in your branch. Make sure all relevant files have been added and committed.

We recommend adding a short, descriptive title to your pull request. The body of the pull request should autofill with our pull request template, which has its own set of directions. Please fill out the relevant sections of the template, including adding a more detailed description of your changes. You can use an LLM to summarize your changes in markdown format and add the description here. Please review and edit for human-readability, including defining jargon. 

Please also link the associated issue with the PR with the phrase `Closes \#ZZZ`. Once the PR is merged, the related issues will also close automatically. 

Once you have filled out the template, click the "Create pull request" button. This will open your pull request on the CREDIT repository.

If you want to receive feedback on code changes but are not ready for them to be merged in a PR, you can post a comment to a relevant issue and link your branch. You can also highlight individual code blocks and right-click to “Get permalink” that will then show up in the comment as a code snippet.

### Pull Request Review Checklist

For those reviewing pull requests, please follow the checklist below to ensure a comprehensive review. This checklist is auto-added to every PR, and please check each box once you have verified.

Required

- [ ] There is a clear description of what issue or feature the pull request is addressing.  
- [ ] Issues covered by the pull request are tagged in the description.  
- [ ] All CI checks and tests pass.  
- [ ] Tests have been updated to cover code changes.  
- [ ] Documentation has been updated to cover code changes and renders properly.  
- [ ] The dependency lists in pyproject.toml and requirements.txt have been updated.   
- [ ] Paths in code and example config files are user generic. Data not included in the repository should have a cloud URL or be located in /glade/campaign/aiml/credit/.   
- [ ] The reviewer has provided both positive and constructive feedback in their review response.  
- [ ] Changes affecting GPU-related functionality have been run by the submitter on Casper and/or Derecho to verify that the code runs as expected.

Recommended

- [ ] Updated public facing methods have full docstrings.  
- [ ] Variable names balance being unambiguous and concise.  
- [ ] Comments have been added to describe more complex operations.  
- [ ] Code minimizes redundancy with the use of loops, function/method calls, and robust data structures.  
- [ ] If dependencies are added, they should not burden the user with additional installation steps or break other parts of the code.  
- [ ] If dependencies are removed, changes to existing code should be tested and any issues addressed.  
- [ ] Tests cover expected outcomes of a function and known edge cases.  
- [ ] Type annotation is added to inputs and outputs of functions/methods.

### Address feedback

After you open your pull request, the CREDIT team will review it and may provide feedback like asking for changes or suggesting improvements. You can address this feedback by making changes to your branch and pushing them to your branch. The pull request will automatically update with your changes. Please also comment with your changes. 

The CREDIT team appreciates your contributions and the interactive process of reviewing pull requests, and will do our best to review your pull request in a timely manner. It is totally normal to have to make several rounds of changes to your pull request before it is ready to be merged, especially if you are new to the project.

Once your pull request is approved by a core maintainer and passes the relevant checks, it will be merged into the main repository!

### Delete your branch

We recommend deleting your branch after your pull request is merged. This will help keep the repo clean and organized, but is not required.

# Backwards Compatibility

In the transition from Gen 1 to Gen 2, CREDIT aims to maintain backwards compatibility with Gen 1-trained models. In v2026.2.0, Gen 1 data processing and application features will still be present, but they will be deprecated sometime in 2027. Before formal deprecation in a future release, any potentially deprecated classes will have a deprecation warning added to them for at least 2 releases prior to formal deprecation. If you continue to use a potentially deprecated feature, please create an Issue to let us know about a potential conflict. We will work with you to transition smoothly to the new functionality and ensure your use case is still supported. 
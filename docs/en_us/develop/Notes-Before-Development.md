# Notes Before Development

> [!NOTE]
>
> You only need to read this page if you want to take part in the development of M9A!
>
> Users should refer to the [M9A User Manual](../manual/newbie.md).
>
> For developing MaaFramework or your own projects, please visit [MaaXYZ/MaaFramework](https://github.com/MaaXYZ/MaaFramework).

## A Brief Overview of the GitHub Pull Request Process

### I don't know programming and just want to make small changes to JSON files/documents. What should I do?

Check out the [GitHub Pull Request Guide for Beginners](https://maa.plus/docs/zh-cn/develop/pr-tutorial.html).

### I have programming experience but haven't participated in related projects. What should I do?

1. If you forked the repository a long time ago, go to `Settings` in your repository, scroll to the bottom, and delete it.

2. Open the [M9A Main Repository](https://github.com/MaaXYZ/M9A), click `Fork`, and then click `Create fork`.

3. Clone your own repository locally and pull the submodules:

    ```bash
    git clone --recursive https://github.com/<your-username>/M9A.git
    ```

    > ⚠
    >
    > **Do not forget `--recursive`! Do not forget `--recursive`! Do not forget `--recursive`!**
    >
    > OCR failures are often caused by forgetting to include `--recursive`.

4. Download the [Release Package](https://github.com/MaaXYZ/MaaFramework/releases) of MaaFramework and extract it into the `deps` folder.

5. Set up the development environment:

    - Download and install VSCode.
    - Optionally install debugging/development tools:

        | Tool | Description |
        | --- | --- |
        | [MaaDebugger](https://github.com/MaaXYZ/MaaDebugger) | Standalone debugging tool |
        | [Maa Pipeline Support](https://marketplace.visualstudio.com/items?itemName=nekosu.maa-support) | VSCode plugin for debugging, screenshots, ROI extraction, color picking, etc. |
        | [ImageCropper](https://github.com/MaaXYZ/MaaFramework/tree/main/tools/ImageCropper) | Standalone tool for screenshots and ROI extraction |
        | [MFA Tools(Not recommended)](https://github.com/SweetSmellFox/MFATools) | Standalone tool for screenshots, ROI extraction, and color picking |

6. Local installation

   1. Install embedded python (only required for Windows)

      ```bash
      ./tools/ci/setup_embed_python.ps1
      ```

   2. Assemble components by running install

      ```bash
      python ./tools/install.py
      ```

   Then you can run M9A in the install folder.

7. Start developing:

    Enjoy coding! Before starting, check out the [Related Reading](#related-reading).

8. Git operations:

    The most commonly used basic commands are:
    - `git add <file>`: Add files to the staging area. Use `*` to represent all files.
    - `git commit -m "message"`: Commit the staged files to the local repository. Please follow the [Conventional Commits Specification](https://www.conventionalcommits.org/en/v1.0.0/) for clear commit messages.
    - `git pull origin <branch>`: Pull updates from the remote repository to the local repository.
    - `git push origin <branch>`: Push local changes to the remote repository.

    > ⚠
    >
    > During development, remember to commit changes regularly with a message.
    > If you're not familiar with Git, you may need to create and switch to a new branch instead of committing directly to `main`.
    > This way, your commits will grow on the new branch without being affected by updates to `main`.

    ```bash
    git checkout -b <branch-name> # Create and switch to a new branch
    ```

    After development, push your modified local branch to the remote repository (your fork):

    ```bash
    git push origin <branch-name>
    ```

    If there are changes in the M9A repository (e.g., commits from others), you may need to sync these changes to your branch:

    1. Link the original M9A repository:

        ```bash
        git remote add upstream https://github.com/MaaXYZ/M9A.git
        ```

    2. Fetch updates from the remote repository:

        ```bash
        git fetch upstream
        ```

    3. Rebase (recommended) or merge the changes:

        ```bash
        git rebase upstream/main # Rebase for a cleaner commit history. Rebase is recommended over merge when completing your personal PR.
        ```

        Or:

        ```bash
        git merge upstream/main
        ```

    Git reference materials:
    - [Git Official Documentation](https://git-scm.com/docs)
    - [Git Simplified Guide](https://www.runoob.com/manual/git-guide/)
    - [Git Tutorial | Runoob](https://www.runoob.com/git/git-tutorial.html)

9. Submit a Pull Request:

    Your modified code has been committed to your repository. Now, you need to submit a Pull Request to the M9A repository and wait for the maintainers to review it.

    [GitHub Pull Request Reference](https://maa.plus/docs/zh-cn/develop/pr-tutorial.html)

## M9A Formatting Requirements

M9A uses a series of formatting tools to ensure that the code and resource files in the repository are clean and consistent, making them easier to maintain and read.

Please ensure that your code is formatted before submission, or [enable Pre-commit Hooks for automatic formatting](#pre-commit-hooks).

Currently enabled formatting tools:

| File Type | Formatting Tool |
| --- | --- |
| JSON/Yaml | [prettier](https://prettier.io/) |
| Markdown | [MarkdownLint](https://github.com/DavidAnson/markdownlint-cli2) |

### Automatically Format Code Using Pre-commit Hooks

<a id="pre-commit-hooks"></a>

1. Ensure that Python and Node environments are installed on your computer.

2. Run the following commands in the project's root directory:

    ```bash
    pip install pre-commit
    pre-commit install
    ```

If `pre-commit` cannot run after installation, ensure that the pip installation path has been added to the PATH environment variable.

From now on, formatting tools will automatically run during each commit to ensure your code meets the formatting standards.

Manually trigger formatting:

```bash
pre-commit run --all-files
```

## Related Reading

- [Project Structure](./Project-Structure.md)
- [Writing interface.json](./Writing-interface.json.md)
- [Writing Pipelines](./Writing-Pipelines.md)
- [Writing Custom Scripts](./Writing-Custom.md)
- [Project Refactoring](./Project-Refactoring.md)
- [Adapting for External Servers](./Adapting-Global-Servers.md)
- [Writing Documentation](./Writing-Documentation.md)

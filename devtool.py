#!/usr/bin/env python3
# -*- coding: utf-8

"""developers tool."""

import subprocess
import os
from typing import List
import fire  # type: ignore
import git

PRECOMMIT_CHECK_EXTENSIONS = [".py"]
PRE_COMMIT_HOOKS = [
    "check-case-conflict",
    "check-merge-conflict",
    "end-of-file-fixer",
    "trailing-whitespace",
    "prettier",
    "black",
    "pylint",
]


def _run_pre_commit_check(hook: str, files: List[str]) -> None:
    """Run pre commit check on selected files

    Args:
        hook (str): name of hook.
        files (List[str]): selected files.
    """
    subprocess.check_call(["pre-commit", "run", hook, "-v", "--files", *files])


class Command:
    """Commands to be run by fire lib."""

    def lint(self, changed_or_staged_files: List[str]) -> None:
        """run linter check by running all PRECOMMIT HOOKS.

        Args:
            changed_or_staged_files (List[str]): list of files to run lint check on.
        """

        files = [item for item in changed_or_staged_files if os.path.splitext(item)[1] in PRECOMMIT_CHECK_EXTENSIONS]

        for hook in PRE_COMMIT_HOOKS:
            _run_pre_commit_check(hook, files)

    def mypy(self, changed_or_staged_files: List[str]) -> None:
        """run mypy on changed files.

        Args:
            changed_or_staged_files (List[str]): list of files to run lint check on.
        """
        try:
            for file in changed_or_staged_files:
                if os.path.splitext(file)[1] == ".py":
                    subprocess.run(["mypy", file], check=True)
        except subprocess.CalledProcessError as err:
            print(f"Error while running mypy: {err}")

    def run(self):
        """Run all the pre-commit checks."""
        repo_path = "."
        repo = git.Repo(repo_path)
        # Get the list of changed or staged files
        changed_or_staged_files = [item.a_path for item in repo.index.diff(None)]
        changed_or_staged_files.extend([item.a_path for item in repo.index.diff("HEAD")])

        self.lint(changed_or_staged_files)
        self.mypy(changed_or_staged_files)
        print("All looks good. 👍 ")


if __name__ == "__main__":
    fire.Fire(Command)

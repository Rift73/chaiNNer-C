"""Runs chainner_pip as `python -m chainner_pip` does, with upstream pip's fix for a
wheel it downloads twice (pip issue 11847), which the bundled chainner_pip 23.2 lacks.

A requirement whose metadata its index serves as a file of its own (PEP 658), as the
PyTorch indexes do for torch and the dependencies they carry, is resolved from that
file. Its wheel is downloaded with the others in one batch after the resolve, and
chainner_pip 23.2 then downloads it again to install it, as it does not record the
batch's files as downloaded: every PyTorch install fetched torch twice (2 GB more on
Windows with CUDA), and triton too on Linux.

store.py runs this file with the interpreter whose packages it installs."""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING

from chainner_pip._internal.operations import prepare

if TYPE_CHECKING:
    from chainner_pip._internal.metadata import BaseDistribution
    from chainner_pip._internal.req.req_install import InstallRequirement


class RequirementPreparer(prepare.RequirementPreparer):
    def _prepare_linked_requirement(
        self, req: InstallRequirement, parallel_builds: bool
    ) -> BaseDistribution:
        # A wheel of the batch is the requirement's local file: record it as
        # downloaded, as upstream pip does.
        link = req.link
        if (
            req.needs_more_preparation
            and link
            and link.is_wheel
            and req.local_file_path
        ):
            self._downloaded[link.url] = req.local_file_path
        return super()._prepare_linked_requirement(req, parallel_builds)


# pip's commands take the class from that module when main() imports them.
prepare.RequirementPreparer = RequirementPreparer

if __name__ == "__main__":
    from chainner_pip._internal.cli.main import main

    sys.exit(main())

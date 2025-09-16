import sys

try:
    # This variable is injected in the __builtins__ by the build
    # process. It is used to enable importing subpackages of mlresearch when
    # the binaries are not built
    # mypy error: Cannot determine type of '__SAFENUDGE_SETUP__'
    __SAFENUDGE_SETUP__  # type: ignore
except NameError:
    __SAFENUDGE_SETUP__ = False

if __SAFENUDGE_SETUP__:
    sys.stderr.write("Partial import of safenudge during the build process.\n")
    # We are not importing the rest of safenudge during the build
    # process, as it may not be compiled yet
else:

    from .new_ctg import CTG
    from .old_ctg import TokenMaskingCTG
    from .wildguard_ctg import WildGuardCTG, WildGuard
    from ._version import __version__

    __all__ = ["CTG", "TokenMaskingCTG", "WildGuardCTG", "WildGuard", "__version__"]

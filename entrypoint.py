import pylint
from pylint.lint import PyLinter

from custom_checker import CustomChecker, register

register(PyLinter)
pylint.modify_sys_path()
pylint.run_pylint()

from astroid import nodes
from typing import TYPE_CHECKING, Optional

from pylint.checkers import BaseChecker

if TYPE_CHECKING:
    from pylint.lint import PyLinter


class PySparkCheckers(BaseChecker):

    name = "spark-unsafe-methods"
    msgs = {
        "W0001": (
            "Calls an unsafe toPandas method.",
            "spark-to-pandas-method",
            "You should use safe method 'toPandas' from 'model_pipe' library.",
        ),
        "W0002": (
            "Calls an unsafe repartition method.",
            "spark-repartition-method",
            "You should use safe method 'repartition' from 'model_pipe' library",
        ),
    }
    options = (
        (
            "ignore-spark-unsafe-methods",
            {
                "default": False,
                "type": "yn",
                "metavar": "<y or n>",
                "help": "Allow using unsafe spark methods",
            },
        ),
    )

    @staticmethod
    def _get_final_node(node: nodes.NodeNG) -> Optional[nodes.Name]:
        next_node = cur_node = node
        while next_node:
            cur_node, next_node = next_node, cur_node.last_child()
        return cur_node
    
    def _get_assignment_call(self, node: nodes.NodeNG, name: str) -> Optional[nodes.Call]:
        """Find first call assignment to variable."""
        parent_frame = node.frame()
        _, assignments = parent_frame.scope_lookup(node, name)
        print(assignments)
        if not assignments:
            return
        first_assignment, *_ = assignments
        if isinstance(first_assignment.parent, nodes.With):
            for call, assign in first_assignment.parent.items:
                if assign.name == name:
                    return call
        return

    def _is_package_method(self, node: nodes.Call, package_name: str):
        """Check if the calling method is from package."""
        class_name = self._get_final_node(node)
        _, imports = node.parent.frame().scope_lookup(node, class_name.name)
        first_import, *_ = imports
        if isinstance(first_import, nodes.Import):
            for original_name, _ in first_import.names:
                if original_name == package_name:
                    return True
        elif isinstance(first_import, nodes.ImportFrom):
            if first_import.modname == package_name:
                return True
        return False
    
    def check_unsafe_method(self, node: nodes.NodeNG, name: str):
        if isinstance(node, nodes.Call) and isinstance(node.func, nodes.Attribute) and node.func.attrname == name:
            final_node = self._get_final_node(node)
            parent_frame = node.frame()
            _, assignments = parent_frame.scope_lookup(node, final_node.name)
            call = self._get_assignment_call(node, final_node.name)
            if call is None:
                return
            return self._is_package_method(call, 'pyspark')

    def visit_call(self, node: nodes.Call) -> None:
        if self.check_unsafe_method(node, 'toPandas'):
            self.add_message("spark-to-pandas-method", node=node)
        if self.check_unsafe_method(node, 'repartition'):
            self.add_message("spark-repartition-method", node=node)


def register(linter: "PyLinter") -> None:
    """This required method auto registers the checker during initialization.
    :param linter: The linter to register the checker to.
    """
    linter.register_checker(PySparkCheckers(linter))

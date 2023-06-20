import ast
from collections import deque, namedtuple
import glob
import logging
from pathlib import Path
from typing import Callable, Deque, Union
from typing import List, Optional

import click
import yaml


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s:%(lineno)d: %(message)s",
)
_logger = logging.getLogger(__name__)


Parameter = namedtuple('Parameter', ['value', 'is_variable'])


class ModuleParserException(Exception):
    pass


class ModuleParser:
    """Class for Python code parsing"""

    def __init__(self, code: str):
        self.source_code = code
        self.module = ast.parse(code)
        self.module_functions = self.get_module_functions()

    def get_module_functions(self) -> List[ast.FunctionDef]:
        nodes: List[ast.FunctionDef] = []
        for obj in self.module.body:
            if isinstance(obj, ast.FunctionDef):
                nodes.append(obj)
        return nodes

    def get_module_function_by_name(self, name: str) -> ast.FunctionDef:
        for func in self.module_functions:
            if func.name == name:
                return func

    def get_module_function_arguments_names(
            self,
            name: str
    ) -> Optional[List[str]]:
        func = self.get_module_function_by_name(name)
        if func:
            return [arg.arg for arg in func.args.args]

    @staticmethod
    def _get_nodes_stacks_recursively(
            start_node: ast.AST,
            find_condition: Callable
    ) -> List[List[ast.AST]]:
        """Get node stacks of ast module.
        Depth first search

        :param find_condition: function which checks find condition
        """
        stacks_list: List[List[ast.AST]] = []
        stack: Deque[ast.AST] = deque([])

        def find_call(node):
            stack.append(node)
            if find_condition(node):
                stacks_list.append(list(stack))
            for child_node in ast.iter_child_nodes(node):
                find_call(child_node) # noqa
                stack.pop()
        find_call(start_node) # noqa
        return stacks_list

    @staticmethod
    def find_variable_in_assignment(
            assign: ast.Assign,
            name: str
    ) -> Optional[ast.Name]:
        for t in assign.targets:
            if isinstance(t, ast.Name) and t.id == name:
                return t

    def find_assignment_to_variable_recursively(
            self,
            name: str
    ) -> List[List[ast.AST]]:
        def condition(node: ast.AST):
            return (
                isinstance(node, ast.Assign) and
                self.find_variable_in_assignment(node, name)
            )

        stacks = self._get_nodes_stacks_recursively(
            start_node=self.module,
            find_condition=condition
        )
        return stacks

    def find_attribute_call_stacks_recursively(
            self,
            name: str
    ) -> List[List[ast.AST]]:
        def condition(node: ast.AST):
            return (
                isinstance(node, ast.Call) and
                isinstance(node.func, ast.Attribute) and
                node.func.attr == name
            )

        stacks = self._get_nodes_stacks_recursively(
            start_node=self.module,
            find_condition=condition
        )
        return stacks

    @staticmethod
    def is_parent_scope(parent_scope: List[ast.AST],
                        child_scope: List[ast.AST]) -> bool:
        if not child_scope or not parent_scope:
            return False
        for i, (child_node, parent_node) in enumerate(zip(child_scope,
                                                          parent_scope)):
            if child_node is not parent_node and i != len(child_scope) - 1:
                return False
        return True

    @staticmethod
    def is_scope(node: ast.AST):
        return (
            isinstance(node, ast.FunctionDef) or
            isinstance(node, ast.AsyncFunctionDef) or
            isinstance(node, ast.ClassDef) or
            isinstance(node, ast.Module)
        )

    def get_parent_scope(self, scope: List[ast.AST]) -> List[ast.AST]:
        if not scope:
            return []
        i = 0
        for i, node in reversed(list(enumerate(scope[:-1]))):
            if self.is_scope(node):
                break
        return scope[:i + 1]

    def check_variable_maybe_accessible_in_scope(
            self,
            variable_stack: List[ast.AST],
            scope_stack: List[ast.AST]
    ) -> bool:
        if (
            not variable_stack or
            not isinstance(variable_stack[-1], ast.Assign)
        ):
            _logger.info('"variable_stack" parameter must be a stack '
                         'to variable assignment')
            return False
        var_parent_stack = self.get_parent_scope(variable_stack)
        return self.is_parent_scope(parent_scope=var_parent_stack,
                                    child_scope=scope_stack)


class ModelCheckError(Exception):
    pass


def parse_config_quality_section(filename):
    path = Path(filename)
    if not path.exists():
        _logger.warning(f'Couldn\'t find config file "{filename}" in project.')
        return {}
    config = yaml.safe_load(path.read_text())
    if 'quality_config' not in config.keys():
        _logger.warning(
            'Couldn\'t find "quality_config" section in config file {filename}'
        )
        return {}
    if not isinstance(config['quality_config'], dict):
        _logger.warning(
            'Couldn\'t read "quality_config" section as dictionary.'
        )
        return {}
    return config['quality_config']


def parse_code_for_tags(parser: ModuleParser) -> Optional[ast.Dict]:
    # TODO: check tags for all call and var stacks (remove temp assumptions)
    # TODO: check parameters in same scope as tags dictionary
    # TODO: raise errors when variable not found, or parameter tags not found
    # TODO: add docstrings
    call_stacks = parser.find_attribute_call_stacks_recursively(
        'save_dq_results'
    )
    if len(call_stacks) == 0:
        return None
    call_stack = call_stacks[-1]  # temporary assumption
    call: ast.Call = call_stack[-1] # noqa
    tags_var_name = None
    for kw in call.keywords:
        if kw.arg != 'tags':
            continue
        if isinstance(kw.value, ast.Dict):
            return kw.value
        elif isinstance(kw.value, ast.Name):
            tags_var_name = kw.value.id
    if tags_var_name is None:
        return None
    var_assign_stacks = parser.find_assignment_to_variable_recursively(
        tags_var_name
    )
    var_assign_stack = var_assign_stacks[-1]  # temporary assumption
    var_assign: ast.Assign = var_assign_stack[-1] # noqa
    # TODO: make tests for function and uncomment check
    # if parser.check_variable_maybe_accessible_in_scope(var_assign_stack,
    #                                                    call_stack):
    tags_dict: ast.Dict = var_assign.value # noqa
    return tags_dict


def parse_main_params(module_parser: ModuleParser):
    arguments = module_parser.get_module_function_arguments_names('main')
    if arguments is not None:
        return arguments
    return []


def parse_variable_from_tags(tags: ast.Dict, key: str) -> Optional[str]:
    for k, v in zip(tags.keys, tags.values):
        k: ast.Str
        if k.s == key and isinstance(v, ast.Name):
            return v.id


def parse_string_from_tags(tags: ast.Dict, key: str) -> Optional[str]:
    for k, v in zip(tags.keys, tags.values):
        k: ast.Str
        if k.s == key and isinstance(v, ast.Str):
            return v.s


def iterate_py_code_in_dir(directory):
    for file_name in glob.iglob(f'{directory}/**/*.py', recursive=True):
        module_parser = ModuleParser(Path(file_name).read_text())
        yield (file_name,
               parse_code_for_tags(module_parser),
               parse_main_params(module_parser))


def get_params_from_tags(tags: ast.Dict,
                         params,
                         main_params,
                         filename) -> Optional[dict]:
    parsed_params = {}
    for param in params:
        str_value = parse_string_from_tags(tags, param)
        var_value = parse_variable_from_tags(tags, param)
        if str_value:
            parsed_params[param] = Parameter(value=str_value,
                                             is_variable=False)
        elif var_value:
            if var_value not in main_params:
                raise ModelCheckError(f'Variable {var_value} must be taken '
                                      f'from main parameters: {main_params} '
                                      f'in file "{filename}"')
            parsed_params[param] = Parameter(value=var_value,
                                             is_variable=True)
        else:
            parsed_params[param] = None
    return parsed_params


def iterate_parsed_params(params, directory, config_file=None):
    parsed_values = {}
    config = parse_config_quality_section(config_file)
    for filename, tags, main_params in iterate_py_code_in_dir(directory):
        _logger.info(f'Trying find tags in "{filename}"')
        if not tags:
            _logger.info(f'Tags in "{filename}" not found')
            continue
        _logger.info(f'Tags in "{filename}" found. Parsing...')
        parsed_params = get_params_from_tags(tags,
                                             params,
                                             main_params,
                                             filename)
        for k, v in parsed_params.items():
            if v is None:
                parsed_values[k] = None
            elif v.is_variable and v.value not in config.keys():
                raise ModelCheckError(f'Variable {v.value} is not set in '
                                      '"quality_config" section of '
                                      f'{config_file}. "quality_config" '
                                      f'section data: {config}')
            elif not v.is_variable:
                parsed_values[k] = v.value
            else:
                parsed_values[k] = config[v.value]
        yield filename, parsed_values


def try_find_params(params, models) -> bool:
    if models and not params.keys() <= models[0].keys():
        raise ModelCheckError('No such params in model.'
                              f'Params: {params.keys()}, '
                              f'model: {models[0].keys()}')
    for model in models:
        if params.items() <= model.items():
            return True
    return False


def run_check(directory, config_file, client):
    models = client.get_models()
    files_counter = 0
    for filename, params in iterate_parsed_params(
            ['usecase', 'pipeline_name', 'model_name', 'type'],
            directory,
            config_file
    ):
        files_counter += 1
        if params.pop('type') == 'date_check':
            _logger.info(f'Skip {filename} because of tags with '
                         'parameter "type" equal "date_check"')
            continue
        for k, v in params.items():
            if v is None:
                raise ModelCheckError(
                    f'Parameter {k} is not set in {filename}'
                )
        _logger.info(f'Compare params from {filename} with registry')
        if not try_find_params(params, models):
            raise ModelCheckError(f'Model with parameters {params} '
                                  'does not exist in registry')
    if files_counter == 0:
        raise ModelCheckError('Saving tags to DQ was not found. At least one '
                              f'.py script in {directory} and subdirectories '
                              'should contain "RestStore.save_dq_results" '
                              'function invocation')


# @click.command()
# @click.option('-d', '--directory', required=True,
#               help='A directory to scan for ds python scripts')
# @click.option('-cf', '--config-file', required=True,
#               help='Airflow dag config file')
# @click.option('-kh', '--keycloak-host', required=True)
# @click.option('-kp', '--keycloak-port', required=True)
# @click.option('-kr', '--keycloak-realm', required=True)
# @click.option('-ku', '--keycloak-username', required=True)
# @click.option('-kpwd', '--keycloak-password', required=True)
# @click.option('-kcid', '--keycloak-client-id', required=True)
# @click.option('-kcs', '--keycloak-client-secret', required=True)
# @click.option('-rah', '--registry-api-host', required=True)
# @click.option('-rap', '--registry-api-port', required=True)
# def main(directory, config_file, keycloak_host, keycloak_port, keycloak_realm,
#          keycloak_username, keycloak_password, keycloak_client_id,
#          keycloak_client_secret, registry_api_host, registry_api_port):
#     run_check(directory=directory,
#               config_file=config_file,
#               client=registry_client)

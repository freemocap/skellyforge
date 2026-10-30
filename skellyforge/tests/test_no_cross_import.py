"""Forge code, tests, tools and dependencies must remain independent of sibling apps."""
from pathlib import Path
import ast
import re
import tomllib

import pytest

_FORBIDDEN = {'skellytracker', 'skellycam', 'freemocap'}
_REPOSITORY = Path(__file__).resolve().parents[2]
_CODE_FOLDERS = ('skellyforge', 'scripts', 'tests', 'test_support', 'diagnostics', 'experiments')


def _imported_modules(source: str):
    tree = ast.parse(source)
    dynamic_modules = {'importlib'}
    dynamic_functions = {'__import__'}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == 'importlib':
                    dynamic_modules.add(alias.asname or alias.name)
                yield alias.name, node.lineno
        elif isinstance(node, ast.ImportFrom):
            if node.module == 'importlib':
                dynamic_functions.update(alias.asname or alias.name for alias in node.names if alias.name == 'import_module')
            yield node.module or '', node.lineno
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not node.args or not isinstance(node.args[0], ast.Constant):
            continue
        function = node.func
        dynamic_call = (isinstance(function, ast.Name) and function.id in dynamic_functions) or (
            isinstance(function, ast.Attribute) and function.attr == 'import_module'
            and isinstance(function.value, ast.Name) and function.value.id in dynamic_modules)
        if dynamic_call and isinstance(node.args[0].value, str):
            yield node.args[0].value, node.lineno


def test_core_tests_and_fixtures_do_not_import_exploratory_tools() -> None:
    offenders = []
    for folder in (_REPOSITORY / 'skellyforge/tests', _REPOSITORY / 'test_support'):
        for path in folder.rglob('*.py'):
            for module, line in _imported_modules(path.read_text(encoding='utf-8')):
                if module.split('.')[0] in ('scripts', 'experiments', 'diagnostics'):
                    offenders.append(f'{path.relative_to(_REPOSITORY)}:{line}')
    assert not offenders, 'Core tests must use shared fixtures or production code: ' + ', '.join(offenders)


def test_forge_source_tests_and_tools_never_import_sibling_apps() -> None:
    paths = list(_REPOSITORY.glob('*.py'))
    for folder in _CODE_FOLDERS:
        paths.extend((_REPOSITORY / folder).rglob('*.py'))
    offenders = []
    for path in sorted(paths):
        for module, line in _imported_modules(path.read_text(encoding='utf-8')):
            if module.split('.')[0] in _FORBIDDEN:
                offenders.append(f'{path.relative_to(_REPOSITORY)}:{line}: {module}')
    assert not offenders, 'Forge must not import sibling apps:\n' + '\n'.join(offenders)


@pytest.mark.parametrize('source', [
    'import skellytracker.core.io',
    'from skellycam import Camera',
    'if TYPE_CHECKING:\n    from freemocap.core import Thing',
    "import importlib as loader\nloader.import_module('skellytracker.core')",
    "from importlib import import_module as load_module\nload_module('freemocap')",
    "__import__('skellycam.core')",
])
def test_import_guard_detects_direct_type_only_and_literal_dynamic_imports(source):
    assert any(module.split('.')[0] in _FORBIDDEN for module, _ in _imported_modules(source))


def test_dependency_declarations_and_lock_exclude_sibling_apps() -> None:
    declaration = tomllib.loads((_REPOSITORY / 'pyproject.toml').read_text(encoding='utf-8'))
    project = declaration['project']
    requirements = list(project['dependencies'])
    for group in project.get('optional-dependencies', {}).values():
        requirements.extend(group)
    for group in declaration.get('dependency-groups', {}).values():
        requirements.extend(item for item in group if isinstance(item, str))
    names = {re.split(r'[\s<>=!~;\[]', item, maxsplit=1)[0].lower().replace('-', '_') for item in requirements}
    names.update(declaration.get('tool', {}).get('uv', {}).get('sources', {}))
    assert not names & _FORBIDDEN
    locked = tomllib.loads((_REPOSITORY / 'uv.lock').read_text(encoding='utf-8'))
    assert not {package['name'] for package in locked['package']} & _FORBIDDEN

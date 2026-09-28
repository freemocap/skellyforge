"""Transport enumerates native properties rather than maintaining a field map."""
from skellyforge import _native
from skellyforge.tools.solver_inspector.inspection import inspection_data
from skellyforge.tests.test_native_window_sequence import seeded


def test_transport_preserves_native_connections_and_all_exposed_properties():
    args,_=seeded();options=_native.ChainSolveOptions();options.inspect_problem=True
    fit=_native.fit_chain_sequence(**args,solve_options=options)
    native=fit.problem_final;data=inspection_data(native)
    for objects,rows in [(native.parameters,data['parameters']),(native.residuals,data['residuals'])]:
        for obj,row in zip(objects,rows,strict=True):
            properties={k for k,v in vars(type(obj)).items() if isinstance(v,property) and not k.startswith('_')}
            assert set(row)==properties
            for key in properties:assert row[key]==getattr(obj,key)

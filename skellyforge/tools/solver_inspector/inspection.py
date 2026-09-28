"""Transport the native read-only properties verbatim; no graph reconstruction."""


def inspection_data(value):
    if value is None or isinstance(value,(str,bool,int,float)):
        return value
    if isinstance(value,(list,tuple)):
        return [inspection_data(v) for v in value]
    if isinstance(value,dict):
        return {k:inspection_data(v) for k,v in value.items()}
    properties={name:descriptor for name,descriptor in vars(type(value)).items()
                if isinstance(descriptor,property) and not name.startswith('_')}
    if not properties or type(value).__module__!='skellyforge._native':
        raise TypeError(f'Not a native inspection object: {type(value)}')
    return {name:inspection_data(getattr(value,name)) for name in properties}

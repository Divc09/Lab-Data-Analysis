"""Strict physical compatibility for operations combining recorded datasets."""
from __future__ import annotations

GROUPS = {
    'time': {'ps':1e-12,'ns':1e-9,'us':1e-6,'ms':1e-3,'s':1.},
    'frequency': {'Hz':1.,'kHz':1e3,'MHz':1e6,'GHz':1e9},
    'length': {'nm':1e-9,'um':1e-6,'mm':1e-3,'m':1.},
}


def unit(value):
    return str(value or '').strip().replace('µ','u').replace('μ','u').replace('�','u')


def unit_scale(source, target):
    a,b=unit(source),unit(target)
    if a==b: return 1.
    for values in GROUPS.values():
        if a in values and b in values: return values[a]/values[b]
    raise ValueError(f'Incompatible units: {source or "unitless"} and {target or "unitless"}')


def compatibility(first, second, *, label='Cross-file operation'):
    if first['domain']!=second['domain'] or first['observable']!=second['observable']:
        raise ValueError(f'{label}: incompatible domain/observable for {first["name"]} and {second["name"]}')
    try:
        return unit_scale(second['x_unit'],first['x_unit']),unit_scale(second['y_unit'],first['y_unit'])
    except ValueError as exc:
        raise ValueError(f'{label}: {first["name"]} and {second["name"]}: {exc}') from exc


def trace_units(trace):
    meta=trace.metadata or {}
    axes=','.join(sorted(str(a).strip().lower() for a in trace.scan_axes))
    domain='elapsed_time' if trace.experiment_type=='TimeTrace' else trace.scan_dim if trace.scan_dim=='time' else trace.scan_dim+':'+axes
    return {'name':trace.file_name,'domain':domain,'observable':trace.mode,
            'x_unit':meta.get('x_unit') or ('ns' if trace.scan_dim=='time' and trace.experiment_type!='TimeTrace' else ''),
            'y_unit':'dimensionless' if trace.mode=='contrast' and not meta.get('reference_only') else meta.get('value_unit') or 'a.u.'}

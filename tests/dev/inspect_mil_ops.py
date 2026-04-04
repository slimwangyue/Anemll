"""Inspect MIL ops in exported prefill models to find ANE-hostile ops."""
import sys
import os

sys.path.insert(0, '/Users/yw68/Anemll')
import coremltools as ct

ANE_HOSTILE = {
    'gather_along_axis', 'gather', 'scatter', 'scatter_along_axis',
    'less', 'less_equal', 'greater', 'greater_equal', 'equal', 'not_equal',
    'where', 'select', 'non_zero', 'topk', 'argsort', 'sort',
    'one_hot', 'cumsum',
}

model_dir = '/Users/yw68/Anemll/qwen3_5_4chunk_lut6_bs512_ctx2048'
for ci in range(4):
    path = os.path.join(model_dir, f'prefill_LUT6_chunk{ci}_bs512.mlpackage')
    if not os.path.exists(path):
        print(f"chunk {ci}: NOT FOUND")
        continue
    print(f"\nLoading chunk {ci}...")
    mlmodel = ct.models.MLModel(path, compute_units=ct.ComputeUnit.CPU_ONLY)
    spec = mlmodel.get_spec()
    prog = ct.models.utils._milproto_to_pymil.load(
        spec,
        spec.specificationVersion,
        mlmodel.weights_dir,
    )
    if prog is None:
        print(f"  Could not get MIL program")
        continue

    op_counts = {}
    hostile_ops = {}
    hostile_details = []
    for func_name in prog.functions:
        func = prog.functions[func_name]
        for op in func.operations:
            op_type = op.op_type
            op_counts[op_type] = op_counts.get(op_type, 0) + 1
            if op_type in ANE_HOSTILE:
                hostile_ops[op_type] = hostile_ops.get(op_type, 0) + 1
                # Collect details about hostile ops
                inputs_info = {}
                for inp_name, inp_val in op.inputs.items():
                    if hasattr(inp_val, 'shape'):
                        inputs_info[inp_name] = f"shape={inp_val.shape}, dtype={inp_val.dtype}"
                    elif hasattr(inp_val, 'val') and inp_val.val is not None:
                        v = inp_val.val
                        inputs_info[inp_name] = f"const={v}" if hasattr(v, '__len__') and len(v) < 10 else f"const(len={len(v) if hasattr(v, '__len__') else '?'})"
                    else:
                        inputs_info[inp_name] = str(type(inp_val).__name__)
                outputs_info = {}
                for out in op.outputs:
                    outputs_info[out.name] = f"shape={out.shape}, dtype={out.dtype}"
                hostile_details.append({
                    'op_type': op_type,
                    'name': op.name,
                    'inputs': inputs_info,
                    'outputs': outputs_info,
                })

    total = sum(op_counts.values())
    print(f"  {total} total MIL ops")
    if hostile_ops:
        print(f"  ANE-HOSTILE: {hostile_ops}")
        for d in hostile_details:
            print(f"    [{d['op_type']}] {d['name']}")
            print(f"      inputs:  {d['inputs']}")
            print(f"      outputs: {d['outputs']}")
    else:
        print(f"  No ANE-hostile ops found")
    print(f"  Unique ops ({len(op_counts)}): {sorted(op_counts.keys())}")
    del mlmodel, prog

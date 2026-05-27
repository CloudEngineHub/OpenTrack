from dataclasses import dataclass

import cyclonedds.idl as idl
import cyclonedds.idl.annotations as annotate
import cyclonedds.idl.types as types


@dataclass
@annotate.final
@annotate.autoid("sequential")
class SimState_(idl.IdlStruct, typename="sim_interface.viewer_dds_.SimState_"):
    qpos: types.array[types.float32, 36]
    qvel: types.array[types.float32, 35]
    tick: types.uint32

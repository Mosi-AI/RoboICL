"""Read source HDF5 states and reconstruct the online world-delta action tensor."""

import numpy as np
from scipy.spatial.transform import Rotation


def state_at(data, frame):
    return {f'{arm}_{dest}':data[f'state/{arm}_{source}'][frame].tolist()
            for arm in ('left','right') for dest,source in
            (('ee_pose','ee_poses'),('arm_joint_state','arm_joint_states'),('ee_joint_state','ee_joint_states'))}


def tensor_at(data, start, count):
    rows=[]
    previous={arm:data[f'state/{arm}_ee_poses'][start] for arm in ('left','right')}
    for frame in range(start,start+count):
        row=[]
        for arm in ('left','right'):
            target=data[f'action/{arm}_ee_poses'][frame]
            before=previous[arm]
            delta_rotation=(Rotation.from_quat(target[[4,5,6,3]]) *
                            Rotation.from_quat(before[[4,5,6,3]]).inv()).as_rotvec()
            row.extend([*(target[:3]-before[:3]), *delta_rotation,
                        float(data[f'action/{arm}_ee_joint_states'][frame,0])])
            previous[arm]=target
        rows.append(row)
    return np.round(rows,6).tolist()

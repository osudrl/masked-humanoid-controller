# Copyright (c) 2018-2022, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


import numpy as np
import os
import yaml

from poselib.poselib.skeleton.skeleton3d import SkeletonMotion
from poselib.poselib.core.rotation3d import *
from isaacgym.torch_utils import *

from utils import torch_utils
from munch import Munch

import torch

from utils.motion_lib import MotionLib, DeviceCache

USE_CACHE = True
print("MOVING MOTION DATA TO GPU, USING CACHE:", USE_CACHE)

if not USE_CACHE:
    old_numpy = torch.Tensor.numpy
    class Patch:
        def numpy(self):
            if self.is_cuda:
                return self.to("cpu").numpy()
            else:
                return old_numpy(self)

    torch.Tensor.numpy = Patch.numpy


def sk_motion_from_ik_file(ik_file_path):
    ik_file = np.load(ik_file_path, allow_pickle=True)
    
    def conv_wxyz_to_xyzw(wxyz):
        return np.concatenate([wxyz[:,1:], wxyz[:,:1]], axis=1)
    
    motion = Munch()

    motion.fps = ik_file['fps']
    motion.tensor = ik_file['dof_pos']
    motion._is_local = False
    motion._skeleton_tree = None

    motion.root_pos = ik_file['root_pos'].squeeze()
    motion.root_rot = conv_wxyz_to_xyzw(ik_file['root_rot'].squeeze())
    motion.root_vel = ik_file['root_vel'].squeeze()
    motion.root_ang_vel = ik_file['root_ang_vel'].squeeze()
    motion.dof_pos = ik_file['dof_pos'].squeeze()
    motion.dof_vel = ik_file['dof_vel'].squeeze()
    motion.global_translations = ik_file['global_translations'].squeeze()
    motion.body_name2id = ik_file['body_name2id']
    motion.id2body_name = ik_file['id2body_name']
    motion.num_joints = len(motion.body_name2id)

    # Dummy Variables
    motion.global_translation =  motion.global_translations
    motion.global_rotation = np.expand_dims(motion.root_rot,1) # only rotations of root is being specified
    motion.local_rotation = motion.root_rot
    motion.global_root_velocity = motion.root_vel
    motion.global_root_angular_velocity = motion.root_ang_vel
    motion.dof_vels = motion.dof_vel

    return motion

class InvKinMotionLib(MotionLib):
    def __init__(self, motion_file, dof_body_ids, dof_offsets,
                 key_body_ids, device):
        
        super().__init__(motion_file, dof_body_ids, dof_offsets,
                         key_body_ids, device)
        
        try:
            self.dps = torch.cat([m.dof_pos for m in self._motions], dim=0).float()
        except:
            print("No dof_pos in motion files")

        return
    
    def get_motion_state(self, motion_ids, motion_times, key_body_ids=None):
        assert len(self._all_unique_motion_types) == 1, "Different motion types simultaneously are not currently supported"
        if self._all_unique_motion_types[0] == "InverseKinematicsMotion":
            return self.get_motion_state_from_inv_kin_motion(motion_ids, motion_times, key_body_ids)
        elif self._all_unique_motion_types[0] == "SkeletonMotion":
            return self.get_motion_state_from_skeleton_motion(motion_ids, motion_times, key_body_ids)
        else:
            assert False , f"Motion Type {self._all_unique_motion_types[0]} not supported yet"
        

    def get_motion_state_from_inv_kin_motion(self, motion_ids, motion_times, key_body_ids=None):
        if key_body_ids is None:
            key_body_ids = self._key_body_ids

        n = len(motion_ids)
        num_bodies = self._get_num_bodies()
        num_key_bodies = self._key_body_ids.shape[0]

        motion_len = self._motion_lengths[motion_ids]
        num_frames = self._motion_num_frames[motion_ids]
        dt = self._motion_dt[motion_ids]

        frame_idx0, frame_idx1, blend = self._calc_frame_blend(motion_times, motion_len, num_frames, dt)

        f0l = frame_idx0 + self.length_starts[motion_ids]
        f1l = frame_idx1 + self.length_starts[motion_ids]

        root_pos0 = self.gts[f0l, 0]
        root_pos1 = self.gts[f1l, 0]

        root_rot0 = self.grs[f0l, 0]
        root_rot1 = self.grs[f1l, 0]

        root_vel = self.grvs[f0l]
        root_ang_vel = self.gravs[f0l]

        key_pos0 = self.gts[f0l.unsqueeze(-1), key_body_ids.unsqueeze(0)]
        key_pos1 = self.gts[f1l.unsqueeze(-1), key_body_ids.unsqueeze(0)]

        dof_pos0 = self.dps[f0l]
        dof_pos1 = self.dps[f1l]

        dof_vel = self.dvs[f0l]

        vals = [root_pos0, root_pos1, root_vel, root_ang_vel, key_pos0, key_pos1, dof_pos0, dof_pos1, dof_vel]
        for v in vals:
            assert v.dtype != torch.float64

        blend = blend.unsqueeze(-1)
        root_pos = (1.0 - blend) * root_pos0 + blend * root_pos1
        root_rot = torch_utils.slerp(root_rot0, root_rot1, blend)

        blend_exp = blend.unsqueeze(-1)
        key_pos = (1.0 - blend_exp) * key_pos0 + blend_exp * key_pos1
        
        # dof_pos = self.dof_pos
        dof_pos = (1.0 - blend) * dof_pos0 + blend * dof_pos1

        return root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel, key_pos

    def get_motion_state_from_skeleton_motion(self, motion_ids, motion_times, key_body_ids = None):
        if key_body_ids is None:
            key_body_ids = self._key_body_ids

        n = len(motion_ids)
        num_bodies = self._get_num_bodies()
        num_key_bodies = key_body_ids.shape[0]

        motion_len = self._motion_lengths[motion_ids]
        num_frames = self._motion_num_frames[motion_ids]
        dt = self._motion_dt[motion_ids]

        frame_idx0, frame_idx1, blend = self._calc_frame_blend(motion_times, motion_len, num_frames, dt)

        f0l = frame_idx0 + self.length_starts[motion_ids]
        f1l = frame_idx1 + self.length_starts[motion_ids]

        root_pos0 = self.gts[f0l, 0]
        root_pos1 = self.gts[f1l, 0]

        root_rot0 = self.grs[f0l, 0]
        root_rot1 = self.grs[f1l, 0]

        local_rot0 = self.lrs[f0l]
        local_rot1 = self.lrs[f1l]

        root_vel = self.grvs[f0l]

        root_ang_vel = self.gravs[f0l]
        
        key_pos0 = self.gts[f0l.unsqueeze(-1), key_body_ids.unsqueeze(0)]
        key_pos1 = self.gts[f1l.unsqueeze(-1), key_body_ids.unsqueeze(0)]

        dof_vel = self.dvs[f0l]

        vals = [root_pos0, root_pos1, local_rot0, local_rot1, root_vel, root_ang_vel, key_pos0, key_pos1]
        for v in vals:
            assert v.dtype != torch.float64


        blend = blend.unsqueeze(-1)

        root_pos = (1.0 - blend) * root_pos0 + blend * root_pos1

        root_rot = torch_utils.slerp(root_rot0, root_rot1, blend)

        blend_exp = blend.unsqueeze(-1)
        key_pos = (1.0 - blend_exp) * key_pos0 + blend_exp * key_pos1
        
        local_rot = torch_utils.slerp(local_rot0, local_rot1, torch.unsqueeze(blend, axis=-1))
        dof_pos = self._local_rotation_to_dof(local_rot)

        return root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel, key_pos
    
    def _load_motions(self, motion_file):
        self._motions = []
        self._motion_lengths = []
        self._motion_weights = []
        self._motion_fps = []
        self._motion_dt = []
        self._motion_num_frames = []
        self._motion_files = []
        self._motion_types = []

        total_len = 0.0

        motion_files, motion_weights = self._fetch_motion_files(motion_file)
        num_motion_files = len(motion_files)
        for f in range(num_motion_files):
            curr_file = motion_files[f]
            print("Loading {:d}/{:d} motion files: {:s}".format(f + 1, num_motion_files, curr_file))
            
            if "inv_kin" in curr_file:
                curr_motion = sk_motion_from_ik_file(curr_file)
                self._motion_types.append("InverseKinematicsMotion")
            else:
                curr_motion = SkeletonMotion.from_file(curr_file)
                self._motion_types.append("SkeletonMotion")

            motion_fps = curr_motion.fps
            curr_dt = 1.0 / motion_fps

            num_frames = curr_motion.tensor.shape[0]
            curr_len = 1.0 / motion_fps * (num_frames - 1)

            self._motion_fps.append(motion_fps)
            self._motion_dt.append(curr_dt)
            self._motion_num_frames.append(num_frames)

            if "inv_kin" in curr_file:
                pass
            else:
                curr_dof_vels = self._compute_motion_dof_vels(curr_motion)
                curr_motion.dof_vels = curr_dof_vels

            # Moving motion tensors to the GPU
            if USE_CACHE:
                curr_motion = DeviceCache(curr_motion, self._device)                
            else:
                curr_motion.tensor = curr_motion.tensor.to(self._device)
                curr_motion._skeleton_tree._parent_indices = curr_motion._skeleton_tree._parent_indices.to(self._device)
                curr_motion._skeleton_tree._local_translation = curr_motion._skeleton_tree._local_translation.to(self._device)
                curr_motion._rotation = curr_motion._rotation.to(self._device)

            self._motions.append(curr_motion)
            self._motion_lengths.append(curr_len)
            
            curr_weight = motion_weights[f]
            self._motion_weights.append(curr_weight)
            self._motion_files.append(curr_file)

            print("Dt:",curr_dt ,"Motion length:", int(curr_len/curr_dt))

        self._motion_lengths = torch.tensor(self._motion_lengths, device=self._device, dtype=torch.float32)

        self._motion_weights = torch.tensor(self._motion_weights, dtype=torch.float32, device=self._device)
        self._motion_weights /= self._motion_weights.sum()

        self._motion_fps = torch.tensor(self._motion_fps, device=self._device, dtype=torch.float32)
        self._motion_dt = torch.tensor(self._motion_dt, device=self._device, dtype=torch.float32)
        self._motion_num_frames = torch.tensor(self._motion_num_frames, device=self._device)


        num_motions = self.num_motions()
        total_len = self.get_total_length()

        print("Loaded {:d} motions with a total length of {:.3f}s.".format(num_motions, total_len))

        self._all_unique_motion_types = list(set(self._motion_types))

        return
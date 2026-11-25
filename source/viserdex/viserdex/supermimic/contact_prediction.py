"""Pre-reset, final-physics-sample object/hand contact labels from live MJWarp constraints.

One channel per fused Revo3 rigid body (palm and 21 links). Convex pieces are
OR-reduced; object-floor and hand self-contacts are not object/hand labels.
"""
import torch


def reduce_contacts(geoms, worlds, active, geom_channel, object_geom, num_envs, channels):
    a, b = geoms.long().unbind(-1)
    valid_geom = (a >= 0) & (b >= 0) & (a < len(geom_channel)) & (b < len(geom_channel))
    a = a.clamp(0, len(geom_channel)-1)
    b = b.clamp(0, len(geom_channel)-1)
    channel = torch.where(object_geom[a], geom_channel[b],
                          torch.where(object_geom[b], geom_channel[a], -1))
    valid = active & valid_geom & (channel >= 0) & (worlds >= 0) & (worlds < num_envs)
    index = (worlds.long()*channels+channel).clamp(0, num_envs*channels-1)
    count = torch.zeros(num_envs*channels, device=geoms.device, dtype=torch.int32)
    count.scatter_add_(0, index, valid.int())
    return count.reshape(num_envs, channels) > 0


class LiveObjectContacts:
    def __init__(self, env, body_names):
        from isaaclab_newton.physics import NewtonManager
        self.env, self.body_names = env, list(body_names)
        solver = NewtonManager._solver
        m = solver.mj_model
        self.data = solver.mjw_data
        channel, objects, manifest = [], [], []
        for i in range(m.ngeom):
            name = m.geom(i).name
            body = m.body(int(m.geom_bodyid[i])).name
            enabled = bool(m.geom_contype[i] or m.geom_conaffinity[i])
            is_object = enabled and 'object' in name.lower()
            matches = [j for j, n in enumerate(body_names) if body.endswith('_Robot_'+n)] if enabled else []
            if len(matches) > 1:
                raise ValueError('Ambiguous live Revo3 contact body')
            channel.append(matches[0] if matches else -1)
            objects.append(is_object)
            if matches or is_object:
                manifest.append(dict(geom=i, name=name, body=body, channel=channel[-1], object=is_object))
        if not any(objects) or set(channel)-{-1} != set(range(len(body_names))):
            raise ValueError('Live contact mapping does not cover every Revo3 body and object')
        self.channel = torch.tensor(channel, device=env.device)
        self.objects = torch.tensor(objects, device=env.device)
        self.manifest = dict(body_names=self.body_names, geoms=manifest,
                             sample='active solver contact constraints at final physics tick, captured before reset')

    def read(self):
        import warp as wp
        d = self.data
        # nacon is the populated prefix, not the per-world nconmax capacity.
        n = int(wp.to_torch(d.nacon)[0])
        return reduce_contacts(wp.to_torch(d.contact.geom)[:n], wp.to_torch(d.contact.worldid)[:n],
            wp.to_torch(d.contact.efc_address)[:n, 0] >= 0, self.channel, self.objects,
            self.env.num_envs, len(self.body_names))

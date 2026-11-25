"""Loss for flattened image keypoints stored as repeating (u, v, depth)."""


def keypoint_loss(predictions, targets, loss_fn, depth_loss_coef=1.0):
    predictions = predictions.unflatten(-1, (-1, 3))
    targets = targets.unflatten(-1, (-1, 3))
    pixel_loss = loss_fn(predictions[..., :2], targets[..., :2])
    depth_loss = loss_fn(predictions[..., 2], targets[..., 2])
    return (pixel_loss + depth_loss_coef * depth_loss) * 100

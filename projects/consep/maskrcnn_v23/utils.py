import torch
import numpy as np
import torchvision.transforms as transforms

from torchvision.utils import (
    draw_bounding_boxes, 
    draw_segmentation_masks,
    draw_keypoints
)

# Set NumPy seed.
np.random.seed(42)

transform = transforms.Compose([
        transforms.ToTensor(),
])

def get_transformed_image(image):
    """
    Converts a NumPy array image to uint8 and float32 tensors.

    :param image: Input image in NumPy format.

    Returns:
        uint8_tensor: Image tensor of type uint8.
        float32_tensor: Batched image tensor of type float32.
    """
    image_transposed = np.transpose(image, [2, 0, 1])
    # Convert to uint8 tensor.
    uint8_tensor = torch.tensor(image_transposed, dtype=torch.uint8)
    # Convert to float32 tensor.
    float32_tensor = transform(image)
    float32_tensor = torch.unsqueeze(float32_tensor, 0)
    return uint8_tensor, float32_tensor

def filter_detections(
    outputs, coco_names, 
    detection_threshold=0.8
):
    """
    Returns the filtered outputs according to the threshold.

    :param outputs: Object detection/instance segmentation outputs.
    :param coco_names: List containing all the MS COCO class names.
    :param detection_threshold: Confidence threshold to filter out boxes

    Returns:
        boxes: The final filtered bounding boxes.
        pred_classes: Class name strings corresponding to the `boxes` detections.
    """
    pred_scores = outputs[0]['scores'].detach().cpu().numpy()
    pred_classes = [coco_names[i] for i in outputs[0]['labels'].cpu().numpy()]
    pred_bboxes = outputs[0]['boxes'].detach().cpu().numpy()
    boxes = pred_bboxes[pred_scores >= detection_threshold].astype(np.int32)
    pred_classes = pred_classes[:len(boxes)]
    labels = outputs[0]['labels'][:len(boxes)]
    return boxes, pred_classes, labels

def draw_boxes(
    boxes, unint8_tensor, pred_classes, 
    labels, colors, fill=False,
    is_instance=False
):
    """
    Annotates and image (tensor) with bounding boxes.

    :param boxes (List): List containing bounding boxes.
    :param uint8_tensor: An uint8 image tensor.
    :param pred_classes: Class name strings.
    :param labels: Class label indices.
    :param colors: List of tuple colors containing RGB values.
    :param fill: Whether to fill the bounding box with same color as that of box.
    :param is_instance: Whether it is instance segmentation output or not. If
          so, create as many random color as number of outputs.

    Returns:
        result_with_boxes: An uint8 tensor with bounding boxes annotated on it.
        plot_colors: List containing the exact RGB colors used to annotate the image.
    """
    if is_instance:
        plot_colors = colors=np.random.randint(0, 255, size=(len(boxes), 3))
        plot_colors = [tuple(color) for color in plot_colors]
    else:
        plot_colors = [colors[label] for label in labels]

    result_with_boxes = draw_bounding_boxes(
        image=unint8_tensor, 
        boxes=torch.tensor(boxes), width=2, 
        colors=plot_colors,
        labels=pred_classes,
        fill=fill
    )
    return result_with_boxes, plot_colors

def get_rgb_mask(outputs):
    """
    Create and return RGB mask for segmentation.

    :param outputs (Tensor): The outputs tensor from a segmentation model.

    Returns:
        all_masks: An RGB mask.
    """
    num_classes = outputs['out'].shape[1]
    masks = outputs['out'][0].cpu()
    class_dim = 0 # 0 as it is a single image and not a batch.
    all_masks = masks.argmax(class_dim) == \
        torch.arange(num_classes)[:, None, None]
    return all_masks

def draw_mask(uint8_tensor, all_masks, colors):
    """
    Draw semantic_segmentation mask on an image tensor.

    :param uint8_tensor: An image tensor of uint8 type.
    :param all_masks: The RGB mask to be overlayed on the image tensor.
    :param colors: List containing RGB color tuples corresponding to the dataset.

    Returns:
        seg_result: The final image tensor with RGB mask overlayed on it.
    """
    seg_result = draw_segmentation_masks(
        uint8_tensor, 
        all_masks,
        colors=colors,
        alpha=0.5
    )
    return seg_result

def draw_instance_mask(outputs, uint8_tensor, colors, threshold):
    """
    Draws segmentatation map on an image tensor which has already been
    annotated with bounding boxes from the outputs of an instance segmentation
    model.

    :param outputs: Outputs of the instance segmentation model.
    :param uint8_tensor: The uint8 image tensor with bounding boxes
          annotated on it.
    :param colors: List containing RGB tuple colors.

    Returns:
        seg_result: The final segmented result with bounding boxes and RGB 
            color masks.
    """
    # Get all the scores.
    scores = list(outputs[0]['scores'].detach().cpu().numpy())
    # Index of those scores which are above a certain threshold.
    thresholded_preds_inidices = [scores.index(i) for i in scores if i > threshold]
    thresholded_preds_count = len(thresholded_preds_inidices)
    masks = outputs[0]['masks']
    final_masks = masks > 0.5
    final_masks = final_masks.squeeze(1)
    # Discard masks for objects which are below threshold.
    final_masks = final_masks[:thresholded_preds_count]
    seg_result = draw_segmentation_masks(
        uint8_tensor, 
        final_masks,
        colors=colors,
        alpha=0.8
    )
    return seg_result

def draw_keypoints_on_image(
    outputs, uint8_tensor, 
    connect_points, colors=(255, 0, 0), threshold=0.8
):
    """
    Draws keypoints and the skeletal lines on an image.

    :param outouts: Outputs of the keypoint detection model.
    :param uint8_tensor: Image tensor to draw keypoints on.
    :param connect_points: List containing tuple values for which keypoint 
          to connect which one.
    :param colors: Color of keypoint circles.
    :param threshold: Detection threshold for filtering.
    """
    keypoints = outputs[0]['keypoints']
    scores = outputs[0]['scores']
    idx = torch.where(scores > threshold)
    keypoints = keypoints[idx]
    keypoint_result = draw_keypoints(
        image=uint8_tensor, 
        keypoints=keypoints, 
        connectivity=connect_points, 
        colors=colors, 
        radius=4, 
        width=3
    )
    return keypoint_result
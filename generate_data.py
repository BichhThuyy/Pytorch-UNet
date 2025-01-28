import numpy as np
import pydicom as pdc
import os
import matplotlib.pyplot as plt
from rt_utils import RTStructBuilder
import cv2


def get_largest_contour(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        return max(contours, key=cv2.contourArea)
    else:
        return None


def scale_contour_to_area(contour, target_area):
    """
    Scales the contour so that its area matches the target area.
    """
    current_area = cv2.contourArea(contour)
    scale_factor = np.sqrt(target_area / current_area)

    moments = cv2.moments(contour)
    if moments['m00'] == 0:
        return contour

    # Calculate the centroid
    cx = int(moments['m10'] / moments['m00'])
    cy = int(moments['m01'] / moments['m00'])

    # Scale contour
    scaled_contour = (contour - [cx, cy]) * scale_factor + [cx, cy]
    return scaled_contour.astype(np.int32)


def estimate_tumor_region(previous_mask, next_mask):
    previous_mask = (previous_mask > 0).astype(np.uint8)
    next_mask = (next_mask > 0).astype(np.uint8)

    contour_prev = get_largest_contour(previous_mask)
    contour_next = get_largest_contour(next_mask)

    if contour_prev is None or contour_next is None:
        return np.zeros_like(previous_mask)

    # Determine the smaller and larger contour
    area_prev = cv2.contourArea(contour_prev)
    area_next = cv2.contourArea(contour_next)

    if area_prev < area_next:
        smaller_contour = contour_prev
        larger_contour = contour_next
    else:
        smaller_contour = contour_next
        larger_contour = contour_prev

    # Compute the average area
    avg_area = (area_prev + area_next) / 2

    # Scale the smaller contour to the average area
    smaller_contour_scaled = scale_contour_to_area(smaller_contour, avg_area)

    # Draw the averaged contour on the current mask
    current_mask = np.zeros_like(previous_mask)
    cv2.drawContours(current_mask, [smaller_contour_scaled], -1, 1, thickness=cv2.FILLED)

    return current_mask


def generate_data():
    target = 'data'
    source = 'T1C'
    total_instances = 0
    total_patient = 0
    for patient in os.listdir(source):
        # Check and create patient folder
        existed_patients = os.listdir(f'{target}/imgs')
        if patient not in existed_patients:
            os.mkdir(f'{target}/imgs/{patient}')
            os.mkdir(f'{target}/masks/{patient}')

        # Create masks array and instances array
        rt_struct = RTStructBuilder.create_from(dicom_series_path=f'{source}/{patient}', rt_struct_path=f'{source}/{patient}/RTSS.dcm')
        axial_masks = None
        try:
            axial_masks = rt_struct.get_roi_mask_by_name('TV') * 1
        except:
            try:
                axial_masks = rt_struct.get_roi_mask_by_name('tv') * 1
            except:
                axial_masks = None
                print('Error patient: ', patient)

        # Create instances array
        if axial_masks is not None:
            instances = sorted([i for i in os.listdir(f'{source}/{patient}') if 'IMG' in i])
            num_instance = len(instances)
            first_instance = 0
            for instance, index in zip(instances, range(0, num_instance)):
                instance_dcm = pdc.read_file(f'{source}/{patient}/{instance}')
                mask = axial_masks[:, :, num_instance - 1 - index]
                if mask.max() == 1 and first_instance == 0:
                    first_instance = index
                prev_mask = axial_masks[:, :, num_instance - 1 - (index - 1)]
                next_mask = axial_masks[:, :, num_instance - 1 - (index + 1)]
                next_mask = next_mask if next_mask.max() == 1 else axial_masks[:, :, num_instance - 1 - (index + 2)]
                if first_instance != 0 and mask.max() != 1 and prev_mask.max() == 1 and next_mask.max() == 1:
                    current_mask = estimate_tumor_region(prev_mask, next_mask)
                    mask[current_mask > 0] = 1

                # Save MRI image
                plt.imsave(
                    f'{target}/imgs/{patient}/{index}.png',
                    instance_dcm.pixel_array,
                    cmap='gray'
                )
                # Save mask
                plt.imsave(
                    f'{target}/masks/{patient}/{index}.png',
                    np.array(mask),
                    cmap='gray'
                )
                total_instances += 1
        total_patient += 1
        print('Done patient =============== ', patient)
    print('Total instances: ', total_instances)
    print('Total patients: ', total_patient)


generate_data()

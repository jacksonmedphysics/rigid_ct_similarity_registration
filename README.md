# rigid_ct_similarity_registration

SITK-based code for rigid registration of CT images based on similarity between selected individual slices. Default will evaluate 20 slices of the fixed image and identify which of the slices in the moving image share highest similarity by SITK correlation score. That determines the average z-axis offset to initialise rigid registration. In cases with variable scan length between fixed and moving images is more reliable than SITK CenteredTransfromInitializerFilter.

To use, read in CT images with sitk 
"""
fixed/moving=sitk.ReadImage('path/to/image')
"""

and output the SITK Euler 3D transform with
"""
outTx=run_slice_trend_rigid_alignment(fixed,moving,show_image=False,n_match_slices=20)
"""

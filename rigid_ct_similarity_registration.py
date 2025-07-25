import SimpleITK as sitk
import numpy as np
##import matplotlib.pyplot as plt
import time
import matplotlib.pyplot as plt
import pandas as pd
from scipy.cluster.vq import kmeans2

###calculate best offset with mutual info and correlation metrics
###only keep location if both metrics in agreement
###Better correspondence if gaussian smoothing at start?
###Detect shrinker parameters based on axial resolution
###Apply gaussian smoothing as pre-process (sigma 10?)
###Split moving image into N array locations equally spaced for offset calc
###Report time required to process
###user can configure time required...



def find_best_offset(image1,image2,slice_index,metric='correlation',shrink_factor='auto',resampling='nearest',iterations=75,
                     learning_rate=1.0, print_offset=True):
    start=time.time()
    ###Preprocessing Steps
    if shrink_factor=='auto':
        shrink_factor=np.round(np.array(image1.GetSize()[:2])/128,0).astype(int).tolist()
        if min(shrink_factor)<1:
            shrink_factor=(1,1)
    shrinker=sitk.ShrinkImageFilter()
    shrinker.SetShrinkFactors(shrink_factor)

    z_min1=image1.GetOrigin()[2]
    z_min2=image2.GetOrigin()[2]
    z_spacing1=image1.GetSpacing()[2]
    z_spacing2=image2.GetSpacing()[2]

    ###Set up Registration Parameters
    R = sitk.ImageRegistrationMethod()
    R.SetOptimizerAsRegularStepGradientDescent(
            learningRate=learning_rate,
            minStep=1e-4,
            numberOfIterations=iterations,
            gradientMagnitudeTolerance=1e-8)
    R.SetOptimizerScalesFromIndexShift()
    R.SetMetricSamplingStrategy(R.RANDOM)
    R.SetMetricSamplingPercentage(0.1)    
    if resampling=='nearest':
        R.SetInterpolator(sitk.sitkNearestNeighbor)
    else:
        R.SetInterpolator(sitk.sitkLinear)
    if metric=='correlation':
        R.SetMetricAsCorrelation()
    elif metric=='mutual_information':
        R.SetMetricAsMattesMutualInformation()
    elif metric=='mean_squares':
        R.SetMetricAsMeanSquares()
    elif metric=='join_histogram_mi':
        R.SetMetricAsJointHistogramMutualInformation()
    elif metric=='ants_neighborhood':
        R.SetMetricAsANTSNeighborhoodCorrelation(2)
    elif metric=='demons':
        R.SetMetricAsDemons()      
    else:
        print('Metrics should be "correlation","mutual_information", or "mean_squares"')
        print('your value',metric,'Setting to "correlation"')
        R.SetMetricAsCorrelation()
   
    #Select fixed slice
    ar1=sitk.GetArrayFromImage(image1)
    ar2=sitk.GetArrayFromImage(image2)
    slice1=sitk.GetImageFromArray(ar1[slice_index,...])
    slice1.SetSpacing(image1.GetSpacing()[:2])
    slice1=shrinker.Execute(slice1)

    #loop 2D registration and similarity scoring
    offsets=[]
    metrics=[]
    for i in range(ar2.shape[0]):
        slice2=sitk.GetImageFromArray(ar2[i,...])
        slice2.SetSpacing(image2.GetSpacing()[:2])
        slice2=shrinker.Execute(slice2)
        tx=sitk.TranslationTransform(2)
        R.SetInitialTransform(tx)
        outTx = R.Execute(slice1, slice2)
        metric=R.GetMetricValue()
        offset=(z_min1+slice_index*z_spacing1)-(z_min2+i*z_spacing2)
        metrics.append(metric)
        offsets.append(offset)
    min_offset=offsets[np.argmin(metrics)]  
    process_time=time.time()-start
    if print_offset:
        print(min_offset,process_time,shrink_factor)
    return offsets,metrics,min_offset,process_time


def run_slice_trend_rigid_alignment(im1,im2,n_match_slices=20,iterations=75,background_value=-1000,gauss_sigma=10,show_image=False):
    total_start=time.time()
    im1=sitk.Cast(im1,sitk.sitkFloat32)
    shrink_factor=np.round(np.array(im1.GetSize()[2])/128,0).astype(int).tolist()
    if shrink_factor<1:
        shrink_factor=1
    shrinker=sitk.ShrinkImageFilter()
    shrinker.SetShrinkFactors((1,1,shrink_factor))
    im1=shrinker.Execute(im1)
    shrink_factor=np.round(np.array(im2.GetSize()[2])/128,0).astype(int).tolist()
    if shrink_factor<1:
        shrink_factor=1
    shrinker=sitk.ShrinkImageFilter()
    shrinker.SetShrinkFactors((1,1,shrink_factor))
    im2=shrinker.Execute(im2)    
    im2=sitk.Cast(im2,sitk.sitkFloat32)
    slice_increment=int(im1.GetSize()[2]/(n_match_slices+1))
    match_indices=[]
    for i in range(n_match_slices):
        match_indices.append((i+1)*slice_increment)

    if gauss_sigma>0:
        print('Gaussian Smoothing image',gauss_sigma,'mm')
        smoother=sitk.SmoothingRecursiveGaussianImageFilter()
        smoother.SetSigma(gauss_sigma)
        im1_smooth=smoother.Execute(im1)
        im2_smooth=smoother.Execute(im2)
        print('Smoothed')


    agreement_values=[]
    df=pd.DataFrame(columns=['minimum_offset','minimum_metric'])
    
    for j in match_indices:
        offsets1,metrics1,min_offset1,process_time1=find_best_offset(im1_smooth,im2_smooth,j,
                                                                 metric='correlation',shrink_factor='auto',
                                                                 resampling='nearest',print_offset=True,iterations=iterations)
        df.loc[len(df)]=[min_offset1,np.min(metrics1)]
    v2=np.expand_dims(df.minimum_offset.values,1)
    clusters=[]
    for k in np.arange(3,8):
        centroid,label=kmeans2(v2, k, minit='points')
        print(k,centroid[np.argmax(np.bincount(label))])
        clusters.append(centroid[np.argmax(np.bincount(label))])
    print('MEDIAN', np.median(clusters))
    average_offset=np.median(clusters)
    
    initial_transform=sitk.Euler3DTransform()
    initial_transform.SetTranslation((0,0,-average_offset))
    R = sitk.ImageRegistrationMethod()
    R.SetOptimizerAsRegularStepGradientDescent(
            learningRate=2.0,
            minStep=1e-4,
            numberOfIterations=100,
            gradientMagnitudeTolerance=1e-8)
    R.SetOptimizerScalesFromIndexShift()
    R.SetMetricAsMattesMutualInformation()
    R.SetMetricSamplingStrategy(R.RANDOM)
    sampling_percentage=0.05*(40000000/np.prod(np.array(im1.GetSize()))) #Decent registration with 0.05 sampling percent and 40M voxel CT image
    if sampling_percentage>1.0:
        sampling_percentage=1.0
    R.SetMetricSamplingPercentage(sampling_percentage)
    R.SetInitialTransform(initial_transform)
    print('Performing 3D Rigid Registration...')
    start_3d=time.time()
    outTx = R.Execute(im1, im2)
    print('3D rigid refinement completed in',round((time.time()-start_3d),1),'seconds')
    print('3D Metric value:', R.GetMetricValue())

    if R.GetMetricValue()>-0.5:
        original_metric=R.GetMetricValue()
        print('Poor initial metric detected, attempting centered initialised transform')
        R.SetMetricSamplingStrategy(R.RANDOM)
        centered_transform=sitk.CenteredTransformInitializer(im1_smooth,im2_smooth,sitk.Euler3DTransform(),
                                                             sitk.CenteredTransformInitializerFilter.GEOMETRY)

        R.SetInitialTransform(centered_transform)
        tx2=R.Execute(im1,im2)
        updated_metric=R.GetMetricValue()
        print('Centered registration metric',updated_metric)
        if updated_metric<original_metric:
            print('selecting updated registration')
            outTx=tx2
    
    total_time=np.round((time.time()-total_start),1)
    print('Total Registration time: ',total_time,'seconds')
    if show_image:
        moved=sitk.Resample(im2,im1,outTx, sitk.sitkLinear, background_value, im2.GetPixelID())
        diff=im1-moved        
        aspect=im1.GetSpacing()[2]/im1.GetSpacing()[0]
        plt.figure(figsize=[10,10])
        plt.imshow(np.flipud(sitk.GetArrayFromImage(diff)[:,int(moved.GetSize()[1]/2),:]),aspect=aspect,clim=[-100,100])
        plt.axis('off')
        plt.title(str(total_time)+' seconds')
        plt.show()
    return outTx


#EXAMPLE USAGE
##fixed=sitk.ReadImage(r"W:\DEEP-PSMA\CHALLENGE_DATA\train_0001\PSMA\CT.nii.gz", sitk.sitkFloat32)
##moving=sitk.ReadImage(r"W:\DEEP-PSMA\CHALLENGE_DATA\train_0001\FDG\CT.nii.gz", sitk.sitkFloat32)
##outTx=run_slice_trend_rigid_alignment(fixed,moving,show_image=True,n_match_slices=20)


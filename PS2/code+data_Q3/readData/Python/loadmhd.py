import SimpleITK as sitk

'''Read in data as a 1, 100, 100, 3 vector field, please change the PATHOFVELOCITY to the directory path of velocity ''' 
velocity = sitk.GetArrayFromImage(sitk.ReadImage(PATHOFVELOCITY))

'''Read in data as a 1, 100, 100  image, please change the PATHOFSOURCE to the directory path of source '''
source= sitk.GetArrayFromImage(sitk.ReadImage(PATHOFSOURCE))
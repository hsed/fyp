from enum import Enum, IntEnum, unique

from itertools import chain


@unique
class BaseDataType(IntEnum):
    '''
        Base-class to define enum types
        Used by dataset and transformer objects
    '''
    DEPTH = 0
    JOINTS = 1
    COM = 2
    ACTION = 3
    NAME = 4
    DEPTH_SEQ = 5
    JOINTS_SEQ = 6
    COM_SEQ = 7
    NAME_SEQ = 8
    ACTION_SEQ = 9

@unique 
class TransformDataType(IntEnum):
    '''
        All negative values are non-standard, mainly used for debugging
        and for transformation

        all these datatypes grow backwards
    '''
    COM_ORIG_MM_SEQ = -17
    DEPTH_ORIG_SEQ = -16
    CROP_TRANSF_MATX_SEQ = -15
    COM_ORIG_PX_SEQ = -14
    JOINTS_ORIG_PX_SEQ = -13
    DEPTH_CROP_SEQ = -12
    DEPTH_CROP_AUG = -11 # final result but without standardisation
    DEPTH_CROP = -10
    AUG_PARAMS = -9
    AUG_MODE = -8
    AUG_TRANSF_MATX = -7
    CROP_TRANSF_MATX = -6
    COM_ORIG_PX = -5
    JOINTS_ORIG_PX = -4
    COM_ORIG_MM = -3
    JOINTS_ORIG_MM = -2
    DEPTH_ORIG = -1
    # All values >= 0 are reserved!


ExtendedDataType = IntEnum('ExtendedDataType', 
                            [(i.name, i.value) for i in chain(TransformDataType, BaseDataType)])

@unique
class BaseDatasetType(Enum):
    TRAIN = 'train'
    TEST = 'test'

@unique
class BaseTaskType(Enum):
    '''
        HPE:
            -> Each raw sample is one frame consisting of:
                1) Depth-map -- 2D Matrix (640 x 480)
                2) Hand Skeleton -- 2D Matrix (21 x 3)
                3) Action Class One-Hot Label (optional to use) -- 1D Matrix (45 x 1)

        HAR:
            -> Each raw sample is a set of F frames consisting of:
                1) Depth-maps from f=1 to t=F -- 3D Matx (F x 640 x 480)
                2) Hand Skeletons from f=1 to t=F -- 3D Matx (F x 21 x 3)
                3) Action Class One-Hot Label (optional to use) -- 2D Matx (F x 45)
                4) Total Frames == F
            -> Unless later decided, F will VARY for each sample
    '''
    HAR = 'har'
    HPE = 'hpe'


@unique
class FHADCameraIntrinsics(Enum):
    # Focal Length
    FX = 475.065948
    FY = 475.065857

    # image center
    UX = 315.944855
    UY = 245.287079


#@unique -- can't put this here as FX==FY
class MSRACameraIntrinsics(Enum):
    # Focal Length
    FX = 241.42
    FY = 241.42

    # image center
    UX = 160.0
    UY = 120.0


@unique
class DepthParameters(Enum):
    OUT_PX=128
    #DPT_RANGE_MM=200
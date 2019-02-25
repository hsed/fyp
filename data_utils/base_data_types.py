from enum import Enum, IntEnum, unique

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
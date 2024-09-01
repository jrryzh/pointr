import numpy as np
import os
import json

"""
    RANSAC for Similarity Transformation Estimation
    Modified from https://github.com/hughw19/NOCS_CVPR2019
    Originally Written by Srinath Sridhar
"""

def estimateSimilarityUmeyama(SourceHom, TargetHom):
    # Copy of original paper is at: http://web.stanford.edu/class/cs273/refs/umeyama.pdf
    SourceCentroid = np.mean(SourceHom[:3, :], axis=1)
    TargetCentroid = np.mean(TargetHom[:3, :], axis=1)
    nPoints = SourceHom.shape[1]
    CenteredSource = SourceHom[:3, :] - np.tile(SourceCentroid, (nPoints, 1)).transpose()
    CenteredTarget = TargetHom[:3, :] - np.tile(TargetCentroid, (nPoints, 1)).transpose()
    CovMatrix = np.matmul(CenteredTarget, np.transpose(CenteredSource)) / nPoints
    if np.isnan(CovMatrix).any():
        print('nPoints:', nPoints)
        print(SourceHom.shape)
        print(TargetHom.shape)
        raise RuntimeError('There are NANs in the input.')

    U, D, Vh = np.linalg.svd(CovMatrix, full_matrices=True)
    d = (np.linalg.det(U) * np.linalg.det(Vh)) < 0.0
    if d:
        D[-1] = -D[-1]
        U[:, -1] = -U[:, -1]
    # rotation
    Rotation = np.matmul(U, Vh)
    # scale
    varP = np.var(SourceHom[:3, :], axis=1).sum()
    Scale = 1 / varP * np.sum(D)
    # translation
    Translation = TargetHom[:3, :].mean(axis=1) - SourceHom[:3, :].mean(axis=1).dot(Scale*Rotation.T)
    # transformation matrix
    OutTransform = np.identity(4)
    OutTransform[:3, :3] = Scale * Rotation
    OutTransform[:3, 3] = Translation

    return Scale, Rotation, Translation, OutTransform


def estimateSimilarityTransform(source: np.array, target: np.array, verbose=False):
    """ Add RANSAC algorithm to account for outliers.
    """
    assert source.shape[0] == target.shape[0], 'Source and Target must have same number of points.'
    SourceHom = np.transpose(np.hstack([source, np.ones([source.shape[0], 1])]))
    TargetHom = np.transpose(np.hstack([target, np.ones([target.shape[0], 1])]))
    # Auto-parameter selection based on source heuristics
    # Assume source is object model or gt nocs map, which is of high quality
    SourceCentroid = np.mean(SourceHom[:3, :], axis=1)
    nPoints = SourceHom.shape[1]
    CenteredSource = SourceHom[:3, :] - np.tile(SourceCentroid, (nPoints, 1)).transpose()
    SourceDiameter = 2 * np.amax(np.linalg.norm(CenteredSource, axis=0))
    InlierT = SourceDiameter / 10.0  # 0.1 of source diameter
    maxIter = 128
    confidence = 0.99

    if verbose:
        print('Inlier threshold: ', InlierT)
        print('Max number of iterations: ', maxIter)

    BestInlierRatio = 0
    BestInlierIdx = np.arange(nPoints)
    for i in range(0, maxIter):
        # Pick 5 random (but corresponding) points from source and target
        RandIdx = np.random.randint(nPoints, size=5)
        Scale, _, _, OutTransform = estimateSimilarityUmeyama(SourceHom[:, RandIdx], TargetHom[:, RandIdx])
        PassThreshold = Scale * InlierT    # propagate inlier threshold to target scale
        Diff = TargetHom - np.matmul(OutTransform, SourceHom)
        ResidualVec = np.linalg.norm(Diff[:3, :], axis=0)
        InlierIdx = np.where(ResidualVec < PassThreshold)[0]
        nInliers = InlierIdx.shape[0]
        InlierRatio = nInliers / nPoints
        # update best hypothesis
        if InlierRatio > BestInlierRatio:
            BestInlierRatio = InlierRatio
            BestInlierIdx = InlierIdx
        if verbose:
            print('Iteration: ', i)
            print('Inlier ratio: ', BestInlierRatio)
        # early break
        if (1 - (1 - BestInlierRatio ** 5) ** i) > confidence:
            break

    if(BestInlierRatio < 0.1):
        #print('[ WARN ] - Something is wrong. Small BestInlierRatio: ', BestInlierRatio)
        return None, None, None, None

    SourceInliersHom = SourceHom[:, BestInlierIdx]
    TargetInliersHom = TargetHom[:, BestInlierIdx]
    Scale, Rotation, Translation, OutTransform = estimateSimilarityUmeyama(SourceInliersHom, TargetInliersHom)

    if verbose:
        print('BestInlierRatio:', BestInlierRatio)
        print('Rotation:\n', Rotation)
        print('Translation:\n', Translation)
        print('Scale:', Scale)

    return Scale, Rotation, Translation, OutTransform

########################## umeyama ################################################


def read_obj_as_numpy(file_path):
    vertices = []
    with open(file_path, 'r') as file:
        for line in file:
            if line.startswith('v '):
                parts = line.split()
                vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
    return np.array(vertices)

def pc_normalize(pc):
    """ pc: NxC, return NxC """
    l = pc.shape[0]
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    scale = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
    pc = pc / scale
    return pc, centroid, scale


PREDICTED_DIR = '/data/nas/zjy/code_repo/pointr/experiments/AdaPoinTr/PartialSpace_ShapeNet55_models/test_test_median/obj_output/'
OUTPUT_DIR = '/data/nas/zjy/code_repo/pointr/tests/output/'

_shapenet_dict = json.load(open('./data/shapenet_synset_dict.json', 'r'))
shapenet_dict = {}
for k, v in _shapenet_dict.items():
    shapenet_dict[v] = k

predicted_obj_lst = [f for f in os.listdir(PREDICTED_DIR) if f.endswith("output.obj")]

pc_path = "/data/nas/zjy/code_repo/pointr/data/ShapeNet55-34/shapenet_pc"

for obj_file in predicted_obj_lst:
    taxonomy = obj_file.split('_')[0]
    taxonomy_id = shapenet_dict[taxonomy]
    model_id = obj_file.split('_')[1]
    
    gt_file = taxonomy_id + '-' + model_id + '.npy'
    
    _predicted_obj = read_obj_as_numpy(os.path.join(PREDICTED_DIR, obj_file))
    _gt_obj = np.load(os.path.join(pc_path, gt_file)).astype(np.float32)
    
    predicted_obj, _, _ = pc_normalize(_predicted_obj)
    gt_obj, _, _ = pc_normalize(_gt_obj)
    
    # DEBUG
    print()

    _, _, _, pred_sRT = estimateSimilarityTransform(gt_obj, predicted_obj, False) 
    
    print(pred_sRT)
    
    if pred_sRT is None:
        pred_sRT = np.identity(4, dtype=float)

    pred_sRT[1, :3] = -pred_sRT[1, :3]
    pred_sRT[2, :3] = -pred_sRT[2, :3]

    '''recovered pose(RT) by Umeyama composes of size factor(s)'''
    s1 = np.cbrt(np.linalg.det(pred_sRT[:3, :3]))
    pred_sRT[:3, :3] = pred_sRT[:3, :3] / s1
    
    transformed_gt_obj = np.dot(pred_sRT[:3, :3], gt_obj.T).T + pred_sRT[:3, 3]
    
    # np.save(os.path.join(OUTPUT_DIR, 'gt_obj.npy'), gt_obj)
    # np.save(os.path.join(OUTPUT_DIR, 'transformed_gt_obj.npy'), transformed_gt_obj)
    with open(os.path.join(OUTPUT_DIR, 'predicted.obj'), 'w') as f:
        for point in predicted_obj:
            f.write('v {:.6f} {:.6f} {:.6f}\n'.format(point[0], point[1], point[2]))

    with open(os.path.join(OUTPUT_DIR, 'transformed_gt_obj.obj'), 'w') as f:
        for point in transformed_gt_obj:
            f.write('v {:.6f} {:.6f} {:.6f}\n'.format(point[0], point[1], point[2]))

    break

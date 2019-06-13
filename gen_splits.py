import os
import glob

def gen_split(root_dir, test_id, stackSize=5):
    trainDatasetF = []
    testDatasetF = []
    trainLabels = []
    testLabels = []
    trainNumFrames = []
    testNumFrames = []
    class_names = []
    for dir_id in range(4):
        dirF = os.path.join(root_dir, 'frames/S' + str(dir_id+1))
        class_id = 0
        for target in sorted(os.listdir(dirF)):
            if target not in class_names:
                class_names.append(target)
            dirF1 = os.path.join(dirF, target)
            for inst in sorted(os.listdir(dirF1)):
                inst_dirF = os.path.join(dirF1, inst)
                numFrames = len(glob.glob1(inst_dirF, '*.jpg'))
                if numFrames >= stackSize:
                    if dir_id+1 == test_id:
                        testDatasetF.append(inst_dirF)
                        testLabels.append(class_id)
                        testNumFrames.append(numFrames)
                    else:
                        trainDatasetF.append(inst_dirF)
                        trainLabels.append(class_id)
                        trainNumFrames.append(numFrames)
            class_id += 1
    return trainDatasetF, testDatasetF, trainLabels, testLabels, trainNumFrames, testNumFrames, class_names

def gen_split_hmdb51(dir, file_split):
        imagesTrain = []
        imagesTest = []
        labelsTrain = []
        labelsTest = []
        numFramesTrain = []
        numFramesTest = []
        file_splitTrain = os.path.join(dir, 'train_test_lists', 'train_split' + str(file_split) + '.txt')
        file_splitTest = os.path.join(dir, 'train_test_lists', 'test_split' + str(file_split) + '.txt')
        fid = open(file_splitTrain)
        vid_files = fid.readlines()
        for ind in range(len(vid_files)):
            img_name_split = vid_files[ind].split(' ')
            imagesTrain.append(os.path.join(dir, 'frames', img_name_split[0]))
            labelsTrain.append(int(img_name_split[2]))
            numFramesTrain.append(int(img_name_split[1]))

        fid = open(file_splitTest)
        vid_files = fid.readlines()
        for ind in range(len(vid_files)):
            img_name_split = vid_files[ind].split(' ')
            imagesTest.append(os.path.join(dir, 'frames', img_name_split[0]))
            labelsTest.append(int(img_name_split[2]))
            numFramesTest.append(int(img_name_split[1]))
        return imagesTrain, imagesTest, labelsTrain, labelsTest, numFramesTrain, numFramesTest

def gen_split_egtea_gazePlus(root_dir, test_inst):
    train_data = open(os.path.join(root_dir, 'train_split' + str(test_inst) + '.txt'), 'r').read().splitlines()
    test_data = open(os.path.join(root_dir, 'test_split' + str(test_inst) + '.txt'), 'r').read().splitlines()
    trainDataset = []
    testDataset = []
    trainLabels = []
    testLabels = []
    trainNumFrames = []
    testNumFrames = []
    for line1 in train_data:
        inst = line1.split(' ')
        trainDataset.append(os.path.join(root_dir, 'frames', inst[0]))
        trainLabels.append(int(inst[1]) - 1)
        trainNumFrames.append(len(glob.glob1(trainDataset[-1], '*.jpg')))
    for line1 in test_data:
        inst = line1.split(' ')
        testDataset.append(os.path.join(root_dir, 'frames', inst[0]))
        testLabels.append(int(inst[1]) - 1)
        testNumFrames.append(len(glob.glob1(testDataset[-1], '*.jpg')))

    return trainDataset, testDataset, trainLabels, testLabels, trainNumFrames, testNumFrames


def gen_split_twoStream(root_dir, test_id, stackSize):
    trainDatasetX = []
    trainDatasetY = []
    trainDatasetF = []
    testDatasetX = []
    testDatasetY = []
    testDatasetF = []
    trainLabels = []
    testLabels = []
    trainNumFrames = []
    testNumFrames = []
    class_names = []
    for dir_id in range(4):
        dirX = os.path.join(root_dir, 'flow_x/S' + str(dir_id+1))
        dirY = os.path.join(root_dir, 'flow_y/S' + str(dir_id+1))
        dirF = os.path.join(root_dir, 'frames/S' + str(dir_id+1))
        class_id = 0
        for target in sorted(os.listdir(dirX)):
            if target not in class_names:
                class_names.append(target)
            dirX1 = os.path.join(dirX, target)
            dirY1 = os.path.join(dirY, target)
            dirF1 = os.path.join(dirF, target)
            for inst in sorted(os.listdir(dirX1)):
                inst_dirX = os.path.join(dirX1, inst)
                inst_dirY = os.path.join(dirY1, inst)
                inst_dirF = os.path.join(dirF1, inst)
                numFrames = len(glob.glob1(inst_dirX, '*.jpg'))
                if numFrames >= stackSize:
                    if dir_id+1 == test_id:
                        testDatasetX.append(inst_dirX)
                        testDatasetY.append(inst_dirY)
                        testDatasetF.append(inst_dirF)
                        testLabels.append(class_id)
                        testNumFrames.append(numFrames)
                    else:
                        trainDatasetX.append(inst_dirX)
                        trainDatasetY.append(inst_dirY)
                        trainDatasetF.append(inst_dirF)
                        trainLabels.append(class_id)
                        trainNumFrames.append(numFrames)
            class_id += 1
    return trainDatasetX, trainDatasetY, trainDatasetF, testDatasetX, testDatasetY, testDatasetF, trainLabels, testLabels, trainNumFrames, testNumFrames, class_names


def gen_split_flow_action(root_dir, test_id, stackSize):
    trainDatasetX = []
    trainDatasetY = []
    testDatasetX = []
    testDatasetY = []
    trainLabels = []
    testLabels = []
    trainNumFrames = []
    testNumFrames = []
    for dir_id in range(4):
        dirX = os.path.join(root_dir, 'flow_x/S' + str(dir_id+1))
        dirY = os.path.join(root_dir, 'flow_y/S' + str(dir_id+1))
        for target in sorted(os.listdir(dirX)):
            dirX1 = os.path.join(dirX, target)
            dirY1 = os.path.join(dirY, target)
            for inst in sorted(os.listdir(dirX1)):
                inst_dirX = os.path.join(dirX1, inst)
                inst_dirY = os.path.join(dirY1, inst)
                numFrames = len(glob.glob1(inst_dirX, '*.jpg'))
                label_fl = open(os.path.join(inst_dirX, 'action_label.txt'), 'r')
                action_label = int(label_fl.readlines()[0])
                label_fl.close()
                if numFrames >= stackSize:
                    if dir_id+1 == test_id:
                        testDatasetX.append(inst_dirX)
                        testDatasetY.append(inst_dirY)
                        testLabels.append(action_label)
                        testNumFrames.append(numFrames)
                    else:
                        trainDatasetX.append(inst_dirX)
                        trainDatasetY.append(inst_dirY)
                        trainLabels.append(action_label)
                        trainNumFrames.append(numFrames)
    return trainDatasetX, trainDatasetY, testDatasetX, testDatasetY, trainLabels, testLabels, trainNumFrames, testNumFrames


def gen_split_egtea_flow_action(root_dir, test_inst):
    train_data = open(os.path.join(root_dir, 'train_split' + str(test_inst) + '.txt'), 'r').read().splitlines()
    test_data = open(os.path.join(root_dir, 'test_split' + str(test_inst) + '.txt'), 'r').read().splitlines()
    trainDatasetX = []
    trainDatasetY = []
    testDatasetX = []
    testDatasetY = []
    trainLabels = []
    testLabels = []
    trainNumFrames = []
    testNumFrames = []
    for line1 in train_data:
        inst = line1.split(' ')
        trainDatasetX.append(os.path.join(root_dir, 'flow_x', inst[0]))
        trainDatasetY.append(os.path.join(root_dir, 'flow_y', inst[0]))
        trainLabels.append(int(inst[2]) - 1)
        trainNumFrames.append(len(glob.glob1(trainDatasetX[-1], '*.jpg')))
    for line1 in test_data:
        inst = line1.split(' ')
        testDatasetX.append(os.path.join(root_dir, 'flow_x', inst[0]))
        testDatasetY.append(os.path.join(root_dir, 'flow_y', inst[0]))
        testLabels.append(int(inst[2]) - 1)
        testNumFrames.append(len(glob.glob1(testDatasetX[-1], '*.jpg')))

    return trainDatasetX, trainDatasetY, testDatasetX, testDatasetY, trainLabels, testLabels, trainNumFrames, testNumFrames

def gen_split_flow_attention(root_dir, test_id, stackSize):
    trainDatasetX = []
    trainDatasetY = []
    testDatasetX = []
    testDatasetY = []
    trainLabels = []
    testLabels = []
    trainNumFrames = []
    testNumFrames = []
    class_id = 0
    for dir_id in range(4):
        dirX = os.path.join(root_dir, 'flow_x/S' + str(dir_id+1))
        dirY = os.path.join(root_dir, 'flow_y/S' + str(dir_id+1))
        class_id = 0
        for target in sorted(os.listdir(dirX)):
            dirX1 = os.path.join(dirX, target)
            dirY1 = os.path.join(dirY, target)
            for inst in sorted(os.listdir(dirX1)):
                inst_dirX = os.path.join(dirX1, inst)
                inst_dirY = os.path.join(dirY1, inst)
                numFrames = len(glob.glob1(inst_dirX, '*.jpg'))
                if numFrames >= stackSize:
                    if dir_id+1 == test_id:
                        testDatasetX.append(inst_dirX)
                        testDatasetY.append(inst_dirY)
                        testLabels.append(class_id)
                        testNumFrames.append(numFrames)
                    else:
                        trainDatasetX.append(inst_dirX)
                        trainDatasetY.append(inst_dirY)
                        trainLabels.append(class_id)
                        trainNumFrames.append(numFrames)
            class_id += 1
    return trainDatasetX, trainDatasetY, testDatasetX, testDatasetY, trainLabels, testLabels, trainNumFrames, testNumFrames

def gen_split_egtea_flow_attention(root_dir, test_inst):
    train_data = open(os.path.join(root_dir, 'train_split' + str(test_inst) + '.txt'), 'r').read().splitlines()
    test_data = open(os.path.join(root_dir, 'test_split' + str(test_inst) + '.txt'), 'r').read().splitlines()
    trainDatasetX = []
    trainDatasetY = []
    testDatasetX = []
    testDatasetY = []
    trainLabels = []
    testLabels = []
    trainNumFrames = []
    testNumFrames = []
    for line1 in train_data:
        inst = line1.split(' ')
        trainDatasetX.append(os.path.join(root_dir, 'flow_x', inst[0]))
        trainDatasetY.append(os.path.join(root_dir, 'flow_y', inst[0]))
        trainLabels.append(int(inst[1]) - 1)
        trainNumFrames.append(len(glob.glob1(trainDatasetX[-1], '*.jpg')))
    for line1 in test_data:
        inst = line1.split(' ')
        testDatasetX.append(os.path.join(root_dir, 'flow_x', inst[0]))
        testDatasetY.append(os.path.join(root_dir, 'flow_y', inst[0]))
        testLabels.append(int(inst[1]) - 1)
        testNumFrames.append(len(glob.glob1(testDatasetX[-1], '*.jpg')))

    return trainDatasetX, trainDatasetY, testDatasetX, testDatasetY, trainLabels, testLabels, trainNumFrames, testNumFrames

def gen_split_egtea_twoStream(root_dir, test_inst):
    train_data = open(os.path.join(root_dir, 'train_split' + str(test_inst) + '.txt'), 'r').read().splitlines()
    test_data = open(os.path.join(root_dir, 'test_split' + str(test_inst) + '.txt'), 'r').read().splitlines()
    trainDatasetX = []
    trainDatasetY = []
    trainDatasetF = []
    testDatasetX = []
    testDatasetY = []
    testDatasetF = []
    trainLabels = []
    testLabels = []
    trainNumFrames = []
    testNumFrames = []
    for line1 in train_data:
        inst = line1.split(' ')
        trainDatasetX.append(os.path.join(root_dir, 'flow_x', inst[0]))
        trainDatasetY.append(os.path.join(root_dir, 'flow_y', inst[0]))
        trainDatasetF.append(os.path.join(root_dir, 'frames', inst[0]))
        trainLabels.append(int(inst[1]) - 1)
        trainNumFrames.append(len(glob.glob1(trainDatasetX[-1], '*.jpg')))
    for line1 in test_data:
        inst = line1.split(' ')
        testDatasetX.append(os.path.join(root_dir, 'flow_x', inst[0]))
        testDatasetY.append(os.path.join(root_dir, 'flow_y', inst[0]))
        testDatasetF.append(os.path.join(root_dir, 'frames', inst[0]))
        testLabels.append(int(inst[1]) - 1)
        testNumFrames.append(len(glob.glob1(testDatasetX[-1], '*.jpg')))

    return trainDatasetX, trainDatasetY, trainDatasetF, testDatasetX, testDatasetY, testDatasetF, trainLabels, testLabels, trainNumFrames, testNumFrames

def gen_dataset_sanity(root_dir, stackSize):
    trainDatasetF = []
    trainLabels = []
    trainNumFrames = []
    class_names = []
    for dir1 in sorted(os.listdir(root_dir)):
        print(dir1)
        dirF = os.path.join(root_dir, dir1)
        class_id = 0
        for target in sorted(os.listdir(dirF)):
            if target not in class_names:
                class_names.append(target)
            dirF1 = os.path.join(dirF, target)
            for inst in sorted(os.listdir(dirF1)):
                inst_dirF = os.path.join(dirF1, inst)
                numFrames = len(glob.glob1(inst_dirF, '*.jpg'))
                if numFrames >= stackSize:
                    trainDatasetF.append(inst_dirF)
                    trainLabels.append(class_id)
                    trainNumFrames.append(numFrames)
            class_id += 1
    return trainDatasetF, trainLabels, trainNumFrames, class_names
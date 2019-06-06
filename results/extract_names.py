import os
from tensorboard.backend.event_processing import event_accumulator
import collections
import csv
import yaml

from tqdm import tqdm
import argparse

def summary_to_csv(events_file, output_folder, summaries=['scalars']):

    inputLogFile = events_file
    outputFolder = output_folder
    ea = event_accumulator.EventAccumulator(inputLogFile,
    size_guidance={
        event_accumulator.COMPRESSED_HISTOGRAMS: 0, # 0 = grab all
        event_accumulator.IMAGES: 0,
        event_accumulator.AUDIO: 0,
        event_accumulator.SCALARS: 0,
        event_accumulator.HISTOGRAMS: 0,
    })
    #print("inputlog", inputLogFile)
    # print(' ')
    # print('Loading events from file*...')
    # print('* This might take a while. Sit back, relax and enjoy a cup of coffee :-)')
    # with Timer():
    ea.Reload() # loads events from file

    # print(' ')
    # print('Log summary:')
    tags = ea.Tags()
    # for t in tags:
    #     tagSum = []
    #     if (isinstance(tags[t],collections.Sequence)):
    #         tagSum = str(len(tags[t])) + ' summaries'
    #     else:
    #         tagSum = str(tags[t])
    #     print('   ' + t + ': ' + tagSum)

    if not os.path.isdir(outputFolder):
        os.makedirs(outputFolder)

    if ('images' in summaries):
        print(' ')
        print('Exporting images...')
        imageDir = outputFolder + 'images'
        print('Image dir: ' + imageDir)
        with Timer():
            imageTags = tags['images']
            for imageTag in imageTags:
                images = ea.Images(imageTag)
                imageTagDir = imageDir + '/' + imageTag
                if not os.path.isdir(imageTagDir):
                    os.makedirs(imageTagDir)
                for image in images:
                    imageFilename = imageTagDir + '/' + str(image.step) + '.png'
                    with open(imageFilename,'wb') as f:
                        f.write(image.encoded_image_string)

    if ('scalars' in summaries):
        csvFileName =  os.path.join(outputFolder,'scalars.csv')
        #print('Exporting scalars to csv-file...')
        #print('   CSV-path: ' + csvFileName)
        scalarTags = tags['scalars']
        if len(scalarTags) == 0:
            print("\nWarning: skipped file %s due to no scalars present, possibly incomplete run..." % inputLogFile)
            return

        with open(csvFileName,'w') as csvfile:
            logWriter = csv.writer(csvfile, delimiter=',')

            # Write headers to columns
            headers = ['wall_time','step']
            for s in scalarTags:
                headers.append(s)
            logWriter.writerow(headers)
    
            vals = ea.Scalars(scalarTags[0])
            for i in range(len(vals)):
                v = vals[i]
                data = [v.wall_time, v.step]
                for s in scalarTags:
                    scalarTag = ea.Scalars(s)
                    if i >= len(scalarTag):
                        print("\nWarning: scalar value missing writing empty string...")
                        data.append('')
                    else:
                        S = scalarTag[i]
                        data.append(S.value)
                logWriter.writerow(data)

def split_tags(dirname):
    pass

    #run_BaselineHPE_0509_1106-tag-train_avg_3d_err_mm.csv

    '''
        run_BaselineHPE_0509_1106-tag-train_avg_3d_err_mm.csv

        split by '-':
        [run_BaselineHPE_0509_1106, tag, train_avg_3d_err_mm]

        return
        {
            model: 'BaselineHPE'
            runs: [
                0509_1106, ..., 0510_1232,
            ]
            tag: 'avg_3d_err_mm

        }
    '''
    assert os.path.isdir(dirname), ("Path %s is not a directory!" % dirname)

    filenames = os.listdir(dirname)
    split_names = [item.split('-') for item in filenames]

    try:
        runs, _, tags = zip(*split_names)
    except ValueError:
        print("Ensure only relevant files are present in dir!")
        print("Split Names:\n", split_names)
    
    runs_split = map(lambda lst: (lst[1], lst[2] + '_' + lst[3]), (run.split('_') for run in runs) )
    runs_zipped = zip(*runs_split) # model_names, runtimes
    model_name, runtimes = (set(item) for item in runs_zipped) 
    
    assert len(model_name) == 1, "Only one model name allowed!"
    
    model_name = list(model_name)[0]
    runtimes = sorted(list(runtimes))
    #names_set = set(model_names)


    tags_zipped = zip(*map(lambda lst: (lst[1], lst[2][:-4]), (tag.split('_') for tag in tags)))
    train_or_val, tag_name = (set(item) for item in runs_zipped) 



def create_csv(base_path):
    dirs = list(filter(lambda path: os.path.isdir(os.path.join(base_path, path)), os.listdir(base_path)))
    #print(list(dirs))

    with tqdm(desc='Reading dirs', total=len(dirs)) as pbar:
        for dir in dirs:
            pbar.desc = 'Current ' + dir
            events = list(filter(lambda item: item[:6] == 'events', os.listdir(os.path.join(base_path, dir))))
            for event in events:
                event_dir = os.path.normpath(os.path.join(base_path, dir))
                event_filename = os.path.normpath(os.path.join(base_path, dir, event))
                #print('event_filename', event_filename)
                summary_to_csv(event_filename, event_dir, summaries=['scalars'])
            pbar.update(1)



if __name__ == '__main__':
    '''
        $> ls
        $> configs models logs [...]
        $> python results\extract_names.py -l "results/hpe_logs"
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--log_dir', required=True, help='The bottom-most path to a folder containing directories of events')
    args = parser.parse_args()
    create_csv(args.log_dir) #'logs/BaselineHPE'

    
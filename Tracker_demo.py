import cv2
import os
import numpy as np
import sys
import ROLO_utils as utils



def main(argv):
    ''' PARAMETERS '''
    num_steps= 6
    test = 11

    [wid, ht, sequence_name, dummy_1, dummy_2] = utils.choose_video_sequence(test)

    img_fold_path = os.path.join('/home/mdeshpa3/Tracker/Data', sequence_name, 'img/')
    gt_file_path= os.path.join('/home/mdeshpa3/Tracker/Data', sequence_name, 'groundtruth_rect.txt')
    yolo_out_path= os.path.join('/home/mdeshpa3/Tracker/Data', sequence_name, 'yolo_out/')
    rolo_out_path= os.path.join('/home/mdeshpa3/Tracker/Data', sequence_name, 'rolo_out_test/')

    paths_imgs = utils.load_folder( img_fold_path)
    paths_rolo= utils.load_folder( rolo_out_path)
    lines = utils.load_dataset_gt( gt_file_path)

    # Define the codec and create VideoWriter object
    fourcc= cv2.cv.CV_FOURCC(*'DIVX')
    video_name = sequence_name + '_test.avi'
    video_path = os.path.join('output/videos/', video_name)
    video = cv2.VideoWriter(video_path, fourcc, 20, (wid, ht))

    total= 0
    rolo_avgloss= 0
    yolo_avgloss= 0
    for i in range(len(paths_rolo)- num_steps):
        id= i + 1
        test_id= id + num_steps - 2  #* num_steps + 1

        path = paths_imgs[test_id]
        img = utils.file_to_img(path)

        if(img is None): break

        yolo_location= utils.find_yolo_location(yolo_out_path, test_id)
        yolo_location= utils.locations_normal( wid, ht, yolo_location)
        print(yolo_location)

        rolo_location= utils.find_rolo_location( rolo_out_path, test_id)
        rolo_location = utils.locations_normal( wid, ht, rolo_location)
        print(rolo_location)

        gt_location = utils.find_gt_location( lines, test_id - 1)d
        print('gt: ' + str(test_id))
        print(gt_location)

        frame = utils.debug_3_locations( img, gt_location, yolo_location, rolo_location)
        video.write(frame)

        utils.createFolder(os.path.join('output/frames/',sequence_name))
        frame_name= os.path.join('output/frames/',sequence_name,str(test_id)+'.jpg')
        print(frame_name)
        cv2.imwrite(frame_name, frame)

        rolo_loss = utils.cal_rolo_IOU(rolo_location, gt_location)
        rolo_avgloss += rolo_loss
        yolo_loss=  utils.cal_yolo_IOU(yolo_location, gt_location)
        yolo_avgloss += yolo_loss
        total += 1

    rolo_avgloss /= total
    yolo_avgloss /= total
    print("yolo_avg_iou = ", yolo_avgloss)
    print("rolo_avg_iou = ", rolo_avgloss)
    video.release()
    cv2.destroyAllWindows()



if __name__=='__main__':
	main(sys.argv)

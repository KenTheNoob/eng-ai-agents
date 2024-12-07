Assignment-3<br>
11/17/2024<br>
assignment-3b.ipynb performs Faster RCNN on the Football match video<br>
The file explains how Faster RCNN works, then runs an untrained version of it on a single image<br>
Next, the file runs a pretrained model of Faster RCNN on a single image<br>
Finally, the model runs the model on all the frames of the video then combines the results into ObjectTracking.mp4<br>
The resulting video has a lot of flickering since no object tracking methods or Kalman Filters were used<br>
Each frame of the resulting video after non-maximum supression is shown in the results folder<br>

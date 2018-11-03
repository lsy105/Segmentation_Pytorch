in='VOC2007/img/000068.jpg'
model='checkpoints/CP30.pth'
python predict.py -m $model -i $in -o output.jpg --cpu --viz

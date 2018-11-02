num_class=21
img_dir='VOC2007/img/'
label_dir='VOC2007/label/'
save_dir='checkpoints/'
learning_rate=0.01
batchsize=1
epochs=30
lr_step=10

#if want use GPU, please add option --gpu
python train.py --img_dir $img_dir --label_dir $label_dir --save_dir $save_dir --num_cls $num_class \
                --epochs $epochs --batch_size $batchsize --learning_rate $learning_rate --lr_step $lr_step 
   

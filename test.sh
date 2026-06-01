datasets=(
    'Pima_Indians_Diabetes_Database'
)


gpu=3


for dataset in "${datasets[@]}"; do
    echo "Running $dataset"
    python train_model_deep.py --pretrain_model_path swift.ckpt --cat_policy indices --normalization none --seed_num 5 --gpu $gpu --dataset $dataset --dataset_path  "../data" > log/${dataset}-tabswift-16.txt
done

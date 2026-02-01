#!/bin/bash
MAX_JOBS=16
GPUS=(0 1 2 3 4 5 6 7)
TOTAL_GPUS=${#GPUS[@]}

get_gpu_allocation(){
    local job_number=$1
    # Calculate which GPU to allocate based on the job number
    local gpu_id=${GPUS[$((job_number % TOTAL_GPUS))]}
    echo $gpu_id
}

check_jobs(){
    while true; do
        jobs_count=$(jobs -p | wc -l)
        if [ "$jobs_count" -lt "$MAX_JOBS" ]; then
            break
        fi
        sleep 1
    done
}

job_number=0

DATA_ROOT=./dataset
OUT_ROOT=.
EXP_NAME=long_term
seed=2023
des='CFPT'

model_name=CFPT
auxi_mode=kernel_balancing
datasets=(M5)



# hyper-parameters
dst=M5

train_epochs=10
patience=3
test_batch_size=1
lambda=1.0
lr=0.0001
batch_size=8
lradj=type1
period=4
beta=0.5
d_model=512
rda=1
rdb=1
kernel_size=3
e_layers=3
dropout=0.0
time_feature_types="['DayOfYear','DayOfMonth']"
rerun=0

# NOTE: M5 settings

train_epochs=100
patience=15
pl_list=(8 12 20 28)


for pl in ${pl_list[@]}; do
    if ! [[ " ${datasets[@]} " =~ " ${dst} " ]]; then
        continue
    fi

    case $pl in
        8) alpha=0.9 lr=0.0002 lradj=type1 batch_size=32 inner_lr=0.0005 inner_optim=adam inner_step=1 auxi_type=akb auxi_loss=AKB J=1 gamma=1.0 kernel_type=exp solver_type=kdiff reg=0.001 C=0.05 normed=1 joint_forecast=1 l2_reg=0.0;;
        12) alpha=0.95 lr=0.001 lradj=type1 batch_size=32 inner_lr=0.0005 inner_optim=adam inner_step=1 auxi_type=akb auxi_loss=AKB J=5 gamma=0.5 kernel_type=exp solver_type=kdiff reg=0.001 C=0.005 normed=1 joint_forecast=1 l2_reg=0.0;;
        20) alpha=0.95 lr=0.001 lradj=type1 batch_size=32 inner_lr=0.0005 inner_optim=adam inner_step=1 auxi_type=akb auxi_loss=AKB J=2 gamma=1.0 kernel_type=exp solver_type=kdiff reg=0.001 C=0.01 normed=1 joint_forecast=1 l2_reg=0.0;;
        28) alpha=0.9 lr=0.001 lradj=constant batch_size=32 inner_lr=0.0005 inner_optim=adam inner_step=1 auxi_type=akb auxi_loss=AKB J=6 gamma=0.5 kernel_type=exp solver_type=kdiff reg=0.001 C=0.001 normed=1 joint_forecast=1 l2_reg=0.0;;
    esac

    case $pl in
        8) beta=0.5 d_model=512 kernel_size=3;;
        12) beta=0.5 d_model=512 kernel_size=3;;
        20) beta=0.7 d_model=512 kernel_size=5;;
        28) beta=0.3 d_model=512 kernel_size=2;;
    esac

    rl=$(echo "1 - $alpha" | bc)
    decimal_places=$(echo "$alpha" | awk -F. '{print length($2)}')
    rl=$(printf "%.${decimal_places}f" $rl)
    ax=$alpha

    JOB_NAME=${model_name}_${dst}_${pl}_${rl}_${ax}_${lr}_${lradj}_${train_epochs}_${patience}_${batch_size}_${inner_lr}_${inner_optim}_${inner_step}_${auxi_type}_${auxi_loss}_${J}_${gamma}_${kernel_type}_${solver_type}_${reg}_${C}_${normed}_${joint_forecast}_${l2_reg}
    OUTPUT_DIR="${OUT_ROOT}/results/${EXP_NAME}/${JOB_NAME}"

    CHECKPOINTS=$OUTPUT_DIR/checkpoints/
    RESULTS=$OUTPUT_DIR/results/
    TEST_RESULTS=$OUTPUT_DIR/test_results/
    LOG_PATH=$OUTPUT_DIR/result_long_term_forecast.txt

    mkdir -p "${OUTPUT_DIR}/"
    # if rerun, remove the previous stdout
    if [ $rerun -eq 1 ]; then
        rm -rf "${OUTPUT_DIR}/stdout.log"
    else
        subdirs=("$RESULTS"/*)
        if [ ${#subdirs[@]} -eq 1 ] && [ -f "${subdirs[0]}/metrics.yaml" ]; then
            echo ">>>>>>> Job: $JOB_NAME already run, skip <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<"
            continue
        fi
    fi


    check_jobs
    # Get GPU allocation for this job
    gpu_allocation=$(get_gpu_allocation $job_number)
    # Increment job number for the next iteration
    ((job_number++))

    echo "Running command for $JOB_NAME"
    {
        # Set CUDA_VISIBLE_DEVICES for this script and run it in the background
        CUDA_VISIBLE_DEVICES=$gpu_allocation python -u run.py \
            --task_name long_term_forecast \
            --is_training 1 \
            --root_path $DATA_ROOT/m5/ \
            --data_path m5.csv \
            --model_id "${dst}_96_${pl}" \
            --model ${model_name} \
            --data_id $dst \
            --data m5 \
            --features M \
            --seq_len 96 \
            --label_len 48 \
            --pred_len ${pl} \
            --enc_in 10 \
            --dec_in 10 \
            --c_out 10 \
            --factor 3 \
            --des ${des} \
            --learning_rate ${lr} \
            --lradj ${lradj} \
            --train_epochs ${train_epochs} \
            --patience ${patience} \
            --batch_size ${batch_size} \
            --test_batch_size ${test_batch_size} \
            --itr 1 \
            --rec_lambda ${rl} \
            --auxi_lambda ${ax} \
            --fix_seed ${seed} \
            --checkpoints $CHECKPOINTS \
            --results $RESULTS \
            --test_results $TEST_RESULTS \
            --log_path $LOG_PATH \
            --rerun $rerun \
            --beta $beta \
            --d_model $d_model \
            --kernel_size $kernel_size \
            --rda $rda \
            --rdb $rdb \
            --e_layers $e_layers \
            --dropout $dropout \
            --period $period \
            --time_feature_types $time_feature_types \
            --meta_lr ${inner_lr} \
            --meta_optim_type ${inner_optim} \
            --meta_inner_steps ${inner_step} \
            --auxi_type $auxi_type \
            --auxi_loss $auxi_loss \
            --auxi_mode $auxi_mode \
            --J $J \
            --C $C \
            --gamma $gamma \
            --kernel_type $kernel_type \
            --solver_type $solver_type \
            --reg_sk $reg \
            --use_norm $normed \
            --joint_forecast $joint_forecast \
            --l2_reg $l2_reg

        sleep 5
    } 2>&1 | tee -a "${OUTPUT_DIR}/stdout.log" &
done






wait
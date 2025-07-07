#!/bin/bash

#SBATCH --job-name=mlslt
#SBATCH --partition=batch
#SBATCH --time=2-0:00:00
#SBATCH --output=out/slurm-%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --mem=128G

# multilingual pretraining
echo "multillingual pre-training"
seeds=(42 43 44)
feat_types=("i3d" "s3d")
for feat_type in "${feat_types[@]}"; do
  for seed in "${seeds[@]}"; do
    config_file="configs/multilingual-pretraining-${feat_type}-seed-${seed}.yaml"
    srun \
      --container-image=/netscratch/enroot/nvcr.io_nvidia_pytorch_20.02-py3.sqsh \
      --container-workdir="$(pwd)" \
      --container-mounts=/netscratch:/netscratch,/ds:/ds:ro,"$(pwd)":"$(pwd)" \
      install.sh python -m signjoey train "$config_file"
    
    srun \
      --container-image=/netscratch/enroot/nvcr.io_nvidia_pytorch_20.02-py3.sqsh \
      --container-workdir="$(pwd)" \
      --container-mounts=/netscratch:/netscratch,/ds:/ds:ro,"$(pwd)":"$(pwd)" \
      install.sh python -m signjoey test "$config_file"

  done
done


# multilingual finetuning
seed=44
feat_type="s3d"
config_file="configs/multilingual-finetuning-${feat_type}-seed-${seed}.yaml"
srun \
  --container-image=/netscratch/enroot/nvcr.io_nvidia_pytorch_20.02-py3.sqsh \
  --container-workdir="$(pwd)" \
  --container-mounts=/netscratch:/netscratch,/ds:/ds:ro,"$(pwd)":"$(pwd)" \
  install.sh python -m signjoey train "$config_file"

srun \
  --container-image=/netscratch/enroot/nvcr.io_nvidia_pytorch_20.02-py3.sqsh \
  --container-workdir="$(pwd)" \
  --container-mounts=/netscratch:/netscratch,/ds:/ds:ro,"$(pwd)":"$(pwd)" \
  install.sh python -m signjoey test "$config_file"


# naive sequential finetuning
echo "Naive sequential finetuning"
mkdir -p $HOME/bin

# Download yq to the local bin directory
wget https://github.com/mikefarah/yq/releases/latest/download/yq_linux_amd64 -O $HOME/bin/yq

# Make yq executable
chmod +x $HOME/bin/yq

# Add the local bin directory to the PATH
export PATH=$HOME/bin:$PATH

# Verify the installation
which yq

seed=44
feat_type="s3d"
permutations=(
    "ASE GSG CSL"
    "ASE CSL GSG"
    "GSG ASE CSL"
    "GSG CSL ASE"
    "CSL ASE GSG"
    "CSL GSG ASE"
)

root_dir='results/naive-seq-finetuning'
config_file="configs/fine-tuning-naive-${feat_type}-seed-${seed}.yaml"
spec_test_config_file="configs/fine-tuning-naive-${feat_type}-seed-${seed}-test-spec.yaml"
test_config_file="configs/fine-tuning-naive-${feat_type}-seed-${seed}-test.yaml"
train_config_file="configs/fine-tuning-naive-${feat_type}-seed-${seed}-train.yaml"

mkdir -p $root_dir

# Iterate over each permutation and print each language with index
first_ind_txt_vocab="results/multilingual-pretraining-${feat_type}-seed-${seed}/txt.vocab"
first_ind_load_model="results/multilingual-pretraining-${feat_type}-seed-${seed}/best.ckpt"
model_dir_prefix="naive-seq-finetuning-${feat_type}-seed-${seed}-"

# Loop through each permutation
for perm in "${permutations[@]}"; do
    IFS=' ' read -r -a languages <<< "$perm"
    joined_perm=$(IFS=_; echo "${languages[*]}")
    model_dir_parent="$root_dir/$joined_perm/$model_dir_prefix"

    # Print each language with its index
    for i in "${!languages[@]}"; do
        index=$((i + 1))
        model_dir="$model_dir_parent${languages[$i]}"

        if [ "$index" -eq 1 ]; then
            txt_vocab="$first_ind_txt_vocab"
            load_model="$first_ind_load_model"
        else
            txt_vocab="$model_dir_parent${languages[$i-1]}/txt.vocab"
            load_model="$model_dir_parent${languages[$i-1]}/best.ckpt"
        fi

        echo "$index: ${languages[$i]}"
        echo "txt_vocab: $txt_vocab, load_model: $load_model, model_dir: $model_dir"

        # Update the training section in the YAML file
        awk -v txt_vocab="$txt_vocab" -v load_model="$load_model" -v model_dir="$model_dir" '
        {
            sub(/txt_vocab: .*/, "txt_vocab: \"" txt_vocab "\"");
            sub(/load_model: .*/, "load_model: \"" load_model "\"");
            sub(/model_dir: .*/, "model_dir: \"" model_dir "\"");
            print $0
        }
        ' "${config_file}" > "${train_config_file}"
        
        # Update the languages field under the training section
        lang="${languages[$i]}"
        yq eval --inplace ".training.languages = []" "${train_config_file}" 
        yq eval --inplace ".training.languages += [\"$lang\"]" "${train_config_file}"

        # training
        srun \
            --container-image=/netscratch/enroot/nvcr.io_nvidia_pytorch_20.02-py3.sqsh \
            --container-workdir="$(pwd)" \
            --container-mounts=/netscratch:/netscratch,/ds:/ds:ro,"$(pwd)":"$(pwd)" \
            install.sh python -m signjoey train "$train_config_file"

        # Update the test section in the YAML file (if needed)
        awk -v txt_vocab="$txt_vocab" -v load_model="$load_model" -v model_dir="$model_dir" \
            '{sub(/txt_vocab: .*/, "txt_vocab: \"" txt_vocab "\""); sub(/load_model: .*/, "load_model: \"" load_model "\""); sub(/model_dir: .*/, "model_dir: \"" model_dir "\"");} 1' "$test_config_file" > "${spec_test_config_file}"

        # testing
        srun \
            --container-image=/netscratch/enroot/nvcr.io_nvidia_pytorch_20.02-py3.sqsh \
            --container-workdir="$(pwd)" \
            --container-mounts=/netscratch:/netscratch,/ds:/ds:ro,"$(pwd)":"$(pwd)" \
            install.sh python -m signjoey test "$spec_test_config_file"

        echo  # Print an empty line for separation
    done
done


# language-specific finetuning
echo "Language-specific finetuning"
mkdir -p $HOME/bin

# Download yq to the local bin directory
wget https://github.com/mikefarah/yq/releases/latest/download/yq_linux_amd64 -O $HOME/bin/yq

# Make yq executable
chmod +x $HOME/bin/yq

# Add the local bin directory to the PATH
export PATH=$HOME/bin:$PATH

# Verify the installation
which yq

seed=44
feat_type="s3d"
languages=(
    "ASE"
    "GSG"
    "CSL"
)

root_dir='results/lang-spec-finetuning'
config_file="configs/lang-spec-fine-tuning-${feat_type}-seed-${seed}.yaml"
spec_test_config_file="configs/lang-spec-fine-tuning-${feat_type}-seed-${seed}-test-spec.yaml"
test_config_file="configs/lang-spec-fine-tuning-${feat_type}-seed-${seed}-test.yaml"
train_config_file="configs/lang-spec-fine-tuning-${feat_type}-seed-${seed}-train.yaml"

mkdir -p $root_dir

# Iterate over each permutation and print each language with index
model_dir_prefix="lang-spec-fine-tuning-${feat_type}-seed-${seed}-"

# Loop through each permutation
for lang in "${languages[@]}"; do
    model_dir_parent="$root_dir/$model_dir_prefix"

    # Print each language with its index
    model_dir="$model_dir_parent${lang}"

    echo "languages: $lang, model_dir: $model_dir"

    # Update the training section in the YAML file
    awk -v model_dir="$model_dir" '
    {
        sub(/model_dir: .*/, "model_dir: \"" model_dir "\"");
        print $0
    }
    ' "${config_file}" > "${train_config_file}"
        
    # Update the languages field under the training section
	yq eval --inplace ".training.languages = []" "${train_config_file}" 
	yq eval --inplace ".training.languages += [\"$lang\"]" "${train_config_file}"

    # training
    srun \
        --container-image=/netscratch/enroot/nvcr.io_nvidia_pytorch_20.02-py3.sqsh \
        --container-workdir="$(pwd)" \
        --container-mounts=/netscratch:/netscratch,/ds:/ds:ro,"$(pwd)":"$(pwd)" \
        install.sh python -m signjoey train "$train_config_file"

    # Update the test section in the YAML file (if needed)
    awk -v model_dir="$model_dir" \
        '{sub(/model_dir: .*/, "model_dir: \"" model_dir "\"");} 1' "$test_config_file" > "${spec_test_config_file}"

    # testing
    srun \
        --container-image=/netscratch/enroot/nvcr.io_nvidia_pytorch_20.02-py3.sqsh \
        --container-workdir="$(pwd)" \
        --container-mounts=/netscratch:/netscratch,/ds:/ds:ro,"$(pwd)":"$(pwd)" \
        install.sh python -m signjoey test "$spec_test_config_file"

    echo  # Print an empty line for separation
done


# joint incremental finetuning
echo "Joint incremental finetuning"
mkdir -p $HOME/bin

# Download yq to the local bin directory
wget https://github.com/mikefarah/yq/releases/latest/download/yq_linux_amd64 -O $HOME/bin/yq

# Make yq executable
chmod +x $HOME/bin/yq

# Add the local bin directory to the PATH
export PATH=$HOME/bin:$PATH

# Verify the installation
which yq

seed=44
feat_type="s3d"
permutations=(
    "ASE"
    "ASE GSG"
    "ASE GSG CSL"
    "ASE"
    "ASE CSL"
    "ASE CSL GSG"
    "GSG"
    "GSG ASE"
    "GSG ASE CSL"
    "GSG"
    "GSG CSL"
    "GSG CSL ASE"
    "CSL"
    "CSL ASE"
    "CSL ASE GSG"
    "CSL"
    "CSL GSG"
    "CSL GSG ASE"
)

root_dir='results/inc-joint-finetuning'
config_file="configs/inc-joint-fine-tuning-${feat_type}-seed-${seed}.yaml"
spec_test_config_file="configs/inc-joint-fine-tuning-${feat_type}-seed-${seed}-test-spec.yaml"
test_config_file="configs/inc-joint-fine-tuning-${feat_type}-seed-${seed}-test.yaml"
train_config_file="configs/inc-joint-fine-tuning-${feat_type}-seed-${seed}-train.yaml"

mkdir -p $root_dir

# Iterate over each permutation and print each language with index
first_ind_txt_vocab="results/multilingual-pretraining-${feat_type}-seed-${seed}/txt.vocab"
first_ind_load_model="results/multilingual-pretraining-${feat_type}-seed-${seed}/best.ckpt"
model_dir_prefix="inc-joint-finetuning-${feat_type}-seed-${seed}-"

# Loop through each permutation
for i in "${!permutations[@]}"; do
    index=$((i + 1))
    IFS=' ' read -r -a languages <<< "${permutations[$i]}"
    joined_perm=$(IFS=-; echo "${languages[*]}")

    model_dir="$root_dir/$index-$joined_perm/$model_dir_prefix${joined_perm}"

    if (( index % 3 == 1 )); then
        txt_vocab="$first_ind_txt_vocab"
        load_model="$first_ind_load_model"
    else
        IFS=' ' read -r -a prev_languages <<< "${permutations[$i-1]}"
        joined_prev_perm=$(IFS=-; echo "${prev_languages[*]}")
        model_dir_parent="$root_dir/$i-$joined_prev_perm/$model_dir_prefix"

        txt_vocab="$model_dir_parent${joined_prev_perm}/txt.vocab"
        load_model="$model_dir_parent${joined_prev_perm}/best.ckpt"
    fi

    echo "$index: ${joined_perm}"
    echo "txt_vocab: $txt_vocab, load_model: $load_model, model_dir: $model_dir"

    # Update the training section in the YAML file
    awk -v txt_vocab="$txt_vocab" -v load_model="$load_model" -v model_dir="$model_dir" '
    {
        sub(/txt_vocab: .*/, "txt_vocab: \"" txt_vocab "\"");
        sub(/load_model: .*/, "load_model: \"" load_model "\"");
        sub(/model_dir: .*/, "model_dir: \"" model_dir "\"");
        print $0
    }
    ' "${config_file}" > "${train_config_file}"
    
    # Update the languages field under the training section
    yq eval --inplace ".training.languages = []" "${train_config_file}" 
    for i in "${!languages[@]}"; do
        lang="${languages[$i]}"
        yq eval --inplace ".training.languages += [\"$lang\"]" "${train_config_file}"
    done

    # training
    srun \
        --container-image=/netscratch/enroot/nvcr.io_nvidia_pytorch_20.02-py3.sqsh \
        --container-workdir="$(pwd)" \
        --container-mounts=/netscratch:/netscratch,/ds:/ds:ro,"$(pwd)":"$(pwd)" \
        install.sh python -m signjoey train "$train_config_file"

    # Update the test section in the YAML file
    if (( index % 3 == 0 )); then
    awk -v txt_vocab="$txt_vocab" -v load_model="$load_model" -v model_dir="$model_dir" \
        '{sub(/txt_vocab: .*/, "txt_vocab: \"" txt_vocab "\""); sub(/load_model: .*/, "load_model: \"" load_model "\""); sub(/model_dir: .*/, "model_dir: \"" model_dir "\"");} 1' "$test_config_file" > "${spec_test_config_file}"

    # testing
    srun \
        --container-image=/netscratch/enroot/nvcr.io_nvidia_pytorch_20.02-py3.sqsh \
        --container-workdir="$(pwd)" \
        --container-mounts=/netscratch:/netscratch,/ds:/ds:ro,"$(pwd)":"$(pwd)" \
        install.sh python -m signjoey test "$spec_test_config_file"
    fi

    echo  # Print an empty line for separation
done


# ewc finetuning
echo "EWC finetuning"
mkdir -p $HOME/bin

# Download yq to the local bin directory
wget https://github.com/mikefarah/yq/releases/latest/download/yq_linux_amd64 -O $HOME/bin/yq

# Make yq executable
chmod +x $HOME/bin/yq

# Add the local bin directory to the PATH
export PATH=$HOME/bin:$PATH

# Verify the installation
which yq

seed=44
feat_type="s3d"
permutations=(
    "ASE GSG CSL"
    "ASE CSL GSG"
    "GSG ASE CSL"
    "GSG CSL ASE"
    "CSL ASE GSG"
    "CSL GSG ASE"
)

root_dir='results/fine-tuning-ewc'
config_file="configs/fine-tuning-ewc-${feat_type}-seed-${seed}.yaml"
spec_test_config_file="configs/fine-tuning-ewc-${feat_type}-seed-${seed}-test-spec.yaml"
test_config_file="configs/fine-tuning-ewc-${feat_type}-seed-${seed}-test.yaml"
train_config_file="configs/fine-tuning-ewc-${feat_type}-seed-${seed}-train.yaml"

mkdir -p $root_dir

# Iterate over each permutation and print each language with index
first_ind_txt_vocab="results/multilingual-pretraining-${feat_type}-seed-${seed}/txt.vocab"
first_ind_load_model="results/multilingual-pretraining-${feat_type}-seed-${seed}/best.ckpt"
model_dir_prefix="finetuning-ewc-${feat_type}-seed-${seed}-"

# Loop through each permutation
for perm in "${permutations[@]}"; do
    IFS=' ' read -r -a languages <<< "$perm"
    joined_perm=$(IFS=_; echo "${languages[*]}")
    model_dir_parent="$root_dir/$joined_perm/$model_dir_prefix"

    # Print each language with its index
    for i in "${!languages[@]}"; do
        index=$((i + 1))
        model_dir="$model_dir_parent${languages[$i]}"

        if [ "$index" -eq 1 ]; then
            txt_vocab="$first_ind_txt_vocab"
            load_model="$first_ind_load_model"
        else
            txt_vocab="$model_dir_parent${languages[$i-1]}/txt.vocab"
            load_model="$model_dir_parent${languages[$i-1]}/best.ckpt"
        fi

        echo "$index: ${languages[$i]}"
        echo "txt_vocab: $txt_vocab, load_model: $load_model, model_dir: $model_dir"

        # Update the training section in the YAML file
        awk -v txt_vocab="$txt_vocab" -v load_model="$load_model" -v model_dir="$model_dir" '
        {
            sub(/txt_vocab: .*/, "txt_vocab: \"" txt_vocab "\"");
            sub(/load_model: .*/, "load_model: \"" load_model "\"");
            sub(/model_dir: .*/, "model_dir: \"" model_dir "\"");
            print $0
        }
        ' "${config_file}" > "${train_config_file}"
        
        # Update the languages field under the training section
        lang="${languages[$i]}"
        yq eval --inplace ".training.languages = []" "${train_config_file}" 
        yq eval --inplace ".training.languages += [\"$lang\"]" "${train_config_file}"

        # training
        srun \
            --container-image=/netscratch/enroot/nvcr.io_nvidia_pytorch_20.02-py3.sqsh \
            --container-workdir="$(pwd)" \
            --container-mounts=/netscratch:/netscratch,/ds:/ds:ro,"$(pwd)":"$(pwd)" \
            install.sh python -m signjoey train "$train_config_file"
    done

        # Update the test section in the YAML file (if needed)
        awk -v txt_vocab="$txt_vocab" -v load_model="$load_model" -v model_dir="$model_dir" \
            '{sub(/txt_vocab: .*/, "txt_vocab: \"" txt_vocab "\""); sub(/load_model: .*/, "load_model: \"" load_model "\""); sub(/model_dir: .*/, "model_dir: \"" model_dir "\"");} 1' "$test_config_file" > "${spec_test_config_file}"

        # testing
        srun \
            --container-image=/netscratch/enroot/nvcr.io_nvidia_pytorch_20.02-py3.sqsh \
            --container-workdir="$(pwd)" \
            --container-mounts=/netscratch:/netscratch,/ds:/ds:ro,"$(pwd)":"$(pwd)" \
            install.sh python -m signjoey test "$spec_test_config_file"

        echo  # Print an empty line for separation
    done
done


#experience replay finetuning
echo "Experience replay finetuning"
mkdir -p $HOME/bin

# Download yq to the local bin directory
wget https://github.com/mikefarah/yq/releases/latest/download/yq_linux_amd64 -O $HOME/bin/yq

# Make yq executable
chmod +x $HOME/bin/yq

# Add the local bin directory to the PATH
export PATH=$HOME/bin:$PATH

# Verify the installation
which yq

seed=44
feat_type="s3d"
permutations=(
    "ASE"
    "ASE GSG"
    "ASE GSG CSL"
    "ASE"
    "ASE CSL"
    "ASE CSL GSG"
    "GSG"
    "GSG ASE"
    "GSG ASE CSL"
    "GSG"
    "GSG CSL"
    "GSG CSL ASE"
    "CSL"
    "CSL ASE"
    "CSL ASE GSG"
    "CSL"
    "CSL GSG"
    "CSL GSG ASE"
)

memory_sizes=(200)
config_file="configs/fine-tuning-er-${feat_type}-seed-${seed}.yaml"
spec_test_config_file="configs/fine-tuning-er-${feat_type}-seed-${seed}-test-spec.yaml"
test_config_file="configs/fine-tuning-er-${feat_type}-seed-${seed}-test.yaml"
train_config_file="configs/fine-tuning-er-${feat_type}-seed-${seed}-train.yaml"

# Iterate over each memory size
for memory_size in "${memory_sizes[@]}"; do
    root_dir="results/fine-tuning-er-$memory_size"
    mkdir -p $root_dir

    # Iterate over each permutation and print each language with index
    first_ind_txt_vocab="results/multilingual-pretraining-${feat_type}-seed-${seed}/txt.vocab"
    first_ind_load_model="results/multilingual-pretraining-${feat_type}-seed-${seed}/best.ckpt"
    model_dir_prefix="er-finetuning-${feat_type}-seed-${seed}-"

    # Loop through each permutation
    for i in "${!permutations[@]}"; do
        index=$((i + 1))
        IFS=' ' read -r -a languages <<< "${permutations[$i]}"
        joined_perm=$(IFS=-; echo "${languages[*]}")

        model_dir="$root_dir/$index-$joined_perm/$model_dir_prefix${joined_perm}"

        if (( index % 3 == 1 )); then
            txt_vocab="$first_ind_txt_vocab"
            load_model="$first_ind_load_model"
        else
            IFS=' ' read -r -a prev_languages <<< "${permutations[$i-1]}"
            joined_prev_perm=$(IFS=-; echo "${prev_languages[*]}")
            model_dir_parent="$root_dir/$i-$joined_prev_perm/$model_dir_prefix"

            txt_vocab="$model_dir_parent${joined_prev_perm}/txt.vocab"
            load_model="$model_dir_parent${joined_prev_perm}/best.ckpt"
        fi

        echo "$index: ${joined_perm}"
        echo "txt_vocab: $txt_vocab, load_model: $load_model, model_dir: $model_dir"
     
        # Update the training section in the YAML file
        awk -v txt_vocab="$txt_vocab" -v load_model="$load_model" -v model_dir="$model_dir" -v memory_size="$memory_size" '
        {
            sub(/txt_vocab: .*/, "txt_vocab: \"" txt_vocab "\"");
            sub(/load_model: .*/, "load_model: \"" load_model "\"");
            sub(/model_dir: .*/, "model_dir: \"" model_dir "\"");
            sub(/memory_size: .*/, "memory_size: " memory_size);
            print $0
        }
        ' "${config_file}" > "${train_config_file}"
        # 
        # Update the languages field under the training section
        yq eval --inplace ".training.languages = []" "${train_config_file}" 
        for i in "${!languages[@]}"; do
            lang="${languages[$i]}"
            yq eval --inplace ".training.languages += [\"$lang\"]" "${train_config_file}"
        done
     
        # training
        srun \
            --container-image=/netscratch/enroot/nvcr.io_nvidia_pytorch_20.02-py3.sqsh \
            --container-workdir="$(pwd)" \
            --container-mounts=/netscratch:/netscratch,/ds:/ds:ro,"$(pwd)":"$(pwd)" \
            install.sh python -m signjoey train "$train_config_file"

        # Update the test section in the YAML file
        if (( index % 3 == 0 )); then
        awk -v txt_vocab="$txt_vocab" -v load_model="$load_model" -v model_dir="$model_dir" -v memory_size="$memory_size" \
            '{sub(/txt_vocab: .*/, "txt_vocab: \"" txt_vocab "\""); sub(/load_model: .*/, "load_model: \"" load_model "\""); sub(/model_dir: .*/, "model_dir: \"" model_dir "\""); sub(/memory_size: .*/, "memory_size: " memory_size);} 1' "$test_config_file" > "${spec_test_config_file}"

        # testing
        srun \
            --container-image=/netscratch/enroot/nvcr.io_nvidia_pytorch_20.02-py3.sqsh \
            --container-workdir="$(pwd)" \
            --container-mounts=/netscratch:/netscratch,/ds:/ds:ro,"$(pwd)":"$(pwd)" \
            install.sh python -m signjoey test "$spec_test_config_file"
        fi
     
        echo  # Print an empty line for separation
    done
done


# adapters finetuning
echo "adapter finetuning"
mkdir -p $HOME/bin

# Download yq to the local bin directory
wget https://github.com/mikefarah/yq/releases/latest/download/yq_linux_amd64 -O $HOME/bin/yq

# Make yq executable
chmod +x $HOME/bin/yq

# Add the local bin directory to the PATH
export PATH=$HOME/bin:$PATH

# Verify the installation
which yq

seed=44
feat_type="s3d"
permutations=(
    "ASE GSG CSL"
    "ASE CSL GSG"
    "GSG ASE CSL"
    "GSG CSL ASE"
    "CSL ASE GSG"
    "CSL GSG ASE"
)

root_dir='results/fine-tuning-adapters'
config_file="configs/fine-tuning-adapters-${feat_type}-seed-${seed}.yaml"
spec_test_config_file="configs/fine-tuning-adapters-${feat_type}-seed-${seed}-test-spec.yaml"
test_config_file="configs/fine-tuning-adapters-${feat_type}-seed-${seed}-test.yaml"
train_config_file="configs/fine-tuning-adapters-${feat_type}-seed-${seed}-train.yaml"

mkdir -p $root_dir

# Iterate over each permutation and print each language with index
first_ind_txt_vocab="results/multilingual-pretraining-${feat_type}-seed-${seed}/txt.vocab"
first_ind_load_model="results/multilingual-pretraining-${feat_type}-seed-${seed}/best.ckpt"
model_dir_prefix="finetuning-adapters-${feat_type}-seed-${seed}-"

# Loop through each permutation
for perm in "${permutations[@]}"; do
    IFS=' ' read -r -a languages <<< "$perm"
    joined_perm=$(IFS=_; echo "${languages[*]}")
    model_dir_parent="$root_dir/$joined_perm/$model_dir_prefix"

    # Print each language with its index
    languages_seen_so_far=()
    for i in "${!languages[@]}"; do
        index=$((i + 1))
        model_dir="$model_dir_parent${languages[$i]}"
        languages_seen_so_far+=("${languages[$i]}")

        if [ "$index" -eq 1 ]; then
            txt_vocab="$first_ind_txt_vocab"
            load_model="$first_ind_load_model"
        else
            txt_vocab="$model_dir_parent${languages[$i-1]}/txt.vocab"
            load_model="$model_dir_parent${languages[$i-1]}/best.ckpt"
        fi

        echo "$index: ${languages[$i]}"
        echo "txt_vocab: $txt_vocab, load_model: $load_model, model_dir: $model_dir"

        # Update the training section in the YAML file
        awk -v txt_vocab="$txt_vocab" -v load_model="$load_model" -v model_dir="$model_dir" '
        {
            sub(/txt_vocab: .*/, "txt_vocab: \"" txt_vocab "\"");
            sub(/load_model: .*/, "load_model: \"" load_model "\"");
            sub(/model_dir: .*/, "model_dir: \"" model_dir "\"");
            print $0
        }
        ' "${config_file}" > "${train_config_file}"
        
        # Update the languages field under the training section
        lang="${languages[$i]}"
        yq eval --inplace ".training.languages = []" "${train_config_file}" 
        yq eval --inplace ".training.languages += [\"$lang\"]" "${train_config_file}"

        # training
        srun \
            --container-image=/netscratch/enroot/nvcr.io_nvidia_pytorch_20.02-py3.sqsh \
            --container-workdir="$(pwd)" \
            --container-mounts=/netscratch:/netscratch,/ds:/ds:ro,"$(pwd)":"$(pwd)" \
            install.sh python -m signjoey train "$train_config_file"
        python -m signjoey train "$train_config_file"
    done

        # Update the test section in the YAML file (if needed)
        awk -v txt_vocab="$txt_vocab" -v load_model="$load_model" -v model_dir="$model_dir" \
            '{sub(/txt_vocab: .*/, "txt_vocab: \"" txt_vocab "\""); sub(/load_model: .*/, "load_model: \"" load_model "\""); sub(/model_dir: .*/, "model_dir: \"" model_dir "\"");} 1' "$test_config_file" > "${spec_test_config_file}"

        # update the adapter_languages filed under the training section with languages seen so far
        yq eval --inplace ".training.adapter_languages = []" "${spec_test_config_file}"
        for lang in "${languages_seen_so_far[@]}"; do
            yq eval --inplace ".training.adapter_languages += [\"$lang\"]" "${spec_test_config_file}"
        done

        for file in "$model_dir"/*.ckpt; do mv "$file" "${file}.adapter"; done
        cp 'results/multilingual-pretraining-s3d-seed-44/best.ckpt' $model_dir
        # copy the adapter_*.pt files from languages_seen_so_far to model_dir
        for lang in "${languages_seen_so_far[@]}"; do
            cp "$model_dir_parent${lang}/adapter_${lang}.pt" "$model_dir"
        done
        # testing
        srun \
            --container-image=/netscratch/enroot/nvcr.io_nvidia_pytorch_20.02-py3.sqsh \
            --container-workdir="$(pwd)" \
            --container-mounts=/netscratch:/netscratch,/ds:/ds:ro,"$(pwd)":"$(pwd)" \
            install.sh python -m signjoey test "$spec_test_config_file"
            python -m signjoey test "$spec_test_config_file"

        mv "$model_dir"/best.ckpt "$model_dir"/multi-pre-trained.best.ckpt
        # remove adapter suffix
        for file in "$model_dir"/*.adapter; do mv "$file" "${file%.adapter}"; done

        echo  # Print an empty line for separation
    done
done